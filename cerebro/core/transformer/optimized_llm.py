# -*- coding: utf-8 -*-
"""
Cerebro: un transformer de lenguaje (estilo GPT) construido desde cero con
NumPy puro. Sin PyTorch, sin TensorFlow, sin autograd: la retropropagación
está derivada a mano con la regla de la cadena y verificada contra
diferencias finitas en las pruebas (tests/tests_final.py).

Arquitectura (pre-LayerNorm, como GPT-2):

    ids -> embedding de token + embedding de posición -> dropout
        -> N x [ LN -> atención multi-cabeza causal -> (+residual)
                 LN -> MLP con GELU                 -> (+residual) ]
        -> LN final -> proyección a vocabulario (pesos compartidos con el
           embedding, opcional) -> logits

Todo funciona por lotes: las entradas son (B, T) y los logits (B, T, V).

Puntos de extensión pensados para crecer:
  * `ModeloConfig` — añade campos (p. ej. n_kv_heads, rotary) sin romper
    checkpoints: los campos desconocidos se ignoran al cargar.
  * `_bloque_forward` / `_bloque_backward` — aquí vive un bloque; cambiar la
    atención o el MLP es tocar solo estas dos funciones.
  * `core/sampling.py` — estrategias de muestreo intercambiables.
"""

import math
from dataclasses import dataclass, asdict, fields

import numpy as np

from core.sampling import muestrear, softmax
from core.tokenizer import ID_PAD, ID_EOS, ID_BOS, ID_UNK

_C_GELU = math.sqrt(2.0 / math.pi)


# =============================================================================
#  Configuración del modelo
# =============================================================================

@dataclass
class ModeloConfig:
    vocab_size: int = 512
    block_size: int = 64          # ventana de contexto máxima
    dim: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dim_ffn: int = 256
    dropout: float = 0.0
    tie_weights: bool = True
    dtype: str = "float32"

    def validar(self):
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) debe ser múltiplo de n_heads ({self.n_heads}).")
        if min(self.vocab_size, self.block_size, self.dim, self.n_layers, self.dim_ffn) < 1:
            raise ValueError("Todas las dimensiones del modelo deben ser >= 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1).")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype debe ser 'float32' o 'float64'.")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        nombres = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in nombres}).validar()


# =============================================================================
#  Piezas numéricas (forward + su backward)
# =============================================================================

def _lineal(x, w, b=None):
    """x: (B, T, Din) @ w: (Din, Dout) [+ b]. Usa un único GEMM 2D (rápido)."""
    B, T, Din = x.shape
    y = x.reshape(B * T, Din) @ w
    if b is not None:
        y += b
    return y.reshape(B, T, w.shape[1])


def _lineal_backward(x, w, dy):
    """Gradientes de y = x @ w + b. Devuelve (dx, dw, db)."""
    B, T, Din = x.shape
    x2 = x.reshape(B * T, Din)
    dy2 = dy.reshape(B * T, dy.shape[-1])
    dw = x2.T @ dy2
    db = dy2.sum(axis=0)
    dx = (dy2 @ w.T).reshape(B, T, Din)
    return dx, dw, db


def _ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd)


def _ln_backward(dy, cache, g):
    xhat, rstd = cache
    dg = (dy * xhat).sum(axis=(0, 1))
    db = dy.sum(axis=(0, 1))
    dxhat = dy * g
    dx = rstd * (dxhat
                 - dxhat.mean(axis=-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dg, db


def _gelu_forward(x):
    """GELU (aproximación tanh de GPT-2). Devuelve (y, tanh_interno)."""
    u = _C_GELU * (x + 0.044715 * x * x * x)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), t


def _gelu_backward(dy, x, t):
    du = _C_GELU * (1.0 + 3.0 * 0.044715 * x * x)
    dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
    return dy * dgelu


def _dividir_cabezas(x, B, T, H, dh):
    """(B, T, D) -> (B, H, T, dh)"""
    return x.reshape(B, T, H, dh).transpose(0, 2, 1, 3)


def _unir_cabezas(x, B, T, D):
    """(B, H, T, dh) -> (B, T, D)"""
    return x.transpose(0, 2, 1, 3).reshape(B, T, D)


# =============================================================================
#  El modelo
# =============================================================================

class OptimizedTransformerLLM:
    def __init__(self, config, tokenizer=None, seed=42):
        self.cfg = config.validar()
        self.tokenizer = tokenizer
        self.dtype = np.dtype(self.cfg.dtype)
        self.params = {}
        self._mascaras = {}
        self.rng = np.random.default_rng(seed)   # solo para dropout
        self._inicializar(seed)

    # -- inicialización ---------------------------------------------------------

    def _inicializar(self, seed):
        c = self.cfg
        rng = np.random.default_rng(seed)
        D, F, V = c.dim, c.dim_ffn, c.vocab_size
        std = 0.02
        std_res = std / math.sqrt(2 * c.n_layers)   # proyecciones residuales, como GPT-2

        def normal(forma, s):
            return (rng.standard_normal(forma) * s).astype(self.dtype)

        def ceros(forma):
            return np.zeros(forma, dtype=self.dtype)

        def unos(forma):
            return np.ones(forma, dtype=self.dtype)

        p = self.params
        p["tok_emb"] = normal((V, D), std)
        p["pos_emb"] = normal((c.block_size, D), std)
        for i in range(c.n_layers):
            l = f"l{i}."
            p[l + "ln1_g"], p[l + "ln1_b"] = unos(D), ceros(D)
            p[l + "wqkv"], p[l + "bqkv"] = normal((D, 3 * D), std), ceros(3 * D)
            p[l + "wo"], p[l + "bo"] = normal((D, D), std_res), ceros(D)
            p[l + "ln2_g"], p[l + "ln2_b"] = unos(D), ceros(D)
            p[l + "w1"], p[l + "b1"] = normal((D, F), std), ceros(F)
            p[l + "w2"], p[l + "b2"] = normal((F, D), std_res), ceros(D)
        p["lnf_g"], p["lnf_b"] = unos(D), ceros(D)
        if not c.tie_weights:
            p["w_out"] = normal((D, V), std)
        p["b_out"] = ceros(V)

    def n_parametros(self):
        return int(sum(v.size for v in self.params.values()))

    def _mascara_causal(self, T):
        m = self._mascaras.get(T)
        if m is None:
            m = np.triu(np.full((T, T), -1e9, dtype=self.dtype), k=1)
            self._mascaras[T] = m
        return m

    def _dropout(self, x, entrenando):
        p = self.cfg.dropout
        if not entrenando or p <= 0.0:
            return x, None
        mascara = (self.rng.random(x.shape) >= p).astype(self.dtype) * (1.0 / (1.0 - p))
        return x * mascara, mascara

    # -- un bloque transformer: forward ---------------------------------------------

    def _bloque_forward(self, i, x, entrenando):
        c, p = self.cfg, self.params
        l = f"l{i}."
        B, T, D = x.shape
        H = c.n_heads
        dh = D // H
        escala = 1.0 / math.sqrt(dh)

        # --- sub-bloque de atención ---
        h1, ln1 = _ln_forward(x, p[l + "ln1_g"], p[l + "ln1_b"])
        qkv = _lineal(h1, p[l + "wqkv"], p[l + "bqkv"])
        q, k, v = np.split(qkv, 3, axis=-1)
        q = _dividir_cabezas(q, B, T, H, dh)
        k = _dividir_cabezas(k, B, T, H, dh)
        v = _dividir_cabezas(v, B, T, H, dh)

        scores = (q @ k.transpose(0, 1, 3, 2)) * escala + self._mascara_causal(T)
        att = softmax(scores)                                   # (B, H, T, T)
        ctx = _unir_cabezas(att @ v, B, T, D)                    # (B, T, D)
        a = _lineal(ctx, p[l + "wo"], p[l + "bo"])
        a, mask_a = self._dropout(a, entrenando)
        x1 = x + a

        # --- sub-bloque MLP ---
        h2, ln2 = _ln_forward(x1, p[l + "ln2_g"], p[l + "ln2_b"])
        f_pre = _lineal(h2, p[l + "w1"], p[l + "b1"])
        f, gelu_t = _gelu_forward(f_pre)
        m = _lineal(f, p[l + "w2"], p[l + "b2"])
        m, mask_m = self._dropout(m, entrenando)
        x2 = x1 + m

        cache = dict(h1=h1, ln1=ln1, q=q, k=k, v=v, att=att, ctx=ctx,
                     mask_a=mask_a, h2=h2, ln2=ln2, f_pre=f_pre, gelu_t=gelu_t,
                     f=f, mask_m=mask_m)
        return x2, cache

    # -- un bloque transformer: backward --------------------------------------------

    def _bloque_backward(self, i, dx2, c_, grads):
        c, p = self.cfg, self.params
        l = f"l{i}."
        B, T, D = dx2.shape
        H = c.n_heads
        dh = D // H
        escala = 1.0 / math.sqrt(dh)

        # --- MLP ---
        dm = dx2 if c_["mask_m"] is None else dx2 * c_["mask_m"]
        df, grads[l + "w2"], grads[l + "b2"] = _lineal_backward(c_["f"], p[l + "w2"], dm)
        df_pre = _gelu_backward(df, c_["f_pre"], c_["gelu_t"])
        dh2, grads[l + "w1"], grads[l + "b1"] = _lineal_backward(c_["h2"], p[l + "w1"], df_pre)
        dx1_ln, grads[l + "ln2_g"], grads[l + "ln2_b"] = _ln_backward(dh2, c_["ln2"], p[l + "ln2_g"])
        dx1 = dx2 + dx1_ln

        # --- atención ---
        da = dx1 if c_["mask_a"] is None else dx1 * c_["mask_a"]
        dctx2, grads[l + "wo"], grads[l + "bo"] = _lineal_backward(c_["ctx"], p[l + "wo"], da)
        dctx = _dividir_cabezas(dctx2, B, T, H, dh)              # (B, H, T, dh)

        att, q, k, v = c_["att"], c_["q"], c_["k"], c_["v"]
        datt = dctx @ v.transpose(0, 1, 3, 2)                    # (B, H, T, T)
        dv = att.transpose(0, 1, 3, 2) @ dctx                    # (B, H, T, dh)
        # Jacobiano del softmax fila a fila (las posiciones enmascaradas tienen
        # att = 0, así que su gradiente es 0 automáticamente).
        dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True)) * escala
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q

        dqkv = np.concatenate(
            [_unir_cabezas(dq, B, T, D), _unir_cabezas(dk, B, T, D),
             _unir_cabezas(dv, B, T, D)], axis=-1)
        dh1, grads[l + "wqkv"], grads[l + "bqkv"] = _lineal_backward(c_["h1"], p[l + "wqkv"], dqkv)
        dx0_ln, grads[l + "ln1_g"], grads[l + "ln1_b"] = _ln_backward(dh1, c_["ln1"], p[l + "ln1_g"])
        return dx1 + dx0_ln

    # -- forward completo -----------------------------------------------------------

    def forward(self, ids, cache=None, entrenando=False):
        """ids: (T,) o (B, T). Devuelve logits (B, T, V). Si `cache` es un
        dict vacío se rellena con lo necesario para `backward`."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        B, T = ids.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Secuencia de {T} tokens excede block_size={self.cfg.block_size}.")
        p = self.params

        x = p["tok_emb"][ids] + p["pos_emb"][:T]
        x, mask_emb = self._dropout(x, entrenando)

        caches = []
        for i in range(self.cfg.n_layers):
            x, c_ = self._bloque_forward(i, x, entrenando)
            caches.append(c_)

        xf, lnf = _ln_forward(x, p["lnf_g"], p["lnf_b"])
        if self.cfg.tie_weights:
            logits = _lineal(xf, p["tok_emb"].T, p["b_out"])
        else:
            logits = _lineal(xf, p["w_out"], p["b_out"])

        if cache is not None:
            cache.update(ids=ids, mask_emb=mask_emb, caches=caches, lnf=lnf, xf=xf)
        return logits

    # kept for API familiarity with earlier versions of the project
    forward_rapido = forward

    # -- pérdida ---------------------------------------------------------------------

    @staticmethod
    def perdida(logits, targets):
        """Entropía cruzada media. `targets` (B, T) o (T,); -1 = ignorar
        (relleno). Devuelve (loss, probs (B*T, V), n_validos)."""
        V = logits.shape[-1]
        lg = logits.reshape(-1, V)
        tg = np.asarray(targets, dtype=np.int64).reshape(-1)
        validos = np.nonzero(tg >= 0)[0]
        n = len(validos)
        m = lg.max(axis=-1, keepdims=True)
        e = np.exp(lg - m)
        s = e.sum(axis=-1, keepdims=True)
        logp = (lg - m) - np.log(s)
        nll = -logp[validos, tg[validos]]
        loss = float(nll.sum(dtype=np.float64) / max(n, 1))
        return loss, e / s, n

    # alias con el nombre que usaba la versión anterior
    def perdida_entropia_cruzada(self, logits, targets):
        loss, probs, _ = self.perdida(logits, targets)
        return loss, probs

    # -- backward ----------------------------------------------------------------------

    def backward(self, cache, targets, probs=None, n_validos=None):
        """Gradientes EXACTOS de la pérdida respecto a cada parámetro."""
        p, c = self.params, self.cfg
        ids = cache["ids"]
        B, T = ids.shape
        V = c.vocab_size
        tg = np.asarray(targets, dtype=np.int64).reshape(-1)
        validos = np.nonzero(tg >= 0)[0]
        n = max(len(validos), 1)

        if probs is None:
            _, probs, _ = self.perdida(cache["logits"], targets)
        dlogits = probs.copy()
        dlogits[validos, tg[validos]] -= 1.0
        mascara_invalidos = np.ones(B * T, dtype=bool)
        mascara_invalidos[validos] = False
        dlogits[mascara_invalidos] = 0.0
        dlogits /= n
        dlogits = dlogits.reshape(B, T, V).astype(self.dtype, copy=False)

        grads = {}
        xf = cache["xf"]
        if c.tie_weights:
            dxf, dw_t, grads["b_out"] = _lineal_backward(xf, p["tok_emb"].T, dlogits)
            grads["tok_emb"] = dw_t.T.copy()       # aporte de la capa de salida
        else:
            dxf, grads["w_out"], grads["b_out"] = _lineal_backward(xf, p["w_out"], dlogits)

        dx, grads["lnf_g"], grads["lnf_b"] = _ln_backward(dxf, cache["lnf"], p["lnf_g"])

        for i in reversed(range(c.n_layers)):
            dx = self._bloque_backward(i, dx, cache["caches"][i], grads)

        if cache["mask_emb"] is not None:
            dx = dx * cache["mask_emb"]

        grads["pos_emb"] = np.zeros_like(p["pos_emb"])
        grads["pos_emb"][:T] = dx.sum(axis=0)
        demb = np.zeros_like(p["tok_emb"])
        np.add.at(demb, ids, dx)
        if "tok_emb" in grads:
            grads["tok_emb"] += demb
        else:
            grads["tok_emb"] = demb
        return grads

    def perdida_y_gradientes(self, ids, targets, entrenando=True):
        """Forward + pérdida + backward en un solo paso. (loss, grads)."""
        cache = {}
        logits = self.forward(ids, cache=cache, entrenando=entrenando)
        cache["logits"] = logits
        loss, probs, n = self.perdida(logits, targets)
        grads = self.backward(cache, targets, probs=probs, n_validos=n)
        return loss, grads

    # -- generación --------------------------------------------------------------------

    def generar_ids(self, ids_prompt, max_nuevos=60, temperatura=0.7, top_k=40,
                    top_p=0.92, penalizacion=1.15, parar_en=("\n",), rng=None,
                    seed=None, ventana_penalizacion=64):
        """Genera ids de forma autoregresiva a partir de `ids_prompt`.
        Devuelve (ids_nuevos, motivo_parada)."""
        if rng is None:
            rng = np.random.default_rng(seed)
        ids = list(ids_prompt)
        if not ids:
            ids = [ID_EOS]
        nuevos = []
        prohibidos = [ID_PAD, ID_BOS, ID_UNK]
        motivo = "max_nuevos"

        for _ in range(max_nuevos):
            contexto = ids[-self.cfg.block_size:]
            logits = self.forward(contexto)[0, -1]
            siguiente = muestrear(
                logits, rng, temperatura=temperatura, top_k=top_k, top_p=top_p,
                penalizacion=penalizacion, ids_previos=ids[-ventana_penalizacion:],
                ids_prohibidos=prohibidos)
            if siguiente == ID_EOS:
                motivo = "eos"
                break
            ids.append(siguiente)
            nuevos.append(siguiente)
            if parar_en and self.tokenizer is not None:
                texto = self.tokenizer.decodificar(nuevos)
                if any(s in texto for s in parar_en):
                    motivo = "parada"
                    break
        return nuevos, motivo

    def generar(self, prompt, max_nuevos=60, temperatura=0.7, top_k=40, top_p=0.92,
                penalizacion=1.15, parar_en=None, seed=None, incluir_prompt=True):
        """Genera texto a partir de un prompt (str). Antepone <EOS> como
        marca de inicio de documento, igual que en el entrenamiento."""
        if self.tokenizer is None:
            raise RuntimeError("El modelo no tiene tokenizador asociado.")
        ids_prompt = [ID_EOS] + self.tokenizer.codificar(prompt)
        nuevos, _ = self.generar_ids(
            ids_prompt, max_nuevos=max_nuevos, temperatura=temperatura,
            top_k=top_k, top_p=top_p, penalizacion=penalizacion,
            parar_en=parar_en, seed=seed)
        texto = self.tokenizer.decodificar(nuevos)
        if parar_en:
            cortes = [texto.index(s) for s in parar_en if s in texto]
            if cortes:
                texto = texto[:min(cortes)]
        return (prompt + texto) if incluir_prompt else texto

    # -- persistencia del estado (los checkpoints en disco viven en core/checkpoint.py) ---

    def state_dict(self):
        return {n: v.copy() for n, v in self.params.items()}

    def load_state_dict(self, estado):
        faltan = set(self.params) - set(estado)
        sobran = set(estado) - set(self.params)
        if faltan or sobran:
            raise ValueError(
                f"El estado no coincide con la arquitectura. Faltan: {sorted(faltan)}; "
                f"sobran: {sorted(sobran)}")
        for n, v in estado.items():
            v = np.asarray(v, dtype=self.dtype)
            if v.shape != self.params[n].shape:
                raise ValueError(f"Forma incorrecta en '{n}': {v.shape} vs {self.params[n].shape}")
            self.params[n][...] = v
