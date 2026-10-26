# -*- coding: utf-8 -*-
"""
Tokenizadores construidos desde cero (solo librería estándar: `re`, `json`,
`collections`). Hay dos, con la misma interfaz:

  * TokenizadorBPE       (recomendado) — Byte-Pair Encoding a nivel de bytes.
        Parte de los 256 bytes posibles, así que CUALQUIER texto (tildes, ñ,
        emojis, código) se puede codificar y decodificar sin pérdida: no
        existe el token "desconocido". Aprende del corpus las secuencias de
        bytes más frecuentes (sílabas, palabras comunes) y las fusiona en
        tokens nuevos hasta llegar al tamaño de vocabulario pedido.

  * TokenizadorPalabras  — el tokenizador original del proyecto: una palabra
        = un token, con <UNK> para lo que no cabe en el vocabulario. Se
        conserva para experimentos y comparaciones (no sirve para el chat,
        porque no conserva los saltos de línea).

Interfaz común (lo único que el resto del proyecto necesita):
    entrenar(textos)            aprende el vocabulario
    codificar(texto) -> ids
    decodificar(ids) -> texto
    bytes_de(ids)    -> nº de bytes de texto que representan esos ids
    to_dict() / from_dict(d)    persistencia (va dentro del checkpoint)
    len(tokenizador)            tamaño real del vocabulario

Ids reservados (iguales en ambos): 0=<PAD>, 1=<UNK>, 2=<BOS>, 3=<EOS>.
"""

import json
import os
import re
from collections import Counter

ESPECIALES = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
ID_PAD, ID_UNK, ID_BOS, ID_EOS = 0, 1, 2, 3
N_ESPECIALES = len(ESPECIALES)


# =============================================================================
#  BPE a nivel de bytes
# =============================================================================

# Pre-tokenización: separa el texto en "piezas" (palabra con su espacio
# inicial, número, símbolos, espacios/saltos de línea). BPE nunca fusiona a
# través de estas fronteras, lo que mantiene los tokens con sentido.
_PATRON_BPE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")


class TokenizadorBPE:
    """BPE byte-level. ids: 0..3 especiales, 4..259 bytes, 260.. fusiones."""

    tipo = "bpe"

    def __init__(self, vocab_size=512):
        if vocab_size < N_ESPECIALES + 256:
            raise ValueError(
                f"vocab_size debe ser al menos {N_ESPECIALES + 256} "
                f"(especiales + 256 bytes)."
            )
        self.vocab_size = vocab_size
        self._reiniciar()

    def _reiniciar(self):
        # id -> bytes que representa. Los especiales no producen texto.
        self.vocab = {i: b"" for i in range(N_ESPECIALES)}
        for b in range(256):
            self.vocab[N_ESPECIALES + b] = bytes([b])
        self.fusiones = []          # lista de (id_a, id_b), en orden de aprendizaje
        self._rangos = {}           # (id_a, id_b) -> prioridad (menor = antes)
        self._cache = {}            # pieza(str) -> tuple(ids)

    # -- entrenamiento ---------------------------------------------------------

    def entrenar(self, textos, verbose=False):
        """Aprende las fusiones más frecuentes del corpus. Determinista: ante
        un empate gana el par de ids más pequeño."""
        self._reiniciar()
        conteo = Counter()
        for texto in textos:
            for pieza in _PATRON_BPE.findall(texto):
                conteo[pieza] += 1

        palabras = [
            [[N_ESPECIALES + b for b in pieza.encode("utf-8")], frecuencia]
            for pieza, frecuencia in conteo.items()
        ]

        while len(self.vocab) < self.vocab_size:
            pares = Counter()
            for ids, frecuencia in palabras:
                for par in zip(ids, ids[1:]):
                    pares[par] += frecuencia
            if not pares:
                break
            (a, b), veces = max(pares.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if veces < 2:
                break  # fusionar algo que solo aparece una vez no ayuda a generalizar
            nuevo = len(self.vocab)
            self.vocab[nuevo] = self.vocab[a] + self.vocab[b]
            self.fusiones.append((a, b))
            for entrada in palabras:
                entrada[0] = self._aplicar_fusion(entrada[0], (a, b), nuevo)
            if verbose and len(self.fusiones) % 50 == 0:
                print(f"  BPE: {len(self.vocab)}/{self.vocab_size} tokens")

        self._rangos = {par: i for i, par in enumerate(self.fusiones)}
        self._cache = {}
        return self

    @staticmethod
    def _aplicar_fusion(ids, par, nuevo):
        a, b = par
        salida, i, n = [], 0, len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                salida.append(nuevo)
                i += 2
            else:
                salida.append(ids[i])
                i += 1
        return salida

    # -- codificación / decodificación ------------------------------------------

    def _codificar_pieza(self, pieza):
        cacheado = self._cache.get(pieza)
        if cacheado is not None:
            return cacheado
        ids = [N_ESPECIALES + b for b in pieza.encode("utf-8")]
        while len(ids) >= 2:
            mejor, mejor_rango = None, None
            for par in zip(ids, ids[1:]):
                rango = self._rangos.get(par)
                if rango is not None and (mejor_rango is None or rango < mejor_rango):
                    mejor, mejor_rango = par, rango
            if mejor is None:
                break
            ids = self._aplicar_fusion(ids, mejor, N_ESPECIALES + 256 + mejor_rango)
        resultado = tuple(ids)
        self._cache[pieza] = resultado
        return resultado

    def codificar(self, texto):
        ids = []
        for pieza in _PATRON_BPE.findall(texto):
            ids.extend(self._codificar_pieza(pieza))
        return ids

    def decodificar(self, ids):
        crudo = b"".join(self.vocab.get(int(i), b"") for i in ids)
        return crudo.decode("utf-8", errors="replace")

    def bytes_de(self, ids):
        return sum(len(self.vocab.get(int(i), b"")) for i in ids)

    # -- persistencia -----------------------------------------------------------

    def to_dict(self):
        return {
            "tipo": self.tipo,
            "vocab_size": self.vocab_size,
            "fusiones": [list(par) for par in self.fusiones],
        }

    @classmethod
    def from_dict(cls, d):
        tok = cls(d["vocab_size"])
        for a, b in d["fusiones"]:
            nuevo = len(tok.vocab)
            tok.vocab[nuevo] = tok.vocab[a] + tok.vocab[b]
            tok.fusiones.append((a, b))
        tok._rangos = {tuple(par): i for i, par in enumerate(tok.fusiones)}
        return tok

    def guardar(self, ruta):
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __len__(self):
        return len(self.vocab)


# =============================================================================
#  Tokenizador por palabras (el original del proyecto, mejorado)
# =============================================================================

_PATRON_PALABRA = re.compile(r"[a-zA-ZáéíóúñÁÉÍÓÚÑüÜ0-9]+|[.,!?;:¿¡]")


class TokenizadorPalabras:
    """Una palabra (o signo de puntuación) = un token. Vocabulario finito:
    al entrenar se quedan las `max_vocab_size` palabras MÁS FRECUENTES; el
    resto se codifica como <UNK>."""

    tipo = "palabras"

    def __init__(self, max_vocab_size=250):
        if max_vocab_size < N_ESPECIALES:
            raise ValueError(
                f"max_vocab_size debe ser al menos {N_ESPECIALES} "
                f"(para los tokens especiales)."
            )
        self.max_vocab_size = max_vocab_size
        self._reiniciar()

    def _reiniciar(self):
        self.stoi = {t: i for i, t in enumerate(ESPECIALES)}
        self.itos = {i: t for t, i in self.stoi.items()}

    @staticmethod
    def _partir(texto):
        return _PATRON_PALABRA.findall(texto.lower())

    def entrenar(self, textos, verbose=False):
        self._reiniciar()
        conteo = Counter()
        for texto in textos:
            conteo.update(self._partir(texto))
        for palabra, _ in sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(self.stoi) >= self.max_vocab_size:
                break
            self.stoi[palabra] = len(self.stoi)
            self.itos[self.stoi[palabra]] = palabra
        return self

    def codificar(self, texto):
        return [self.stoi.get(p, ID_UNK) for p in self._partir(texto)]

    def decodificar(self, ids):
        palabras = [self.itos.get(int(i), "<UNK>") for i in ids if int(i) != ID_PAD]
        palabras = [p for p in palabras if p not in ("<BOS>", "<EOS>")]
        texto = " ".join(palabras)
        return re.sub(r" ([.,!?;:])", r"\1", texto)

    def bytes_de(self, ids):
        # Aproximación: caracteres de la palabra + 1 espacio de separación.
        return sum(len(self.itos.get(int(i), "")) + 1
                   for i in ids if int(i) >= N_ESPECIALES)

    def to_dict(self):
        return {"tipo": self.tipo, "max_vocab_size": self.max_vocab_size,
                "stoi": self.stoi}

    @classmethod
    def from_dict(cls, d):
        tok = cls(d["max_vocab_size"])
        tok.stoi = dict(d["stoi"])
        tok.itos = {i: t for t, i in tok.stoi.items()}
        return tok

    def guardar(self, ruta):
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __len__(self):
        return len(self.stoi)


# Compatibilidad con el nombre que usaba la versión anterior del proyecto.
TokenizadorUnificado = TokenizadorPalabras


# =============================================================================
#  Fábrica (punto de extensión: añade aquí tu propio tokenizador)
# =============================================================================

TOKENIZADORES = {"bpe": TokenizadorBPE, "palabras": TokenizadorPalabras}


def crear_tokenizador(tipo, tamano):
    if tipo not in TOKENIZADORES:
        raise ValueError(f"Tokenizador desconocido '{tipo}'. Opciones: {list(TOKENIZADORES)}")
    return TOKENIZADORES[tipo](tamano)


def tokenizador_desde_dict(d):
    tipo = d.get("tipo")
    if tipo not in TOKENIZADORES:
        raise ValueError(f"Tipo de tokenizador desconocido en el checkpoint: {tipo!r}")
    return TOKENIZADORES[tipo].from_dict(d)
