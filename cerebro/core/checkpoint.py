# -*- coding: utf-8 -*-
"""
Checkpoints: un único archivo .npz que contiene TODO lo necesario para
reconstruir el modelo exactamente:

    p__<nombre>   pesos del modelo
    m__ / v__     estado del optimizador (opcional, solo para reanudar)
    meta          JSON: arquitectura, tokenizador completo, métricas, versión

Como la arquitectura y el tokenizador viajan dentro del checkpoint, un
modelo se carga correctamente aunque hayas cambiado config.py después de
entrenarlo. No se usa pickle (np.load con allow_pickle=False): cargar un
checkpoint ajeno no puede ejecutar código.

Los .npz se escriben de forma atómica (archivo temporal + os.replace), así
que una interrupción a mitad de guardado no deja un checkpoint corrupto.
"""

import datetime
import json
import os

import numpy as np

from core.tokenizer import tokenizador_desde_dict
from core.transformer.optimized_llm import ModeloConfig, OptimizedTransformerLLM

FORMATO = "cerebro-ckpt-v4"


def _jsonable(obj):
    """Convierte estados de NumPy RNG y escalares NumPy a JSON puro."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def guardar_checkpoint(ruta, modelo, optimizador=None, meta_extra=None):
    """Guarda modelo + tokenizador (+ optimizador) en `ruta` (.npz)."""
    if modelo.tokenizer is None:
        raise ValueError("El modelo no tiene tokenizador: no se puede guardar un checkpoint completo.")
    meta = {
        "formato": FORMATO,
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": modelo.cfg.to_dict(),
        "tokenizador": modelo.tokenizer.to_dict(),
        "n_parametros": modelo.n_parametros(),
    }
    if optimizador is not None:
        meta["optimizador_t"] = int(optimizador.t)
    if meta_extra:
        meta.update(meta_extra)

    arrays = {"meta": np.array(json.dumps(_jsonable(meta), ensure_ascii=False))}
    for nombre, valor in modelo.params.items():
        arrays["p__" + nombre] = valor
    if optimizador is not None:
        for nombre, valor in optimizador.m.items():
            arrays["m__" + nombre] = valor
        for nombre, valor in optimizador.v.items():
            arrays["v__" + nombre] = valor

    directorio = os.path.dirname(os.path.abspath(ruta))
    os.makedirs(directorio, exist_ok=True)
    tmp = ruta + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, ruta)


def leer_meta(ruta):
    """Lee solo los metadatos (rápido, sin cargar los pesos)."""
    with np.load(ruta, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def cargar_checkpoint(ruta, con_optimizador=True):
    """Devuelve dict(modelo, meta, opt_state). `opt_state` es None si el
    checkpoint no lo trae o si `con_optimizador` es False."""
    with np.load(ruta, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("formato") not in (FORMATO, "cerebro-ckpt-v3"):
            raise ValueError(
                f"Formato de checkpoint no soportado: {meta.get('formato')!r} "
                f"(se esperaba {FORMATO!r}). Los checkpoints JSON de la versión "
                f"anterior del proyecto no son compatibles: vuelve a entrenar.")
        cfg = ModeloConfig.from_dict(meta["config"])
        tok = tokenizador_desde_dict(meta["tokenizador"])
        modelo = OptimizedTransformerLLM(cfg, tokenizer=tok)
        estado = {k[3:]: z[k] for k in z.files if k.startswith("p__")}
        modelo.load_state_dict(estado)

        opt_state = None
        rng_state = meta.get("rng_state")
        if con_optimizador and any(k.startswith("m__") for k in z.files):
            opt_state = {
                "t": meta.get("optimizador_t", 0),
                "m": {k[3:]: z[k] for k in z.files if k.startswith("m__")},
                "v": {k[3:]: z[k] for k in z.files if k.startswith("v__")},
            }
    return {"modelo": modelo, "meta": meta, "opt_state": opt_state, "rng_state": rng_state}
