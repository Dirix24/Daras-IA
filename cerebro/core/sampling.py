# -*- coding: utf-8 -*-
"""
Estrategias de muestreo para elegir el siguiente token a partir de los
logits del modelo. Todo en NumPy puro.

Orden de aplicación (el estándar):
    penalización de repetición -> temperatura -> top-k -> top-p -> muestreo
"""

import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def muestrear(logits, rng, temperatura=1.0, top_k=None, top_p=None,
              penalizacion=1.0, ids_previos=None, ids_prohibidos=None):
    """Devuelve el id del siguiente token.

    logits          vector (V,) con los logits del último paso.
    rng             np.random.Generator (así la generación es reproducible).
    temperatura     <1 más conservador, >1 más creativo, 0 = voraz (argmax).
    top_k           solo se considera los k tokens más probables.
    top_p           solo se considera el núcleo de tokens cuya probabilidad
                    acumulada llega a p (nucleus sampling).
    penalizacion    >1 desalienta repetir tokens ya usados (estilo CTRL).
    ids_previos     tokens recientes sobre los que se aplica la penalización.
    ids_prohibidos  tokens que nunca se deben elegir (p. ej. <PAD>).
    """
    logits = np.array(logits, dtype=np.float64, copy=True)

    if ids_prohibidos:
        logits[list(ids_prohibidos)] = -np.inf

    if penalizacion and penalizacion != 1.0 and ids_previos is not None and len(ids_previos):
        vistos = np.unique(np.asarray(ids_previos, dtype=np.int64))
        vistos = vistos[np.isfinite(logits[vistos])]
        valores = logits[vistos]
        logits[vistos] = np.where(valores > 0, valores / penalizacion, valores * penalizacion)

    if temperatura is None or temperatura <= 1e-6:
        return int(np.argmax(logits))
    logits = logits / temperatura

    if top_k is not None and 0 < top_k < logits.size:
        umbral = np.partition(logits, -top_k)[-top_k]
        logits[logits < umbral] = -np.inf

    probs = softmax(logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        orden = np.argsort(-probs)
        acumulada = np.cumsum(probs[orden])
        # Se conservan los tokens hasta cruzar p (siempre al menos uno).
        corte = int(np.searchsorted(acumulada, top_p) + 1)
        mantener = orden[:corte]
        filtrado = np.zeros_like(probs)
        filtrado[mantener] = probs[mantener]
        probs = filtrado / filtrado.sum()

    return int(rng.choice(probs.size, p=probs))
