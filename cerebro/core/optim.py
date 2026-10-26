# -*- coding: utf-8 -*-
"""
Optimizadores y utilidades de entrenamiento, escritos a mano en NumPy.

  * AdamW  — Adam (Kingma & Ba, 2015) con decaimiento de pesos DESACOPLADO
             (Loshchilov & Hutter, 2019): el decaimiento se aplica
             directamente a los pesos, no pasa por los momentos.
  * SGD    — descenso de gradiente con momentum (simple y fácil de razonar).
  * recortar_norma_global — limita la norma total del gradiente.
  * lr_coseno_con_calentamiento — calentamiento lineal + decaimiento coseno.

Los optimizadores actualizan los arrays del modelo IN PLACE.
"""

import math

import numpy as np


def recortar_norma_global(grads, max_norm):
    """Si la norma L2 conjunta de todos los gradientes supera `max_norm`,
    los reescala proporcionalmente. Devuelve la norma original."""
    total = 0.0
    for g in grads.values():
        total += float(np.sum(g.astype(np.float64) ** 2))
    norma = math.sqrt(total)
    if max_norm and norma > max_norm:
        factor = max_norm / (norma + 1e-12)
        for nombre in grads:
            grads[nombre] = grads[nombre] * factor
    return norma


def lr_coseno_con_calentamiento(paso, total, lr_max, warmup_frac=0.05, min_factor=0.1):
    """Learning rate en el paso `paso` (0-indexado) de `total` pasos."""
    warmup = max(1, int(total * warmup_frac))
    if paso < warmup:
        return lr_max * (paso + 1) / warmup
    progreso = (paso - warmup) / max(1, total - warmup)
    progreso = min(max(progreso, 0.0), 1.0)
    coseno = 0.5 * (1.0 + math.cos(math.pi * progreso))
    return lr_max * (min_factor + (1.0 - min_factor) * coseno)


class AdamW:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8,
                 weight_decay=0.0):
        self.params = params            # dict nombre -> ndarray (se modifican in place)
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {n: np.zeros_like(p) for n, p in params.items()}
        self.v = {n: np.zeros_like(p) for n, p in params.items()}

    @staticmethod
    def _decae(nombre, p):
        # Solo las matrices de pesos (no sesgos ni LayerNorm) sufren decaimiento.
        return p.ndim >= 2

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        b1, b2, t = self.beta1, self.beta2, self.t
        correccion1 = 1.0 - b1 ** t
        correccion2 = 1.0 - b2 ** t
        for nombre, g in grads.items():
            p = self.params[nombre]
            m, v = self.m[nombre], self.v[nombre]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            if self.weight_decay and self._decae(nombre, p):
                p *= (1.0 - lr * self.weight_decay)
            p -= lr * (m / correccion1) / (np.sqrt(v / correccion2) + self.eps)

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, estado):
        self.t = int(estado["t"])
        for n in self.params:
            if n in estado["m"]:
                self.m[n] = np.array(estado["m"][n], dtype=self.params[n].dtype)
                self.v[n] = np.array(estado["v"][n], dtype=self.params[n].dtype)


class SGD:
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.t = 0
        self.m = {n: np.zeros_like(p) for n, p in params.items()}
        self.v = {}  # no se usa; existe para compartir formato de checkpoint

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for nombre, g in grads.items():
            p = self.params[nombre]
            if self.weight_decay and p.ndim >= 2:
                p *= (1.0 - lr * self.weight_decay)
            self.m[nombre] *= self.momentum
            self.m[nombre] += g
            p -= lr * self.m[nombre]

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": {}}

    def load_state_dict(self, estado):
        self.t = int(estado["t"])
        for n in self.params:
            if n in estado["m"]:
                self.m[n] = np.array(estado["m"][n], dtype=self.params[n].dtype)


OPTIMIZADORES = {"adamw": AdamW, "sgd": SGD}


def crear_optimizador(nombre, params, lr, weight_decay=0.0, beta1=0.9,
                      beta2=0.99, eps=1e-8, momentum=0.9):
    if nombre == "adamw":
        return AdamW(params, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                     weight_decay=weight_decay)
    if nombre == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Optimizador desconocido '{nombre}'. Opciones: {list(OPTIMIZADORES)}")
