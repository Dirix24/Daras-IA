# -*- coding: utf-8 -*-
"""
Datos: carga del corpus, división train/validación y generación de lotes.

Formato del corpus (texto plano UTF-8):
  * Un DOCUMENTO es un bloque de líneas separado de los demás por una línea
    en blanco. Puede ser una frase, un párrafo o una conversación:

        usuario: hola
        cerebro: hola, ¿en qué puedo ayudarte?

        los transformadores usan atención para relacionar palabras.

  * Puedes añadir todos los archivos .txt que quieras en data/corpus/ :
    se cargan automáticamente junto con data/corpus.txt.

  * VALIDACIÓN: si existe data/validacion.txt, esos documentos se usan SOLO
    para medir (nunca se entrena con ellos). Si no existe, se aparta un 10%
    de los documentos al azar (de forma determinista).

Cada documento se codifica y se le añade el token <EOS> al final; todo se
concatena en un único "stream" de tokens del que se muestrean ventanas
aleatorias para entrenar (así no se desperdicia nada en relleno).
"""

import os
import re

import numpy as np

import config
from core.tokenizer import ID_EOS

SEED_DIVISION = 0   # fijo: todas las corridas comparten la misma validación


def documentos_desde_texto(texto):
    bloques = re.split(r"\n\s*\n", texto.replace("\r\n", "\n"))
    docs = []
    for b in bloques:
        b = "\n".join(linea.rstrip() for linea in b.strip().split("\n"))
        if b:
            docs.append(b)
    return docs


def _leer(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return documentos_desde_texto(f.read())


def cargar_documentos(ruta=None):
    """Carga documentos. Sin `ruta`: data/corpus.txt + data/corpus/*.txt.
    Con `ruta`: solo ese archivo (o todos los .txt de ese directorio)."""
    docs = []
    if ruta:
        if os.path.isdir(ruta):
            for nombre in sorted(os.listdir(ruta)):
                if nombre.endswith(".txt"):
                    docs.extend(_leer(os.path.join(ruta, nombre)))
        elif os.path.exists(ruta):
            docs.extend(_leer(ruta))
        else:
            raise FileNotFoundError(f"No existe el corpus: {ruta}")
    else:
        if os.path.exists(config.CORPUS_FILE):
            docs.extend(_leer(config.CORPUS_FILE))
        if os.path.isdir(config.CORPUS_DIR):
            for nombre in sorted(os.listdir(config.CORPUS_DIR)):
                if nombre.endswith(".txt"):
                    docs.extend(_leer(os.path.join(config.CORPUS_DIR, nombre)))
    if not docs:
        raise FileNotFoundError(
            "No hay corpus: añade texto a data/corpus.txt o archivos .txt en data/corpus/")
    if len(docs) < 5:
        # Corpus con pocas "líneas en blanco": se toma cada línea como documento.
        lineas = [l.strip() for d in docs for l in d.split("\n") if l.strip()]
        if len(lineas) > len(docs):
            docs = lineas
    return docs


def dividir_documentos(docs, frac_val=0.1, seed=SEED_DIVISION):
    """División train/val por DOCUMENTO, con barajado determinista."""
    if len(docs) < 2:
        return list(docs), list(docs)   # imposible separar: se valida sobre lo mismo
    orden = np.random.default_rng(seed).permutation(len(docs))
    n_val = min(max(1, int(round(len(docs) * frac_val))), len(docs) - 1)
    val = [docs[i] for i in orden[:n_val]]
    train = [docs[i] for i in orden[n_val:]]
    return train, val


def cargar_validacion():
    """Documentos de data/validacion.txt, o None si el archivo no existe."""
    if os.path.exists(config.VALIDACION_FILE):
        docs = _leer(config.VALIDACION_FILE)
        return docs or None
    return None


def cargar_test():
    """Conjunto de prueba final: nunca se usa para elegir checkpoints."""
    if os.path.exists(config.TEST_FILE):
        return _leer(config.TEST_FILE) or None
    return None


def cargar_generalizacion():
    """Diálogos totalmente nuevos (data/generalizacion.txt) para medir cuánto
    conocimiento NUEVO tiene el modelo. Nunca se usan para entrenar."""
    if os.path.exists(config.GENERALIZACION_FILE):
        return _leer(config.GENERALIZACION_FILE) or None
    return None


def cargar_datos(ruta=None, frac_val=0.1):
    """Devuelve (docs_train, docs_val, origen_val).

    Si hay un archivo de validación explícito (data/validacion.txt) y no se
    pidió un corpus alternativo, se usa; en otro caso se aparta una fracción
    al azar. Un documento de validación jamás aparece en el entrenamiento."""
    docs = cargar_documentos(ruta)
    explicita = cargar_validacion() if ruta is None else None
    if explicita:
        vistos = set(explicita)
        train = [d for d in docs if d not in vistos]
        return train, explicita, "archivo data/validacion.txt"
    train, val = dividir_documentos(docs, frac_val)
    return train, val, f"{int(round(frac_val * 100))}% de documentos al azar"


class Stream:
    """Todos los documentos codificados y concatenados.

    ids   : tokens (int32).
    mask  : bool por token; True = "este token cuenta como objetivo de
            predicción". En los diálogos las líneas del USUARIO se enmascaran
            (el modelo las lee pero no se le pide predecirlas): así toda su
            capacidad se dedica a aprender a RESPONDER, como en el ajuste fino
            supervisado de los chatbots. La prosa normal cuenta entera.
    """

    def __init__(self, ids, mask):
        self.ids = np.asarray(ids, dtype=np.int32)
        self.mask = np.asarray(mask, dtype=bool)

    def __len__(self):
        return len(self.ids)

    def n_objetivos(self):
        return int(self.mask[1:].sum())


def _es_dialogo(lineas):
    return all(l.startswith("usuario:") or l.startswith("cerebro:") for l in lineas)


def construir_stream(docs, tokenizador, enmascarar_usuario=True):
    """[EOS] doc1 [EOS] doc2 [EOS] ... como Stream."""
    ids, mask = [ID_EOS], [False]
    salto = tokenizador.codificar("\n")
    for d in docs:
        lineas = d.split("\n")
        if enmascarar_usuario and _es_dialogo(lineas):
            for k, linea in enumerate(lineas):
                cuenta = linea.startswith("cerebro:")
                toks = tokenizador.codificar(linea)
                ids.extend(toks)
                mask.extend([cuenta] * len(toks))
                if k < len(lineas) - 1:
                    ids.extend(salto)
                    mask.extend([cuenta] * len(salto))
        else:
            toks = tokenizador.codificar(d)
            ids.extend(toks)
            mask.extend([True] * len(toks))
        ids.append(ID_EOS)
        mask.append(True)
    return Stream(ids, mask)


def _objetivos(stream, inicio, fin):
    """(x, y) para posiciones inicio..fin-1; y = -1 donde el objetivo está enmascarado."""
    x = stream.ids[inicio:fin]
    y = np.where(stream.mask[inicio + 1:fin + 1], stream.ids[inicio + 1:fin + 1], -1)
    return x, y


class CargadorBatches:
    """Muestrea ventanas aleatorias (x, y) del stream, con y = x desplazado 1
    (y = -1 en los objetivos enmascarados)."""

    def __init__(self, stream, block_size, batch_size, seed=0):
        if len(stream) < 3 or stream.n_objetivos() < 1:
            raise ValueError("El corpus es demasiado pequeño para entrenar.")
        self.stream = stream
        self.T = min(block_size, len(stream) - 1)
        self.B = batch_size
        self.rng = np.random.default_rng(seed)

    def siguiente(self):
        while True:
            inicios = self.rng.integers(0, len(self.stream) - self.T, size=self.B)
            pares = [_objetivos(self.stream, i, i + self.T) for i in inicios]
            x = np.stack([p[0] for p in pares]).astype(np.int64)
            y = np.stack([p[1] for p in pares]).astype(np.int64)
            if (y >= 0).any():
                return x, y


def ventanas_secuenciales(stream, block_size):
    """Cubre TODO el stream sin solaparse: lista de (x, y) de igual longitud
    T más, si sobra cola, una última ventana más corta."""
    n = len(stream)
    T = min(block_size, n - 1)
    completas, i = [], 0
    while i + T + 1 <= n:
        completas.append(_objetivos(stream, i, i + T))
        i += T
    cola = _objetivos(stream, i, n - 1) if i < n - 1 else None
    return completas, cola
