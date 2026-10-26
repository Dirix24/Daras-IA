# -*- coding: utf-8 -*-
"""
Memoria de Cerebro.

  * Corto plazo: los últimos turnos de la conversación (en RAM). Se usan para
    construir el prompt, así el modelo "ve" lo que se acaba de decir.
  * Largo plazo: datos con nombre (clave -> valor + categoría, importancia y
    fecha) guardados en un JSON. Se escribe de forma atómica (archivo
    temporal + os.replace) para no corromper la memoria si se corta el
    programa a mitad de guardado.
"""

import datetime
import json
import os

import config


def _ahora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AdvancedMemoryManager:
    def __init__(self, filepath=None, max_turnos=None):
        self.filepath = filepath or config.MEMORY_FILE
        self.max_turnos = max_turnos or config.MEMORIA_CORTO_PLAZO_TURNOS * 2
        self.short_term_context = []   # [{"rol": "usuario"|"ia", "contenido": str}]
        self.long_term_memory = {}
        self.cargar_memoria()

    # -- corto plazo ---------------------------------------------------------------

    def agregar_contexto(self, rol, contenido):
        self.short_term_context.append({"rol": rol, "contenido": contenido})
        exceso = len(self.short_term_context) - self.max_turnos
        if exceso > 0:
            del self.short_term_context[:exceso]

    def limpiar_corto_plazo(self):
        self.short_term_context = []

    # -- largo plazo ---------------------------------------------------------------

    def guardar_largo_plazo(self, clave, valor, categoria="general", importancia=1):
        self.long_term_memory[clave] = {
            "valor": valor, "categoria": categoria,
            "importancia": importancia, "fecha": _ahora(),
        }
        self.persistir()

    def recuperar_largo_plazo(self, clave):
        entrada = self.long_term_memory.get(clave)
        return entrada["valor"] if entrada else None

    def actualizar_largo_plazo(self, clave, nuevo_valor):
        if clave in self.long_term_memory:
            self.long_term_memory[clave]["valor"] = nuevo_valor
            self.long_term_memory[clave]["fecha"] = _ahora()
            self.persistir()
            return True
        return False

    def eliminar_largo_plazo(self, clave):
        if clave in self.long_term_memory:
            del self.long_term_memory[clave]
            self.persistir()
            return True
        return False

    def buscar(self, texto, limite=5):
        """Recuperación ligera por relevancia léxica. No requiere embeddings.
        Puntúa coincidencias de palabras en clave/valor/categoría y devuelve
        como máximo `limite` entradas, preservando la memoria explícita.
        """
        import re
        palabras = {p for p in re.findall(r"[\wáéíóúñü]+", texto.lower()) if len(p) > 2}
        resultados = []
        for clave, entrada in self.long_term_memory.items():
            campo = f"{clave} {entrada.get('valor', '')} {entrada.get('categoria', '')}".lower()
            score = sum(1 for p in palabras if p in campo)
            if score:
                score += 0.01 * float(entrada.get("importancia", 1))
                resultados.append((score, clave, entrada["valor"]))
        resultados.sort(reverse=True)
        return {k: v for _, k, v in resultados[:max(0, limite)]}

    # -- persistencia ---------------------------------------------------------------

    def persistir(self):
        directorio = os.path.dirname(os.path.abspath(self.filepath))
        os.makedirs(directorio, exist_ok=True)
        tmp = self.filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"long_term": self.long_term_memory}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.filepath)

    def cargar_memoria(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.long_term_memory = json.load(f).get("long_term", {})
            except (OSError, ValueError):
                self.long_term_memory = {}
