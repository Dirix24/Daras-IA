# -*- coding: utf-8 -*-
"""
Lógica de la conversación de Cerebro (separada de main.py para poder probarla).

Cómo decide qué hacer con cada mensaje:
  1. Si empieza por "/" es un comando explícito.
  2. Si es una cuenta o una pregunta por la hora, el agente usa la
     herramienta correspondiente (reglas, no el modelo: un modelo tan pequeño
     no sabe calcular).
  3. Si el usuario dice cómo se llama, se guarda en la memoria de largo plazo.
  4. Todo lo demás lo responde el modelo de lenguaje, condicionado por los
     últimos turnos de la conversación.
"""

import re

import config
from agent.intents import detectar_intencion
from core.tokenizer import ID_EOS

AYUDA = """Comandos:
  /ayuda                    esta ayuda
  /estado                   información del modelo y de la memoria
  /memoria                  muestra lo que hay en la memoria de largo plazo
  /recordar clave = valor   guarda un dato
  /olvidar clave            borra un dato
  /calc 12 * (3 + 4)        calculadora segura
  /hora                     fecha y hora del sistema
  /archivos                 lista los archivos del espacio de trabajo
  /leer nombre.txt          lee un archivo del espacio de trabajo
  /escribir nombre.txt txt  escribe un archivo en el espacio de trabajo
  /temp 0.7                 cambia la temperatura de generación (0 = determinista)
  /limpiar                  olvida la conversación actual (no la memoria de largo plazo)
  /salir                    termina
También entiendo cosas como «cuánto es 12 por 7» o «qué hora es»."""

_NOMBRE = re.compile(
    r"\b(?:me\s+llamo|mi\s+nombre\s+es|soy)\s+([A-ZÁÉÍÓÚÑa-záéíóúñ]{2,20})\b", re.IGNORECASE)
_PREGUNTA_NOMBRE = re.compile(r"(c[oó]mo\s+me\s+llamo|cu[aá]l\s+es\s+mi\s+nombre|sabes\s+mi\s+nombre)",
                              re.IGNORECASE)
_NO_SON_NOMBRES = {"un", "una", "el", "la", "tu", "yo", "muy", "bien", "pequeño", "cerebro",
                   "nuevo", "programador", "estudiante", "de"}


class Conversacion:
    def __init__(self, modelo, memoria, agente, info_checkpoint=None, seed=None):
        self.modelo = modelo
        self.memoria = memoria
        self.agente = agente
        self.info = info_checkpoint or {}
        self.gen = dict(max_nuevos=config.GEN_MAX_NUEVOS, temperatura=config.GEN_TEMPERATURA,
                        top_k=config.GEN_TOP_K, top_p=config.GEN_TOP_P,
                        penalizacion=config.GEN_PENALIZACION_REPETICION)
        import numpy as np
        self.rng = np.random.default_rng(seed)

    # -- entrada principal ------------------------------------------------------------

    def procesar(self, entrada):
        """Devuelve (texto de respuesta, terminar: bool)."""
        entrada = entrada.strip()
        if not entrada:
            return "", False
        if entrada.startswith("/"):
            return self._comando(entrada)

        plan = detectar_intencion(entrada)
        if plan:
            return self._responder_con_agente(plan), False

        if _PREGUNTA_NOMBRE.search(entrada):
            nombre = self.memoria.recuperar_largo_plazo("nombre")
            return (f"Te llamas {nombre}." if nombre
                    else "Todavía no sé tu nombre. Dímelo con «me llamo ...»."), False
        m = _NOMBRE.search(entrada)
        if m and m.group(1).lower() not in _NO_SON_NOMBRES:
            nombre = m.group(1).capitalize()
            self.memoria.guardar_largo_plazo("nombre", nombre, categoria="usuario", importancia=3)
            return f"Mucho gusto, {nombre}. Lo guardé en mi memoria. ¿En qué te puedo ayudar?", False

        return self.responder_modelo(entrada), False

    # -- respuesta del modelo -----------------------------------------------------------

    def _armar_prompt_ids(self, mensaje, max_nuevos):
        tok, B = self.modelo.tokenizer, self.modelo.cfg.block_size
        turnos = list(self.memoria.short_term_context)
        recuerdos = self.memoria.buscar(mensaje, limite=4)
        bloque_memoria = ""
        if recuerdos:
            bloque_memoria = "memoria: " + "; ".join(
                f"{k}={v}" for k, v in recuerdos.items()) + "\n"
        while True:
            partes = [f"{'usuario' if t['rol'] == 'usuario' else 'cerebro'}: {t['contenido']}\n"
                      for t in turnos]
            texto = bloque_memoria + "".join(partes) + f"usuario: {mensaje}\ncerebro:"
            ids = [ID_EOS] + tok.codificar(texto)
            if len(ids) <= B - max_nuevos or not turnos:
                break
            turnos = turnos[2:] if len(turnos) >= 2 else []   # descarta el turno más viejo
        return ids[-(B - max_nuevos):]

    def responder_modelo(self, mensaje):
        g = self.gen
        max_nuevos = max(8, min(g["max_nuevos"], self.modelo.cfg.block_size - 40))
        ids = self._armar_prompt_ids(mensaje, max_nuevos)
        nuevos, _ = self.modelo.generar_ids(
            ids, max_nuevos=max_nuevos, temperatura=g["temperatura"], top_k=g["top_k"],
            top_p=g["top_p"], penalizacion=g["penalizacion"], parar_en=("\n",), rng=self.rng)
        texto = self.modelo.tokenizer.decodificar(nuevos).split("\n")[0].strip()
        if not texto:
            texto = "No sé qué responder a eso."
        self.memoria.agregar_contexto("usuario", mensaje)
        self.memoria.agregar_contexto("ia", texto)
        return texto

    # -- agente -------------------------------------------------------------------------

    def _responder_con_agente(self, plan):
        historial = self.agente.ejecutar_plan_autonomo(plan)
        paso = historial[-1]
        if "error" in paso:
            return f"No pude hacerlo: {paso['error']}"
        res = paso["resultado"]
        if "error" in res:
            return f"No pude hacerlo: {res['error']}"
        if paso["herramienta"] == "calculator":
            valor = res["result"]
            if isinstance(valor, float):
                valor = round(valor, 10)
            return f"{plan[0]['arguments']['expression']} = {valor}"
        if paso["herramienta"] == "clock":
            return f"Ahora mismo son: {res['time']}"
        return str(res)

    # -- comandos -----------------------------------------------------------------------

    def _comando(self, entrada):
        partes = entrada.split(None, 1)
        cmd = partes[0].lower()
        resto = partes[1].strip() if len(partes) > 1 else ""
        a = self.agente

        if cmd in ("/salir", "/exit", "/quit"):
            return "¡Hasta pronto!", True
        if cmd in ("/ayuda", "/help"):
            return AYUDA, False
        if cmd in ("/estado", "/status"):
            return self.estado(), False
        if cmd in ("/memoria", "/memory"):
            if not self.memoria.long_term_memory:
                return "La memoria de largo plazo está vacía. Usa /recordar clave = valor.", False
            return "\n".join(f"  {k} = {v['valor']}   ({v['fecha']})"
                             for k, v in self.memoria.long_term_memory.items()), False
        if cmd == "/recordar":
            if "=" not in resto:
                return "Uso: /recordar clave = valor", False
            clave, valor = (s.strip() for s in resto.split("=", 1))
            if not clave or not valor:
                return "Uso: /recordar clave = valor", False
            self.memoria.guardar_largo_plazo(clave, valor)
            return f"Guardado: {clave} = {valor}", False
        if cmd == "/olvidar":
            return (f"Olvidé «{resto}»." if self.memoria.eliminar_largo_plazo(resto)
                    else f"No tengo nada guardado como «{resto}»."), False
        if cmd == "/calc":
            return self._responder_con_agente(
                [{"tool": "calculator", "arguments": {"expression": resto}}]), False
        if cmd == "/hora":
            return self._responder_con_agente([{"tool": "clock", "arguments": {}}]), False
        if cmd == "/archivos":
            return self._responder_con_agente([{"tool": "list_files", "arguments": {}}]), False
        if cmd == "/leer":
            h = a.ejecutar_plan_autonomo([{"tool": "read_file", "arguments": {"filename": resto}}])[-1]
            r = h.get("resultado", h)
            return r.get("content", f"No pude hacerlo: {r.get('error')}"), False
        if cmd == "/escribir":
            p = resto.split(None, 1)
            if len(p) < 2:
                return "Uso: /escribir nombre.txt texto a guardar", False
            h = a.ejecutar_plan_autonomo(
                [{"tool": "write_file", "arguments": {"filename": p[0], "content": p[1]}}])[-1]
            r = h.get("resultado", h)
            return r.get("message", f"No pude hacerlo: {r.get('error')}"), False
        if cmd == "/temp":
            try:
                self.gen["temperatura"] = max(0.0, float(resto))
                return f"Temperatura = {self.gen['temperatura']}", False
            except ValueError:
                return "Uso: /temp 0.7", False
        if cmd == "/limpiar":
            self.memoria.limpiar_corto_plazo()
            return "Conversación reiniciada.", False
        return f"Comando desconocido: {cmd}. Escribe /ayuda.", False

    def estado(self):
        m = self.modelo
        c = m.cfg
        lineas = [
            "=== ESTADO DE CEREBRO ===",
            f"Modelo: transformer {c.n_layers} capas x {c.n_heads} cabezas, dim {c.dim}, "
            f"contexto {c.block_size}, {m.n_parametros():,} parámetros",
            f"Tokenizador: {m.tokenizer.tipo}, {len(m.tokenizer)} tokens",
            f"Memoria de largo plazo: {len(self.memoria.long_term_memory)} datos",
            f"Temperatura: {self.gen['temperatura']}",
        ]
        if "val_bpb" in self.info:
            lineas.append(f"Calidad en validación: {self.info['val_bpb']:.3f} bits/byte "
                          f"(perplejidad {self.info.get('val_ppl', float('nan')):.1f})")
        lineas.append(f"Permisos del agente: {', '.join(sorted(self.agente.permisos_concedidos))}")
        return "\n".join(lineas)
