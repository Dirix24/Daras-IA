# -*- coding: utf-8 -*-
"""Agente con ejecución controlada de herramientas.

Límites que SÍ se hacen cumplir (no son decorativos):
  * máximo de pasos por plan            (MAX_AGENT_STEPS)
  * máximo de llamadas a herramientas   (MAX_TOOL_CALLS)
  * permisos: cada herramienta exige un permiso (READ / WRITE / CALCULATE) y
    el agente solo ejecuta las que tiene concedidas (PERMISOS_AGENTE).
  * un plan mal formado o que falla se detiene: el agente falla cerrado.
"""

import config
from agent.tools.registry import ToolRegistry


class ControlledAgent:
    def __init__(self, max_steps=None, max_tool_calls=None, permisos=None, registry=None):
        self.max_steps = max_steps or config.MAX_AGENT_STEPS
        self.max_tool_calls = max_tool_calls or config.MAX_TOOL_CALLS
        self.permisos_concedidos = set(config.PERMISOS_AGENTE if permisos is None else permisos)
        self.registry = registry or ToolRegistry()

    @property
    def permisos_herramientas(self):
        return {n: t["permiso"] for n, t in self.registry.tools.items()}

    def verificar_permiso(self, herramienta):
        """True si el agente tiene concedido el permiso que exige la herramienta."""
        tool = self.registry.tools.get(herramienta)
        return tool is not None and tool["permiso"] in self.permisos_concedidos

    def ejecutar_plan_autonomo(self, intenciones):
        """Ejecuta una lista de {"tool": nombre, "arguments": {...}} respetando
        los límites. Devuelve el historial paso a paso."""
        historial = []
        llamadas = 0
        if not isinstance(intenciones, (list, tuple)):
            return [{"paso": 0, "error": "El plan debe ser una lista."}]
        if len(intenciones) > self.max_steps:
            historial.append({"paso": 0, "aviso": f"Plan recortado a {self.max_steps} pasos "
                                                  f"(MAX_AGENT_STEPS)."})

        for paso, intencion in enumerate(intenciones[:self.max_steps], start=1):
            if not isinstance(intencion, dict) or not isinstance(intencion.get("tool"), str):
                historial.append({"paso": paso, "error": "Intención mal formada."})
                break
            herramienta = intencion["tool"]

            if herramienta not in self.registry.tools:
                historial.append({"paso": paso,
                                  "error": f"Herramienta '{herramienta}' no autorizada o inexistente."})
                break
            if not self.verificar_permiso(herramienta):
                historial.append({"paso": paso, "error": f"Permiso denegado para '{herramienta}' "
                                  f"(requiere {self.registry.tools[herramienta]['permiso']})."})
                break
            if llamadas >= self.max_tool_calls:
                historial.append({"paso": paso, "error": "Límite máximo de llamadas a "
                                                         "herramientas alcanzado (MAX_TOOL_CALLS)."})
                break

            resultado = self.registry.ejecutar(herramienta, intencion.get("arguments", {}))
            llamadas += 1
            historial.append({"paso": paso, "herramienta": herramienta, "resultado": resultado})
            if isinstance(resultado, dict) and "error" in resultado:
                break
        return historial

    def ejecutar(self, herramienta, **argumentos):
        """Atajo: un plan de un solo paso. Devuelve el resultado o {"error": ...}."""
        h = self.ejecutar_plan_autonomo([{"tool": herramienta, "arguments": argumentos}])
        ultimo = h[-1]
        return ultimo.get("resultado", {"error": ultimo.get("error", "Sin resultado.")})
