# -*- coding: utf-8 -*-
"""
Detección de intenciones por REGLAS (no por el modelo de lenguaje): decide si
un mensaje del usuario es una operación para la calculadora o una pregunta
por la hora. Es un enrutador honesto y determinista: el modelo pequeño no
sabe calcular, así que las cuentas se delegan a la herramienta segura.

Las herramientas de archivos NUNCA se activan por lenguaje natural: solo con
comandos explícitos (/leer, /escribir).
"""

import re

_OPERADORES = re.compile(r"[-+*/^%×÷]|\bpor\b|\bmás\b|\bmas\b|\bmenos\b|\bentre\b")
_PREFIJO_CALCULO = re.compile(
    r"^(?:cu[aá]nto\s+es|cu[aá]nto\s+son|calcula(?:r)?|resuelve|cu[aá]l\s+es\s+el\s+resultado\s+de)\s+(.+?)\s*\??\s*$",
    re.IGNORECASE)
_SOLO_EXPRESION = re.compile(r"^[\d\s.,+\-*/()^%×÷]+$")
_HORA = re.compile(r"\b(qu[eé]\s+hora\s+es|hora\s+actual|qu[eé]\s+d[ií]a\s+es\s+hoy|qu[eé]\s+fecha)\b",
                   re.IGNORECASE)

_PALABRAS = [(r"\bpor\b", "*"), (r"\bm[aá]s\b", "+"), (r"\bmenos\b", "-"),
             (r"\bentre\b", "/"), (r"\bdividido\s+(?:entre|por)\b", "/"),
             (r"\belevado\s+a\b", "**"), (r"\by\s+medio\b", "+0.5")]


def _a_expresion(texto):
    t = texto.lower()
    t = re.sub(r"\bdividido\s+(?:entre|por)\b", "/", t)
    for patron, reemplazo in _PALABRAS:
        t = re.sub(patron, reemplazo, t)
    return t.strip().rstrip("?").strip()


def detectar_intencion(mensaje):
    """Devuelve un plan [{"tool":..., "arguments":...}] o None si el mensaje
    debe responderlo el modelo de lenguaje."""
    texto = mensaje.strip()
    if _HORA.search(texto):
        return [{"tool": "clock", "arguments": {}}]

    m = _PREFIJO_CALCULO.match(texto)
    candidato = m.group(1) if m else texto
    expr = _a_expresion(candidato)
    if (_SOLO_EXPRESION.match(expr) and re.search(r"\d", expr)
            and (m or _OPERADORES.search(expr))):
        return [{"tool": "calculator", "arguments": {"expression": expr}}]
    return None
