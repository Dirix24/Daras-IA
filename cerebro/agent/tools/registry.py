# -*- coding: utf-8 -*-
"""
Registro de herramientas del agente.

Seguridad (todo verificado en tests/tests_final.py):
  * La calculadora NO usa eval(): analiza la expresión con `ast` y solo
    acepta números, operadores aritméticos y unas pocas funciones de una
    lista blanca. Además limita el tamaño de los resultados, así que
    "9**9**9" no puede colgar la máquina.
  * Los archivos viven en un WORKSPACE: la ruta se resuelve con realpath
    (siguiendo enlaces simbólicos) y se comprueba con commonpath que sigue
    dentro. Tienen un tamaño máximo por archivo.
  * Cada herramienta declara el permiso que exige (ver ControlledAgent).
"""

import ast
import datetime
import math
import operator
import os
import re

import config

MAX_LARGO_EXPRESION = 200
MAX_BITS_RESULTADO = 20000      # tope para enteros enormes
MAX_EXPONENTE = 10000

_BINARIOS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARIOS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCIONES = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "log": math.log, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil,
}
_CONSTANTES = {"pi": math.pi, "e": math.e}


def normalizar_expresion(expr):
    """Traduce notación cotidiana: 2^8 -> 2**8, × -> *, ÷ -> /, 3,5 -> 3.5."""
    expr = expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = expr.replace("−", "-")
    return re.sub(r"(?<=\d),(?=\d)", ".", expr)


def _es_numero(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _revisar(v):
    if isinstance(v, int) and v.bit_length() > MAX_BITS_RESULTADO:
        raise ValueError("Resultado demasiado grande.")
    return v


def _evaluar_nodo(nodo):
    if isinstance(nodo, ast.Constant):
        if _es_numero(nodo.value):
            return nodo.value
        raise ValueError("Solo se permiten números.")
    if isinstance(nodo, ast.Name):
        if nodo.id in _CONSTANTES:
            return _CONSTANTES[nodo.id]
        raise ValueError(f"Nombre no permitido: {nodo.id}")
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _BINARIOS:
        a, b = _evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right)
        if isinstance(nodo.op, ast.Pow):
            if abs(b) > MAX_EXPONENTE:
                raise ValueError("Exponente demasiado grande.")
            if isinstance(a, int) and isinstance(b, int) and b > 0 \
                    and abs(a).bit_length() * b > MAX_BITS_RESULTADO:
                raise ValueError("Resultado demasiado grande.")
        return _revisar(_BINARIOS[type(nodo.op)](a, b))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _UNARIOS:
        return _UNARIOS[type(nodo.op)](_evaluar_nodo(nodo.operand))
    if (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
            and nodo.func.id in _FUNCIONES and len(nodo.args) == 1 and not nodo.keywords):
        return _FUNCIONES[nodo.func.id](_evaluar_nodo(nodo.args[0]))
    raise ValueError("Expresión no permitida.")


def evaluar_expresion_segura(expresion):
    """Evalúa una expresión aritmética sin usar eval()."""
    if not isinstance(expresion, str) or len(expresion) > MAX_LARGO_EXPRESION:
        raise ValueError("Expresión vacía o demasiado larga.")
    arbol = ast.parse(expresion.strip(), mode="eval")
    return _evaluar_nodo(arbol.body)


class ToolRegistry:
    def __init__(self, workspace=None, max_file_bytes=None):
        self.workspace = os.path.realpath(workspace or config.WORKSPACE_DIR)
        self.max_file_bytes = max_file_bytes or config.MAX_FILE_BYTES
        self.tools = {}
        self.registrar_base()

    def registrar(self, nombre, desc, func, schema, permiso="READ"):
        """Punto de extensión: añade aquí tus propias herramientas."""
        self.tools[nombre] = {"desc": desc, "func": func, "schema": schema,
                              "permiso": permiso}

    # -- rutas seguras ------------------------------------------------------------

    def resolver_ruta(self, nombre):
        """Ruta absoluta real dentro del workspace, o ValueError."""
        if not isinstance(nombre, str) or not nombre.strip() or "\0" in nombre:
            raise ValueError("Nombre de archivo inválido.")
        os.makedirs(self.workspace, exist_ok=True)
        real = os.path.realpath(os.path.join(self.workspace, nombre))
        try:
            dentro = os.path.commonpath([real, self.workspace]) == self.workspace
        except ValueError:      # p. ej. distintas unidades en Windows
            dentro = False
        if not dentro or real == self.workspace:
            raise ValueError("Acceso denegado fuera del workspace.")
        return real

    # -- herramientas -------------------------------------------------------------

    def registrar_base(self):
        def calc(expression):
            try:
                valor = evaluar_expresion_segura(normalizar_expresion(expression))
                return {"result": valor}
            except ZeroDivisionError:
                return {"error": "División entre cero."}
            except (ValueError, SyntaxError, TypeError, OverflowError,
                    RecursionError, MemoryError) as e:
                return {"error": f"Expresión inválida: {e}"}
        self.registrar("calculator", "Calculadora matemática segura", calc,
                       {"expression": str}, permiso="CALCULATE")

        def clock():
            return {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self.registrar("clock", "Hora local del sistema", clock, {}, permiso="READ")

        def read_file(filename):
            try:
                ruta = self.resolver_ruta(filename)
            except ValueError as e:
                return {"error": str(e)}
            if not os.path.isfile(ruta):
                return {"error": "El archivo no existe."}
            if os.path.getsize(ruta) > self.max_file_bytes:
                return {"error": "Archivo demasiado grande."}
            with open(ruta, "r", encoding="utf-8", errors="replace") as f:
                return {"content": f.read()}
        self.registrar("read_file", "Lee un archivo del workspace", read_file,
                       {"filename": str}, permiso="READ")

        def write_file(filename, content):
            try:
                ruta = self.resolver_ruta(filename)
            except ValueError as e:
                return {"error": str(e)}
            if len(content.encode("utf-8")) > self.max_file_bytes:
                return {"error": "Contenido demasiado grande."}
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "message": f"Escrito {filename}"}
        self.registrar("write_file", "Escribe un archivo en el workspace", write_file,
                       {"filename": str, "content": str}, permiso="WRITE")

        def list_files():
            os.makedirs(self.workspace, exist_ok=True)
            nombres = []
            for raiz, _, archivos in os.walk(self.workspace):
                for a in archivos:
                    nombres.append(os.path.relpath(os.path.join(raiz, a), self.workspace))
            return {"files": sorted(nombres)}
        self.registrar("list_files", "Lista los archivos del workspace", list_files,
                       {}, permiso="READ")

    def ejecutar(self, nombre, args):
        if nombre not in self.tools:
            return {"error": f"Herramienta '{nombre}' no encontrada."}
        if not isinstance(args, dict):
            return {"error": "Los argumentos deben ser un diccionario."}
        tool = self.tools[nombre]
        for param, tipo in tool["schema"].items():
            if param not in args or not isinstance(args[param], tipo):
                return {"error": f"Parámetro requerido faltante o tipo incorrecto: '{param}'"}
        extra = set(args) - set(tool["schema"])
        if extra:
            return {"error": f"Parámetros no permitidos: {sorted(extra)}"}
        try:
            return tool["func"](**args)
        except Exception as e:
            return {"error": str(e)}
