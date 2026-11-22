"""Punto de entrada JSON v1 para investigación y planificación de DARAS."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import random
import sys

from companiona import find_local, init_db, open_database, research_online

VERSION = 1
TAREAS = (
    ("alcance", "Definir requisitos y criterios de aceptación", (), 1, 2, 4),
    ("investigacion", "Contrastar documentación y restricciones", ("alcance",), 1, 3, 6),
    ("diseno", "Diseñar y revisar alternativas", ("investigacion",), 2, 4, 8),
    ("prototipo", "Construir un prototipo", ("diseno",), 3, 6, 12),
    ("evaluacion", "Ejecutar pruebas y documentar resultados", ("prototipo",), 2, 4, 8),
)


def plan(completadas=()):
    ids = {t[0] for t in TAREAS}
    if not isinstance(completadas, (tuple, list)) or any(type(i) is not str or i not in ids for i in completadas):
        raise ValueError("Identificador de tarea desconocido.")
    finalizadas = set(completadas)
    if any(i in finalizadas and not set(deps) <= finalizadas for i, _, deps, *_ in TAREAS):
        raise ValueError("Las tareas completadas deben incluir sus dependencias.")
    return [{"id": i, "titulo": titulo, "depende_de": list(deps), "horas": [bajo, moda, alto],
             "estado": "completada" if i in finalizadas else
             "lista" if set(deps) <= finalizadas else "bloqueada"}
            for i, titulo, deps, bajo, moda, alto in TAREAS]


def simular(pasos, semilla, muestras):
    if type(semilla) is not int or not 0 <= semilla < 2**32:
        raise ValueError("Semilla fuera de rango.")
    if type(muestras) is not int or not 10 <= muestras <= 10_000:
        raise ValueError("Indica entre 10 y 10 000 muestras.")
    rng = random.Random(semilla)
    restantes = [t["horas"] for t in pasos if t["estado"] != "completada"]
    totales = sorted(sum(rng.triangular(bajo, alto, moda) for bajo, moda, alto in restantes)
                      for _ in range(muestras))
    return {"tipo": "simulacion", "metodo": "Monte Carlo triangular independiente, tareas seriales",
            "supuestos": "Duraciones de ejemplo en horas; sustituir por estimaciones del equipo.",
            "semilla": semilla, "muestras": muestras,
            "p50_horas": round(totales[(muestras - 1) // 2], 2),
            "p90_horas": round(totales[(muestras - 1) * 9 // 10], 2)}


def coordinar(peticion):
    if not isinstance(peticion, dict) or type(peticion.get("version")) is not int or peticion["version"] != VERSION:
        raise ValueError("Se espera una solicitud versión 1.")
    if peticion.get("accion") == "capacidades":
        return {"version": VERSION, "especialista": "DARAS", "estado": "ok",
                "capacidades": ["planificar", "replanificar", "investigar", "simular"]}
    objetivo = peticion.get("objetivo")
    if not isinstance(objetivo, str) or not 1 <= len(objetivo.strip()) <= 600:
        raise ValueError("Indica un objetivo de hasta 600 caracteres.")
    accion = peticion.get("accion")
    if accion not in ("planificar", "replanificar", "investigar", "simular"):
        raise ValueError("Acción DARAS desconocida.")
    red = peticion.get("red") is True and accion == "investigar"
    init_db()
    fuentes, avisos = [], []
    if accion == "investigar":
        fuentes, avisos = research_online(objetivo) if red else (find_local(objetivo, limit=5), [])
    pasos = plan(peticion.get("completadas", []))
    resultado = {"version": VERSION, "especialista": "DARAS", "estado": "ok", "accion": accion,
                 "objetivo": objetivo, "plan": pasos,
                 "simulacion": simular(pasos, peticion.get("semilla", 0), peticion.get("muestras", 1000)),
                 "evidencia": [{"tipo": "fuente", "origen": item.get("source"),
                                "titulo": item.get("title"), "url": item.get("url"),
                                "extracto": str(item.get("extract", ""))[:400]}
                               for item in fuentes[:8]],
                 "red_autorizada": red, "avisos": avisos}
    if accion == "investigar" and not fuentes:
        resultado["estado"] = "parcial"
        resultado["avisos"].append("No se encontraron fuentes para este objetivo.")
    cuerpo = json.dumps(resultado, ensure_ascii=False, sort_keys=True)
    identificador = sha256(cuerpo.encode("utf-8")).hexdigest()
    with open_database() as db:
        db.execute("CREATE TABLE IF NOT EXISTS specialist_runs (id TEXT PRIMARY KEY, fecha TEXT, resultado TEXT)")
        db.execute("INSERT OR IGNORE INTO specialist_runs VALUES (?, ?, ?)",
                   (identificador, datetime.now(timezone.utc).isoformat(), cuerpo))
    return {"id": identificador, **resultado}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        bruto = sys.stdin.buffer.read(32_769)
        if len(bruto) > 32_768:
            raise ValueError("Solicitud demasiado grande.")
        respuesta = coordinar(json.loads(bruto.decode("utf-8")))
    except ValueError as error:
        respuesta = {"version": VERSION, "especialista": "DARAS", "estado": "error", "error": str(error)}
    print(json.dumps(respuesta, ensure_ascii=True))


if __name__ == "__main__":
    main()
