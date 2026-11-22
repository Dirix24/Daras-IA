import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import companiona
from especialista import coordinar, plan, simular


class Especialista(unittest.TestCase):
    def test_replanificacion_y_simulacion(self):
        base = plan()
        actualizado = plan(["alcance"])
        self.assertEqual(actualizado[1]["estado"], "lista")
        self.assertEqual(simular(base, 42, 200), simular(base, 42, 200))
        self.assertLess(simular(actualizado, 42, 200)["p90_horas"],
                        simular(base, 42, 200)["p90_horas"])
        with self.assertRaises(ValueError):
            plan(["diseno"])

    def test_persistencia_y_fuentes_sin_red(self):
        objetivo = {"version": 1, "accion": "investigar", "objetivo": "Robots"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(companiona, "DB_PATH", Path(tmp) / "daras.db"):
            companiona.init_db()
            companiona.save_research_results([{"url": "https://ejemplo.org/robot", "source": "Manual",
                                              "title": "Robots", "extract": "Documentación."}])
            with patch("especialista.research_online") as online:
                primero = coordinar(objetivo)
                self.assertEqual(coordinar(objetivo), primero)
                online.assert_not_called()
            self.assertEqual(primero["evidencia"][0]["url"], "https://ejemplo.org/robot")
            with companiona.open_database() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM specialist_runs").fetchone()[0], 1)
