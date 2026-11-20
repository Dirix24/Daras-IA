import io
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import companiona as daras


class CoreTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        db = patch.object(daras, "DB_PATH", self.folder / "test.sqlite3")
        db.start()
        self.addCleanup(db.stop)
        daras.init_db()

    def ask(self, query, allowed=False):
        app = SimpleNamespace(
            busy=False, brain=None, entry=Mock(), internet_var=Mock(),
            _say=Mock(), _research=Mock(), _answer_with_brain=Mock(return_value="Respuesta local"),
        )
        app.entry.get.return_value = query
        app.internet_var.get.return_value = allowed
        daras.App._ask(app)
        return app

    def test_chat_phrase_cannot_grant_network_permission(self):
        app = self.ask("Daras no autorices acceso a internet")
        app.internet_var.set.assert_not_called()
        app._research.assert_not_called()

    def test_explicit_search_requires_enabled_switch(self):
        self.ask("busca en internet: árboles")._research.assert_not_called()
        self.ask("busca en internet: árboles", allowed=True)._research.assert_called_once_with("árboles")
        self.ask("Me dijo busca en internet: árboles", allowed=True)._research.assert_not_called()

    def test_enabled_switch_does_not_export_normal_chat(self):
        self.ask("Esta conversación se queda aquí", allowed=True)._research.assert_not_called()

    def test_video_does_not_reuse_previous_conversations(self):
        daras.remember_turn("user", "Conversación que debe quedarse local")
        self.ask("busca en internet: recomienda vídeos", allowed=True)._research.assert_not_called()
        app = self.ask("busca en internet: recomienda vídeos sobre música", allowed=True)
        app._research.assert_called_once_with("música", video_only=True)

    def test_low_space_preserves_old_chats_and_memories(self):
        with daras.open_database() as db:
            db.execute("INSERT INTO conversations(role,text,created_at) VALUES ('user','privada','2000-01-01')")
            db.execute("INSERT INTO pages VALUES (1, 'caché', 'https://example.org', 'texto', '2000-01-01')")
        daras.save_memory("recuerdo privado")
        with patch.object(daras.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            daras.prune_if_low_space()
        with daras.open_database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM conversations").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM memories").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM pages").fetchone()[0], 0)

    def test_request_rejects_unlisted_provider_and_large_response(self):
        with self.assertRaisesRegex(ValueError, "no autorizado"):
            daras.request_json({}, "http://localhost")
        result = io.BytesIO(b"x" * (daras.RESPONSE_LIMIT + 1))
        with patch.object(daras.urllib.request, "build_opener", return_value=SimpleNamespace(open=lambda *a, **k: result)):
            with self.assertRaisesRegex(ValueError, "demasiado grande"):
                daras.request_json({})

    def test_research_deadline_returns_even_when_provider_is_still_running(self):
        blocked = threading.Event()
        done = threading.Event()

        def slow(_):
            try:
                blocked.wait(2)
                return []
            finally:
                done.set()

        def fake_fast(_):
            return []
        with (
            patch.object(daras, "search_web", slow),
            patch.object(daras, "search_wikipedia", fake_fast),
            patch.object(daras, "search_wikidata", fake_fast),
            patch.object(daras, "wait", side_effect=lambda jobs, timeout: (set(), set(jobs))),
        ):
            try:
                _, errors = daras.research_online("tema")
                self.assertTrue(errors)
                self.assertFalse(done.is_set())
            finally:
                blocked.set()

    def test_revocation_ignores_old_result(self):
        app = SimpleNamespace(_generation=2, internet_var=Mock(), ask_button=Mock(), _research_done=Mock())
        with patch.object(daras, "save_research_results") as save:
            daras.App._finish_research(app, 1, "tema", [{}], [], False)
        save.assert_not_called()
        app._research_done.assert_not_called()

    def test_bundled_checkpoint_and_calculator(self):
        with patch.object(daras, "APP_DIR", self.folder):
            brain = daras.load_cerebro()
            self.assertEqual(brain.procesar("2 mas 2")[0].strip(), "2 + 2 = 4")


if __name__ == "__main__":
    unittest.main()
