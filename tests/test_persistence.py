"""Offline SQLite checks using real files and graph, mocking only model I/O."""

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_demo import CERTIFICATE, SAMPLES
from src.core.state import Document, Requirement, UserEvent, new_run
from src.memory.checkpointer import persistent_graph, session_config


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.database = Path(self.folder.name) / "nested/checkpoints.sqlite"

    def run_certificate(self, graph, session, payload):
        with patch("processor.Client.chat", return_value=SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)))):
            return graph.invoke(new_run(UserEvent(kind="upload", file_path=str(
                SAMPLES / "student_enrollment.txt")), session), session_config(session))

    def test_reopen_preserves_typed_state_and_isolates_sessions(self):
        with persistent_graph(self.database) as graph:
            self.run_certificate(graph, "accepted", CERTIFICATE)
            self.run_certificate(graph, "review", {**CERTIFICATE, "student_id": None})
            connection = graph.checkpointer.conn
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        with persistent_graph(self.database) as graph:
            accepted = graph.get_state(session_config("accepted")).values
            review = graph.get_state(session_config("review")).values
            self.assertIsInstance(accepted["documents"][0], Document)
            self.assertIsInstance(accepted["relevant_requirements"][0], Requirement)
            self.assertEqual(accepted["validation_errors"], [])
            self.assertTrue(accepted["relevant_requirements"][0].satisfied)
            self.assertTrue(accepted["relevant_requirements"][0].source_metadata)
            self.assertTrue(review["validation_errors"])
            self.assertIn("student_id", review["extracted_facts"]["student_enrollment.txt"]["missing_fields"])
            self.assertFalse(review["relevant_requirements"][0].satisfied)
            self.assertEqual(graph.get_state(session_config("unused")).values, {})
            # A subsequent event retains previously extracted facts in its session.
            result = graph.invoke(new_run(UserEvent(kind="message", text="Check again"), "accepted"),
                                  session_config("accepted"))
            self.assertEqual(result["extracted_facts"], accepted["extracted_facts"])
            self.assertTrue(result["relevant_requirements"][0].satisfied)

    def test_resume_pending_rules_after_reopen(self):
        with persistent_graph(self.database) as graph:
            with patch("src.workers.rules_engine.evaluate_r014", side_effect=RuntimeError("simulated interruption")):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    self.run_certificate(graph, "resume", CERTIFICATE)
        with persistent_graph(self.database) as graph:
            config = session_config("resume")
            self.assertIn("rules_engine", graph.get_state(config).next)
            with patch("processor.Client.chat") as chat:
                result = graph.invoke(None, config)
                chat.assert_not_called()
            self.assertTrue(result["relevant_requirements"][0].satisfied)
            self.assertIn("R014", result["final_response"])

    def test_connection_closes_after_error(self):
        with self.assertRaisesRegex(RuntimeError, "test"):
            with persistent_graph(self.database) as graph:
                connection = graph.checkpointer.conn
                raise RuntimeError("test")
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_empty_session_is_rejected(self):
        with self.assertRaises(ValueError):
            session_config("  ")

    def test_persistence_cli_offline_then_inspect(self):
        from scripts.persistence_demo import main

        for mode in ("--offline", "--inspect"):
            output = io.StringIO()
            with patch("sys.argv", ["persistence_demo", "--db", str(self.database),
                                    "--session", "cli-test", mode]), redirect_stdout(output):
                main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["session_id"], "cli-test")
            self.assertEqual(report["pending_nodes"], [])
            self.assertEqual(report["rule_results_and_provenance"][0]["evaluation_result"],
                             "supplied_and_accepted")
            self.assertTrue(report["extracted_facts"])
            if mode == "--inspect":
                self.assertEqual(report["graph_path_this_execution"], [])


if __name__ == "__main__":
    unittest.main()
