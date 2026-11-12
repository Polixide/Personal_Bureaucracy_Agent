"""Verify demo reporting using the real graph, fixtures, and deterministic rules."""

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_demo import CERTIFICATE, SAMPLES, main, run_offline_demo


class DemoTests(unittest.TestCase):
    def test_all_offline_scenarios(self):
        with redirect_stdout(io.StringIO()):
            reports = run_offline_demo()
        self.assertEqual(len(reports), 7)
        self.assertTrue(all(r["test_result"] == "PASS" for r in reports))
        good = reports[0]
        self.assertIn("document_analyzer", good["graph_path"])
        self.assertIn("rules_engine", good["graph_path"])
        self.assertTrue(good["rule_results_and_provenance"][0]["source_metadata"])
        for report in reports[1:]:
            self.assertNotIn("human_approval", report["graph_path"])
        for report in reports[2:5]:
            facts = next(iter(report["extracted_facts"].values()))
            self.assertTrue(facts["missing_fields"] or facts["ambiguous_fields"])

    def test_live_cli_contract_without_live_server(self):
        output = io.StringIO()
        with patch("sys.argv", ["run_demo", str(SAMPLES / "student_enrollment.txt")]), \
                patch("processor.Client.chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(CERTIFICATE)))) as chat, \
                redirect_stdout(output):
            main()
        report = json.loads(output.getvalue())
        self.assertEqual(chat.call_args.kwargs["model"], "qwen3:8b")
        self.assertEqual(report["execution_mode"], "LIVE: real Ollama")
        self.assertIn("MOCKED", report["mode_note"])


if __name__ == "__main__":
    unittest.main()
