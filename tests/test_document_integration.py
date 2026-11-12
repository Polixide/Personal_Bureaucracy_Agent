"""Execute the real graph and processor, mocking only the Ollama boundary."""

import json
from datetime import date
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import pymupdf
from ollama import ResponseError

from src.core.graph import build_graph
from src.core.state import Document, UserEvent, new_run

SAMPLES = Path(__file__).resolve().parents[1] / "data/synthetic_samples"
ENROLLMENT = dict(document_type="student_enrollment_certificate", holder_name="Alex Example",
                  student_id="TEST-0001", institution="Example University", issue_date="2026-10-01")
INSURANCE = dict(document_type="health_insurance_certificate", holder_name="Alex Example",
                 policy_number="TEST-POL-0001", institution="Example Insurance",
                 issue_date="2026-10-01", expiry_date="2027-09-30")


class DocumentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def run_graph(self, path, payload=None, error=None, **state_updates):
        state = {**new_run(UserEvent(kind="upload", file_path=str(path)), "test-user"),
                 **state_updates}
        with patch("processor.Client.chat") as chat:
            chat.side_effect = error
            chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
            states = list(self.graph.stream(state, stream_mode="values", config={
                "configurable": {"document_model": "qwen3:8b", "document_timeout": 5}}))
        return states[-1], states, chat

    def test_enrollment_mapping_and_fact_preservation(self):
        path = SAMPLES / "student_enrollment.txt"
        result, _, chat = self.run_graph(path, ENROLLMENT,
            extracted_facts={"appointment_date": date(2026, 11, 20), "existing": {"value": 1}})
        document = result["documents"][0]
        self.assertEqual(document.doc_type, "student_certificate")
        self.assertEqual(document.status, "analyzed")
        self.assertEqual(result["extracted_facts"][path.name]["student_id"], "TEST-0001")
        self.assertEqual(result["extracted_facts"]["existing"], {"value": 1})
        self.assertEqual(result["extracted_facts"]["appointment_date"], date(2026, 11, 20))
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(chat.call_args.kwargs["model"], "qwen3:8b")
        self.assertIn("Student Enrollment Certificate", chat.call_args.kwargs["messages"][1]["content"])

    def test_insurance_pdf_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "insurance.PDF"
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((72, 72),
                    "Synthetic Health Insurance Certificate\nHolder name: Alex Example\n"
                    "Institution: Example Insurance\nPolicy number: TEST-POL-0001\n"
                    "Issue date: 2026-10-01\nExpiry date: 2027-09-30")
                pdf.save(path)
            result, _, chat = self.run_graph(path, INSURANCE)
        self.assertEqual(result["documents"][0].doc_type, "health_insurance")
        self.assertEqual(result["documents"][0].status, "analyzed")
        self.assertIn("Synthetic Health Insurance", chat.call_args.kwargs["messages"][1]["content"])

    def assert_rejected(self, result, states):
        self.assertEqual(result["documents"][0].status, "rejected")
        self.assertTrue(result["validation_errors"])
        self.assertIn("Stopped safely", result["final_response"])
        self.assertFalse(any("blockers" in state for state in states))  # state_update bypassed
        self.assertTrue(all(not r.satisfied for r in result["relevant_requirements"]))

    def test_quality_flags_survive_and_block_success(self):
        for payload in (
            {**ENROLLMENT, "student_id": None},
            {**ENROLLMENT, "ambiguous_fields": ["holder_name"]},
            {**ENROLLMENT, "expiry_date": "2026-09-01"},
            {**ENROLLMENT, "document_type": "unknown"},
            {**ENROLLMENT, "issue_date": None, "ambiguous_fields": ["issue_date"]},
        ):
            with self.subTest(payload=payload):
                result, states, _ = self.run_graph(SAMPLES / "student_enrollment.txt", payload)
                self.assert_rejected(result, states)
                facts = result["extracted_facts"]["student_enrollment.txt"]
                self.assertTrue(facts["missing_fields"] or facts["ambiguous_fields"] or facts["validation_issues"])

    def test_model_failures_become_graph_issues(self):
        for error in (httpx.ConnectError("offline"), httpx.ReadTimeout("timeout"),
                      ResponseError("model not found", status_code=404)):
            with self.subTest(error=error):
                result, states, _ = self.run_graph(SAMPLES / "student_enrollment.txt", error=error)
                self.assert_rejected(result, states)
                self.assertEqual(result["validation_errors"][0].code, "SCHEMA_ERROR")

    def test_invalid_schema_becomes_graph_issue(self):
        result, states, _ = self.run_graph(SAMPLES / "student_enrollment.txt", {**ENROLLMENT, "issue_date": 0})
        self.assert_rejected(result, states)

    def test_malformed_model_json_becomes_graph_issue(self):
        path = SAMPLES / "student_enrollment.txt"
        with patch("processor.Client.chat", return_value=SimpleNamespace(
                message=SimpleNamespace(content="not JSON"))):
            states = list(self.graph.stream(
                new_run(UserEvent(kind="upload", file_path=str(path)), "test-user"),
                stream_mode="values"))
        self.assert_rejected(states[-1], states)

    def test_missing_file_clears_old_facts_and_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.txt"
            old = Document(doc_id=path.name, file_path=str(path), status="analyzed",
                           doc_type="student_certificate", issue_date=date(2026, 1, 1))
            result, states, chat = self.run_graph(path, ENROLLMENT, documents=[old],
                                                extracted_facts={path.name: ENROLLMENT})
        self.assert_rejected(result, states)
        self.assertIsNone(result["extracted_facts"][path.name])
        self.assertIsNone(result["documents"][0].doc_type)
        self.assertIsNone(result["documents"][0].issue_date)
        chat.assert_not_called()

    def test_empty_and_invalid_files_skip_ollama(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, content in (("empty.txt", ""), ("invalid.pdf", "not a PDF")):
                with self.subTest(name=name):
                    path = Path(folder) / name
                    path.write_text(content, encoding="utf-8")
                    result, states, chat = self.run_graph(path)
                    self.assert_rejected(result, states)
                    chat.assert_not_called()

    def test_images_rejected_at_gate_zero(self):
        for extension in ("png", "jpg"):
            with self.subTest(extension=extension):
                result, _, chat = self.run_graph(f"image.{extension}")
                self.assertEqual(result["validation_errors"][0].source, "gate_0")
                self.assertIn("Stopped safely", result["final_response"])
                chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
