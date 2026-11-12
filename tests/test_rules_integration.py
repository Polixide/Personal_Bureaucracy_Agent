"""R014 evaluation and offline A/B/C integration, with only Ollama mocked."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from schemas import DocumentInfo
from src.core.graph import build_graph
from src.core.state import Document, UserEvent, new_run
from src.workers.rules_engine import evaluate_r014, load_r014

SAMPLE = Path(__file__).resolve().parents[1] / "data/synthetic_samples/student_enrollment.txt"
CERTIFICATE = dict(document_type="student_enrollment_certificate", holder_name="Alex Example",
                   student_id="TEST-0001", institution="Example University", issue_date="2026-10-01")


class RulesIntegrationTests(unittest.TestCase):
    def test_existing_knowledge_and_provenance(self):
        rule, sources = load_r014()
        self.assertEqual(rule["rule_id"], "R014")
        self.assertEqual([s["source_id"] for s in sources], rule["source_ids"])
        self.assertTrue(all(s["url"].startswith("https://") for s in sources))

    def test_complete_graph_supplied_missing_and_uncertain(self):
        graph = build_graph()
        for payload, expected in (
            (CERTIFICATE, "supplied_and_accepted"),
            ({**CERTIFICATE, "student_id": None}, "manual_review_required"),
            ({**CERTIFICATE, "ambiguous_fields": ["holder_name"]}, "manual_review_required"),
            ({**CERTIFICATE, "issue_date": "2026-02-30"}, "manual_review_required"),
            ({**CERTIFICATE, "issue_date": None}, "manual_review_required"),
            ({**CERTIFICATE, "expiry_date": "2026-09-01"}, "manual_review_required"),
            ({**CERTIFICATE, "document_type": "unknown"}, "manual_review_required"),
            (None, "missing"),
        ):
            with self.subTest(expected=expected, payload=payload):
                event = (UserEvent(kind="upload", file_path=str(SAMPLE)) if payload else
                         UserEvent(kind="message", text="Check my certificate"))
                with patch("processor.Client.chat", return_value=SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload)))) as chat:
                    result = graph.invoke(new_run(event, "demo-user"))
                requirement, = result["relevant_requirements"]
                self.assertEqual(requirement.evaluation_result, expected)
                self.assertEqual(requirement.rule_id, "R014")
                self.assertEqual(requirement.satisfied, expected == "supplied_and_accepted")
                self.assertEqual(requirement.source_metadata[0]["source_id"], requirement.source_id)
                self.assertIn(expected, result["final_response"])
                self.assertIn("MOCKED", result["final_response"])
                self.assertEqual(bool(result["validation_errors"]), expected != "supplied_and_accepted")
                if payload is None:
                    chat.assert_not_called()

    def test_unverified_state_never_satisfies_rule(self):
        document = Document(doc_id="certificate", file_path="unused.txt", status="analyzed",
                            doc_type="student_certificate")
        for raw in (None, {}, {**CERTIFICATE, "student_id": None},
                    {**CERTIFICATE, "ambiguous_fields": ["holder_name"]},
                    {**CERTIFICATE, "validation_issues": ["needs review"]},
                    {**CERTIFICATE, "issue_date": 0}):
            with self.subTest(raw=raw):
                result = evaluate_r014({"documents": [document], "extracted_facts": {"certificate": raw}})
                self.assertFalse(result.satisfied)
                self.assertEqual(result.evaluation_result, "manual_review_required")

    def test_unrelated_document_is_not_a_certificate(self):
        document = Document(doc_id="insurance", file_path="unused.txt", status="analyzed",
                            doc_type="health_insurance")
        self.assertEqual(evaluate_r014({"documents": [document]}).evaluation_result, "missing")

    def test_analyzed_label_without_facts_does_not_override_valid_certificate(self):
        documents = [Document(doc_id=name, file_path="unused.txt", status="analyzed",
                              doc_type="student_certificate") for name in ("missing", "valid")]
        result = evaluate_r014({"documents": documents, "extracted_facts": {
            "valid": DocumentInfo(**CERTIFICATE).model_dump(mode="json")}})
        self.assertTrue(result.satisfied)


if __name__ == "__main__":
    unittest.main()
