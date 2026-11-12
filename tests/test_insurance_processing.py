"""Synthetic Turkish insurance regression checks, with model replies mocked."""

from datetime import date
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

from pydantic import ValidationError

from processor import main, process_document
from schemas import DocumentInfo
from src.core.graph import build_graph
from src.core.state import UserEvent, new_run

SAMPLE = Path(__file__).resolve().parents[1] / "data/synthetic_samples/insurance_turkish.txt"
INSURANCE = dict(document_type="health_insurance_certificate", holder_name="Alex Example",
                 institution="Example Insurance", policy_number="TEST-TR-0001",
                 issue_date="2026-10-01", coverage_start_date="2026-10-10",
                 coverage_end_date="2027-10-09", source_snippets={
                     "issue_date": "POLİÇE ONAY TARİHİ: 2026-10-01\nDÜZENLEME TARİHİ: 2026-10-02",
                     "coverage_end_date": "SEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ: 2026-10-10 - 2027-10-09"})


class InsuranceProcessingTests(unittest.TestCase):
    def process(self, payload, sample=SAMPLE):
        with patch("processor.Client.chat", return_value=SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)))) as chat:
            result = process_document(sample)
        return result, chat.call_args.kwargs

    def test_turkish_request_and_date_distinctions(self):
        result, request = self.process(INSURANCE)
        prompt = request["messages"][0]["content"]
        for label in ("SAĞLIK SİGORTASI POLİÇESİ", "POLİÇE ONAY TARİHİ", "DÜZENLEME TARİHİ",
                      "SEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ"):
            self.assertIn(label, prompt)
            self.assertIn(label, request["messages"][1]["content"])
        self.assertEqual(result.issue_date, date(2026, 10, 1))
        self.assertEqual(result.coverage_start_date, date(2026, 10, 10))
        self.assertEqual(result.coverage_end_date, date(2027, 10, 9))
        self.assertEqual(result.expiry_date, result.coverage_end_date)
        self.assertEqual(result.source_snippets["issue_date"], "POLİÇE ONAY TARİHİ: 2026-10-01")
        self.assertEqual(result.validation_issues, [])
        self.assertEqual(result.missing_fields, [])

    def test_invalid_numeric_and_ambiguous_coverage_dates(self):
        for name in ("coverage_start_date", "coverage_end_date"):
            for value in (0, 0.0, True, False):
                with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                    DocumentInfo(**{**INSURANCE, name: value})
            invalid = DocumentInfo(**{**INSURANCE, name: "2026-02-30"})
            self.assertIsNone(getattr(invalid, name))
            self.assertTrue(invalid.validation_issues)
            ambiguous = DocumentInfo(**{**INSURANCE, "ambiguous_fields": [name]})
            self.assertIsNone(getattr(ambiguous, name))
            if name == "coverage_end_date":
                self.assertIsNone(ambiguous.expiry_date)

    def test_conflicting_and_reversed_coverage_dates(self):
        result = DocumentInfo(**{**INSURANCE, "expiry_date": "2027-11-01"})
        self.assertIn("expiry_date conflicts with coverage_end_date", result.validation_issues)
        result = DocumentInfo(**{**INSURANCE, "coverage_start_date": "2028-01-01"})
        self.assertIn("coverage_end_date precedes coverage_start_date", result.validation_issues)

    def test_missing_dates_and_identifiers_are_not_invented(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.txt"
            path.write_text("SAĞLIK SİGORTASI POLİÇESİ\nNo identifiers or dates supplied.", encoding="utf-8")
            result, _ = self.process({"document_type": "unknown"}, path)
        for name in ("policy_number", "issue_date", "expiry_date", "coverage_start_date", "coverage_end_date"):
            self.assertIsNone(getattr(result, name))
        self.assertIn("policy_number", result.missing_fields)
        self.assertIn("expiry_date", result.missing_fields)

    def test_unmatched_source_snippet_is_removed_and_flagged(self):
        result, _ = self.process({**INSURANCE, "source_snippets": {"policy_number": "invented quote"}})
        self.assertNotIn("invented quote", result.source_snippets.values())
        self.assertTrue(any("source snippet does not support" in note for note in result.correction_notes))

    def test_non_list_validation_issues_raise_validation_error(self):
        with self.assertRaises(ValidationError):
            DocumentInfo(validation_issues=None)

    def test_snippet_schema_exposes_only_exact_field_keys(self):
        schema = DocumentInfo.model_json_schema()["properties"]["source_snippets"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("issue_date", schema["properties"])
        self.assertNotIn("Issue date", schema["properties"])
        with self.assertRaises(ValidationError):
            DocumentInfo(source_snippets={"Issue date": "2026-10-01"})

    def test_turkish_insurance_is_preserved_but_cannot_satisfy_r014(self):
        with patch("processor.Client.chat", return_value=SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(INSURANCE)))):
            result = build_graph().invoke(new_run(
                UserEvent(kind="upload", file_path=str(SAMPLE)), "synthetic-policy-user"))
        self.assertEqual(result["documents"][0].status, "analyzed")
        facts = result["extracted_facts"][SAMPLE.name]
        self.assertEqual(facts["coverage_end_date"], "2027-10-09")
        self.assertEqual(facts["source_snippets"]["coverage_end_date"], INSURANCE["source_snippets"]["coverage_end_date"])
        self.assertFalse(result["relevant_requirements"][0].satisfied)
        self.assertEqual(result["relevant_requirements"][0].evaluation_result, "missing")
        self.assertIn("Stopped safely", result["final_response"])

    def test_turkish_json_cli_works_with_windows_legacy_encoding(self):
        buffer = io.BytesIO()
        output = io.TextIOWrapper(buffer, encoding="cp1252", write_through=True)
        with patch("sys.argv", ["processor.py", str(SAMPLE)]), \
                patch("processor.Client.chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(INSURANCE)))), redirect_stdout(output):
            main()
        report = json.loads(buffer.getvalue().decode("cp1252"))
        self.assertEqual(report["source_snippets"]["issue_date"], "POLİÇE ONAY TARİHİ: 2026-10-01")
        output.close()

    def test_dual_issuance_labels_cannot_silently_verify_model_choice(self):
        result, _ = self.process({**INSURANCE, "issue_date": "2026-10-02",
            "source_snippets": {"issue_date": "DÜZENLEME TARİHİ: 2026-10-02"}})
        self.assertEqual(result.issue_date, date(2026, 10, 1))  # unique, labeled source evidence
        self.assertTrue(any("model value rejected" in note for note in result.correction_notes))
        self.assertEqual(result.source_snippets["issue_date"], "POLİÇE ONAY TARİHİ: 2026-10-01")


if __name__ == "__main__":
    unittest.main()
