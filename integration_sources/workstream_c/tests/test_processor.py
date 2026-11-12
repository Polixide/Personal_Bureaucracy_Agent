"""Offline integration tests: real extraction and validation, mocked model output."""

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from processor import DocumentProcessingError, process_document
from schemas import DocumentInfo

SAMPLES = Path(__file__).resolve().parents[1] / "data/synthetic_samples"
ENROLLMENT = dict(document_type="student_enrollment_certificate", holder_name="Alex Example",
                  student_id="TEST-0001", institution="Example University", issue_date="2026-10-01")
INSURANCE = dict(document_type="health_insurance_certificate", holder_name="Alex Example",
                 institution="Example Insurance", policy_number="TEST-POL-0001",
                 issue_date="2026-10-01", expiry_date="2027-09-30")


class ProcessorTests(unittest.TestCase):
    def run_sample(self, filename: str, payload: dict) -> DocumentInfo:
        with patch("processor.Client") as client:
            client.return_value.chat.return_value = SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)))
            result = process_document(SAMPLES / filename)
            request = client.return_value.chat.call_args.kwargs
            self.assertEqual(request["model"], "qwen3:4b")
            self.assertEqual(request["format"], DocumentInfo.model_json_schema())
            self.assertEqual(request["messages"][1]["content"],
                             (SAMPLES / filename).read_text(encoding="utf-8"))
            return result

    def test_enrollment(self):
        result = self.run_sample("student_enrollment.txt", ENROLLMENT)
        self.assertEqual(result.issue_date, date(2026, 10, 1))
        self.assertEqual(result.missing_fields, [])

    def test_insurance(self):
        result = self.run_sample("health_insurance.txt", INSURANCE)
        self.assertEqual(result.policy_number, "TEST-POL-0001")
        self.assertEqual(result.validation_issues, [])

    def test_missing_fields(self):
        payload = {**ENROLLMENT, "student_id": None, "issue_date": None,
                   "missing_fields": ["policy_number"], "validation_issues": ["model claim"]}
        result = self.run_sample("enrollment_missing.txt", payload)
        self.assertEqual(result.missing_fields, ["student_id", "issue_date"])
        self.assertEqual(result.validation_issues, [])

    def test_ambiguity_removes_uncertain_values(self):
        result = self.run_sample("insurance_ambiguous.txt", {
            **INSURANCE, "policy_number": "TEST-A", "issue_date": None,
            "ambiguous_fields": ["policy_number", "issue_date"]})
        self.assertIsNone(result.policy_number)
        self.assertIsNone(result.issue_date)
        self.assertIn("policy_number", result.missing_fields)
        self.assertTrue(any("ambiguous" in issue for issue in result.validation_issues))

    def test_inconsistent_dates(self):
        result = self.run_sample("insurance_inconsistent.txt", {
            **INSURANCE, "expiry_date": "2026-09-01"})
        self.assertIn("expiry_date precedes issue_date", result.validation_issues)

    def test_invalid_date(self):
        result = self.run_sample("insurance_invalid_date.txt", {
            **INSURANCE, "issue_date": "2026-02-30"})
        self.assertIsNone(result.issue_date)
        self.assertIn("issue_date", result.missing_fields)
        self.assertTrue(any("invalid date" in issue for issue in result.validation_issues))

    def test_unknown_and_empty_fields(self):
        result = DocumentInfo(holder_name="  ")
        self.assertIsNone(result.holder_name)
        self.assertTrue(result.validation_issues)

    def test_schema_rejects_invalid_output(self):
        for payload in ({"document_type": "passport"}, {"unexpected": True},
                        {"ambiguous_fields": ["invented_field"]}, {"student_id": 123}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                DocumentInfo.model_validate(payload)

    def test_model_and_host_configuration(self):
        with patch("processor.Client") as client:
            client.return_value.chat.return_value = SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(ENROLLMENT)))
            process_document(SAMPLES / "student_enrollment.txt", model="qwen3:8b", timeout=10)
            self.assertEqual(client.return_value.chat.call_args.kwargs["model"], "qwen3:8b")
            client.assert_called_once_with(host="http://127.0.0.1:11434", timeout=10)

    def test_connection_and_bad_json(self):
        with patch("processor.Client") as client:
            client.return_value.chat.side_effect = httpx.ConnectError("offline")
            with self.assertRaises(DocumentProcessingError):
                process_document(SAMPLES / "student_enrollment.txt")
            client.return_value.chat.side_effect = None
            for content in ("not JSON", "[]"):
                client.return_value.chat.return_value = SimpleNamespace(
                    message=SimpleNamespace(content=content))
                with self.assertRaises(DocumentProcessingError):
                    process_document(SAMPLES / "student_enrollment.txt")

    def test_empty_text_skips_model(self):
        with patch("processor.extract_text", return_value=" "), patch("processor.Client") as client:
            with self.assertRaisesRegex(ValueError, "no text"):
                process_document("empty.txt")
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
