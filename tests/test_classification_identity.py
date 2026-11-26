"""Document identity vs referenced attachments; only model replies are mocked."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from document_classifier import classify_document
from processor import process_document

ROOT = Path(__file__).resolve().parents[1] / "data/synthetic_samples/classification_identity"
CASES = {
    "application_attachments": "ikamet_application_form",
    "student_requirements": "student_enrollment_certificate",
    "insurance_legislation": "health_insurance_certificate",
    "checklist": "unknown",
    "insurance_misleading": "health_insurance_certificate",
    "unknown": "unknown",
    "conflicting_titles": "unknown",
}


class IdentityClassificationTests(unittest.TestCase):
    def test_synthetic_identity_cases(self):
        for name, expected in CASES.items():
            with self.subTest(name=name), patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "true"}):
                client = Mock()
                client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='{"document_type":"unknown"}'))
                result = classify_document((ROOT / (name + ".txt")).read_text(encoding="utf-8"), client, "qwen3:8b")
                self.assertEqual(result.document_type, expected)
                self.assertEqual(result.ambiguous, expected == "unknown")
                self.assertLessEqual(client.chat.call_count, 1)
                if name != "unknown":
                    client.chat.assert_not_called()
                    self.assertEqual(result.method, "indicators")

    def test_primary_quotes_do_not_include_attachment_items(self):
        client = Mock()
        result = classify_document((ROOT / "application_attachments.txt").read_text(encoding="utf-8"), client, "qwen3:8b")
        self.assertTrue(any("Residence Permit Application Form" in quote for quote in result.evidence))
        self.assertFalse(any("health insurance policy" in quote.lower() for quote in result.evidence))

    def test_purpose_without_fixed_title_can_identify_application(self):
        client = Mock()
        result = classify_document("This form confirms that the applicant made a residence application.\n"
            "Attachments\nPrivate health insurance policy", client, "qwen3:8b")
        self.assertEqual(result.document_type, "ikamet_application_form")
        client.chat.assert_not_called()

    def test_two_distinct_populated_roles_identify_untitled_document(self):
        client = Mock()
        result = classify_document("Record\nApplication number: DEMO-1\nPassport number: DEMO-P1\n"
            "Attachments\nHealth Insurance Policy", client, "qwen3:8b")
        self.assertEqual(result.document_type, "ikamet_application_form")
        client.chat.assert_not_called()

    def test_repeated_identifier_alone_is_not_a_vote_count(self):
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='{"document_type":"unknown"}'))
        with patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "false"}):
            result = classify_document("Policy number: DEMO-1\nPolicy number: DEMO-1\nPolicy number: DEMO-1", client, "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")
        client.chat.assert_not_called()

    def test_placeholder_fields_do_not_establish_identity(self):
        client = Mock()
        with patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "false"}):
            result = classify_document("Application number: N/A\nPassport number: not provided", client, "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")

    def test_uncertain_semantic_fallback_once_with_thinking_disabled(self):
        text = (ROOT / "semantic_student.txt").read_text(encoding="utf-8")
        quote = text.splitlines()[-1]
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "document_type": "student_enrollment_certificate", "evidence": [quote]})))
        with patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "true", "PBA_CLASSIFIER_THINK": "false"}):
            result = classify_document(text, client, "qwen3:8b")
        self.assertEqual(result.document_type, "student_enrollment_certificate")
        self.assertEqual(result.method, "model")
        client.chat.assert_called_once()
        self.assertFalse(client.chat.call_args.kwargs["think"])

    def test_semantic_fallback_cannot_cite_an_attachment(self):
        text = "Administrative notes.\nAttachments\nPrivate health insurance policy"
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "document_type": "health_insurance_certificate", "evidence": ["Private health insurance policy"]})))
        result = classify_document(text, client, "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")
        self.assertTrue(result.validation_issues)

    def test_broad_model_quote_cannot_launder_attachment_identity(self):
        text = "Administrative notes.\nAttachments\nPrivate health insurance policy"
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "document_type": "health_insurance_certificate", "evidence": [text]})))
        result = classify_document(text, client, "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")

    def test_page_boundary_exposes_real_second_document_identity(self):
        client = Mock()
        result = classify_document("Residence Permit Application Form\nAttachments\nHealth Insurance Policy\n"
            "\f\nHealth Insurance Policy\nPolicy number: DEMO-1\nInsured: Casey Sample", client, "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")
        client.chat.assert_not_called()

    def test_fallback_configuration_does_not_change_extraction_timeout(self):
        text = (ROOT / "semantic_student.txt").read_text(encoding="utf-8")
        replies = [dict(document_type="student_enrollment_certificate", evidence=[text.splitlines()[-1]]), {}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "note.txt"
            path.write_text(text, encoding="utf-8")
            with patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "true", "PBA_CLASSIFIER_THINK": "true"}), \
                    patch("processor.Client") as client:
                client.return_value.chat.side_effect = [SimpleNamespace(message=SimpleNamespace(content=json.dumps(r))) for r in replies]
                process_document(path, timeout=7)
                client.assert_called_once_with(host="http://127.0.0.1:11434", timeout=7)
                calls = client.return_value.chat.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertTrue(calls[0].kwargs["think"])
                self.assertFalse(calls[1].kwargs["think"])


if __name__ == "__main__":
    unittest.main()
