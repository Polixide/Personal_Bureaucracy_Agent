"""Synthetic regressions for PDF reading order, context and evidence retention."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from document_classifier import classify_document
from document_context import select_context
from document_identity import identity_evidence
from processor import process_document
from document_profiles import REGISTRY
from document_evidence import labeled_name_evidence
from scripts.private_document_check import check


class PDFPipelineRecoveryTests(unittest.TestCase):
    def test_separate_labeled_name_components_have_exact_source_quote(self):
        text = "Surname\n:\nSample Family\nGiven Name\n:\nAda"
        evidence = labeled_name_evidence("Ada Sample Family", text)
        self.assertIsNotNone(evidence)
        self.assertIn(evidence["snippet"], text)
        self.assertEqual(evidence["normalized_value"], "Ada Sample Family")

    def test_unlabeled_or_invented_name_components_are_not_grounded(self):
        self.assertIsNone(labeled_name_evidence("Ada Example", "Ada\nExample"))
        self.assertIsNone(labeled_name_evidence("Ada Invented", "Given Name: Ada\nSurname: Example"))

    def test_conflicting_people_do_not_supply_composed_name(self):
        text = "Given Name: Ada\nSurname: Example\nGiven Name: Bea\nSurname: Other"
        self.assertIsNone(labeled_name_evidence("Ada Example", text))

    def test_name_components_cannot_cross_pages_or_distant_sections(self):
        self.assertIsNone(labeled_name_evidence("Ada Example", "Given Name: Ada\n\f\nSurname: Example"))
        text = "Given Name: Ada\n" + "Other field\n" * 15 + "Surname: Example"
        self.assertIsNone(labeled_name_evidence("Ada Example", text))

    def test_name_components_preserve_unicode(self):
        evidence = labeled_name_evidence("Çağla Örnek", "Adı: Çağla\nSoyadı: Örnek")
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["normalized_value"], "Çağla Örnek")

    def test_composed_application_name_reconciles_bad_model_quote(self):
        text = "Residence Permit Application Form\nGiven Name: Ada\nSurname: Example"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.txt"
            path.write_text(text, encoding="utf-8")
            reply = {"applicant_name": "Ada Example", "source_snippets": {"applicant_name": "Invented citation"}}
            with patch("processor.Client.chat", return_value=SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))):
                result = process_document(path)
        self.assertEqual(result.applicant_name, "Ada Example")
        self.assertNotIn("applicant_name", result.ambiguous_fields)
        self.assertFalse(any("applicant_name:" in issue for issue in result.validation_issues))
        self.assertIn(result.source_snippets["applicant_name"], text)
        self.assertTrue(any("snippet" in note for note in result.correction_notes))

    def test_wrapped_policy_title_late_in_extracted_order(self):
        text = "\n".join(["Synthetic table row" for _ in range(40)]) + "\nHealth Insurance\nPolicy"
        client = Mock()
        result = classify_document(text, client, "qwen3:8b")
        self.assertEqual(result.document_type, "health_insurance_certificate")
        self.assertTrue(all(quote in text for quote in result.evidence))
        client.chat.assert_not_called()

    def test_numbered_self_description_is_not_an_attachment_reference(self):
        text = ("Application details\n" + "\n".join(["Synthetic row" for _ in range(30)]) +
            "\n2. This form confirms that its holder made a residence application. "
            "The holder must keep a copy for later administrative steps.\n\f\n" +
            "\n".join(["Form field row" for _ in range(20)]) + "\nPrivate health insurance\npolicy")
        result = classify_document(text, Mock(), "qwen3:8b")
        self.assertEqual(result.document_type, "ikamet_application_form")
        self.assertTrue(any(item["role"] == "purpose" for item in identity_evidence(text)["primary"]))

    def test_inline_attachment_mention_does_not_start_a_persistent_section(self):
        text = ("The applicant must retain supporting documents for later use.\n"
                "This form shows that its holder made a residence application.")
        result = classify_document(text, Mock(), "qwen3:8b")
        self.assertEqual(result.document_type, "ikamet_application_form")

    def test_wrapped_attachment_title_does_not_escape_list_context(self):
        text = ("Residence Permit Application Form\nRequired supporting documents\n"
                "1. Private Health Insurance\nPolicy\n2. Student Enrollment Certificate")
        self.assertEqual(classify_document(text, Mock(), "qwen3:8b").document_type, "ikamet_application_form")

    def test_real_contradictory_titles_still_require_review(self):
        result = classify_document("Residence Permit Application Form\nHealth Insurance Policy", Mock(), "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")

    def test_unsupported_control_document_retains_generic_evidence(self):
        text = ("Declaration and control document\nSupporting documents\n"
                "Residence Permit Application Form\nStudent Enrollment Certificate\nRecord date: 2028-09-18")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "control.txt"
            path.write_text(text, encoding="utf-8")
            with patch("processor.Client.chat") as chat:
                result = process_document(path)
                chat.assert_not_called()
        self.assertEqual(result.document_type, "unknown")
        self.assertIsNone(result.application_number)
        self.assertTrue(result.validation_issues)
        self.assertTrue(any(item.original_value == "2028-09-18" and item.field is None for item in result.source_evidence))

    def test_secondary_declaration_heading_does_not_override_main_certificate(self):
        result = classify_document("Student Enrollment Certificate\nDeclaration\nStudent ID: DEMO-1", Mock(), "qwen3:8b")
        self.assertEqual(result.document_type, "student_enrollment_certificate")

    def test_bounded_context_retains_first_page_and_identity_quote(self):
        first = "Application number: DEMO-A1\nApplicant name: Sample Person\n"
        text = first + "\f\n" + "\n".join(["Long unrelated instruction text" for _ in range(1000)]) + "\nResidence Permit Application Form"
        context, info = select_context(text, 2000, quotes=["Residence Permit Application Form"])
        self.assertLessEqual(len(context), 2000)
        self.assertIn("Application number: DEMO-A1", context)
        self.assertIn("Residence Permit Application Form", context)
        self.assertFalse(info["full_text"])
        self.assertEqual(info["pages"], [1, 2])

    def test_classification_schema_requires_quotes_without_free_prose_method(self):
        text = "Example University attests that a student is registered."
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "document_type": "student_enrollment_certificate", "evidence": [text], "ambiguous": False})))
        with patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "true"}):
            result = classify_document(text, client, "qwen3:8b")
        request = client.chat.call_args.kwargs
        self.assertEqual(result.document_type, "student_enrollment_certificate")
        self.assertEqual(set(request["format"]["required"]), {"document_type", "evidence", "ambiguous"})
        self.assertNotIn("method", request["format"]["properties"])
        self.assertGreater(request["options"]["num_ctx"], 4096)

    def test_known_type_without_model_quotes_is_not_blindly_accepted(self):
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='{"document_type":"student_enrollment_certificate"}'))
        with patch.dict(os.environ, {"PBA_CLASSIFIER_FALLBACK": "true"}):
            result = classify_document("Example University attests student registration.", client, "qwen3:8b")
        self.assertEqual(result.document_type, "unknown")

    def test_private_report_never_contains_source_values_or_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "PRIVATE-FILENAME.txt"
            path.write_text("Student Enrollment Certificate\nHolder name: PRIVATE PERSON\n"
                "Student ID: PRIVATE-ID-42\nInstitution: Sample University\nIssue date: 2028-09-18", encoding="utf-8")
            reply = dict(holder_name="PRIVATE PERSON", student_id="PRIVATE-ID-42", institution="Sample University", issue_date="2028-09-18")
            with patch("ollama.Client.chat", return_value=SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))):
                report = check(path, model="qwen3:8b", host="http://127.0.0.1:11434", timeout=5)
        serialized = json.dumps(report)
        for private in ("PRIVATE PERSON", "PRIVATE-ID-42", "PRIVATE-FILENAME"):
            self.assertNotIn(private, serialized)
        self.assertEqual(report["document_type"], "student_enrollment_certificate")
        self.assertEqual(len(report["calls"]), 1)

    def test_extraction_contract_requires_nullable_values_not_unbounded_prose(self):
        schema = REGISTRY["ikamet_application_form"].schema.model_json_schema()
        self.assertEqual(set(schema["required"]), {"applicant_name", "application_number", "passport_number", "application_date", "permit_type"})
        self.assertNotIn("source_snippets", schema["required"])
        self.assertFalse(schema["properties"]["source_snippets"]["additionalProperties"])
        self.assertEqual(schema["properties"]["source_snippets"]["properties"]["applicant_name"]["maxLength"], 160)
        self.assertIn({"type": "null"}, schema["properties"]["application_number"]["anyOf"])

    def test_runtime_context_and_generation_limits_are_configurable(self):
        from config import ollama_options
        with patch.dict(os.environ, {"PBA_OLLAMA_NUM_CTX": "6144", "PBA_OLLAMA_MAX_TOKENS": "512"}):
            self.assertEqual(ollama_options(), {"temperature": 0, "num_ctx": 6144, "num_predict": 512})


if __name__ == "__main__":
    unittest.main()
