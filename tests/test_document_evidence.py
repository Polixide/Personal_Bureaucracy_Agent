"""Source-grounded hybrid regressions; all documents are synthetic."""

from datetime import date
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf

from document_evidence import normalize_date
from processor import process_document

SAMPLE = Path(__file__).resolve().parents[1] / "data/synthetic_samples/insurance_numeric_dates.txt"


class EvidenceTests(unittest.TestCase):
    def process_text(self, text, payload):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.txt"
            path.write_text(text, encoding="utf-8")
            with patch("processor.Client.chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload)))) as chat:
                result = process_document(path)
        return result, chat.call_args.kwargs

    def test_numeric_formats_and_calendar_validity(self):
        for raw, expected in (("18/09/2028", "2028-09-18"), ("2028-09-18", "2028-09-18"),
                              ("18.09.2028", "2028-09-18"), ("18-09-2028", "2028-09-18"),
                              ("31/02/2028", None), ("03/04/2028", None),
                              ("2028-02-30", None)):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_date(raw), expected)
        self.assertEqual(normalize_date("03/04/2028", "dmy"), "2028-04-03")
        self.assertEqual(normalize_date("03/04/2028", "mdy"), "2028-03-04")

    def test_labeled_source_fills_missing_insurance_fields(self):
        text = SAMPLE.read_text(encoding="utf-8")
        result, request = self.process_text(text, {"document_type": "unknown"})
        self.assertEqual(result.document_type, "health_insurance_certificate")
        self.assertEqual(result.policy_number, "DEMO-TR-2048")
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertEqual(result.coverage_start_date, date(2028, 9, 20))
        self.assertEqual(result.coverage_end_date, date(2029, 6, 30))
        self.assertEqual(result.expiry_date, result.coverage_end_date)
        self.assertIn("holder_name", result.missing_fields)
        self.assertIn("institution", result.missing_fields)
        self.assertEqual(request["messages"][1]["content"], text)
        supplied = json.loads(request["messages"][2]["content"].split("\n", 1)[1])
        self.assertEqual(supplied["date_order"], "dmy")
        self.assertTrue(any(e.original_value == "18/09/2028" and e.field == "issue_date"
                            for e in result.source_evidence))
        self.assertTrue(any(e.field is None and e.original_value == "15/03/1994"
                            for e in result.source_evidence))
        self.assertTrue(all(e.snippet in text for e in result.source_evidence))

    def test_digit_rearrangement_is_rejected_and_audited(self):
        result, _ = self.process_text(SAMPLE.read_text(encoding="utf-8"), {
            "document_type": "unknown", "issue_date": "1809-09-28",
            "coverage_start_date": "2009-09-28", "coverage_end_date": "3006-06-29"})
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertEqual(result.coverage_start_date, date(2028, 9, 20))
        self.assertEqual(result.coverage_end_date, date(2029, 6, 30))
        self.assertTrue(any("model value rejected" in note for note in result.correction_notes))

    def test_unsupported_model_date_without_labeled_evidence_becomes_null(self):
        result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\nKanun: 18/09/2028", {
            "document_type": "unknown", "issue_date": "2028-09-18", "policy_number": "999999"})
        self.assertIsNone(result.issue_date)
        self.assertIsNone(result.policy_number)
        self.assertIn("issue_date", result.missing_fields)
        self.assertTrue(result.validation_issues)

    def test_ambiguous_and_conflicting_dates_remain_null(self):
        for body in ("POLİÇE ONAY TARİHİ: 03/04/2028",
                     "POLİÇE ONAY TARİHİ: 18/09/2028\nPOLİÇE ONAY TARİHİ: 19/09/2028",
                     "POLİÇE ONAY TARİHİ: 31/02/2028"):
            with self.subTest(body=body):
                result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\n" + body, {"document_type": "unknown"})
                self.assertIsNone(result.issue_date)
                self.assertIn("issue_date", result.ambiguous_fields)

    def test_explicit_format_establishes_ambiguous_numeric_order(self):
        result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\nFormat: DD/MM/YYYY\n"
                                      "POLİÇE ONAY TARİHİ: 03/04/2028", {"document_type": "unknown"})
        self.assertEqual(result.issue_date, date(2028, 4, 3))

    def test_legal_date_does_not_establish_policy_date_order(self):
        result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\nKanun: 18/09/1994\n"
                                      "POLİÇE ONAY TARİHİ: 03/04/2028", {"document_type": "unknown"})
        self.assertIsNone(result.issue_date)

    def test_coverage_pair_across_lines_and_reversed_period(self):
        text = "SAĞLIK SİGORTASI POLİÇESİ\nSEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ\n30/06/2029-00:00 - 20/09/2028-23:59"
        result, _ = self.process_text(text, {"document_type": "unknown"})
        self.assertEqual(result.coverage_start_date, date(2029, 6, 30))
        self.assertIn("coverage_end_date precedes coverage_start_date", result.validation_issues)

    def test_incomplete_period_cannot_fill_both_dates(self):
        result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\n"
                                      "SEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ: 18/09/2028", {"document_type": "unknown"})
        self.assertIsNone(result.coverage_start_date)
        self.assertIsNone(result.coverage_end_date)
        self.assertIsNone(result.expiry_date)

    def test_policy_labels_do_not_capture_tax_or_identity_numbers(self):
        text = "SAĞLIK SİGORTASI POLİÇESİ\nVERGİ NO: 1234567890\nT.C. KİMLİK NO: 99999999999"
        result, _ = self.process_text(text, {"document_type": "unknown"})
        self.assertIsNone(result.policy_number)
        result, _ = self.process_text(text + "\nPOLİÇE NO\nDEMO-42", {"document_type": "unknown"})
        self.assertEqual(result.policy_number, "DEMO-42")
        result, _ = self.process_text(text + "\nPOLİÇE NO: DEMO-42\nPOLİÇE NO: DEMO-43", {"document_type": "unknown"})
        self.assertIsNone(result.policy_number)

    def test_law_date_cannot_be_selected_using_a_true_but_irrelevant_quote(self):
        result, _ = self.process_text(SAMPLE.read_text(encoding="utf-8"), {
            "document_type": "health_insurance_certificate", "issue_date": "1994-03-15",
            "source_snippets": {"issue_date": "Yasal referans: 15/03/1994 tarihli kanun, sayılı madde."}})
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertIn("18/09/2028", result.source_snippets["issue_date"])
        self.assertTrue(result.correction_notes)

    def test_source_evidence_cannot_be_supplied_by_model(self):
        result, _ = self.process_text("Student Enrollment Certificate", {
            "source_evidence": [{"invented": True}], "document_type": "unknown"})
        self.assertEqual(result.source_evidence, [])
        self.assertTrue(result.validation_issues)

    def test_real_pdf_page_reference_without_ocr(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((72, 72), "Health Insurance Certificate")
                pdf.new_page().insert_text((72, 72), "Issue date: 2028-09-18\nPolicy number: DEMO-42")
                pdf.save(path)
            with patch("processor.Client.chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content='{"document_type":"unknown"}'))):
                result = process_document(path)
        issue = next(e for e in result.source_evidence if e.field == "issue_date")
        self.assertEqual(issue.page_number, 2)
        self.assertEqual(issue.original_value, "2028-09-18")

    def test_explicit_expiry_conflicting_with_coverage_is_not_overwritten(self):
        text = SAMPLE.read_text(encoding="utf-8") + "\nExpiry date: 2029-07-01"
        result, _ = self.process_text(text, {"document_type": "unknown"})
        self.assertEqual(result.expiry_date, date(2029, 7, 1))
        self.assertEqual(result.coverage_end_date, date(2029, 6, 30))
        self.assertIn("expiry_date conflicts with coverage_end_date", result.validation_issues)

    def test_conflicting_supported_headings_do_not_force_classification(self):
        result, _ = self.process_text("Student Enrollment Certificate\nSAĞLIK SİGORTASI POLİÇESİ", {
            "document_type": "health_insurance_certificate"})
        self.assertEqual(result.document_type, "unknown")
        self.assertTrue(result.validation_issues)

    def test_labeled_dates_do_not_cross_pdf_page_boundaries(self):
        from document_evidence import collect_evidence

        evidence = collect_evidence("POLİÇE ONAY TARİHİ\n\f\n18/09/2028", paginated=True)
        issue = next(e for e in evidence["candidates"] if e["field"] == "issue_date")
        self.assertIsNone(issue["normalized_value"])
        self.assertEqual(issue["page_number"], 1)

    def test_unknown_document_does_not_retain_unverified_policy_number(self):
        result, _ = self.process_text("Unclassified administrative note", {
            "document_type": "unknown", "policy_number": "UNSUPPORTED-42"})
        self.assertIsNone(result.policy_number)
        self.assertTrue(any("required fields cannot be determined" in issue for issue in result.validation_issues))

    def test_normalized_model_equivalent_does_not_create_a_disagreement(self):
        result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ: 18/09/2028", {
            "document_type": "health_insurance_certificate", "issue_date": "18/09/2028"})
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertFalse(any("model value rejected" in issue for issue in result.validation_issues))

    def test_fabricated_quote_with_a_real_date_is_still_flagged(self):
        result, _ = self.process_text("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ: 18/09/2028", {
            "document_type": "health_insurance_certificate", "issue_date": "2028-09-18",
            "source_snippets": {"issue_date": "Fabricated wording 18/09/2028"}})
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertNotIn("Fabricated", result.source_snippets["issue_date"])
        self.assertTrue(any("model source snippet not found" in note for note in result.correction_notes))
        self.assertFalse(any("model source snippet not found" in issue for issue in result.validation_issues))


if __name__ == "__main__":
    unittest.main()
