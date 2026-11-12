"""Synthetic layout regressions; only Ollama chat replies are mocked."""

from datetime import date
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

from document_evidence import snippet_in_source
from processor import process_document

SAMPLE = Path(__file__).resolve().parents[1] / "data/synthetic_samples/insurance_multiline_layout.txt"


class ReconciliationTests(unittest.TestCase):
    def process(self, text, payload=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "layout.txt"
            path.write_text(text, encoding="utf-8")
            with patch("processor.Client.chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload or {"document_type": "unknown"})))):
                return process_document(path)

    def test_multiline_fixture_recovers_unique_values_despite_model_ambiguity(self):
        result = self.process(SAMPLE.read_text(encoding="utf-8"), {
            "document_type": "health_insurance_certificate",
            "holder_name": "Taylor Example", "institution": "Synthetic Insurance",
            "ambiguous_fields": ["policy_number", "issue_date", "coverage_start_date", "coverage_end_date", "expiry_date"]})
        self.assertEqual(result.policy_number, "DEMO-LAYOUT-730")
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertEqual(result.coverage_start_date, date(2028, 9, 20))
        self.assertEqual(result.coverage_end_date, date(2029, 6, 30))
        self.assertEqual(result.expiry_date, result.coverage_end_date)
        self.assertEqual(result.ambiguous_fields, [])
        self.assertEqual(result.missing_fields, [])
        self.assertEqual(result.validation_issues, [])
        self.assertTrue(result.correction_notes)
        policy = next(e for e in result.source_evidence if e.field == "policy_number")
        self.assertEqual(policy.original_value, ": DEMO-LAYOUT-730")
        self.assertEqual(policy.normalized_value, result.policy_number)
        self.assertTrue(all(snippet_in_source(s, SAMPLE.read_text(encoding="utf-8"))
                            for s in result.source_snippets.values()))

    def test_identical_issue_dates_and_empty_repeated_labels_are_not_conflicts(self):
        text = ("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ\n"
                "POLİÇE ONAY TARİHİ: 18/09/2028\nPOLİÇE ONAY TARİHİ: 2028-09-18")
        result = self.process(text)
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertNotIn("issue_date", result.ambiguous_fields)
        self.assertFalse(any(i.startswith("issue_date:") for i in result.validation_issues))

    def test_distinct_issue_dates_remain_ambiguous(self):
        text = ("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ: 18/09/2028\n"
                "POLİÇE ONAY TARİHİ: 19/09/2028")
        result = self.process(text, {"document_type": "health_insurance_certificate", "issue_date": "2028-09-18"})
        self.assertIsNone(result.issue_date)
        self.assertIn("issue_date", result.missing_fields)
        self.assertIn("issue_date", result.ambiguous_fields)
        self.assertNotIn("issue_date", result.source_snippets)
        self.assertEqual(len([i for i in result.validation_issues if i.startswith("issue_date:")]), 1)

    def test_model_invalid_normalization_is_corrected_with_note(self):
        result = self.process(SAMPLE.read_text(encoding="utf-8"), {
            "document_type": "health_insurance_certificate", "issue_date": "2028-02-30",
            "coverage_start_date": "2009-09-28", "coverage_end_date": "3006-06-29"})
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertEqual(result.coverage_end_date, date(2029, 6, 30))
        self.assertTrue(any("issue_date: model value rejected" in n for n in result.correction_notes))
        self.assertNotIn("issue_date", result.ambiguous_fields)

    def test_whitespace_and_unicode_equivalent_model_quote_is_supported(self):
        text = SAMPLE.read_text(encoding="utf-8")
        quote = unicodedata.normalize("NFD", "POLİÇE ONAY TARİHİ : 18/09/2028")
        result = self.process(text, {"document_type": "health_insurance_certificate",
                                   "issue_date": "2028-09-18", "source_snippets": {"issue_date": quote}})
        self.assertFalse(any("snippet" in i for i in result.validation_issues))
        self.assertIn("18/09/2028", result.source_snippets["issue_date"])
        self.assertFalse(snippet_in_source("POLİÇE ONAY TARİHİ 19/09/2028", text))

    def test_decomposed_turkish_label_keeps_policy_value_offsets_correct(self):
        text = unicodedata.normalize("NFD", "SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE   NO\n： DEMO-42")
        result = self.process(text)
        self.assertEqual(result.policy_number, "DEMO-42")

    def test_split_heading_and_standalone_period_separator(self):
        text = ("SAĞLIK SİGORTASI POLİÇESİ\nSEYAHAT BAŞLANGIÇ-\nBİTİŞ TARİHİ\n"
                "18/09/2028-00:00\n-\n20/09/2028-23:59")
        result = self.process(text)
        self.assertEqual(result.coverage_start_date, date(2028, 9, 18))
        self.assertEqual(result.coverage_end_date, date(2028, 9, 20))

    def test_birth_legal_and_footer_dates_cannot_complete_a_period(self):
        for following in ("Doğum tarihi: 20/09/2028", "Kanun: 20/09/2028", "Sayfa 1: 20/09/2028"):
            with self.subTest(following=following):
                result = self.process("SAĞLIK SİGORTASI POLİÇESİ\n"
                    "SEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ\n18/09/2028\n" + following)
                self.assertIsNone(result.coverage_start_date)
                self.assertIsNone(result.coverage_end_date)

    def test_more_than_two_period_dates_are_not_silently_truncated(self):
        result = self.process("SAĞLIK SİGORTASI POLİÇESİ\nSEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ\n"
                              "18/09/2028\n20/09/2028\n21/09/2028")
        self.assertIsNone(result.coverage_start_date)
        self.assertIsNone(result.coverage_end_date)

    def test_invalid_or_ambiguous_source_is_not_rescued_by_model(self):
        for value in ("31/02/2028", "03/04/2028"):
            with self.subTest(value=value):
                result = self.process("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ: " + value,
                    {"document_type": "health_insurance_certificate", "issue_date": "2028-04-03"})
                self.assertIsNone(result.issue_date)
                self.assertTrue(result.validation_issues)

    def test_policy_label_cannot_consume_an_identity_or_phone_field(self):
        for label in ("T.C. KİMLİK NO", "VERGİ NO", "TELEFON"):
            with self.subTest(label=label):
                result = self.process("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE NO\n" + label + ": 1234567890")
                self.assertIsNone(result.policy_number)

    def test_spaces_around_date_and_label_punctuation_are_harmless(self):
        result = self.process("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ: 18 / 09 / 2028", {
            "document_type": "health_insurance_certificate", "issue_date": "2028-09-18",
            "source_snippets": {"issue_date": "POLİÇE ONAY TARİHİ : 18/09/2028"}})
        self.assertEqual(result.issue_date, date(2028, 9, 18))
        self.assertFalse(any(i.startswith("issue_date:") for i in result.validation_issues))
        evidence = next(e for e in result.source_evidence if e.field == "issue_date")
        self.assertEqual(evidence.original_value, "18 / 09 / 2028")

    def test_footer_and_birth_dates_do_not_supply_or_disambiguate_issue_date(self):
        for body in ("Page 1 issue date: 18/09/2028", "Doğum tarihi: 18/09/1994"):
            with self.subTest(body=body):
                result = self.process("SAĞLIK SİGORTASI POLİÇESİ\n" + body +
                                      "\nPOLİÇE ONAY TARİHİ: 03/04/2028")
                self.assertIsNone(result.issue_date)

    def test_verified_coverage_values_keep_bad_model_citations_as_diagnostics(self):
        result = self.process(SAMPLE.read_text(encoding="utf-8"), {
            "document_type": "health_insurance_certificate", "holder_name": "Taylor Example",
            "institution": "Synthetic Insurance", "coverage_start_date": "2028-09-20",
            "coverage_end_date": "2029-06-30", "source_snippets": {
                "coverage_start_date": "Invented start citation 20/09/2028",
                "coverage_end_date": "Invented end citation 30/06/2029"}})
        self.assertEqual(result.coverage_start_date, date(2028, 9, 20))
        self.assertEqual(result.coverage_end_date, date(2029, 6, 30))
        self.assertEqual(result.validation_issues, [])
        self.assertEqual(result.missing_fields, [])
        for field in ("coverage_start_date", "coverage_end_date"):
            self.assertTrue(any(note.startswith(field + ": model source snippet not found")
                                and "independently verified" in note for note in result.correction_notes))
            self.assertNotIn("Invented", result.source_snippets[field])

    def test_genuinely_fabricated_date_and_citation_still_require_review(self):
        result = self.process("SAĞLIK SİGORTASI POLİÇESİ\nNo dates supplied.", {
            "document_type": "health_insurance_certificate", "coverage_start_date": "2028-09-20",
            "source_snippets": {"coverage_start_date": "Invented coverage 20/09/2028"}})
        self.assertIsNone(result.coverage_start_date)
        self.assertIn("coverage_start_date", result.ambiguous_fields)
        self.assertTrue(any("model source snippet not found" in issue for issue in result.validation_issues))
        self.assertFalse(any("independently verified" in note for note in result.correction_notes))

    def test_conflicting_source_and_bad_model_citation_still_require_review(self):
        result = self.process("SAĞLIK SİGORTASI POLİÇESİ\nPOLİÇE ONAY TARİHİ: 18/09/2028\n"
                              "POLİÇE ONAY TARİHİ: 19/09/2028", {
            "document_type": "health_insurance_certificate", "issue_date": "2028-09-18",
            "source_snippets": {"issue_date": "Fabricated quote 18/09/2028"}})
        self.assertIsNone(result.issue_date)
        self.assertTrue(any("model source snippet not found" in issue for issue in result.validation_issues))

    def test_reversed_coverage_does_not_downgrade_citation_failures(self):
        result = self.process("SAĞLIK SİGORTASI POLİÇESİ\n"
                              "SEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ: 20/09/2028 - 18/09/2028", {
            "document_type": "health_insurance_certificate", "source_snippets": {
                "coverage_end_date": "Fabricated end 18/09/2028",
                "expiry_date": "Fabricated expiry 18/09/2028"}})
        self.assertIn("coverage_end_date precedes coverage_start_date", result.validation_issues)
        for field in ("coverage_end_date", "expiry_date"):
            self.assertTrue(any(issue.startswith(field + ": model source snippet not found")
                                for issue in result.validation_issues))

    def test_quote_only_holder_name_has_no_independent_candidate_verification(self):
        result = self.process(SAMPLE.read_text(encoding="utf-8"), {
            "document_type": "health_insurance_certificate", "holder_name": "Taylor Example",
            "source_snippets": {"holder_name": "Invented holder quote"}})
        self.assertTrue(any(issue.startswith("holder_name: model source snippet not found")
                            for issue in result.validation_issues))


if __name__ == "__main__":
    unittest.main()
