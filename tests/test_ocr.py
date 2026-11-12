"""Synthetic PDFs and mocked OCR TextPages; no Tesseract installation required."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from extractor import DocumentExtractionError, extract_text


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "synthetic.pdf"
        self.ocr_document = pymupdf.open()
        self.addCleanup(self.ocr_document.close)
        self.ocr_document.new_page().insert_text((72, 72), "Synthetic scanned certificate")
        self.ocr_page = self.ocr_document[0]
        self.textpage = self.ocr_page.get_textpage()

    def mocked_ocr(self, page, **kwargs):
        self.textpage.parent = page
        return self.textpage

    def make_pdf(self, pages):
        with pymupdf.open() as document:
            for text in pages:
                page = document.new_page()
                if text is None:
                    # Rasterize synthetic text to create an actual image-only page.
                    image = self.ocr_document[0].get_pixmap().tobytes("png")
                    page.insert_image(page.rect, stream=image)
                elif text:
                    page.insert_text((72, 72), text)
            document.save(self.path)

    def test_digital_pdf_never_calls_ocr(self):
        self.make_pdf(["Digital certificate"])
        with patch("pymupdf.Page.get_textpage_ocr") as ocr:
            self.assertIn("Digital certificate", extract_text(self.path))
            ocr.assert_not_called()

    def test_scanned_pdf_default_english(self):
        self.make_pdf([None])
        with patch.dict(os.environ, {"PBA_OCR_LANGUAGES": "", "TESSDATA_PREFIX": ""}), \
                patch("pymupdf.Page.get_textpage_ocr", autospec=True, side_effect=self.mocked_ocr) as ocr:
            self.assertIn("Synthetic scanned certificate", extract_text(self.path))
            self.assertEqual(ocr.call_count, 1)
            self.assertEqual(ocr.call_args.kwargs, dict(language="eng", dpi=300, full=True, tessdata=None))

    def test_mixed_pdf_preserves_page_order_and_language_settings(self):
        self.make_pdf(["Digital first", None, "Digital last"])
        with patch.dict(os.environ, {"PBA_OCR_LANGUAGES": "eng+tur", "TESSDATA_PREFIX": "test-tessdata"}), \
                patch("pymupdf.Page.get_textpage_ocr", autospec=True, side_effect=self.mocked_ocr) as ocr:
            text = extract_text(self.path)
            self.assertLess(text.index("Digital first"), text.index("Synthetic scanned"))
            self.assertLess(text.index("Synthetic scanned"), text.index("Digital last"))
            self.assertEqual(ocr.call_count, 1)
            self.assertEqual(ocr.call_args.kwargs, dict(language="eng+tur", dpi=300, full=True, tessdata="test-tessdata"))

    def test_ocr_failure_never_returns_partial_digital_text(self):
        self.make_pdf(["Digital first", None])
        for error in (RuntimeError("OCR unavailable"), ValueError("missing language data"),
                      OSError("cannot load data")):
            with self.subTest(error=error), \
                    patch("pymupdf.Page.get_textpage_ocr", side_effect=error), \
                    self.assertRaisesRegex(DocumentExtractionError, "Page 2.*TESSDATA_PREFIX"):
                extract_text(self.path)

    def test_empty_ocr_result_raises(self):
        self.make_pdf([None])
        with pymupdf.open() as blank:
            blank.new_page()
            blank_page = blank[0]
            self.textpage = blank_page.get_textpage()
            with patch("pymupdf.Page.get_textpage_ocr", autospec=True, side_effect=self.mocked_ocr), \
                    self.assertRaisesRegex(DocumentExtractionError, "no extractable text after OCR"):
                extract_text(self.path)

    def test_replacement_glyphs_trigger_ocr(self):
        self.make_pdf([None])
        original = pymupdf.Page.get_text
        def unreadable(page, *args, **kwargs):
            return original(page, *args, **kwargs) if "textpage" in kwargs else "\ufffd \ufffd"
        with patch("pymupdf.Page.get_text", unreadable), \
                patch("pymupdf.Page.get_textpage_ocr", autospec=True, side_effect=self.mocked_ocr) as ocr:
            self.assertIn("Synthetic scanned", extract_text(self.path))
            ocr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
