"""Only synthetic documents are used; PDFs are generated in a temporary folder."""

from pathlib import Path
import tempfile
import unittest

import pymupdf

from extractor import DocumentExtractionError, extract_text


class ExtractTextTests(unittest.TestCase):
    def test_synthetic_certificate(self) -> None:
        sample = Path(__file__).resolve().parents[1] / "data/synthetic_samples/student_enrollment.txt"
        text = extract_text(sample)
        self.assertIn("Student Enrollment Certificate", text)
        self.assertIn("Alex Example", text)
        self.assertEqual(text, sample.read_text(encoding="utf-8"))

    def test_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                extract_text(Path(folder) / "missing.txt")

    def test_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "document.csv"
            path.write_text("synthetic", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported extension"):
                extract_text(path)

    def test_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "certificate.PDF"
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "Synthetic enrollment certificate")
                document.new_page().insert_text((72, 72), "Student: Alex Example")
                document.save(path)
            text = extract_text(str(path))
            self.assertIn("Synthetic enrollment certificate", text)
            self.assertIn("Student: Alex Example", text)

    def test_pdf_without_text(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.pdf"
            with pymupdf.open() as document:
                document.new_page()
                document.save(path)
            with self.assertRaisesRegex(DocumentExtractionError, "no extractable text"):
                extract_text(path)

    def test_invalid_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.pdf"
            path.write_text("This is not a PDF.", encoding="utf-8")
            with self.assertRaises(DocumentExtractionError):
                extract_text(path)


if __name__ == "__main__":
    unittest.main()
