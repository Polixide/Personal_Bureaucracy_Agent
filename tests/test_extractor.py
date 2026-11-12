"""Only synthetic documents are used; PDFs are generated in a temporary folder."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from config import OLLAMA_MODEL

import pymupdf

from src.document_processing.extractor import DocumentExtractionError, extract_text, extract_pages
from src.document_processing.extractor import _create_ocr_engine as create_engine_for_test


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


def test_page_boundaries_include_blank_pages(tmp_path):
    path = tmp_path / "pages.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72,72), "First page")
        document.new_page()
        document.new_page().insert_text((72,72), "Third page")
        document.save(path)
    pages = extract_pages(path)
    assert [p.number for p in pages] == [1,2,3]
    assert "First page" in pages[0].text
    assert not pages[1].text.strip()
    assert "Third page" in pages[2].text


def scanned_pdf(path, *, embedded_text=None):
    """Synthetic raster page; OCR is always mocked in tests."""
    with pymupdf.open() as document:
        page = document.new_page()
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
        image.clear_with(255)
        page.insert_image(page.rect, stream=image.tobytes("png"))
        if embedded_text:
            page.insert_text((72,72), embedded_text)
        document.save(path)


def ocr_result(text):
    from types import SimpleNamespace
    return SimpleNamespace(txts=tuple(text.splitlines()))


def test_adequate_text_pdf_with_image_does_not_invoke_ocr(tmp_path):
    path = tmp_path / "text.pdf"
    scanned_pdf(path, embedded_text="Student Enrollment Certificate with adequate readable embedded text")
    with patch("src.document_processing.extractor._create_ocr_engine") as initialize:
        assert "Student Enrollment" in extract_pages(path)[0].text
    initialize.assert_not_called()


def test_image_only_pdf_uses_ocr_and_preserves_page_number(tmp_path):
    from unittest.mock import Mock
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    engine = Mock(return_value=ocr_result("Recognized scan text"))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine) as initialize:
        pages = extract_pages(path)
    assert pages[0].number == 1 and pages[0].text == "Recognized scan text"
    assert engine.call_count == initialize.call_count == 1
    assert engine.call_args.args[0].startswith(b"\x89PNG")


def test_ocr_unavailable_produces_controlled_error(tmp_path):
    import pytest
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    with patch("src.document_processing.extractor._create_ocr_engine", side_effect=ImportError("onnxruntime unavailable")):
        with pytest.raises(DocumentExtractionError, match="RapidOCR failed on page 1"):
            extract_pages(path)


def test_ocr_timeout_is_a_controlled_extraction_error(tmp_path):
    import pytest
    from unittest.mock import Mock
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    engine = Mock(side_effect=TimeoutError("backend timeout"))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine):
        with pytest.raises(DocumentExtractionError, match="OCR timed out"):
            extract_pages(path)


def test_ocr_output_uses_existing_structured_processing_and_grounding(tmp_path):
    import json
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src.document_processing.processor import process_document
    from src.document_processing.schemas import StudentCertificate
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    text = "Student Enrollment Certificate\nHolder name: Alex Example\nStudent ID: S123\nInstitution: Example University\nIssue date: 01.11.2026"
    payload = {"holder_name": "Alex Example", "student_id": "FABRICATED-999",
               "institution": "Example University", "issue_date": "2026-11-01"}
    replies = [{"document_type": "student_enrollment_certificate"}, payload]
    engine = Mock(return_value=ocr_result(text))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine), \
         patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [SimpleNamespace(message=SimpleNamespace(content=json.dumps(r))) for r in replies]
        info = process_document(path)
    assert isinstance(info, StudentCertificate)
    assert info.issue_date == date(2026,11,1)
    assert info.holder_name == "Alex Example"
    assert info.student_id is None
    assert any("student_id" in issue for issue in info.validation_issues)
    assert info.selected_pages == [1]
    assert info.source_pages["issue_date"] == 1
    assert client.return_value.chat.call_count == 2


def test_mixed_pdf_ocr_only_runs_on_scanned_page(tmp_path):
    from unittest.mock import Mock
    path = tmp_path / "mixed.pdf"
    scan = tmp_path / "scan.pdf"
    scanned_pdf(scan)
    with pymupdf.open() as document:
        document.new_page().insert_text((72,72), "Adequate embedded document text with multiple readable words")
        with pymupdf.open(scan) as source:
            document.insert_pdf(source)
        document.new_page()
        document.save(path)
    engine = Mock(return_value=ocr_result("OCR page two"))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine):
        pages = extract_pages(path)
    assert [page.number for page in pages] == [1,2,3]
    assert pages[1].text == "OCR page two"
    assert pages[2].text == ""
    assert engine.call_count == 1


def test_ocr_initializes_once_for_multiple_scanned_pages(tmp_path):
    from unittest.mock import Mock
    scan = tmp_path / "scan.pdf"; path = tmp_path / "scans.pdf"
    scanned_pdf(scan)
    with pymupdf.open() as document, pymupdf.open(scan) as source:
        document.insert_pdf(source); document.insert_pdf(source)
        document.save(path)
    engine = Mock(side_effect=[ocr_result("First scanned page"), ocr_result("Second scanned page")])
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine) as initialize:
        pages = extract_pages(path)
    assert initialize.call_count == 1
    assert engine.call_count == 2
    assert [(p.number, p.text) for p in pages] == [(1,"First scanned page"),(2,"Second scanned page")]


def test_ocr_failure_leaves_document_rejected_not_analyzed(tmp_path):
    from unittest.mock import Mock
    from src.core.state import Document
    from src.workers.document_analyzer import document_analyzer
    path = tmp_path / "scan.pdf"; scanned_pdf(path)
    engine = Mock(side_effect=RuntimeError("synthetic inference failure"))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine):
        result = document_analyzer({"documents":[Document(doc_id="scan",file_path=str(path))]})
    assert result["documents"][0].status == "rejected"
    assert "RapidOCR failed" in result["documents"][0].processing_error
    assert not result["extracted_facts"]


def test_empty_ocr_result_is_a_controlled_failure(tmp_path):
    import pytest
    from unittest.mock import Mock
    path = tmp_path / "scan.pdf"; scanned_pdf(path)
    engine = Mock(return_value=ocr_result(""))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine):
        with pytest.raises(DocumentExtractionError, match="no readable text"):
            extract_pages(path)


def test_rapidocr_factory_uses_local_weights_and_cpu_only(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import sys
    directory = tmp_path / "models"; directory.mkdir()
    for filename in ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx", "ch_ppocr_mobile_v2.0_cls_mobile.onnx"):
        (directory / filename).write_bytes(b"synthetic model placeholder")
    factory = Mock()
    package = SimpleNamespace(__file__=str(tmp_path / "__init__.py"), RapidOCR=factory,
                              EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"))
    with patch.dict(sys.modules, {"rapidocr": package}):
        assert create_engine_for_test() is factory.return_value
    params = factory.call_args.kwargs["params"]
    assert params["EngineConfig.onnxruntime.use_cuda"] is False
    for stage in ("Det", "Rec", "Cls"):
        assert params[f"{stage}.engine_type"] == "onnxruntime"
        assert Path(params[f"{stage}.model_path"]).is_file()


def test_missing_bundled_models_fail_without_downloading(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import sys
    import pytest
    factory = Mock()
    package = SimpleNamespace(__file__=str(tmp_path / "__init__.py"), RapidOCR=factory,
                              EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"))
    with patch.dict(sys.modules, {"rapidocr": package}):
        with pytest.raises(DocumentExtractionError, match="bundled model files are missing"):
            create_engine_for_test()
    factory.assert_not_called()


def test_rapidocr_insurance_text_uses_labelled_evidence_reconciliation(tmp_path):
    import json
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src.document_processing.processor import process_document
    path=tmp_path / "insurance-scan.pdf"; scanned_pdf(path)
    text=("Health Insurance Certificate\nInsured: Taylor Example\nInsurer: Example Insurance\n"
          "Policy number: DEMO-OCR-42\nPolicy approval date: 18/09/2028\n"
          "Coverage start date: 20/09/2028\nCoverage end date: 30/06/2029")
    engine=Mock(return_value=ocr_result(text))
    with patch("src.document_processing.extractor._create_ocr_engine",return_value=engine), patch("src.document_processing.processor.Client") as client:
        payloads=[{"document_type":"health_insurance_certificate"}, {"policy_number":"FABRICATED-99", "source_snippets":{"policy_number":"Invented quotation"}}]
        client.return_value.chat.side_effect=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value))) for value in payloads]
        info=process_document(path)
    assert info.policy_number == "DEMO-OCR-42"
    assert info.coverage_start_date == date(2028,9,20)
    assert info.coverage_end_date == date(2029,6,30)
    assert info.source_pages["policy_number"] == 1
    assert info.source_snippets["policy_number"] == "Policy number: DEMO-OCR-42"
    assert any("policy_number" in note for note in info.correction_notes)
    assert not info.validation_issues
    assert info.source_evidence and all(e.page_number == 1 for e in info.source_evidence)
