"""Deterministic text extraction, independent of future LLM processing."""

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import pymupdf
from src.core.llm_diagnostics import _emit
from config import DOCUMENT_OCR_DPI, DOCUMENT_MAX_IMAGE_SIDE, DOCUMENT_MIN_TEXT_CHARS, DOCUMENT_OCR_MIN_CONFIDENCE


class DocumentExtractionError(ValueError):
    """A supported document could not provide readable text."""


@dataclass(frozen=True)
class TextPage:
    number: int  # One-based original page; TXT is one logical page.
    text: str
    extraction_method: str = "native_text"
    native_text_character_count: int | None = None
    ocr_used: bool = False
    ocr_duration_seconds: float = 0.0
    ocr_confidence: float | None = None
    rotation_degrees: int = 0
    quality_issues: tuple[str, ...] = ()
    native_text: str | None = None
    form_fields: tuple[tuple[str, str], ...] = ()
    low_confidence_lines: tuple[str, ...] | None = None
    ocr_bounds: tuple[tuple[float, float, float, float], ...] = ()


def text_quality_issues(text):
    visible=[c for c in text if not c.isspace()]
    alphanumeric=sum(c.isalnum() for c in visible)
    issues=[]
    if alphanumeric<DOCUMENT_MIN_TEXT_CHARS:issues.append("insufficient_text")
    if visible and (alphanumeric/len(visible)<0.45 or text.count("\ufffd")/len(visible)>0.05):issues.append("malformed_text")
    return issues


def render_page(page, rotation=0):
    """Reuse PyMuPDF for bounded RGB rendering; no second PDF/image backend."""
    if not 200<=DOCUMENT_OCR_DPI<=300 or DOCUMENT_MAX_IMAGE_SIDE<256:
        raise DocumentExtractionError("Invalid document rendering configuration")
    scale=min(DOCUMENT_OCR_DPI/72,DOCUMENT_MAX_IMAGE_SIDE/max(page.rect.width,page.rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(scale,scale).prerotate(rotation),colorspace=pymupdf.csRGB,alpha=False).tobytes("png")


def render_pdf_pages(file_path, numbers):
    """Render only explicitly selected original pages for optional vision input."""
    numbers=list(dict.fromkeys(numbers))
    try:
        with pymupdf.open(file_path) as document:
            if document.needs_pass:raise DocumentExtractionError("Password-protected PDFs are not supported")
            if any(type(n) is not int or not 1<=n<=len(document) for n in numbers):
                raise DocumentExtractionError("Invalid selected PDF page number")
            return {n:render_page(document[n-1]) for n in numbers}
    except (pymupdf.FileDataError,RuntimeError) as error:
        raise DocumentExtractionError("Cannot render selected PDF pages") from error


def _create_ocr_engine():
    """Import/initialize only on the first scanned page; CPU inference only."""
    import rapidocr
    from rapidocr import EngineType, RapidOCR

    # RapidOCR 3.10 wheels bundle these models. Explicit local paths prevent
    # missing weights from silently triggering a network download at runtime.
    directory = Path(rapidocr.__file__).resolve().parent / "models"
    files = {"Det": "PP-OCRv6_det_small.onnx", "Rec": "PP-OCRv6_rec_small.onnx",
             "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx"}
    params = {"EngineConfig.onnxruntime.use_cuda": False,
              "EngineConfig.onnxruntime.intra_op_num_threads": 2,
              "EngineConfig.onnxruntime.inter_op_num_threads": 1}
    for stage, filename in files.items():
        path = directory / filename
        if not path.is_file():
            raise DocumentExtractionError("RapidOCR bundled model files are missing. Reinstall the project requirements.")
        params[f"{stage}.model_path"] = str(path)
        params[f"{stage}.engine_type"] = EngineType.ONNXRUNTIME
    return RapidOCR(params=params)


@lru_cache(maxsize=1)
def get_ocr_engine():
    """Reuse a single local engine across pages/reruns; never used for text PDFs."""
    return _create_ocr_engine()


def ocr_page(page, number, *, metadata=None):
    """OCR one scanned page; returned text is still untrusted document input."""
    try:
        engine = get_ocr_engine()
        best_text = ""
        best_quality, best_confidence, best_angle, best_low_lines = -1, None, 0, None
        best_bounds = ()
        for angle in (0, 90, 180, 270):
            result = engine(render_page(page, angle))
            lines = result.txts if result.txts is not None else ()
            candidate = "\n".join(line.strip() for line in lines if line.strip())
            scores = getattr(result, "scores", None)
            confidence = (sum(float(score) for score in scores) / len(scores) if scores is not None and len(scores)
                          else 1.0 if candidate else 0.0)
            quality = confidence * max(1, sum(char.isalnum() for char in candidate)) ** 0.5
            if candidate and quality > best_quality:
                best_text, best_quality, best_angle = candidate, quality, angle
                best_confidence = confidence if scores is not None and len(scores) else None
                best_low_lines = (
                    tuple(line.strip() for line, score in zip(lines, scores)
                          if line.strip() and float(score) < DOCUMENT_OCR_MIN_CONFIDENCE)
                    if scores is not None and len(lines) == len(scores) else None
                )
                boxes = getattr(result, "boxes", None)
                best_bounds = ()
                if boxes is not None and len(boxes) == len(lines):
                    bounds = []
                    for line, box in zip(lines, boxes):
                        if line.strip():
                            x, y = zip(*box)
                            bounds.append((min(map(float, x)), min(map(float, y)), max(map(float, x)), max(map(float, y))))
                    best_bounds = tuple(bounds)
            if angle == 0 and candidate and confidence >= 0.85:
                break
        if metadata is not None:
            metadata.update(ocr_confidence=best_confidence, rotation_degrees=best_angle,
                            low_confidence_lines=best_low_lines, ocr_bounds=best_bounds)
        text=best_text
        if not text:
            raise DocumentExtractionError(f"RapidOCR found no readable text on page {number}. Try a clearer scan.")
        return text
    except DocumentExtractionError:
        raise
    except TimeoutError as error:
        raise DocumentExtractionError(f"OCR timed out on page {number}: {error}") from error
    except Exception as error:
        raise DocumentExtractionError(
            f"RapidOCR failed on page {number}. Check the Python OCR dependencies, "
            f"bundled model files and available memory. {error}"
        ) from error


def extract_pages(file_path: str | Path, *, allow_ocr_failure: bool = False,
                  local_ocr: bool = True) -> list[TextPage]:
    """Read UTF-8 TXT or PDF text, retaining original one-based page numbers.

    Raises FileNotFoundError for missing paths, ValueError for unsupported
    extensions, and DocumentExtractionError for unreadable or textless PDFs.
    Filesystem errors (such as permission errors) propagate to the caller.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")

    extension = path.suffix.lower()
    if extension == ".txt":
        text=path.read_text(encoding="utf-8-sig")
        return [TextPage(1,text,native_text_character_count=len(text),quality_issues=tuple(text_quality_issues(text)))]
    if extension != ".pdf":
        raise ValueError(f"Unsupported extension '{extension}'. Use PDF or TXT.")

    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise DocumentExtractionError("Password-protected PDFs are not supported.")
            pages = []
            for i, page in enumerate(document):
                text = "\n".join(re.sub(r"[ \t]{2,}","  ",line).strip() for line in page.get_text(sort=True).splitlines())
                native_text = text
                native_count = len(native_text)
                form_fields = tuple(
                    ((widget.field_label or widget.field_name or "").replace("_", " "), widget.field_value)
                    for widget in page.widgets() or ()
                    if widget.field_type == pymupdf.PDF_WIDGET_TYPE_TEXT
                    and isinstance(widget.field_value, str) and widget.field_value.strip()
                )
                metadata={}
                used_ocr=local_ocr and bool(text_quality_issues(text)) and bool(page.get_images(full=True) or page.get_image_info())
                duration=0.0
                issues=[]
                _emit(f"[DOCUMENT] native_page page={i+1} native_text_characters={native_count} ocr_triggered={used_ocr}")
                if used_ocr:
                    started=perf_counter()
                    try:text=ocr_page(page,i+1,metadata=metadata) or text
                    except DocumentExtractionError:
                        issues.append("ocr_failed")
                        if not allow_ocr_failure:raise
                    finally:
                        duration=perf_counter()-started
                        _emit(f"[DOCUMENT] ocr page={i+1} elapsed_seconds={duration:.3f} failure={bool(issues)}")
                issues.extend(text_quality_issues(text))
                if metadata.get("ocr_confidence") is not None and metadata["ocr_confidence"]<DOCUMENT_OCR_MIN_CONFIDENCE:
                    issues.append("low_ocr_confidence")
                pages.append(TextPage(i+1,text,"rapidocr" if used_ocr else "native_text",native_count,used_ocr,duration,
                                      metadata.get("ocr_confidence"),metadata.get("rotation_degrees",0),
                                      tuple(dict.fromkeys(issues)), native_text, form_fields,
                                      metadata.get("low_confidence_lines"), metadata.get("ocr_bounds", ())))
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentExtractionError(f"Cannot read PDF: {path}") from error

    if not allow_ocr_failure and not any(page.text.strip() or page.form_fields for page in pages):
        raise DocumentExtractionError(
            f"PDF contains no extractable text: {path}. No readable text was recovered."
        )
    return pages


def extract_text(file_path: str | Path) -> str:
    """Compatibility text API; processing uses extract_pages to retain boundaries."""
    return "\n".join(page.text for page in extract_pages(file_path))


def main() -> None:
    """Run the extractor from a terminal."""
    parser = argparse.ArgumentParser(description="Extract text from a PDF or UTF-8 TXT file.")
    parser.add_argument("file_path", type=Path)
    args = parser.parse_args()
    try:
        print(extract_text(args.file_path))
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
