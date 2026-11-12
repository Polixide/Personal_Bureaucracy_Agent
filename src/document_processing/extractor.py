"""Deterministic text extraction, independent of future LLM processing."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf


class DocumentExtractionError(ValueError):
    """A supported document could not provide readable text."""


@dataclass(frozen=True)
class TextPage:
    number: int  # One-based original page; TXT is one logical page.
    text: str


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


def ocr_page(page, number):
    """OCR one scanned page; returned text is still untrusted document input."""
    try:
        engine = get_ocr_engine()
        image = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False).tobytes("png")
        result = engine(image)
        text = "\n".join(line.strip() for line in (result.txts or ()) if line.strip())
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


def extract_pages(file_path: str | Path) -> list[TextPage]:
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
        return [TextPage(1, path.read_text(encoding="utf-8-sig"))]
    if extension != ".pdf":
        raise ValueError(f"Unsupported extension '{extension}'. Use PDF or TXT.")

    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise DocumentExtractionError("Password-protected PDFs are not supported.")
            pages = []
            for i, page in enumerate(document):
                text = page.get_text()
                # Short text-only pages (and blank pages) keep their existing
                # behavior. OCR only raster-containing pages with little text.
                if sum(c.isalnum() for c in text) < 30 and page.get_images(full=True):
                    text = ocr_page(page, i + 1) or text
                pages.append(TextPage(i + 1, text))
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentExtractionError(f"Cannot read PDF: {path}") from error

    if not any(page.text.strip() for page in pages):
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
