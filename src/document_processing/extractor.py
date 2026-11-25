"""Text of every page: the PDF text layer, or the text recognised from the page image for scans and photos.

Recognition is RapidOCR (local) by default. With DOCUMENT_PROVIDER=cloud the caller passes `transcribe`, a cloud
model that reads the page image (src/document_processing/transcribe.py); RapidOCR stays the fallback.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf

from config import CLOUD_IMAGE_JPEG_QUALITY


class DocumentExtractionError(ValueError):
    """A supported document could not provide readable text."""


@dataclass(frozen=True)
class TextPage:
    number: int  # One-based original page; TXT is one logical page.
    text: str
    ocr: bool = False  # True when the text was recognised from the page image (a scan or a photo)


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
        best_text="";best_quality=-1
        for angle in (0,90,180,270):
            pixmap=page.get_pixmap(dpi=200,colorspace=pymupdf.csRGB,alpha=False) if angle==0 else page.get_pixmap(matrix=pymupdf.Matrix(200/72,200/72).prerotate(angle),colorspace=pymupdf.csRGB,alpha=False)
            result=engine(pixmap.tobytes("png"))
            candidate="\n".join(line.strip() for line in (result.txts or ()) if line.strip())
            scores=getattr(result,"scores",None)
            confidence=sum(float(v) for v in scores)/len(scores) if scores is not None and len(scores) else (1.0 if candidate else 0.0)
            quality=confidence*max(1,sum(c.isalnum() for c in candidate))**0.5
            if candidate and quality>best_quality: best_text,best_quality=candidate,quality
            if angle==0 and candidate and confidence>=0.85: break
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_TEXT_CHARS = 30          # fewer letters/digits than this on a page with an image: OCR it
IMAGE_PAGE_TEXT_CHARS = 50   # a page mostly covered by an image (a scan) with only a watermark or header: OCR too
IMAGE_COVERAGE = 0.5


def image_coverage(page) -> float:
    """Share of the page area covered by images (0..1); scans are close to 1."""
    area = abs(page.rect.width * page.rect.height) or 1.0
    covered = 0.0
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            covered += abs(rect.width * rect.height)
    return min(covered / area, 1.0)


def needs_ocr(page, text: str) -> bool:
    """OCR raster pages whose text layer is missing, or is only a watermark/header over a scan."""
    letters = sum(c.isalnum() for c in text)
    if not page.get_images(full=True):
        return False
    if letters < MIN_TEXT_CHARS:
        return True
    return letters < IMAGE_PAGE_TEXT_CHARS and image_coverage(page) >= IMAGE_COVERAGE


def is_picture_page(page, text: str) -> bool:
    """A scan or a photo: OCR is needed, or an image covers most of the page (scanner apps add their own,
    often poor, text layer; a cloud model reads the image better)."""
    if needs_ocr(page, text):
        return True
    return bool(page.get_images(full=True)) and image_coverage(page) >= IMAGE_COVERAGE


def read_page(page, number: int, transcribe: Callable | None = None) -> TextPage:
    """One page: its text layer, or its OCR text when it is a scan (see page_text).

    With `transcribe` (a cloud model), picture pages are read from their image; "" from it means it failed, and the
    page follows the local path.
    """
    text = page.get_text()
    if transcribe is not None and is_picture_page(page, text):
        transcribed = transcribe(page, number)
        if transcribed.strip():
            return TextPage(number, transcribed, ocr=True)
    if not needs_ocr(page, text):
        return TextPage(number, text)
    return TextPage(number, page_text(page, number, text), ocr=True)


def page_pixmap(page, max_side: int):
    """The page rendered so its longer side is max_side pixels (at most 4x the PDF size)."""
    longest = max(page.rect.width, page.rect.height) or 1.0
    zoom = min(max_side / longest, 4.0)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)


def page_png(page, max_side: int) -> bytes:
    """PNG of one page (for the local vision model)."""
    return page_pixmap(page, max_side).tobytes("png")


def page_jpeg(page, max_side: int) -> bytes:
    """JPEG of one page, for a cloud model: a photographed page is about 1 MB instead of 6 MB as PNG (providers
    refuse images above a few MB, and a smaller upload is faster)."""
    return page_pixmap(page, max_side).tobytes("jpg", jpg_quality=CLOUD_IMAGE_JPEG_QUALITY)


def page_text(page, number: int, text: str) -> str:
    """OCR text of a scanned page; appended to a short text layer (a watermark or a header) when there is one."""
    recognised = ocr_page(page, number) or ""
    if text.strip() and recognised:
        return text.rstrip() + "\n" + recognised
    return recognised or text


def open_document(path: Path):
    """A PDF, or a photo (JPG/PNG) converted to a one-page PDF so it follows the scanned-page path."""
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        with pymupdf.open(path) as image:
            return pymupdf.open("pdf", image.convert_to_pdf())
    return pymupdf.open(path)


def extract_pages(file_path: str | Path, transcribe: Callable | None = None) -> list[TextPage]:
    """Read UTF-8 TXT, PDF or a photo (JPG/PNG), retaining original one-based page numbers.

    `transcribe(page, number) -> str` replaces RapidOCR for scans and photos (DOCUMENT_PROVIDER=cloud).
    Raises FileNotFoundError for missing paths, ValueError for unsupported
    extensions, and DocumentExtractionError for unreadable or textless files.
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
    if extension != ".pdf" and extension not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{extension}'. Use PDF, TXT, JPG or PNG.")

    try:
        with open_document(path) as document:
            if document.needs_pass:
                raise DocumentExtractionError("Password-protected PDFs are not supported.")
            pages = [read_page(page, i + 1, transcribe) for i, page in enumerate(document)]
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentExtractionError(f"Cannot read file: {path}") from error

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
