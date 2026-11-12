"""Deterministic text extraction, independent of future LLM processing."""

import argparse
import os
from pathlib import Path

import pymupdf


class DocumentExtractionError(ValueError):
    """A supported document could not provide readable text."""


def _usable_text(text: str) -> bool:
    """Treat whitespace, punctuation, and replacement glyphs alone as unreadable."""
    return any(character.isalnum() for character in text)


def _page_text(page: pymupdf.Page) -> str:
    text = page.get_text()
    if _usable_text(text):
        return text
    languages = os.environ.get("PBA_OCR_LANGUAGES", "eng").strip() or "eng"
    tessdata = os.environ.get("TESSDATA_PREFIX") or None
    try:
        textpage = page.get_textpage_ocr(
            language=languages, dpi=300, full=True, tessdata=tessdata,
        )
        text = page.get_text(textpage=textpage)
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        raise DocumentExtractionError(
            f"Page {page.number + 1} has no extractable text and OCR failed: {error}. "
            "Install Tesseract and the requested language data; set TESSDATA_PREFIX "
            "to the folder containing eng.traineddata (and tur.traineddata for Turkish). "
            "Set PBA_OCR_LANGUAGES=eng or eng+tur. See document_processing/README.md."
        ) from error
    if not _usable_text(text):
        raise DocumentExtractionError(
            f"Page {page.number + 1} has no extractable text after OCR. "
            "Check scan quality and PBA_OCR_LANGUAGES; blank pages also produce this error."
        )
    return text


def extract_text(file_path: str | Path) -> str:
    """Read UTF-8 TXT or PDF text, using local OCR for unreadable PDF pages.

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
        return path.read_text(encoding="utf-8-sig")
    if extension != ".pdf":
        raise ValueError(f"Unsupported extension '{extension}'. Use PDF or TXT.")

    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise DocumentExtractionError("Password-protected PDFs are not supported.")
            # A form-feed preserves PDF page boundaries without changing the str API.
            text = "\n\f\n".join(_page_text(page) for page in document)
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentExtractionError(f"Cannot read PDF: {path}") from error

    if not text.strip():
        raise DocumentExtractionError(
            f"PDF contains no extractable text: {path}."
        )
    return text


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
