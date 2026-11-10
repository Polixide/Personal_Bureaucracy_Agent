"""Deterministic text extraction, independent of future LLM processing."""

import argparse
from pathlib import Path

import pymupdf


class DocumentExtractionError(ValueError):
    """A supported document could not provide readable text."""


def extract_text(file_path: str | Path) -> str:
    """Read a UTF-8 TXT or text-based PDF file and return its text.

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
            text = "\n".join(page.get_text() for page in document)
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentExtractionError(f"Cannot read PDF: {path}") from error

    if not text.strip():
        raise DocumentExtractionError(
            f"PDF contains no extractable text: {path}. OCR is not supported."
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
