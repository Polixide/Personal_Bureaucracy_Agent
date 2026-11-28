"""Gate 0: deterministic checks on what the user sends, before any model call or Twin write. No model, no state.

Gate 1 (src/agent/gate1.py) checks what the MODEL says; Gate 0 checks what the USER sends. Each refusal is an Issue
with a code and a plain message for the user:
  EMPTY_MESSAGE      a chat message with no text
  MESSAGE_TOO_LONG   more than GATE0_MAX_MESSAGE_CHARS characters (one turn is one request; long text goes as a file)
  UNSUPPORTED_FILE   an extension the document readers do not handle (UPLOAD_TYPES)
  EMPTY_FILE         a file with no bytes
  FILE_TOO_LARGE     more than GATE0_MAX_UPLOAD_MB
  WRONG_CONTENT      the bytes do not match the extension (a renamed file: ".pdf" that is not a PDF, ".png" not a PNG)
  TOO_MANY_PAGES     a PDF with more than GATE0_MAX_PDF_PAGES pages
  INVALID_PROFILE    an empty or over-long profile name
The model's own inputs are checked the same way at the tool boundary: `run_tool` (src/tools/registry.py) validates
every tool call with Pydantic and an invalid call never runs. Gate 0 is that rule applied to the user's input.
"""
from pathlib import Path

import pymupdf

from config import GATE0_MAX_MESSAGE_CHARS, GATE0_MAX_PDF_PAGES, GATE0_MAX_UPLOAD_MB, UPLOAD_TYPES
from src.models import Issue

PROFILE_MAX_CHARS = 60
SIGNATURES = {"pdf": (b"%PDF",), "png": (b"\x89PNG",), "jpg": (b"\xff\xd8",), "jpeg": (b"\xff\xd8",)}


def check_message(text: str) -> list[Issue]:
    """A chat message: not empty, not longer than one turn should be."""
    stripped = (text or "").strip()
    if not stripped:
        return [Issue(code="EMPTY_MESSAGE", message="The message is empty: write what you need.")]
    if len(stripped) > GATE0_MAX_MESSAGE_CHARS:
        return [Issue(code="MESSAGE_TOO_LONG", item=str(len(stripped)),
                      message=f"The message is too long ({len(stripped)} characters, at most "
                              f"{GATE0_MAX_MESSAGE_CHARS}). Attach long text as a TXT or PDF file instead.")]
    return []


def check_upload(name: str, content: bytes) -> list[Issue]:
    """A document upload: a supported type, not empty, not too large, bytes that match the extension."""
    extension = Path(name or "").suffix.lower().lstrip(".")
    if extension not in UPLOAD_TYPES:
        return [Issue(code="UNSUPPORTED_FILE", item=extension or None,
                      message="Unsupported file. Choose a PDF, a TXT file or a photo (JPG or PNG).")]
    if not content:
        return [Issue(code="EMPTY_FILE", message="The uploaded file is empty.")]
    megabytes = len(content) / (1024 * 1024)
    if megabytes > GATE0_MAX_UPLOAD_MB:
        return [Issue(code="FILE_TOO_LARGE", item=f"{megabytes:.1f} MB",
                      message=f"The file is too large ({megabytes:.1f} MB, at most {GATE0_MAX_UPLOAD_MB} MB).")]
    if extension in SIGNATURES and not content.startswith(SIGNATURES[extension]):
        return [Issue(code="WRONG_CONTENT", item=extension,
                      message=f"This file is not a real .{extension} (it may have been renamed). Upload the original.")]
    if extension == "pdf":
        return check_pdf_pages(content)
    return []


def check_pdf_pages(content: bytes) -> list[Issue]:
    """At most GATE0_MAX_PDF_PAGES pages. A PDF that cannot be opened passes: the reader explains that problem."""
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = document.page_count
    except Exception:              # damaged or unusual PDF: src/document_processing gives the precise reason
        return []
    if pages > GATE0_MAX_PDF_PAGES:
        return [Issue(code="TOO_MANY_PAGES", item=str(pages),
                      message=f"The PDF has {pages} pages (at most {GATE0_MAX_PDF_PAGES}). "
                              "Upload only the pages of your documents.")]
    return []


def check_profile(name: str) -> list[Issue]:
    """A profile name: it identifies the user's Digital Twin, so it must not be empty or endless."""
    stripped = (name or "").strip()
    if not stripped or len(stripped) > PROFILE_MAX_CHARS:
        return [Issue(code="INVALID_PROFILE", message=f"Use a profile name of 1 to {PROFILE_MAX_CHARS} characters.")]
    return []
