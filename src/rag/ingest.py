"""Knowledge base ingestion: official web pages and documents -> procedures/<id>/sources_text/<SOURCE_ID>.txt.

Each text file starts with a short header (source, URL, retrieval date, content hash) followed by the page
text. The hash lets `check_sources` tell when an official page changed, so the rules citing it can be reviewed.
Only the standard library, httpx (already used by ollama) and PyMuPDF are used.
"""
import hashlib
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import httpx
from pydantic import BaseModel

from config import RAG_FETCH_TIMEOUT, RAG_USER_AGENT
from src.models import Source

HEADER_END = "\n\n"
SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "form", "select", "button",
             "iframe", "template", "head"}
BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "table",
              "section", "article", "main", "br", "dt", "dd", "blockquote", "pre", "hr", "caption"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
VOID_TAGS = {"br", "img", "meta", "link", "input", "hr", "source", "area", "base", "col", "embed", "param",
             "track", "wbr"}
# Class/id names are split into words ("menu-content" -> menu, content); the first word decides.
CONTENT_WORDS = {"content", "article", "detail", "main", "post", "entry", "text", "body"}
NOT_CONTENT_WORDS = {"menu", "nav", "navbar", "navigation", "modal", "header", "footer", "sidebar", "breadcrumb",
                     "breadcrumbs", "cookie", "cookies", "social", "share", "search", "skip"}
MIN_MAIN_CHARS = 200


class IngestError(RuntimeError):
    """A source could not be fetched or read."""


class SourceText(BaseModel):
    """The readable text of one source and the hash of that text."""
    source_id: str
    text: str
    sha256: str


# ------------------------------------------------------------ HTML -> text --
def first_words(names: str) -> set[str]:
    """'nav justify-content-end menu-content' -> {'nav', 'justify', 'menu'} (lower case)."""
    return {re.split(r"[-_]", name)[0] for name in names.lower().split() if name}


class _PageText(HTMLParser):
    """Collects visible text, one line per block, and remembers which content container it is in."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int | None, bool]] = []   # (tag, container id, skipped)
        self.pieces: list[tuple[str, tuple[int, ...]]] = []
        self.next_id = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a block, remember content containers, and skip unwanted parts (menus, scripts...)."""
        if tag in BLOCK_TAGS:
            self._mark("\n\n" if tag in HEADING_TAGS else "\n")
        if tag in VOID_TAGS:
            return
        words = first_words(" ".join(value or "" for key, value in attrs if key in ("class", "id")))
        unwanted = tag not in ("html", "body") and bool(words & NOT_CONTENT_WORDS)
        container = None
        if tag in ("main", "article") or (words & CONTENT_WORDS and not unwanted):
            container, self.next_id = self.next_id, self.next_id + 1
        self.stack.append((tag, container, tag in SKIP_TAGS or unwanted))

    def handle_endtag(self, tag: str) -> None:
        """Close a block and the matching open tag (unclosed inner tags are dropped too)."""
        if tag in BLOCK_TAGS:
            self._mark("\n\n" if tag in HEADING_TAGS else "\n")
        for index in range(len(self.stack) - 1, -1, -1):   # tolerate unclosed tags
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if any(skipped for _, _, skipped in self.stack):
            return
        text = " ".join(data.split())
        if text:
            self.pieces.append((text, self._containers()))

    def _mark(self, separator: str) -> None:
        self.pieces.append((separator, self._containers()))

    def _containers(self) -> tuple[int, ...]:
        return tuple(container for _, container, _ in self.stack if container is not None)


def _main_container(pieces: list[tuple[str, tuple[int, ...]]]) -> int | None:
    """The content container holding the most text, or None when the page has none."""
    sizes: dict[int, int] = {}
    for text, containers in pieces:
        if text.strip():
            for container in containers:
                sizes[container] = sizes.get(container, 0) + len(text)
    return max(sizes, key=lambda container: sizes[container]) if sizes else None


def tidy_lines(raw: str) -> str:
    """Single spaces, no empty runs of more than one blank line, no trailing spaces."""
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str) -> str:
    """Visible text of the page's main content (menus, scripts, headers and footers removed)."""
    parser = _PageText()
    parser.feed(html)
    parser.close()
    main = _main_container(parser.pieces)
    text = join_pieces([text for text, containers in parser.pieces if main is None or main in containers])
    if len(text) < MIN_MAIN_CHARS:   # the guessed container was too small: keep the whole page instead
        text = join_pieces([text for text, _ in parser.pieces])
    return text


def join_pieces(pieces: list[str]) -> str:
    """Join text pieces with single spaces; block separators keep their line breaks."""
    joined = ""
    for text in pieces:
        joined += text if text.startswith("\n") or joined.endswith(("\n", " ")) or not joined else " " + text
    return tidy_lines(joined)


# ------------------------------------------------------------- documents --
def pdf_to_text(content: bytes) -> str:
    """Text of a text-based PDF, page by page (scanned PDFs give little or no text)."""
    import pymupdf

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = [page.get_text() for page in document]
    except Exception as error:   # PyMuPDF raises several types for broken files
        raise IngestError(f"Cannot read PDF: {error}") from error
    return tidy_lines("\n\n".join(pages))


def bytes_to_text(content: bytes, kind: str) -> str:
    """Readable text of HTML, PDF or plain-text content."""
    if kind == "pdf":
        return pdf_to_text(content)
    decoded = content.decode("utf-8", errors="replace")
    return html_to_text(decoded) if kind == "html" else tidy_lines(decoded)


def content_kind(name_or_type: str) -> str:
    """'pdf', 'html' or 'text' from a file name, URL or HTTP content type."""
    value = name_or_type.lower()
    if "pdf" in value:
        return "pdf"
    if "html" in value or value.endswith((".htm", "/")):
        return "html"
    return "text"


# ------------------------------------------------------------------ fetch --
def fetch_url(url: str, client: httpx.Client | None = None) -> tuple[bytes, str]:
    """Download one public page: (content, kind). A plain client name is sent, never user data."""
    owned = client is None
    client = client or httpx.Client(timeout=RAG_FETCH_TIMEOUT, follow_redirects=True,
                                    headers={"User-Agent": RAG_USER_AGENT})
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise IngestError(f"Cannot fetch {url}: {error}") from error
    finally:
        if owned:
            client.close()
    content_type = response.headers.get("content-type", "")
    return response.content, content_kind(content_type if content_type else url)


def source_text(source_id: str, content: bytes, kind: str) -> SourceText:
    """Readable text and its hash; an empty result is an error (nothing to cite)."""
    text = bytes_to_text(content, kind)
    if len(text) < 40:
        raise IngestError(f"{source_id}: no readable text found (scanned or script-only page?)")
    return SourceText(source_id=source_id, text=text, sha256=text_sha256(text))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------- the files --
def header(source: Source, retrieved: date, sha256: str) -> str:
    """The first lines of a sources_text file: what it is and where it comes from."""
    organization = f" ({source.organization})" if source.organization else ""
    return (f"SOURCE: {source.title}{organization}\nURL: {source.url}\nRETRIEVED: {retrieved.isoformat()}\n"
            f"SHA256: {sha256}")


def write_source_text(folder: Path, source: Source, page: SourceText, retrieved: date) -> Path:
    """Save the text as sources_text/<SOURCE_ID>.txt (replacing an older copy)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source.source_id}.txt"
    path.write_text(header(source, retrieved, page.sha256) + HEADER_END + page.text + "\n", encoding="utf-8")
    return path


def stored_text(path: Path) -> str:
    """The saved text without its header ('' when the file does not exist)."""
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8")
    first, separator, rest = content.partition(HEADER_END)
    return rest.strip() if separator and first.startswith("SOURCE:") else content.strip()
