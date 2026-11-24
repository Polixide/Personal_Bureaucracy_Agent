"""Web scraping for procedures that have no Procedure Pack: a small, polite crawler of official sites.

    python -m src.rag crawl            (seeds from knowledge/seeds.json)

- It starts from the seed pages and follows links only on the same site and under the seed's path prefix,
  up to `max_depth` links away and `max_pages` pages per seed. Paths matching `skip` are left out.
- It reads each site's robots.txt and obeys it, waits CRAWL_DELAY seconds between requests and sends a generic
  client name (no user or project details).
- Pages with readable text (HTML or PDF) are saved as knowledge/corpus/<WEB_ID>.txt with the same header as a
  pack's sources_text, and listed in knowledge/corpus.json (title, url, topic, retrieval date, SHA-256).
The corpus feeds the assistant for questions outside the active procedure and for procedures without a pack:
the answers cite these pages and say that nothing was checked against the user's documents. Scraped text never
becomes a rule by itself: a person writes rules (draft-rules can propose them).
"""
import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from pydantic import BaseModel, Field

from config import CORPUS_DIR, CRAWL_DELAY, RAG_USER_AGENT
from src.models import Source
from src.rag.ingest import (NOT_CONTENT_WORDS, VOID_TAGS, IngestError, fetch_url, first_words, source_text,
                             write_source_text)

MIN_PAGE_CHARS = 300
NAVIGATION_TAGS = {"nav", "header", "footer", "aside"}
PAGE_EXTENSIONS = ("", ".html", ".htm", ".php", ".aspx", ".pdf")


class Seed(BaseModel):
    """Where to start, how far to go, and what the pages are about."""
    url: str
    topic: str
    max_depth: int = 1
    max_pages: int = 25
    prefix: str = ""                       # path prefix links must keep (default: the seed's path)
    skip: list[str] = Field(default_factory=list)   # path fragments to leave out


class CorpusPage(BaseModel):
    """One saved page of the corpus."""
    source_id: str
    title: str
    url: str
    topic: str
    organization: str
    retrieved_at: date
    content_sha256: str


class CrawlReport(BaseModel):
    saved: list[CorpusPage] = Field(default_factory=list)
    refused_by_robots: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    too_short: list[str] = Field(default_factory=list)


# --------------------------------------------------------------- pages --
class _Links(HTMLParser):
    """The <title>, the first <h1> and the <a href> of the page's content.

    Links inside navigation, header, footer, menus and sidebars are left out (the same class/id words as the
    text extraction of src/rag/ingest.py), so the crawler follows the page's own links, not the site menu.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.title = ""
        self.h1 = ""
        self._in: str | None = None
        self._stack: list[bool] = []      # one entry per open element: is it (or a parent) navigation?

    def _skipping(self) -> bool:
        return bool(self._stack) and self._stack[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        names = f"{values.get('class') or ''} {values.get('id') or ''} {values.get('role') or ''}"
        navigation = tag in NAVIGATION_TAGS or bool(first_words(names) & NOT_CONTENT_WORDS)
        if tag == "a" and values.get("href") and not (self._skipping() or navigation):
            self.links.append(values["href"])
        if tag in ("title", "h1"):
            self._in = tag
        if tag not in VOID_TAGS:
            self._stack.append(self._skipping() or navigation)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._in:
            self._in = None
        if tag not in VOID_TAGS and self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._in == "title" and not self.title:
            self.title = data.strip()
        elif self._in == "h1" and not self.h1:
            self.h1 = data.strip()


def parse_page(html: str) -> _Links:
    parser = _Links()
    parser.feed(html)
    return parser


def normalise(url: str) -> str:
    """No fragment, no trailing slash: one spelling per page."""
    return url.split("#", 1)[0].rstrip("/")


def follows(url: str, seed: Seed) -> bool:
    """A link the crawler may follow from this seed: same site, under the prefix, a page or a PDF, not skipped."""
    start, target = urlparse(seed.url), urlparse(url)
    prefix = seed.prefix or start.path
    extension = Path(target.path).suffix.lower()
    return (target.scheme in ("http", "https") and target.netloc == start.netloc and target.path.startswith(prefix)
            and extension in PAGE_EXTENSIONS and not target.query and not any(part in target.path for part in seed.skip))


def web_id(url: str) -> str:
    """WEB_<SITE>_<PATH>, readable and stable (a short hash keeps long paths unique)."""
    parsed = urlparse(url)
    words = re.sub(r"[^A-Za-z0-9]+", "_", f"{parsed.netloc.split('.')[0]}_{parsed.path}").strip("_").upper()
    if len(words) > 60:
        words = words[:52] + "_" + hashlib.sha1(url.encode()).hexdigest()[:7].upper()
    return f"WEB_{words}"


# -------------------------------------------------------------- robots --
class Robots:
    """robots.txt of each site, read once with the same fetcher; a missing file allows everything."""

    def __init__(self, fetch: Callable[[str], tuple[bytes, str]]):
        self.fetch = fetch
        self.sites: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        site = urlparse(url)
        key = f"{site.scheme}://{site.netloc}"
        if key not in self.sites:
            self.sites[key] = self._read(key)
        parser = self.sites[key]
        return parser is None or parser.can_fetch(RAG_USER_AGENT, url)

    def _read(self, site: str) -> RobotFileParser | None:
        try:
            content, _ = self.fetch(f"{site}/robots.txt")
        except IngestError:
            return None
        text = content.decode("utf-8", errors="replace")
        if "<html" in text[:500].lower():          # a web page instead of robots.txt: no rules published
            return None
        parser = RobotFileParser()
        parser.parse(text.splitlines())
        return parser


# --------------------------------------------------------------- crawl --
def crawl_seed(seed: Seed, fetch: Callable, robots: Robots, delay: float, seen: set[str],
               report: CrawlReport, today: date, folder: Path) -> None:
    """Breadth-first from one seed, saving every readable page."""
    queue = [(normalise(seed.url), 0)]
    saved = 0
    while queue and saved < seed.max_pages:
        url, depth = queue.pop(0)
        if url in seen and depth > 0:      # a seed reached from another seed is still expanded
            continue
        seen.add(url)
        if not robots.allowed(url):
            report.refused_by_robots.append(url)
            continue
        time.sleep(delay)
        try:
            content, kind = fetch(url)
        except IngestError:
            report.failed.append(url)
            continue
        links = parse_page(content.decode("utf-8", errors="replace")) if kind == "html" else None
        if save_page(url, content, kind, links, seed, report, today, folder):
            saved += 1
        if links is not None and depth < seed.max_depth:
            for href in links.links:
                target = normalise(urljoin(url, href))
                if follows(target, seed) and target not in seen:
                    queue.append((target, depth + 1))


def save_page(url: str, content: bytes, kind: str, links: _Links | None, seed: Seed, report: CrawlReport,
              today: date, folder: Path) -> bool:
    """Save one page's text; False when it has too little text to cite."""
    source_id = web_id(url)
    if any(page.source_id == source_id for page in report.saved):
        return False
    try:
        page = source_text(source_id, content, kind)
    except IngestError:
        report.too_short.append(url)
        return False
    if len(page.text) < MIN_PAGE_CHARS:
        report.too_short.append(url)
        return False
    title = (links.h1 or links.title) if links is not None else Path(urlparse(url).path).stem
    organization = urlparse(url).netloc
    source = Source(source_id=source_id, title=title or url, url=url, retrieved_at=today.isoformat(),
                    authority=organization, organization=organization)
    write_source_text(folder / "corpus", source, page, today)
    report.saved.append(CorpusPage(source_id=source_id, title=source.title, url=url, topic=seed.topic,
                                   organization=organization, retrieved_at=today, content_sha256=page.sha256))
    return True


def crawl(seeds: list[Seed], folder: Path = CORPUS_DIR, fetch: Callable = fetch_url, delay: float = CRAWL_DELAY,
          today: date | None = None) -> CrawlReport:
    """Crawl every seed and write knowledge/corpus.json (pages of earlier crawls not seen again are kept)."""
    today = today or date.today()
    report = CrawlReport()
    robots = Robots(fetch)
    seen: set[str] = set()
    for seed in seeds:
        crawl_seed(seed, fetch, robots, delay, seen, report, today, folder)
    pages = {page.source_id: page for page in load_corpus(folder)}
    pages.update({page.source_id: page for page in report.saved})
    save_corpus(folder, list(pages.values()))
    return report


# --------------------------------------------------------------- files --
def load_seeds(folder: Path = CORPUS_DIR) -> list[Seed]:
    path = folder / "seeds.json"
    return [Seed.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_corpus(folder: Path = CORPUS_DIR) -> list[CorpusPage]:
    """The pages listed in corpus.json ([] before the first crawl)."""
    path = folder / "corpus.json"
    if not path.is_file():
        return []
    return [CorpusPage.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_corpus(folder: Path, pages: list[CorpusPage]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    data = [page.model_dump(mode="json") for page in sorted(pages, key=lambda page: page.source_id)]
    (folder / "corpus.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
