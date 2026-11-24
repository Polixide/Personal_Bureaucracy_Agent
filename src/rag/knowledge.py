"""The knowledge base of a Procedure Pack: its official sources, their saved texts, and change checks.

- `fetch_sources` downloads the pages in sources.json and saves their text (sources_text/<SOURCE_ID>.txt).
- `add_source` adds a new official page or document (URL or local PDF/TXT) to sources.json and saves its text.
- `check_sources` downloads again, compares with the saved text, and lists the rules to review when a page changed.

Nothing here changes rules.json: rules are only changed by a person (see the add-rule skill).
"""
import difflib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from config import PROCEDURES_DIR
from src.models import Pack, Source
from src.pack import load_pack
from src.rag.ingest import (IngestError, SourceText, content_kind, fetch_url, source_text, stored_text,
                            write_source_text)

Fetcher = Callable[[str], tuple[bytes, str]]
DIFF_PREVIEW_LINES = 8


class FetchResult(BaseModel):
    """What happened to one source during a fetch."""
    source_id: str
    ok: bool
    chars: int = 0
    changed: bool = False
    error: str = ""


class SourceCheck(BaseModel):
    """Whether one official page still says what the saved copy says."""
    source_id: str
    status: str                      # unchanged | changed | not_saved | unreachable
    rules_to_review: list[str] = Field(default_factory=list)
    preview: list[str] = Field(default_factory=list)
    error: str = ""


# --------------------------------------------------------- sources.json --
def sources_path(pack: Pack) -> Path:
    return pack.root / "sources.json"


def text_folder(pack: Pack) -> Path:
    return pack.root / "sources_text"


def save_sources(pack: Pack, sources: list[Source]) -> None:
    """Write sources.json back (2-space indent, UTF-8 kept readable)."""
    data = [source.model_dump(exclude_defaults=True) for source in sources]
    sources_path(pack).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rules_citing(pack: Pack, source_id: str) -> list[str]:
    """Ids of the rules that list this source."""
    return [rule.rule_id for rule in pack.rules if source_id in rule.source_ids]


# ---------------------------------------------------------------- fetch --
def fetch_one(pack: Pack, source: Source, fetch: Fetcher, today: date) -> tuple[FetchResult, Source]:
    """Download one source and save its text; returns the result and the updated source entry."""
    try:
        content, kind = fetch(source.url)
        page = source_text(source.source_id, content, kind)
    except IngestError as error:
        return FetchResult(source_id=source.source_id, ok=False, error=str(error)), source
    changed = page.sha256 != source.content_sha256
    write_source_text(text_folder(pack), source, page, today)
    updated = source.model_copy(update={"retrieved_at": today.isoformat(), "content_sha256": page.sha256})
    return FetchResult(source_id=source.source_id, ok=True, chars=len(page.text), changed=changed), updated


def fetch_sources(procedure_id: str, source_ids: list[str] | None = None, fetch: Fetcher = fetch_url,
                  today: date | None = None, root: Path | None = None) -> list[FetchResult]:
    """Download the pack's sources (all, or the given ids) and save their texts and hashes."""
    pack = load_pack(procedure_id, root or PROCEDURES_DIR)
    today = today or date.today()
    wanted = set(source_ids or [source.source_id for source in pack.sources])
    unknown = wanted - {source.source_id for source in pack.sources}
    results = [FetchResult(source_id=item, ok=False, error="not in sources.json") for item in sorted(unknown)]
    updated_sources = []
    for source in pack.sources:
        if source.source_id in wanted:
            result, source = fetch_one(pack, source, fetch, today)
            results.append(result)
        updated_sources.append(source)
    save_sources(pack, updated_sources)
    return results


def fetch_from_file(procedure_id: str, source_id: str, file_path: Path, today: date | None = None,
                    root: Path | None = None) -> FetchResult:
    """Save the text of an existing source from a local copy (for sites that refuse automatic downloads)."""
    [result] = fetch_sources(procedure_id, [source_id], _file_reader(Path(file_path)), today, root)
    return result


# ------------------------------------------------------------------ add --
def add_source(procedure_id: str, source: Source, file_path: Path | None = None, fetch: Fetcher = fetch_url,
               today: date | None = None, root: Path | None = None) -> FetchResult:
    """Add a source to sources.json and save its text, from its URL or from a local copy of the document.

    The URL is required even for a local file: it says where the official document comes from.
    """
    pack = load_pack(procedure_id, root or PROCEDURES_DIR)
    if pack.source(source.source_id) is not None:
        raise ValueError(f"{source.source_id} is already in sources.json; use fetch to update it")
    today = today or date.today()
    reader = fetch if file_path is None else _file_reader(Path(file_path))
    result, saved = fetch_one(pack, source, reader, today)
    if result.ok:
        save_sources(pack, [*pack.sources, saved])
    return result


def _file_reader(path: Path) -> Fetcher:
    """A fetcher that reads one local file instead of the network."""
    def read(_url: str) -> tuple[bytes, str]:
        if not path.is_file():
            raise IngestError(f"No such file: {path}")
        return path.read_bytes(), content_kind(path.name)
    return read


# ---------------------------------------------------------------- check --
def check_sources(procedure_id: str, fetch: Fetcher = fetch_url, root: Path | None = None) -> list[SourceCheck]:
    """Download every source again and compare it with the saved text. Nothing is written."""
    pack = load_pack(procedure_id, root or PROCEDURES_DIR)
    return [check_one(pack, source, fetch) for source in pack.sources]


def check_one(pack: Pack, source: Source, fetch: Fetcher) -> SourceCheck:
    """unchanged / changed (with the rules citing it and a short diff) / not_saved / unreachable."""
    if not source.content_sha256:
        return SourceCheck(source_id=source.source_id, status="not_saved")
    try:
        content, kind = fetch(source.url)
        page: SourceText = source_text(source.source_id, content, kind)
    except IngestError as error:
        return SourceCheck(source_id=source.source_id, status="unreachable", error=str(error))
    if page.sha256 == source.content_sha256:
        return SourceCheck(source_id=source.source_id, status="unchanged")
    old = stored_text(text_folder(pack) / f"{source.source_id}.txt")
    return SourceCheck(source_id=source.source_id, status="changed",
                       rules_to_review=rules_citing(pack, source.source_id), preview=diff_preview(old, page.text))


def diff_preview(old: str, new: str, limit: int = DIFF_PREVIEW_LINES) -> list[str]:
    """The first added or removed lines between two texts ('+ ...' / '- ...')."""
    lines = []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        lines.append(line[:160])
        if len(lines) >= limit:
            break
    return lines
