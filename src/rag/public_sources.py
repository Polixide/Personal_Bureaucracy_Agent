"""Load Daniele's committed, public source snapshots without fetching or indexing uploads."""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from src.rag.schemas import SourceChunk

ROOT = Path(__file__).resolve().parents[2]
PACKS = ("ikamet_first", "ikamet_extension", "tc_kimlik_first", "tc_kimlik_renewal")
CASE_WEB_IDS = frozenset({
    "WEB_IRO_EN_ADDRESS_REGISTRATION", "WEB_IRO_EN_OFFICIAL_DOCUMENTS",
    "WEB_IRO_EN_EXCHANGE_STUDENTS_GUIDE",
    "WEB_IRO_EN_EXCHANGE_STUDENTS_GUIDE_HEALTH_INSURANCE",
    "WEB_IRO_EN_EXCHANGE_STUDENTS_GUIDE_IMPORTANT_CONTACTS",
    "WEB_IRO_EN_EXCHANGE_STUDENTS_GUIDE_RESIDENCE_PERMIT",
    "WEB_IRO_EN_EXCHANGE_STUDENTS_GUIDE_VISA_AND_PASSPORT_REQ_6FCBF29",
})
SOURCE_ID = re.compile(r"[A-Z0-9_]{1,100}")


@dataclass(frozen=True)
class PublicSources:
    chunks: tuple[SourceChunk, ...]
    sources: int
    loaded: int
    issues: tuple[str, ...]
    fingerprint: str


def _locations(root):
    yield "public_web", root / "knowledge" / "corpus.json", root / "knowledge" / "corpus"
    for pack in PACKS:
        folder = root / "procedures" / pack
        yield pack, folder / "sources.json", folder / "sources_text"


def chunk_text(text, limit=800):
    """Daniele's paragraph-first chunks, retaining the order of source text."""
    block = ""
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        while len(paragraph) > limit:
            end = max(paragraph.rfind("\n", 1, limit + 1), paragraph.rfind(" ", limit // 2, limit + 1))
            end = end if end > 0 else limit
            if block:
                yield block
                block = ""
            yield paragraph[:end].strip()
            paragraph = paragraph[end:].lstrip()
        if paragraph:
            if block and len(block) + len(paragraph) + 2 > limit:
                yield block
                block = ""
            block = block + "\n\n" + paragraph if block else paragraph
    if block:
        yield block


def _safe_text(path, root):
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("Source points outside its public corpus")
    return path.read_text(encoding="utf-8")


def _body(text):
    header, divider, body = text.partition("\n\n")
    return body.strip() if divider and header.startswith("SOURCE:") else text.strip()


def _manifest(path, root):
    data = json.loads(_safe_text(path, root))
    if not isinstance(data, list):
        raise ValueError("Expected a list of public source manifest entries")
    return data


def _document(scope, record, folder, root):
    source_id = record["source_id"]
    if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
        raise ValueError("Invalid public source ID")
    path = folder / (source_id + ".txt")
    parsed = urlparse(record["url"])
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid source URL")
    body = _body(_safe_text(path, root))
    actual = sha256(body.encode("utf-8")).hexdigest()
    if not body or actual != record.get("content_sha256"):
        raise ValueError("Content checksum mismatch")

    authority = "sabanci_operational" if scope == "public_web" else record["authority"]
    if scope == "public_web" and parsed.hostname != "iro.sabanciuniv.edu":
        raise ValueError("Unexpected public web source")
    kind = "reference_guidance" if authority == "crosscheck_only" else "official_guidance"
    for index, text in enumerate(chunk_text(body)):
        identity = json.dumps([scope, source_id, actual, index])
        yield SourceChunk(chunk_id=sha256(identity.encode("utf-8")).hexdigest()[:24], source_id=source_id,
                          title=record["title"], url=record["url"], authority=authority,
                          content_kind=kind, content=text, retrieved_at=record.get("retrieved_at"),
                          procedure=scope, file_ref=path.relative_to(root).as_posix(),
                          organization=record.get("organization"), topic=record.get("topic"),
                          source_version=actual[:12])


@lru_cache(maxsize=4)
def _load(root, signatures):
    chunks, issues = [], []
    sources = loaded = 0
    for scope, manifest, folder in _locations(root):
        try:
            records = _manifest(manifest, root)
        except (OSError, ValueError):
            issues.append(f"{manifest.relative_to(root).as_posix()} · unavailable or invalid manifest")
            continue
        seen = set()
        for record in records:
            sources += 1
            source_id = record.get("source_id") if isinstance(record, dict) else None
            if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id) or source_id in seen:
                issues.append(f"{manifest.relative_to(root).as_posix()} · invalid or repeated source ID")
                continue
            seen.add(source_id)
            path = (folder / (source_id + ".txt")).relative_to(root).as_posix()
            try:
                rows = list(_document(scope, record, folder, root))
            except (OSError, ValueError, TypeError, KeyError):
                issues.append(f"{path} · missing, invalid, or checksum mismatch")
                continue
            loaded += 1
            chunks.extend(rows)
    fingerprint = sha256("".join(chunk.chunk_id for chunk in chunks).encode()).hexdigest()
    return PublicSources(tuple(chunks), sources, loaded, tuple(issues), fingerprint)


def load_public_sources(root=None):
    root = (ROOT if root is None else Path(root)).resolve()
    paths = []
    for _, manifest, folder in _locations(root):
        paths.append(manifest)
        paths.extend(sorted(folder.glob("*.txt")))
    signatures = []
    for path in paths:
        try:
            stat = path.stat()
            signatures.append((path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
        except OSError:
            signatures.append((path, None))
    return _load(root, tuple(signatures))


def relevant_to_case(chunk):
    return (chunk.procedure == "ikamet_first" and chunk.authority != "crosscheck_only"
            or chunk.procedure == "public_web" and chunk.source_id in CASE_WEB_IDS)
