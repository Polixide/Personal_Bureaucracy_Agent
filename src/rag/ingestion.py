"""Offline ingestion of canonical rule excerpts and optional authoritative text.

No remote fetch runs at import/startup. Cache inputs reference an existing
knowledge source; URLs/authority are resolved from the canonical source catalogue.
"""
import hashlib
import json
from pathlib import Path
from config import KNOWLEDGE_DIR, PROJECT_ROOT
from src.rag.schemas import GuidanceInput, SourceChunk

GUIDANCE_CACHE = PROJECT_ROOT / "data" / "official_guidance.json"
ALLOWED_AUTHORITIES = {"government_primary", "sabanci_operational", "sabanci_direct_instruction"}


def source_catalogue():
    data = json.loads((KNOWLEDGE_DIR / "sources.json").read_text(encoding="utf-8"))
    return {row["source_id"]: row for row in data["sources"] if row["authority_level"] in ALLOWED_AUTHORITIES}


def _chunks(text, size=1400, overlap=180):
    text = text.strip()
    offset = 0
    while offset < len(text):
        end = min(len(text), offset + size)
        if end < len(text):
            boundary = text.rfind(" ", offset + size // 2, end)
            if boundary > offset: end = boundary
        yield text[offset:end]
        if end == len(text): break
        offset = max(offset + 1, end - overlap)


def _chunk(source, text, kind, *, rule_ids=(), verified_at=None):
    digest = hashlib.sha256((source["source_id"] + text).encode()).hexdigest()[:20]
    return SourceChunk(chunk_id=digest, source_id=source["source_id"], title=source["title"],
        url=source["url"], authority=source["authority_level"], content=text,
        content_kind=kind, retrieved_at=source.get("retrieved_at"), verified_at=verified_at,
        rule_ids=list(rule_ids))


def ingest_guidance(records, path=GUIDANCE_CACHE):
    """Validate an explicit local source cache; never alter knowledge/rules."""
    catalogue = source_catalogue()
    checked = [GuidanceInput.model_validate(record) for record in records]
    if any(record.source_id not in catalogue for record in checked):
        raise ValueError("Guidance must reference a canonical government or Sabanci source")
    target = Path(path).resolve()
    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError("Guidance cache must stay inside the project")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([record.model_dump(mode="json") for record in checked], ensure_ascii=False, indent=2), encoding="utf-8")
    return len(checked)


def build_corpus(cache_path=GUIDANCE_CACHE):
    """Build small inspectable chunks without embeddings or a second rule engine."""
    catalogue = source_catalogue()
    rule_data = json.loads((KNOWLEDGE_DIR / "rules.json").read_text(encoding="utf-8"))
    corpus = []
    for rule in rule_data["rules"]:
        sources = [catalogue[sid] for sid in rule["source_ids"] if sid in catalogue]
        if not sources: continue
        sources.sort(key=lambda s: (s["authority_level"] != "government_primary", s["source_id"]))
        if rule["rule_id"]=="R017":sources.sort(key=lambda s:s["authority_level"]!="sabanci_direct_instruction")
        content = rule["description"] + (" " + rule["notes"] if rule.get("notes") else "")
        corpus.append(_chunk(sources[0], content, "project_operational_policy" if sources[0]["authority_level"]=="sabanci_direct_instruction" else "project_rule_excerpt", rule_ids=[rule["rule_id"]]))
    cache = Path(cache_path)
    if cache.is_file():
        records = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(records, list): raise ValueError("Guidance cache must contain a list")
        for raw in records:
            record = GuidanceInput.model_validate(raw)
            if record.source_id not in catalogue: raise ValueError("Unapproved source in guidance cache")
            corpus.extend(_chunk(catalogue[record.source_id], text, "official_guidance", verified_at=record.verified_at)
                          for text in _chunks(record.content))
    return corpus
