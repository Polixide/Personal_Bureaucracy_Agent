"""Local allow-listed snapshot comparison; never writes active knowledge."""
from hashlib import sha256
import json
from config import KNOWLEDGE_DIR
from src.rag.ingestion import build_corpus,source_catalogue
from src.workers.rule_loader import load_rules


def compare_snapshot(source_id,source_url,content):
    source=source_catalogue().get(source_id)
    if source is None or source.get("authority_level") not in {"government_primary","sabanci_operational"} or source.get("url")!=source_url:
        raise ValueError("Snapshot must match an allow-listed official source ID and exact URL")
    old="\n".join(row.content for row in build_corpus() if row.source_id==source_id and row.content_kind=="official_guidance")
    if not old:return {"status":"BASELINE_REQUIRED","source_id":source_id,"source_url":source_url,"message":"No comparable guidance snapshot exists. Human review must establish a baseline; rule excerpts are not source snapshots."}
    digest=lambda text:sha256(text.encode()).hexdigest()
    changed=digest(old)!=digest(content)
    rules=[r["rule_id"] for r in load_rules() if source_id in r["source_ids"]]
    version=json.loads((KNOWLEDGE_DIR/"rules.json").read_text(encoding="utf-8"))["dataset_version"]
    return {"status":"RULE_CHANGE_CANDIDATE" if changed else "UNCHANGED","source_id":source_id,"source_url":source_url,"before_hash":digest(old),"after_hash":digest(content),"active_version":version,"affected_rule_ids":rules,"new_snapshot":content if changed else None,"message":"Content difference is a review candidate, not proof of a changed law. A human must compare passages, update/version the reviewed knowledge explicitly and run regression tests before promotion. Active rules were not changed."}
