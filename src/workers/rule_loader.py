"""Load the active root knowledge contract; never execute condition strings."""
import json
from config import KNOWLEDGE_DIR


def load_rules(snapshot=None):
    if snapshot is not None:return snapshot["rules"]
    data=json.loads((KNOWLEDGE_DIR/"rules.json").read_text(encoding="utf-8"))
    rules=data["rules"]
    ids=[r["rule_id"] for r in rules]
    if len(ids)!=len(set(ids)) or not ids:
        raise ValueError("Active rule IDs must be unique and nonempty")
    sources={s["source_id"] for s in json.loads((KNOWLEDGE_DIR/"sources.json").read_text(encoding="utf-8"))["sources"]}
    if any(set(r["dependencies"])-set(ids) or set(r["source_ids"])-sources for r in rules):
        raise ValueError("Unknown active rule dependency or source")
    return rules


def load_dependencies(snapshot=None):
    if snapshot is not None:return {key:set(values) for key,values in snapshot["dependencies"].items()}
    rules=load_rules()
    result={r["rule_id"]:set(r["dependencies"]) for r in rules}
    edges=json.loads((KNOWLEDGE_DIR/"dependencies.json").read_text(encoding="utf-8"))["edges"]
    for prerequisite,dependent in edges:
        if prerequisite not in result or dependent not in result:raise ValueError("Unknown dependency edge")
        result[dependent].add(prerequisite)
    return result


def knowledge_snapshot():
    """Host-captured immutable-by-convention rule contract for one checkpointed run."""
    before={name:(KNOWLEDGE_DIR/name).read_bytes() for name in ("rules.json","dependencies.json","sources.json")}
    rules=load_rules()
    version=json.loads((KNOWLEDGE_DIR/"rules.json").read_text(encoding="utf-8"))["dataset_version"]
    dependencies={key:sorted(values) for key,values in load_dependencies().items()}
    if any((KNOWLEDGE_DIR/name).read_bytes()!=value for name,value in before.items()):raise ValueError("Knowledge changed while capturing this run; retry after the reviewed update completes")
    return {"version":version,"rules":rules,"dependencies":dependencies}
