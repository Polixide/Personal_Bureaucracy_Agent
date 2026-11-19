"""Load only the canonical root knowledge; never execute condition strings."""
import json
from config import KNOWLEDGE_DIR

def load_rules():
    data = json.loads((KNOWLEDGE_DIR / "rules.json").read_text(encoding="utf-8"))
    rules = data["rules"]
    if {r["rule_id"] for r in rules} != {f"R{i:03}" for i in range(1, 30)}:
        raise ValueError("Unsupported knowledge rule contract")
    return rules

def load_dependencies():
    # Union both root declarations so neither prerequisite list is silently lost.
    edges = json.loads((KNOWLEDGE_DIR / "dependencies.json").read_text(encoding="utf-8"))["edges"]
    result = {r["rule_id"]: set(r["dependencies"]) for r in load_rules()}
    for prerequisite, dependent in edges:
        result[dependent].add(prerequisite)
    return result
