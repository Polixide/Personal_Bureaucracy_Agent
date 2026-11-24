"""Regression oracle: the generic evaluator reproduces the milestone-1 engine's verdicts.

tests/data/legacy_verdicts.json was generated once from the old engine (src/workers/rules_engine.py,
default scope) by scripts/snapshot_legacy_verdicts.py (deleted after the migration; see commit 486917b)
and is never edited by hand. Until phase 9 a second
test also re-ran the old engine against the snapshot; the old engine was deleted in phase 9 after both
tests had passed together (commits fdc5ebd .. 486917b).
Known, approved divergence: the old code-only "Milestone-1 scoped" R015 variant is not reproduced
(docs/migration_plan.md, decision on R015); the snapshot uses the default mode.
"""
import json
from pathlib import Path

import pytest

from src.pack import coerce_fact, load_pack
from src.rules import evaluate_rules

SNAPSHOT = Path(__file__).parent / "data" / "legacy_verdicts.json"
PACK = load_pack("ikamet_first")


def load_cases() -> list[dict]:
    lines = SNAPSHOT.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[1:] if line.strip()]


CASES = load_cases()


def new_verdicts(context: dict) -> dict[str, str]:
    facts = {name: coerce_fact(PACK, name, value) for name, value in context.items()}
    output = {}
    for result in evaluate_rules(PACK, facts):
        if result.status == "unknown":
            output[result.rule_id] = "unknown:" + ",".join(result.missing_fields)
        else:
            output[result.rule_id] = result.status
    return output


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_new_evaluator_matches_legacy_snapshot(case):
    assert new_verdicts(case["context"]) == case["verdicts"]


def test_snapshot_exercises_every_rule_in_every_status():
    seen = {rule_id: set() for rule_id in PACK.rule_ids()}
    for case in CASES:
        for rule_id, verdict in case["verdicts"].items():
            seen[rule_id].add(verdict.split(":")[0])
    assert all(statuses == {"pass", "fail", "unknown"} for statuses in seen.values()), seen
