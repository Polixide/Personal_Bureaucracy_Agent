"""Consistency of the knowledge layer (knowledge/*.json).

These do not evaluate rules (that is the Rules Engine's job); they check the
rule base is internally consistent, so a broken edit is caught before use.
"""
import json
from datetime import date
from pathlib import Path

import pytest

from src.twin.codec import decode_date


def load_rule_cases() -> list[dict]:
    """knowledge/test_cases.json with every `*_date` input parsed strictly to a `date`.

    Moved here from Neda's src/evaluation/harness.py (retired in phase 3) without changes.
    """
    path = Path(__file__).resolve().parents[1] / "knowledge" / "test_cases.json"
    cases = json.loads(path.read_text())["tests"]
    for case in cases:
        case["input"] = {k: decode_date(v) if k.endswith("_date") else v
                         for k, v in case["input"].items()}
    return cases

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"


def load(name):
    return json.loads((KNOWLEDGE / name).read_text())


RULES = {r["rule_id"]: r for r in load("rules.json")["rules"]}
SOURCES = {s["source_id"] for s in load("sources.json")["sources"]}
EDGES = load("dependencies.json")["edges"]


def test_rule_ids_are_unique():
    assert len(RULES) == len(load("rules.json")["rules"])


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_rule_sources_exist(rule_id):
    assert set(RULES[rule_id]["source_ids"]) <= SOURCES


def test_dependency_edges_reference_known_rules():
    unknown = {r for edge in EDGES for r in edge} - set(RULES)
    assert not unknown


def test_dependency_graph_has_no_cycle():
    graph = {r: [] for r in RULES}
    for before, after in EDGES:
        graph[before].append(after)
    state = {}                                       # 1 = visiting, 2 = done

    def visit(node, trail):
        if state.get(node) == 1:
            pytest.fail(f"cycle: {' -> '.join(trail + [node])}")
        if state.get(node) != 1 and state.get(node) != 2:
            state[node] = 1
            for nxt in graph[node]:
                visit(nxt, trail + [node])
            state[node] = 2

    for node in graph:
        visit(node, [])


def test_rule_test_cases_reference_known_rules_and_valid_outcomes():
    cases = load_rule_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        assert case["rule"] in RULES, case["id"]
        assert case["expected"] in ("PASS", "FAIL"), case["id"]


def test_rule_test_case_dates_are_parsed_strictly():
    cases = {c["id"]: c for c in load_rule_cases()}
    assert cases["T02"]["input"]["application_date"] == date(2026, 11, 10)
    assert all(isinstance(v, date) for c in cases.values()
               for k, v in c["input"].items() if k.endswith("_date"))
