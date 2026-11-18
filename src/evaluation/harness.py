"""Scenario harness: replay synthetic students through the graph with restarts.

For each scenario:
  1. seed the stored Digital Twin with `initial_twin`;
  2. for each session: open a NEW store connection and compile a NEW graph
     (a simulated application restart), run the event, close everything;
  3. reopen the store and compare what was persisted with `expect`.

So every check is made on persisted data, never on in-memory state.

Usage:  python -m scripts.run_eval            (prints a report, exit 1 on failure)
"""
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.graph import build_graph
from src.core.state import UserEvent
from src.memory.checkpointer import TwinStore
from src.memory.codec import decode_date, decode_timestamp
from src.memory.session import run_event

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_FILE = ROOT / "data" / "synthetic_samples" / "scenarios.json"
RULE_CASES_FILE = ROOT / "knowledge" / "test_cases.json"


# --------------------------------------------------------------- loading --
def from_typed_json(value: Any) -> Any:
    """Decode {"$date": ...} / {"$datetime": ...} markers, recursively. Strict."""
    if isinstance(value, dict):
        if set(value) == {"$date"}:
            return decode_date(value["$date"])
        if set(value) == {"$datetime"}:
            return decode_timestamp(value["$datetime"])
        return {k: from_typed_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_typed_json(v) for v in value]
    return value


def load_scenarios(path: Path = SCENARIOS_FILE) -> list[dict]:
    return from_typed_json(json.loads(Path(path).read_text()))["scenarios"]


def load_rule_cases(path: Path = RULE_CASES_FILE) -> list[dict]:
    """knowledge/test_cases.json with every `*_date` input parsed to a `date`.

    For the Rules Engine owner: feed `case["input"]` to the rule evaluator and
    compare with `case["expected"]` ("PASS" / "FAIL"). A malformed date fails here,
    at load time, instead of being compared as a string.
    """
    cases = json.loads(Path(path).read_text())["tests"]
    for case in cases:
        case["input"] = {k: decode_date(v) if k.endswith("_date") else v
                         for k, v in case["input"].items()}
    return cases


# ---------------------------------------------------------------- checks --
@dataclass
class Check:
    scenario_id: str
    session: int                 # 1-based; 0 = scenario-level
    name: str
    expected: Any
    actual: Any

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


@dataclass
class ScenarioResult:
    scenario_id: str
    checks: list[Check] = field(default_factory=list)
    error: str | None = None     # crash while running the scenario

    @property
    def ok(self) -> bool:
        return self.error is None and all(c.ok for c in self.checks)


def _compare_twin(sid: str, n: int, expect: dict, twin: dict, runs: list) -> list[Check]:
    checks = []
    if "blockers" in expect:
        checks.append(Check(sid, n, "blockers", expect["blockers"], twin.get("blockers", [])))
    if "facts" in expect:
        facts = twin.get("extracted_facts", {})
        for key, value in expect["facts"].items():
            checks.append(Check(sid, n, f"fact {key}", value, facts.get(key, "<missing>")))
    if "documents" in expect:
        stored = {d.doc_id: d for d in twin.get("documents", [])}
        checks.append(Check(sid, n, "document ids", sorted(expect["documents"]), sorted(stored)))
        for doc_id, fields in expect["documents"].items():
            doc = stored.get(doc_id)
            for name, value in fields.items():
                checks.append(Check(sid, n, f"{doc_id}.{name}", value,
                                    getattr(doc, name) if doc else "<missing>"))
    if "runs" in expect:
        counts = Counter(r.outcome for r in runs)
        checks.append(Check(sid, n, "stored runs",
                            expect["runs"], {"pass": counts["pass"], "fail": counts["fail"]}))
    return checks


def run_scenario(scenario: dict, db_path: Path,
                 graph_factory: Callable = build_graph) -> ScenarioResult:
    sid, user_id = scenario["scenario_id"], scenario["user_id"]
    result = ScenarioResult(sid)
    try:
        with TwinStore(db_path) as store:
            store.save_twin({"user_id": user_id, **scenario.get("initial_twin", {})})

        for n, session in enumerate(scenario["sessions"], start=1):
            expect, path = session["expect"], []
            with TwinStore(db_path) as store:                    # restart: new connection
                final = run_event(graph_factory(), store, user_id,  # and newly compiled graph
                                  UserEvent(**session["event"]), path=path)
            errors = final.get("validation_errors") or []
            if "outcome" in expect:
                result.checks.append(Check(sid, n, "outcome", expect["outcome"],
                                           "fail" if errors else "pass"))
            if "error_codes" in expect:
                result.checks.append(Check(sid, n, "error codes", expect["error_codes"],
                                           [e.code for e in errors]))
            for node in expect.get("path_includes", []):
                result.checks.append(Check(sid, n, f"ran {node}", True, node in path))
            for node in expect.get("path_excludes", []):
                result.checks.append(Check(sid, n, f"skipped {node}", True, node not in path))

            with TwinStore(db_path) as store:                    # read back after restart
                twin, runs = store.load_twin(user_id), store.list_runs(user_id)
            result.checks += _compare_twin(sid, n, expect, twin, runs)
    except Exception as exc:                                     # report, don't abort the suite
        result.error = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------- report --
def evaluate(scenarios: list[dict] | None = None, workdir: Path | None = None) -> dict:
    """Run every scenario in its own database file and summarise."""
    scenarios = load_scenarios() if scenarios is None else scenarios
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(workdir or tmp)
        results = [run_scenario(s, base / f"{s['scenario_id']}.sqlite") for s in scenarios]

    checks = [c for r in results for c in r.checks]
    expected_fail = [c for c in checks if c.name == "outcome" and c.expected == "fail"]
    expected_pass = [c for c in checks if c.name == "outcome" and c.expected == "pass"]
    return {
        "results": results,
        "scenarios_passed": sum(r.ok for r in results),
        "scenarios_total": len(results),
        "checks_passed": sum(c.ok for c in checks),
        "checks_total": len(checks),
        # guardrail view: did gates stop what they should, and only that?
        "failures_caught": sum(c.actual == "fail" for c in expected_fail),
        "failures_expected": len(expected_fail),
        "false_alarms": sum(c.actual == "fail" for c in expected_pass),
        "persistence_checks_passed": sum(c.ok for c in checks if c.name not in
                                         ("outcome", "error codes") and
                                         not c.name.startswith(("ran ", "skipped "))),
        "persistence_checks_total": sum(1 for c in checks if c.name not in
                                        ("outcome", "error codes") and
                                        not c.name.startswith(("ran ", "skipped "))),
    }


def format_report(report: dict) -> str:
    lines = ["Scenario evaluation", "-" * 60]
    for r in report["results"]:
        failed = [c for c in r.checks if not c.ok]
        lines.append(f"{'PASS' if r.ok else 'FAIL'}  {r.scenario_id}  "
                     f"({len(r.checks) - len(failed)}/{len(r.checks)} checks)")
        if r.error:
            lines.append(f"      crashed: {r.error}")
        for c in failed:
            lines.append(f"      session {c.session} {c.name}: "
                         f"expected {c.expected!r}, got {c.actual!r}")
    lines += ["-" * 60,
              f"Scenarios passed      : {report['scenarios_passed']}/{report['scenarios_total']}",
              f"Checks passed         : {report['checks_passed']}/{report['checks_total']}",
              f"Persisted-state checks: {report['persistence_checks_passed']}/"
              f"{report['persistence_checks_total']}",
              f"Failures caught       : {report['failures_caught']}/{report['failures_expected']}",
              f"False alarms          : {report['false_alarms']}"]
    return "\n".join(lines)
