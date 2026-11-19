"""Copy the checkpointed state into the twin record after each graph invocation.

    started = utc_now()
    try:
        graph.invoke(payload, config)
    finally:
        record_after_invoke(store, graph, config, started)

or simply `invoke_and_record(graph, store, payload, config)`.

The checkpoint stays the live source of truth; this only mirrors it. Because
every call mirrors the whole twin, a missed sync (crash, error) heals on the
next successful one; only that run's log row would be missing.

Outcome of a run, as stored in `runs.outcome`:
  awaiting_approval  paused at the human_approval interrupt
  incomplete         paused anywhere else (e.g. a node raised)
  fail               validation errors or a failing run_status
  pass               otherwise

Reflection memory (plan 3.4): when a run needed repairs and the final plan then
passed Gate 1, one validated lesson per distinct Gate 1 error is written,
once per run.
"""
import json
from collections.abc import Mapping
from datetime import datetime
from functools import cache
from typing import Any

from config import KNOWLEDGE_DIR
from src.memory.codec import utc_now
from src.memory.twin_store import Lesson, TwinStore

FAIL_STATUSES = {"INVALID_INPUT", "APPROVAL_DENIED", "HUMAN_REVIEW_REQUIRED",
                 "PLANNING_UNAVAILABLE", "PLAN_INVALID"}
GATE_1_PASSED = {"VALIDATED", "APPROVED"}


def run_outcome(values: Mapping[str, Any], next_nodes: tuple[str, ...]) -> str:
    if next_nodes:
        return "awaiting_approval" if tuple(next_nodes) == ("human_approval",) else "incomplete"
    if values.get("validation_errors") or values.get("run_status") in FAIL_STATUSES:
        return "fail"
    return "pass"


@cache
def _rule_versions() -> dict[str, str]:
    rules = json.loads((KNOWLEDGE_DIR / "rules.json").read_text(encoding="utf-8"))["rules"]
    return {rule["rule_id"]: rule["version"] for rule in rules}


def lessons_from_run(values: Mapping[str, Any]) -> list[Lesson]:
    """Validated lessons from a repaired run; [] if no repair or Gate 1 never passed."""
    history = values.get("repair_history") or []
    if not history or values.get("run_status") not in GATE_1_PASSED:
        return []
    plan = values.get("candidate_plan") or []
    corrective = "Revised plan: " + "; ".join(step.action for step in plan) if plan else \
        "Revised plan without the rejected actions"
    outcome = f"Revised plan passed Gate 1 after {values.get('retry_count', len(history))} repair(s)"
    lessons, seen = [], set()
    for attempt in history:
        for error in attempt.get("errors", []):
            key = (error.get("code"), error.get("item"), error.get("rule_id"))
            if key in seen:
                continue
            seen.add(key)
            rule_id = error.get("rule_id")
            lessons.append(Lesson(
                user_id=values["user_id"], procedure=values.get("active_procedure") or "unknown",
                failure_type=error["code"], affected_item=error.get("item"),
                validator_error=error["message"], rule_or_source_id=rule_id,
                rule_version=_rule_versions().get(rule_id) if rule_id else None,
                corrective_action=corrective[:1000], outcome=outcome,
                correction_validated=True, trace_id=values.get("trace_id")))
    return lessons


def record_after_invoke(store: TwinStore, graph, config: dict,
                        started_at: datetime) -> str | None:
    """Mirror the thread's current checkpoint; returns the run outcome (None if no state)."""
    snapshot = graph.get_state(config)
    values = snapshot.values
    if not values or "user_id" not in values:
        return None
    outcome = run_outcome(values, tuple(snapshot.next))
    store.commit_snapshot(values, outcome, started_at)
    lessons = lessons_from_run(values)
    if lessons and values.get("trace_id"):
        store.add_lessons_once(values["trace_id"], lessons)
    return outcome


def invoke_and_record(graph, store: TwinStore, payload, config: dict):
    """graph.invoke(payload, config), then mirror the result, even if the run raised."""
    started = utc_now()
    try:
        return graph.invoke(payload, config)
    finally:
        record_after_invoke(store, graph, config, started)
