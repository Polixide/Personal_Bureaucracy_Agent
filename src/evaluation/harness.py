"""Evaluation harness: curated scenarios + multi-session restarts, scored.

Every scenario runs the REAL graph, rules engine, gates, repair loop, approval
interrupt, LangGraph SQLite checkpoint and twin record. Only model transport is
mocked (document extraction and planner proposals), so results are exact and
reproducible offline.

Compared with asserting inside a test, the harness:
  * collects every expectation as a Check, so one run reports all differences;
  * restarts the application (closes both databases, recompiles the graph)
    between the approval pause and the user's decision;
  * checks that the twin record equals the checkpoint after reopening, that
    each run is logged with its rule results, and that lessons are written
    exactly when a repaired plan passed Gate 1;
  * aggregates metrics (python -m scripts.run_eval).
"""
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langgraph.types import Command

from config import PROJECT_ROOT
from src.core.graph import build_graph
from src.core.state import ResidencePermitContext, UserEvent, new_run
from src.guardrails.gate_1 import gate_1 as real_gate_1
from src.memory.checkpointer import reset_thread, sqlite_checkpointer
from src.memory.twin_store import TwinStore, twin_view
from src.memory.twin_sync import invoke_and_record
from src.evaluation.scenarios import READY_CONTEXT, SCENARIOS, candidate, response

RULE_IDS = [f"R{i:03}" for i in range(1, 30)]
CERTIFICATE_TEXT = ("Student Enrollment Certificate\nHolder name: Alex Example\n"
                    "Student ID: SYN-001\nSabancı University\nIssue date: 2026-11-01")
CERTIFICATE_RESPONSES = [
    {"document_type": "student_enrollment_certificate"},
    {"holder_name": "Alex Example", "student_id": "SYN-001",
     "institution": "Sabancı University", "issue_date": "2026-11-01"}]


# ---------------------------------------------------------------- checks --
@dataclass
class Check:
    name: str
    expected: Any
    actual: Any

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


@dataclass
class ScenarioResult:
    scenario_id: str
    purpose: str
    kind: str                                  # "curated" | "multi-session"
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def check(self, name: str, expected: Any, actual: Any) -> None:
        self.checks.append(Check(name, expected, actual))

    @property
    def ok(self) -> bool:
        return self.error is None and all(c.ok for c in self.checks)


# --------------------------------------------------------- app lifecycle --
class App:
    """One 'application session': checkpoint + twin record + compiled graph."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self._stack = ExitStack()

    def __enter__(self) -> "App":
        saver = self._stack.enter_context(sqlite_checkpointer(self.workdir / "checkpoints.db"))
        self.store = self._stack.enter_context(TwinStore(self.workdir / "twin_record.db"))
        self.graph = build_graph(saver)
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()

    def run(self, payload, config):
        return invoke_and_record(self.graph, self.store, payload, config)


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


@contextmanager
def mocked_models(document_responses: list[dict], proposals: list, gate_log: list | None = None):
    """Mock only model transport; optionally record every Gate 1 output."""
    def gate_1(state):
        output = real_gate_1(state)
        if gate_log is not None:
            gate_log.append(output)
        return output

    with patch("src.document_processing.processor.Client") as document_client, \
         patch("src.agents.planner.create_action_plan") as planner, \
         patch("src.core.graph.gate_1", gate_1):
        document_client.return_value.chat.side_effect = [response(r) for r in document_responses]
        planner.side_effect = proposals
        yield document_client, planner


def check_record_matches_checkpoint(result: ScenarioResult, workdir: Path, thread_id: str) -> dict:
    """Reopen both databases (a restart) and compare; returns the checkpoint values."""
    with App(workdir) as app:
        snapshot = app.graph.get_state(config_for(thread_id))
        record = app.store.load_twin(thread_id)
        result.check("record == checkpoint after restart",
                     True, twin_view(record) == twin_view(snapshot.values))
        result.metrics["record_matches_checkpoint"] = twin_view(record) == twin_view(snapshot.values)
        return snapshot.values


# ------------------------------------------------------- curated scenarios --
def run_curated(scenario: dict, workdir: Path) -> ScenarioResult:
    result = ScenarioResult(scenario["id"], scenario["purpose"], "curated")
    thread_id = scenario["id"]
    config = config_for(thread_id)
    try:
        doc_path = workdir / "synthetic.txt"
        if scenario.get("unknown_document"):
            doc_path.write_text("ciao sono marco", encoding="utf-8")
            doc_responses = [{"document_type": "unknown"}]
        else:
            doc_path.write_text(CERTIFICATE_TEXT, encoding="utf-8")
            doc_responses = CERTIFICATE_RESPONSES
        good = candidate(scenario, submission=scenario.get("plan") == "submit")
        initial = candidate(scenario, submission=True) if scenario.get("premature") else good
        proposals = [initial]
        if scenario.get("premature"):
            if scenario.get("exhaust"):
                for attempt in range(1, 4):
                    revision = initial.model_copy(deep=True)
                    revision.actions[-1].reason = (f"Proposed submission after repair {attempt}; "
                                                   "prerequisites remain unverified")
                    proposals.append(revision)
            else:
                proposals.append(good)

        context = {} if scenario.get("empty_context") else READY_CONTEXT
        request = new_run(UserEvent(kind="upload", file_path=str(doc_path)), thread_id)
        request["residence_permit_context"] = ResidencePermitContext(
            **{**context, **scenario["changes"]})

        gates: list[dict] = []
        expected_statuses = {rule: "passed" for rule in RULE_IDS} | scenario["statuses"]
        with mocked_models(doc_responses, proposals, gates) as (document_client, planner):
            with App(workdir) as app:                                   # session 1
                state = app.run(request, config)
            actual_statuses = {r.rule_id: r.status for r in state["relevant_requirements"] or []}
            blockers = sorted(r.rule_id for r in state["relevant_requirements"] or []
                              if r.severity == "blocking" and r.passed is not True)
            gate_sequence = [not g["validation_errors"] for g in gates]
            paused = bool(state.get("__interrupt__"))
            if paused:
                with App(workdir) as app:                               # session 2, after restart
                    state = app.run(Command(resume=True), config)

            correct_rules = sum(actual_statuses.get(r) == s for r, s in expected_statuses.items())
            result.check("rule statuses", expected_statuses, actual_statuses)
            result.check("blockers", scenario["blockers"], blockers)
            result.check("gate 1 sequence", scenario["gates"], gate_sequence)
            if scenario.get("error"):
                result.check(f"gate 1 caught {scenario['error']}", True,
                             scenario["error"] in {e.code for e in gates[0]["validation_errors"]})
            result.check("paused for approval", scenario["gates"][-1], paused)
            result.check("final status", scenario["final"], state.get("run_status"))
            result.check("repairs used", len(scenario["gates"]) - 1, state.get("retry_count"))
            result.check("approved plan written",
                         scenario["final"] == "APPROVED", state.get("approved_plan") is not None)
            if scenario["final"] == "APPROVED":
                result.check("approved plan is the valid proposal",
                             [a.step_id for a in good.actions],
                             [a.step_id for a in state["approved_plan"].actions])
            result.check("planner calls", len(scenario["gates"]), planner.call_count)
            result.check("model extraction calls", len(doc_responses),
                         document_client.return_value.chat.call_count)
            result.metrics.update(
                rules_correct=correct_rules, rules_total=len(expected_statuses),
                final_ok=scenario["final"] == state.get("run_status"),
                invalid_plans_expected=scenario["gates"].count(False),
                invalid_plans_caught=sum(not want and not got for want, got
                                         in zip(scenario["gates"], gate_sequence)),
                false_alarms=sum(want and not got for want, got
                                 in zip(scenario["gates"], gate_sequence)),
                repair_expected=len(scenario["gates"]) > 1 and scenario["gates"][-1],
                repaired=len(gate_sequence) > 1 and gate_sequence[-1:] == [True],
                limit_expected=not scenario["gates"][-1],
                stopped_at_limit=(not paused and state.get("retry_count") == 3
                                  and state.get("run_status") == "HUMAN_REVIEW_REQUIRED"))

        final = check_record_matches_checkpoint(result, workdir, thread_id)
        with TwinStore(workdir / "twin_record.db") as store:
            runs = store.list_runs(thread_id)
            lessons = store.find_lessons(thread_id, "ikamet", limit=50)
        expected_outcome = "pass" if scenario["final"] == "APPROVED" else "fail"
        result.check("runs logged", [expected_outcome], [r.outcome for r in runs])
        result.check("rule results logged", expected_statuses,
                     runs[0].requirements if runs else None)
        repaired_ok = len(scenario["gates"]) > 1 and scenario["gates"][-1]
        result.check("lesson written iff a repair passed Gate 1", repaired_ok, bool(lessons))
        if repaired_ok and scenario.get("error"):
            result.check("lesson names the caught error", True,
                         any(l.failure_type == scenario["error"] for l in lessons))
        result.check("checkpoint finished", None, final.get("__interrupt__"))
        result.metrics["lessons"] = len(lessons)
    except Exception as exc:                                  # report, don't abort the suite
        result.error = f"{type(exc).__name__}: {exc}"
    return result


# -------------------------------------------------- multi-session scenarios --
def _upload_request(workdir: Path, thread_id: str) -> dict:
    path = workdir / f"{thread_id}.txt"
    path.write_text(CERTIFICATE_TEXT, encoding="utf-8")
    return new_run(UserEvent(kind="upload", file_path=str(path)), thread_id)


def _proposal():
    return candidate({"corrections": [("R003", "Provide entry details and requested stay dates")]})


def ms_three_sessions(workdir: Path) -> ScenarioResult:
    """Upload, quit; come back and approve, quit; come back and re-evaluate."""
    result = ScenarioResult("M1-three-sessions",
                            "Documents, approval and run log survive restarts; re-evaluation voids the approval.",
                            "multi-session")
    thread_id, config = "M1-student", config_for("M1-student")
    with mocked_models(CERTIFICATE_RESPONSES, [_proposal()]):
        with App(workdir) as app:
            first = app.run(_upload_request(workdir, thread_id), config)
        result.check("session 1 paused for approval", True, bool(first.get("__interrupt__")))
        with App(workdir) as app:
            second = app.run(Command(resume=True), config)
        result.check("session 2 approved after restart", "APPROVED", second.get("run_status"))
        with TwinStore(workdir / "twin_record.db") as store:
            result.check("approved plan recorded", True,
                         store.load_twin(thread_id).get("approved_plan") is not None)
        with App(workdir) as app:
            third = app.run(new_run(UserEvent(kind="reevaluate"), thread_id), config)
    result.check("session 3 still has the document", ["student_certificate"],
                 [d.doc_type for d in third.get("documents", [])])
    # new_run() clears approved_plan on re-evaluation: an approval of an older
    # assessment must not survive new facts. The record must follow.
    result.check("re-evaluation clears the old approval", None, third.get("approved_plan"))
    check_record_matches_checkpoint(result, workdir, thread_id)
    with TwinStore(workdir / "twin_record.db") as store:
        runs = store.list_runs(thread_id)
    result.check("run log", [("upload", "pass"), ("reevaluate", "pass")],
                 [(r.event_kind, r.outcome) for r in runs])
    return result


def ms_denial_is_remembered(workdir: Path) -> ScenarioResult:
    result = ScenarioResult("M2-denied-after-restart",
                            "A plan denied after a restart is logged and never written back.",
                            "multi-session")
    thread_id, config = "M2-student", config_for("M2-student")
    with mocked_models(CERTIFICATE_RESPONSES, [_proposal()]):
        with App(workdir) as app:
            app.run(_upload_request(workdir, thread_id), config)
        with TwinStore(workdir / "twin_record.db") as store:
            result.check("paused run logged", ["awaiting_approval"],
                         [r.outcome for r in store.list_runs(thread_id)])
        with App(workdir) as app:
            final = app.run(Command(resume=False), config)
    result.check("final status", "APPROVAL_DENIED", final.get("run_status"))
    result.check("no approved plan", None, final.get("approved_plan"))
    check_record_matches_checkpoint(result, workdir, thread_id)
    with TwinStore(workdir / "twin_record.db") as store:
        [run] = store.list_runs(thread_id) or [None]
        result.check("same run updated to fail", ("fail", "APPROVAL_DENIED"),
                     (run.outcome, run.run_status) if run else None)
        result.check("record has no approved plan", None,
                     store.load_twin(thread_id).get("approved_plan"))
    return result


def ms_reset_forgets_one_student(workdir: Path) -> ScenarioResult:
    result = ScenarioResult("M3-reset-one-student",
                            "Reset deletes one student's checkpoint and record; others stay.",
                            "multi-session")
    with mocked_models(CERTIFICATE_RESPONSES * 2, [_proposal(), _proposal()]):
        with App(workdir) as app:
            for thread_id in ("M3-keep", "M3-forget"):
                app.run(_upload_request(workdir, thread_id), config_for(thread_id))
    reset_thread("M3-forget", "M3-forget", workdir / "checkpoints.db")
    with App(workdir) as app:
        result.check("checkpoint deleted", {}, app.graph.get_state(config_for("M3-forget")).values)
        result.check("record deleted", {}, app.store.load_twin("M3-forget"))
        result.check("run log deleted", [], app.store.list_runs("M3-forget"))
        result.check("other student kept", 1, len(app.store.load_twin("M3-keep")["documents"]))
    return result


MULTI_SESSION: list[Callable[[Path], ScenarioResult]] = [
    ms_three_sessions, ms_denial_is_remembered, ms_reset_forgets_one_student]


def run_multi_session(scenario: Callable[[Path], ScenarioResult], workdir: Path) -> ScenarioResult:
    try:
        return scenario(workdir)
    except Exception as exc:
        return ScenarioResult(scenario.__name__, scenario.__doc__ or "", "multi-session",
                              error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- report --
def evaluate(workdir: Path | None = None) -> dict:
    """Run every scenario in its own directory and aggregate metrics."""
    scratch = PROJECT_ROOT / ".test-temp"          # checkpoints must stay inside the project
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        base = Path(workdir or tmp)
        results = []
        for scenario in SCENARIOS:
            (base / scenario["id"]).mkdir(parents=True, exist_ok=True)
            results.append(run_curated(scenario, base / scenario["id"]))
        for scenario in MULTI_SESSION:
            (base / scenario.__name__).mkdir(parents=True, exist_ok=True)
            results.append(run_multi_session(scenario, base / scenario.__name__))

    curated = [r for r in results if r.kind == "curated" and r.error is None]
    total = lambda key: sum(r.metrics.get(key, 0) for r in curated)
    return {
        "results": results,
        "scenarios_passed": sum(r.ok for r in results), "scenarios_total": len(results),
        "checks_passed": sum(c.ok for r in results for c in r.checks),
        "checks_total": sum(len(r.checks) for r in results),
        "final_status_correct": total("final_ok"), "curated_total": len(SCENARIOS),
        "rules_correct": total("rules_correct"), "rules_total": total("rules_total"),
        "invalid_plans_caught": total("invalid_plans_caught"),
        "invalid_plans_expected": total("invalid_plans_expected"),
        "false_alarms": total("false_alarms"),
        "repairs_expected": total("repair_expected"), "repairs_succeeded": total("repaired"),
        "limit_expected": total("limit_expected"), "stopped_at_limit": total("stopped_at_limit"),
        "record_matches": sum(bool(r.metrics.get("record_matches_checkpoint")) for r in results),
        "lessons_written": total("lessons"),
    }


def _ratio(a: int, b: int) -> str:
    return f"{a}/{b}" + (f" ({a / b:.0%})" if b else "")


def format_report(report: dict) -> str:
    lines = ["Evaluation (real graph, rules, gates, approval and databases; model calls mocked)",
             "-" * 78]
    for r in report["results"]:
        failed = [c for c in r.checks if not c.ok]
        lines.append(f"{'PASS' if r.ok else 'FAIL'}  {r.scenario_id:<28} "
                     f"{len(r.checks) - len(failed):>2}/{len(r.checks):<2} checks  {r.purpose}")
        if r.error:
            lines.append(f"      crashed: {r.error}")
        for c in failed:
            lines.append(f"      {c.name}: expected {c.expected!r}, got {c.actual!r}")
    lines += ["-" * 78,
              f"Scenarios passed           : {_ratio(report['scenarios_passed'], report['scenarios_total'])}",
              f"Checks passed              : {_ratio(report['checks_passed'], report['checks_total'])}",
              f"Final status correct       : {_ratio(report['final_status_correct'], report['curated_total'])}",
              f"Rule statuses correct      : {_ratio(report['rules_correct'], report['rules_total'])}",
              f"Invalid plans caught (G1)  : {_ratio(report['invalid_plans_caught'], report['invalid_plans_expected'])}",
              f"False alarms (G1)          : {report['false_alarms']}",
              f"Repaired, then passed G1   : {report['repairs_succeeded']} (expected {report['repairs_expected']})",
              f"Stopped at 3-repair limit  : {report['stopped_at_limit']} (expected {report['limit_expected']})",
              f"Record == checkpoint       : {_ratio(report['record_matches'], report['scenarios_total'] - 1)}"
              "  (after restart; reset scenario excluded)",
              f"Validated lessons written  : {report['lessons_written']}"]
    return "\n".join(lines)
