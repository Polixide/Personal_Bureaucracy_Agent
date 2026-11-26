"""Offline regressions for the target branch's opaque planning contract."""
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from scripts.evaluate_agent import compare_exports, export_results, load_suite
from src.agents.planner import PlannerDraft, PlannerError, _ground_draft, compact_planner_context, planner
from src.core.legal_stay import INSURANCE_PREPARATION_ADVISORY, insurance_explanation
from src.core.llm_diagnostics import collect_model_calls, log_planner_fallback, logged_chat, ollama_usage
from src.core.state import ActionPlan, UserEvent
from src.evaluation.metrics import aggregate, exact_action_score, retrieval_metrics, summarize_calls
from src.evaluation.router import (
    WorkflowTrace, _accepted_input, _case_outcome, _document_fixture, _extraction_scores, _mock_transport, run_case,
)
from src.guardrails.gate_0 import gate_0
from src.guardrails.gate_1 import gate_1, validate_plan
from src.memory.reflection import reflection
from src.workers.document_analyzer import document_analyzer
from src.workers.rules_engine import rules_engine


@pytest.fixture(scope="module")
def suite():
    return load_suite()


def _scenario(suite, name):
    return next(spec for spec in suite["scenarios"] if spec["id"] == name)


def _assessed(suite, spec):
    state = _accepted_input(suite, spec, UserEvent(kind="generate_plan"))
    state.update(gate_0(state))
    if not state.get("validation_errors"):
        state.update(rules_engine(state))
    return state


def _planning_attempt(state, trace):
    update = planner(state)
    trace.observe("planner", update)
    state.update(update)
    update = gate_1(state)
    trace.observe("gate_1", update)
    state.update(update)


def test_exact_five_action_reply_uses_opaque_ids_and_logs_canonical_rules(suite, capsys):
    spec = _scenario(suite, "F")
    state = _assessed(suite, spec)
    trace = WorkflowTrace()
    requirements = deepcopy(state["relevant_requirements"])
    documents = deepcopy(state["documents"])
    with _mock_transport(suite, spec) as chat, collect_model_calls() as calls:
        _planning_attempt(state, trace)
    assert state["run_status"] == "VALIDATED"
    assert chat.call_count == 1 and state["retry_count"] == 0
    assert [action.grounding_ids for action in state["candidate_plan"]] == [["R006"], ["R008"], ["R019"]]
    assert [action.title for action in state["candidate_plan"]] == [
        "Upload Passport Identity Page", "Upload Turkey Entry Stamp", "Upload Declaration Control Document",
    ]
    assert "expiry date" in state["candidate_plan"][0].reason
    assert "entry-stamp page" in state["candidate_plan"][1].reason
    assert state["candidate_plan"][2].reason == "The signed declaration is required."
    assert state["relevant_requirements"] == requirements and state["documents"] == documents
    assert all(step.requires_approval for step in state["candidate_plan"]) and not state.get("approved_plan")
    assert calls[0].structured_output_valid and not calls[0].candidate_contract_valid
    assert calls[0].duplicate_candidate_references == 2
    err = capsys.readouterr().err
    assert '[PLANNER] grounded_rule_ids=["R006", "R008", "R019"]' in err
    assert '"candidate_001"' in err and "<invalid_rule_id>" not in err


@pytest.mark.parametrize("shape", ["raw_rule", "repeated_inside_row", "identical_rows", "overlapping_group"])
def test_opaque_selector_guardrails_remain_strict(suite, shape):
    state = _assessed(suite, _scenario(suite, "F"))
    payload = compact_planner_context(state)
    first, second = [row["candidate_id"] for row in payload["canonical_candidates"]][:2]
    row = {"title": "Upload passport identity page", "reason": "Provide evidence for review."}
    selectors = {
        "raw_rule": [["R006"]],
        "repeated_inside_row": [[first, first]],
        "identical_rows": [[first], [first]],
        "overlapping_group": [[first, second], [first]],
    }[shape]
    draft = PlannerDraft(actions=[{**row, "candidate_ids": keys} for keys in selectors], summary="Review the evidence.")
    with pytest.raises(PlannerError):
        _ground_draft(draft, state, payload)


def test_distinct_candidates_sharing_a_rule_keep_their_separate_tasks(suite):
    state = _assessed(suite, _scenario(suite, "F"))
    from src.agents.planner import canonical_candidates

    original = next(row for row in canonical_candidates(state) if row["candidate_id"] == "R006")
    replacement = {**original, "candidate_id": "R006:expiry_confirmation", "action_type": "REQUEST_USER_CONFIRMATION",
                   "fields": ["passport_expiry_date"], "documents": []}
    with patch("src.agents.planner.canonical_candidates", side_effect=lambda _: deepcopy([original, replacement])):
        payload = compact_planner_context(state)
        first, second = [row["candidate_id"] for row in payload["canonical_candidates"]]
        draft = PlannerDraft(actions=[
            {"candidate_ids": [first], "title": "Upload passport identity page", "reason": "Provide the page."},
            {"candidate_ids": [second], "title": "Confirm your passport expiry date", "reason": "Confirm the expiry."},
        ], summary="Review the evidence.")
        plan = _ground_draft(draft, state, payload)
        assert not validate_plan(state, plan)
    assert [step.grounding_ids for step in plan.actions] == [["R006"], ["R006:expiry_confirmation"]]


def test_gate_1_rejects_a_duplicate_even_if_a_plan_is_already_materialized(suite):
    spec = _scenario(suite, "F")
    state = _assessed(suite, spec)
    with _mock_transport(suite, spec):
        state.update(planner(state))
    steps = state["candidate_plan"]
    repeat = steps[0].model_copy(update={"step_id": "repeat", "priority": 4})
    errors = validate_plan(state, ActionPlan(actions=[*steps, repeat], summary=state["plan_summary"]))
    assert [(error.code, error.item) for error in errors] == [
        ("DUPLICATE_CANDIDATE_REFERENCE", steps[0].step_id), ("DUPLICATE_CANDIDATE_REFERENCE", "repeat"),
    ]


@pytest.mark.parametrize("scenario", ["F-repair", "F-stop"])
def test_reflection_receives_opaque_feedback_and_stops_unchanged_duplicates(suite, scenario, tmp_path):
    spec = _scenario(suite, scenario)
    state = _assessed(suite, spec)
    trace = WorkflowTrace()
    with patch("config.TWIN_RECORD_PATH", tmp_path / "synthetic-reflection.db"):
        with _mock_transport(suite, spec) as chat, collect_model_calls() as calls:
            _planning_attempt(state, trace)
            assert state["planner_diagnostics"]["schema_issue"] == "duplicate_candidate_reference"
            rejected = state["rejected_model_plan"]
            state.update(reflection(state))
            _planning_attempt(state, trace)
    feedback = json.loads(chat.call_args.kwargs["messages"][1]["content"])["repair"]
    assert feedback["affected_candidate_ids"] == ["candidate_002"]
    assert feedback["previous_model_plan"] == rejected
    assert "duplicate_candidate_reference" in feedback["validation_errors"][0]["message"]
    assert [call.task for call in calls] == ["planner", "reflection_replan"]
    assert calls[1].retry == 1 and state["retry_count"] == 1
    if scenario == "F-repair":
        assert state["run_status"] == "VALIDATED" and trace.metrics(state)["recovered_by_llm_repair"]
        assert [step.grounding_ids for step in state["candidate_plan"]] == [["R006"], ["R008"], ["R019"]]
    else:
        assert state["validation_errors"][0].code == "UNCHANGED_INVALID_PLAN"
        assert trace.metrics(state)["unchanged_invalid_replans"] == 1
        assert state["candidate_plan"] is None
    assert not state.get("approved_plan")


@pytest.mark.parametrize("scenario", ["A", "B", "C", "D", "E", "G", "H"])
def test_remaining_preparation_fixtures_drive_target_workers(suite, scenario):
    spec = _scenario(suite, scenario)
    state = _assessed(suite, spec)
    trace = WorkflowTrace()
    trace.nodes["gate_0"] += 1
    if not state["validation_errors"]:
        with _mock_transport(suite, spec), collect_model_calls() as calls:
            _planning_attempt(state, trace)
    else:
        calls = []
    # The unit evaluates observed worker outputs. Real interrupt behavior is
    # covered by the separate graph integration test below.
    approval_eligible = bool(trace.gates and trace.gates[-1].accepted)
    result = _case_outcome(state, trace, [asdict(call) for call in calls], suite, spec, approval_eligible)
    assert result["task_success"], result["assertions"]
    assert result["rag_enabled"] and result["rag_queries"] == 0
    assert result["retrieval_latency_seconds"] is None
    if scenario == "D":
        assert insurance_explanation(state) == INSURANCE_PREPARATION_ADVISORY
        assert INSURANCE_PREPARATION_ADVISORY in state["candidate_plan"][0].reason
        assert result["rule_outcomes"]["R015"] == "FAIL" and "R015" in result["blocker_rule_ids"]
        assert approval_eligible
    if scenario == "H":
        assert result["fallback_used"] and result["timeouts"] == 1 and calls[0].fallback_used
        assert result["validated_plan_origin"] == "deterministic_fallback"
    if scenario == "E":
        assert calls == [] and state["candidate_plan"] == []
    if scenario == "G":
        assert state["run_status"] == "INVALID_INPUT" and not approval_eligible


@pytest.mark.parametrize("file_format", ["txt", "pdf"])
def test_document_benchmark_uses_real_local_backend_and_sanitized_exports(suite, tmp_path, file_format):
    spec = _scenario(suite, "I")
    path = _document_fixture(suite, spec, tmp_path, file_format)
    state = _accepted_input(suite, spec, UserEvent(kind="analyze_document", file_path=str(path)))
    state.update(gate_0(state))
    with patch("config.DOCUMENT_CACHE_ENABLED", False):
        with patch("src.document_processing.processor.DOCUMENT_AI_PROVIDER", "local-only"):
            with _mock_transport(suite, spec), collect_model_calls() as calls:
                state.update(document_analyzer(state))
    state.update(rules_engine(state))
    scores = _extraction_scores(state, suite, spec)
    assert scores["extraction_reported_fields"] == scores["extraction_exact_fields"] == 5
    assert scores["document_source_accuracy"] == 1
    assert [call.task for call in calls] == ["document_classification", "document_extraction"]
    assert all(call.structured_output_valid for call in calls)
    assert next(rule for rule in state["relevant_requirements"] if rule.rule_id == "R015").passed is None
    result = _case_outcome(state, WorkflowTrace(), [asdict(call) for call in calls], suite, spec, False)
    assert result["task_success"], result["assertions"]
    json_path, csv_path = export_results(tmp_path, "synthetic", {"runs": [{"scenario_id": "I", **result}]})
    serialized = json_path.read_text(encoding="utf-8")
    assert json.loads(serialized)["runs"][0]["total_tokens"] is None
    assert not any(secret in serialized for secret in ("Synthetic Fixture Person", "SYNTH-COVER-01", "Coverage start"))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["total_tokens"] == row["retrieval_latency_seconds"] == ""


def test_model_metrics_separate_timeout_fallback_and_native_usage():
    client = Mock()
    client.chat.side_effect = [SimpleNamespace(prompt_eval_count=5, eval_count=3, eval_duration=500_000_000),
                               httpx.ReadTimeout("Synthetic provider failure")]
    with collect_model_calls() as calls, patch("src.core.llm_diagnostics.perf_counter", side_effect=[1, 3, 4, 7]):
        logged_chat(client, component="planner", model="fake")
        with pytest.raises(httpx.ReadTimeout):
            logged_chat(client, component="reflection_replan", retry=1, model="fake")
        log_planner_fallback()
    assert calls[0].total_tokens == 8 and calls[0].generation_tokens_per_second == 6
    assert calls[1].timed_out and calls[1].fallback_used and calls[1].total_tokens is None
    assert [(call.retry, call.wall_seconds) for call in calls] == [(0, 2), (1, 3)]
    assert all(call.ttft_seconds is None for call in calls)
    tally = summarize_calls([asdict(call) for call in calls])
    assert tally["model_outputs_received"] == 1 and tally["transport_failures"] == 1
    assert tally["total_tokens"] is None and tally["llm_latency_seconds"] == 5
    assert ollama_usage({"prompt_eval_count": 0, "eval_count": False})["total_tokens"] is None


def test_metric_denominators_and_peer_comparison():
    assert exact_action_score(["R006", "R019"], [["R006"], ["R008", "R019"]])["action_f1"] == pytest.approx(2 / 3)
    assert exact_action_score([], [[]])["action_f1"] == 1
    measured = retrieval_metrics(True, [{"retrieved": ["x", "a"], "relevant": ["a", "b"]}], k=2)
    assert measured["retrieval_precision_at_k"] == measured["retrieval_recall_at_k"] == 0.5
    assert retrieval_metrics(False)["retrieval_precision_at_k"] is None
    rows = [
        {"scenario_id": "A", "task_success": True, "gate_submissions": 2, "materialized_gate_submissions": 1,
         "gate_acceptances": 1, "schema_outputs_evaluated": 3, "schema_valid_outputs": 2,
         "end_to_end_latency_seconds": 1, "total_tokens": None},
        {"scenario_id": "B", "task_success": False, "gate_submissions": 1, "materialized_gate_submissions": 1,
         "gate_acceptances": 1, "schema_outputs_evaluated": 1, "schema_valid_outputs": 1,
         "end_to_end_latency_seconds": 3, "total_tokens": 7},
    ]
    report = aggregate(rows)
    assert report["plan_validation_rate"] == 1 and report["gate_attempt_acceptance_rate"] == pytest.approx(2 / 3)
    assert report["schema_validity_rate"] == 0.75 and report["task_success_rate"] == 0.5
    assert report["mean_total_tokens"] is report["p95_end_to_end_latency_seconds"] is None
    assert aggregate(rows * 10)["p95_end_to_end_latency_seconds"] == 3
    export = {"benchmark_version": "test", "metadata": {"architecture": "router", "execution_mode": "offline"}, "runs": rows}
    other = deepcopy(export)
    other["metadata"]["architecture"] = "orchestrator"
    assert compare_exports(export, other)["scenario_repeats"] == {"A": 1, "B": 1}
    other["runs"].pop()
    with pytest.raises(ValueError, match="coverage"):
        compare_exports(export, other)


@pytest.mark.parametrize("scenario", ["F", "F-repair", "F-stop", "I"])
def test_native_router_benchmark_uses_real_interrupts_and_disposable_state(suite, scenario, tmp_path):
    pytest.importorskip("langgraph.graph", exc_type=ImportError)
    spec = _scenario(suite, scenario)
    mode = "document-offline" if scenario == "I" else "offline"
    result = run_case(suite, spec, mode, tmp_path)
    assert result["task_success"], result["assertions"]
    assert result["human_approval_reached"] == (scenario in {"F", "F-repair"})
    assert result["human_review_required"] == (scenario == "F-stop")
    assert result["safety_boundary_intact"]
    assert not list(tmp_path.iterdir())
