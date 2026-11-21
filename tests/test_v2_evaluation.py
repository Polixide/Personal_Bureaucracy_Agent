"""Deterministic scenario outcomes, independent semantic rubric and efficiency."""
from copy import deepcopy
import json
import pytest
from pydantic import ValidationError
from src.evaluation.scenarios import GOLDEN_SCENARIOS, run_scenario, evaluate_scenarios
from src.evaluation.metrics import trajectory_metrics, reliability_metrics, state_metrics
from src.evaluation.rag_eval import evaluate_rag


@pytest.mark.parametrize("name",list(GOLDEN_SCENARIOS))
def test_curated_v2_golden_outcome(name):
    result=run_scenario(name)
    assert result["passed"], result
    assert result["elapsed_seconds"]>=0


def test_reliability_distinguishes_any_success_from_consistent_success():
    result=reliability_metrics([True,False,True],k=3)
    assert result["single_run_success"]==pytest.approx(2/3)
    assert result["pass_at_k"]==1.0 and result["pass_power_k"]==0.0
    assert reliability_metrics([True]*3,k=3)["pass_power_k"]==1.0
    assert reliability_metrics([],k=3)["single_run_success"] is None
    with pytest.raises(ValueError):reliability_metrics(["yes"],k=1)


def test_trajectory_counts_real_attempts_and_excludes_approval_preview_and_replay():
    rows=[{"name":"read_eikamet_sandbox","status":"SUCCEEDED","fingerprint":"same","elapsed_seconds":0.1},
        {"name":"read_eikamet_sandbox","status":"SUCCEEDED","fingerprint":"same","elapsed_seconds":0.1},
        {"name":"create_calendar_reminder","status":"APPROVAL_REQUIRED"},
        {"name":"create_calendar_reminder","status":"SUCCEEDED","replayed":True},
        {"name":"bad","status":"FAILED","failure_category":"TOOL_SCHEMA_ERROR","fingerprint":"bad"},
        {"kind":"reflection"}]
    result=trajectory_metrics(rows,expected_tools=["read_eikamet_sandbox"])
    assert result["tool_calls"]==3 and result["duplicate_tool_calls"]==1
    assert result["schema_adherence"]==pytest.approx(2/3)
    assert result["tool_selection_precision"]==pytest.approx(2/3)
    assert result["reflection_count"]==1 and result["approval_previews"]==1


def test_rag_metrics_no_judge_default_and_strict_independent_rubric():
    sources=[{"source_id":"official","url":"https://example.test/official","content":"Supported guidance"}]
    result=evaluate_rag("why", "See https://example.test/official", sources, expected_source_ids=["official"])
    assert result["expected_source_recall"]==result["citation_coverage"]==1.0
    assert not result["judge_used"] and result["triad"] is None
    calls=[]
    def judge(payload):
        calls.append(payload)
        return {"context_relevance":4,"groundedness":3,"answer_relevance":4,"rationale":"Supported with one qualification","unsupported_claims":[]}
    result=evaluate_rag("why","Supported answer",sources,judge=judge)
    assert result["judge_used"] and result["triad"]["groundedness"]==3
    assert "schema" in calls[0] and "rubric" in calls[0]
    with pytest.raises(ValidationError):
        evaluate_rag("why","answer",sources,judge=lambda payload:{"context_relevance":5,"groundedness":4,"answer_relevance":4,"rationale":"bad"})


def test_efficiency_summary_does_not_mutate_canonical_state():
    state={"tool_outcomes":[],"repair_history":[{"attempt":1}],"assistant_diagnostics":{"llm_calls":1,"input_chars":100,"elapsed_seconds":0.5}}
    before=deepcopy(state)
    result=state_metrics(state)
    assert result["reflection_count"]==1 and result["input_characters"]==100
    assert result["chat_elapsed_seconds"]==0.5 and state==before


def test_repeat_runner_reports_all_outcomes_without_live_transports():
    result=evaluate_scenarios(repeats=3,names=["all_prerequisites_missing","calendar_reminder"])
    assert result["run_count"]==6 and result["passed_runs"]==6
    assert not result["live_model_requested"]


def test_golden_trajectory_measures_actual_tool_selection_and_arguments():
    result=run_scenario("sandbox_field_diff")
    assert result["trajectory"]["tool_selection_precision"]==1.0
    assert result["trajectory"]["argument_accuracy"]==1.0
    wrong=trajectory_metrics([{"name":"simulate_fill_eikamet_field","status":"SUCCEEDED","arguments":{"field_id":"wrong"}}],
        expected_tools=["simulate_fill_eikamet_field"],expected_arguments={"simulate_fill_eikamet_field":{"field_id":"passport_number"}})
    assert wrong["argument_accuracy"]==0.0


def test_actual_assistant_diagnostics_are_counted_by_the_evaluator():
    from src.evaluation.scenarios import run_scenario
    result=run_scenario("freeform_tool_proposal")
    assert result["passed"]
    assert result["trajectory"]["llm_calls"]==2
    assert result["trajectory"]["input_characters"]>0


def test_actual_assistant_circuit_categories_are_visible_in_state_metrics():
    from src.evaluation.metrics import state_metrics,trajectory_metrics
    diagnostics={"llm_calls":2,"llm_input_chars":3200,"input_chars":500,"failure_categories":["IDENTICAL_TOOL_FAILURE_LIMIT"]}
    result=state_metrics({"assistant_diagnostics":diagnostics})
    assert result["loop_stopped"] and result["llm_calls"]==2 and result["input_characters"]==3200
    assert trajectory_metrics([{"failure_category":"TOOL_INVOCATION_LIMIT"}])["loop_stopped"]
