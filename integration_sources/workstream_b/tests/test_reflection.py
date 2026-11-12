"""Mock model output and check every correction goes through validation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.action_planner import PlannerError
from src.plan import ActionPlan, PlanAction
from src.plan_validator import validate_plan
from src.reflection import plan_with_reflection, reflect_plan
from src.requirement_engine import RuleResult
from src.state import State


def fixtures():
    state = State(user_goal="Apply", iteration_count=4)
    results = [RuleResult(rule_id="R003", passed=False, status="failed",
                          error_code="PASSPORT_VALIDITY_INSUFFICIENT", message="Insufficient validity")]
    bad = ActionPlan(actions=[], summary="No actions")
    good = ActionPlan(actions=[PlanAction(
        action="Review passport renewal options", priority=1,
        reason="Resolve insufficient passport validity", related_rule_id="R003",
    )], summary="Address the passport requirement")
    return state, results, bad, good


def response(plan):
    return SimpleNamespace(message=SimpleNamespace(content=plan.model_dump_json()))


def test_valid_plan_needs_no_reflection():
    state, results, _, good = fixtures()
    with patch("src.reflection.action_planner.create_action_plan", return_value=good), \
         patch("src.reflection.ollama.chat") as chat:
        outcome = plan_with_reflection(state, results)
    assert outcome.status == "VALID"
    assert outcome.attempts == 0
    assert state.iteration_count == 4
    chat.assert_not_called()


def test_valid_correction_is_revalidated():
    state, results, bad, good = fixtures()
    with patch("src.reflection.action_planner.create_action_plan", return_value=bad), \
         patch("src.reflection.ollama.chat", return_value=response(good)) as chat:
        outcome = plan_with_reflection(state, results)
    assert outcome.status == "VALID"
    assert outcome.validation.valid
    assert outcome.attempts == 1
    assert state.iteration_count == 5
    assert not outcome.history[0].validation.valid
    assert outcome.history[1].validation.valid
    assert "validation" in chat.call_args.kwargs["messages"][1]["content"]


def test_invalid_corrections_rejected_and_circuit_breaker():
    state, results, bad, _ = fixtures()
    with patch("src.reflection.action_planner.create_action_plan", return_value=bad), \
         patch("src.reflection.ollama.chat", return_value=response(bad)) as chat, \
         patch("src.reflection.validate_plan", wraps=validate_plan) as validator:
        outcome = plan_with_reflection(state, results)
    assert outcome.status == "HUMAN_REVIEW_REQUIRED"
    assert not outcome.validation.valid
    assert outcome.attempts == chat.call_count == 3
    assert validator.call_count == 4
    assert state.iteration_count == 7


def test_unknown_rule_correction_is_revalidated():
    state, results, bad, good = fixtures()
    good.actions[0].related_rule_id = "R999"
    with patch("src.reflection.action_planner.create_action_plan", return_value=bad), \
         patch("src.reflection.ollama.chat", return_value=response(good)):
        outcome = plan_with_reflection(state, results)
    assert outcome.status == "HUMAN_REVIEW_REQUIRED"
    assert any(error.code == "UNKNOWN_RULE_ID" for error in outcome.validation.errors)


def test_malformed_correction_is_bounded():
    state, results, bad, _ = fixtures()
    malformed = SimpleNamespace(message=SimpleNamespace(content="not JSON"))
    with patch("src.reflection.action_planner.create_action_plan", return_value=bad), \
         patch("src.reflection.ollama.chat", return_value=malformed) as chat:
        outcome = plan_with_reflection(state, results)
    assert outcome.status == "HUMAN_REVIEW_REQUIRED"
    assert outcome.attempts == chat.call_count == 3
    assert any(error.code == "REFLECTION_ERROR" for error in outcome.validation.errors)


def test_initial_planner_error():
    state, results, _, _ = fixtures()
    with patch("src.reflection.action_planner.create_action_plan", side_effect=PlannerError("offline")):
        outcome = plan_with_reflection(state, results)
    assert outcome.status == "HUMAN_REVIEW_REQUIRED"
    assert outcome.plan is None
    assert outcome.attempts == 0


def test_reflection_returns_candidate_without_self_validation():
    state, results, bad, _ = fixtures()
    validation = validate_plan(state, results, bad)
    with patch("src.reflection.ollama.chat", return_value=response(bad)), \
         patch("src.reflection.validate_plan") as validator:
        candidate = reflect_plan(state, results, bad, validation)
    assert candidate == bad
    validator.assert_not_called()


def test_reflection_connection_error():
    state, results, bad, _ = fixtures()
    with patch("src.reflection.ollama.chat", side_effect=ConnectionError("offline")):
        with pytest.raises(PlannerError, match="reflection request failed"):
            reflect_plan(state, results, bad, validate_plan(state, results, bad))


def test_prompt_includes_explicit_errors_and_repair_instructions():
    state, results, bad, good = fixtures()
    validation = validate_plan(state, results, bad)
    with patch("src.reflection.ollama.chat", return_value=response(good)) as chat:
        reflect_plan(state, results, bad, validation)
    messages = chat.call_args.kwargs["messages"]
    for error in validation.errors:
        assert f"{error.code}: {error.message}" in messages[1]["content"]
    prompt = messages[0]["content"]
    assert "UNSUPPORTED_DEADLINE: set the action deadline to null" in prompt
    assert "PREMATURE_SUBMISSION: remove or replace" in prompt
    assert "Do not return the original plan unchanged" in prompt
    assert 'Invalid: "Submit documents"' in prompt
    assert 'Corrected: "Re-check document completeness after blockers are resolved."' in prompt
