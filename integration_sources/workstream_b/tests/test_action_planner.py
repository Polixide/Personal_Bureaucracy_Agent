"""Planner tests mock Ollama; no running model or network is required."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.action_planner import PlannerError, create_action_plan
from src.plan import ActionPlan
from src.requirement_engine import RuleResult
from src.state import State


def mock_response(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def requirement(status="failed"):
    return RuleResult(
        rule_id="R003", passed=False if status == "failed" else None,
        status=status,
        error_code="PASSPORT_VALIDITY_INSUFFICIENT" if status == "failed" else None,
        message="Passport validity insufficient" if status == "failed" else "Missing passport expiry",
    )


def action(priority=1):
    return {
        "action": "Review passport renewal options",
        "priority": priority,
        "reason": "The engine reports insufficient passport validity.",
        "related_rule_id": "R003", "deadline": None,
    }


def test_valid_structured_plan():
    content = json.dumps({"actions": [action(2), action(1)], "summary": "Review passport validity."})
    state = State(user_goal="Apply")
    before = state.model_dump()
    with patch("src.action_planner.ollama.chat", return_value=mock_response(content)) as chat:
        plan = create_action_plan(state, [requirement()])
    assert isinstance(plan, ActionPlan)
    assert [item.priority for item in plan.actions] == [1, 2]
    assert chat.call_args.kwargs["format"] == ActionPlan.model_json_schema()
    assert state.model_dump() == before


@pytest.mark.parametrize("content", [
    "Here is your plan", "{}", '{"actions": "renew passport", "summary": "Plan"}',
    json.dumps({"actions": [dict(action(), priority="urgent")], "summary": "Plan"}),
    json.dumps({"actions": [dict(action(), deadline="tomorrow")], "summary": "Plan"}),
    json.dumps({"actions": [], "summary": "Plan", "unexpected": True}),
])
def test_invalid_output(content):
    with patch("src.action_planner.ollama.chat", return_value=mock_response(content)):
        with pytest.raises(PlannerError, match="invalid ActionPlan"):
            create_action_plan(State(user_goal="Apply"), [requirement()])


def test_empty_action_list():
    with patch("src.action_planner.ollama.chat", return_value=mock_response(
        '{"actions": [], "summary": "No next action identified."}'
    )):
        assert create_action_plan(State(user_goal="Apply"), []).actions == []


@pytest.mark.parametrize("status", ["failed", "missing_information"])
def test_requirement_status_is_preserved_in_prompt(status):
    result = requirement(status)
    content = json.dumps({"actions": [], "summary": "Await review."})
    with patch("src.action_planner.ollama.chat", return_value=mock_response(content)) as chat:
        create_action_plan(State(user_goal="Apply"), [result])
    messages = chat.call_args.kwargs["messages"]
    payload = json.loads(messages[1]["content"])
    assert payload["requirement_results"] == [result.model_dump(mode="json")]
    assert payload["rules"][0]["rule_id"] == "R003"
    assert "UNKNOWN, not FAIL" in messages[0]["content"]
    assert "corrective" in messages[0]["content"]


def test_unknown_rule_reference_is_rejected():
    content = json.dumps({"actions": [dict(action(), related_rule_id="R999")], "summary": "Plan"})
    with patch("src.action_planner.ollama.chat", return_value=mock_response(content)):
        with pytest.raises(PlannerError, match="rule ID"):
            create_action_plan(State(user_goal="Apply"), [requirement()])


def test_connection_error():
    with patch("src.action_planner.ollama.chat", side_effect=ConnectionError("offline")):
        with pytest.raises(PlannerError, match="request failed"):
            create_action_plan(State(user_goal="Apply"), [])
