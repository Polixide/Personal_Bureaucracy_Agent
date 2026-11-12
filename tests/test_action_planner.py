import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from config import OLLAMA_MODEL
from src.agents.planner import create_action_plan, PlannerError
from src.core.state import ActionPlan, PlanStep
from src.workers.rules_engine import evaluate_requirements


@pytest.mark.parametrize("override,expected", [(None, "qwen3:4b"), ("qwen3:8b", "qwen3:8b")])
def test_model_config_default_and_environment_override(override, expected):
    import os
    import subprocess
    import sys
    from config import PROJECT_ROOT
    environment = dict(os.environ)
    environment.pop("OLLAMA_MODEL", None)
    if override is not None:
        environment["OLLAMA_MODEL"] = override
    result = subprocess.run([sys.executable, "-c", "import config; print(config.OLLAMA_MODEL)"],
                            cwd=PROJECT_ROOT, env=environment, capture_output=True,
                            text=True, check=True, timeout=30)
    assert result.stdout.strip() == expected


def action(**changes):
    return PlanStep(**dict(step_id="ask",action="Provide entry details",priority=1,reason="Entry details are unknown",related_rule_id="R003",**changes))

def response(payload):
    return SimpleNamespace(message=SimpleNamespace(content=payload))

def test_structured_plan_and_unknown_prompt():
    state={"relevant_requirements":evaluate_requirements({})}
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(ActionPlan(actions=[action()],summary="Ask for information").model_dump_json())
        plan=create_action_plan(state)
        request=client.return_value.chat.call_args.kwargs
    assert isinstance(plan,ActionPlan)
    assert request["model"] == OLLAMA_MODEL
    assert request["format"] == ActionPlan.model_json_schema()
    payload=json.loads(request["messages"][1]["content"])
    assert payload["requirement_results"][2]["status"] == "missing_information"
    assert "UNKNOWN" in request["messages"][0]["content"]

@pytest.mark.parametrize("content",["not json","{}",'{"actions":[],"summary":"x","extra":true}', '{"actions":"bad","summary":"x"}'])
def test_invalid_output(content):
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(content)
        with pytest.raises(PlannerError): create_action_plan({})

def test_unknown_rule_reference():
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(ActionPlan(actions=[action()],summary="x").model_dump_json())
        with pytest.raises(PlannerError,match="Rule ID"): create_action_plan({})

def test_connection_error():
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=ConnectionError("offline")
        with pytest.raises(PlannerError): create_action_plan({})

def test_empty_plan():
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response('{"actions":[],"summary":"No actions"}')
        assert create_action_plan({}).actions == []


def test_sorting_and_no_state_mutation():
    first=action(); second=action(); second.step_id="other"; second.priority=2
    state={"relevant_requirements":evaluate_requirements({})}
    before=[r.model_dump() for r in state["relevant_requirements"]]
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(ActionPlan(actions=[second,first],summary="x").model_dump_json())
        result=create_action_plan(state)
    assert [a.priority for a in result.actions] == [1,2]
    assert [r.model_dump() for r in state["relevant_requirements"]] == before

@pytest.mark.parametrize("field,value",[("priority","urgent"),("due_date","tomorrow"),("requires_approval",None)])
def test_invalid_action_fields(field,value):
    payload=action().model_dump(mode="json"); payload[field]=value
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(json.dumps({"actions":[payload],"summary":"x"}))
        with pytest.raises(PlannerError): create_action_plan({})
