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
    from src.agents.planner import planner_output_schema,compact_planner_context
    assert request["format"] == planner_output_schema(compact_planner_context(state))
    payload=json.loads(request["messages"][1]["content"])
    assert payload["canonical_candidates"]
    assert all(row["status"]=="missing_information" for row in payload["canonical_candidates"])
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
    first=action(); second=action(); second.step_id="other"; second.priority=2; second.related_rule_id="R001"
    state={"relevant_requirements":evaluate_requirements({})}
    before=[r.model_dump() for r in state["relevant_requirements"]]
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(ActionPlan(actions=[second,first],summary="x").model_dump_json())
        result=create_action_plan(state)
    assert [a.priority for a in result.actions] == [1,2]
    assert [a.related_rule_id for a in result.actions] == ["R001","R003"]
    assert [r.model_dump() for r in state["relevant_requirements"]] == before

@pytest.mark.parametrize("field,value",[("title",None),("reason",[]),("candidate_ids",["unknown_candidate"])])
def test_invalid_action_fields(field,value):
    payload={"candidate_ids":["R003"],"title":"Confirm your current location and passport entry","reason":"Location and entry information is missing"}; payload[field]=value
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(json.dumps({"actions":[payload],"summary":"x"}))
        with pytest.raises(PlannerError): create_action_plan({"relevant_requirements":evaluate_requirements({})})



def test_planner_transport_is_compact_prioritized_and_preserves_current_gaps():
    from copy import deepcopy
    from src.core.state import ResidencePermitContext
    state={"residence_permit_context":ResidencePermitContext(is_in_turkiye=True),"relevant_requirements":evaluate_requirements({"residence_permit_context":{"is_in_turkiye":True}})}
    before=deepcopy(state)
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response(ActionPlan(actions=[action()],summary="Ask about entry").model_dump_json())
        create_action_plan(state)
        payload=json.loads(client.return_value.chat.call_args.kwargs["messages"][1]["content"])
    assert "requirement_results" not in payload and "state" not in payload
    assert len(payload["canonical_candidates"])<=5
    assert payload["relevant_context"].get("is_in_turkiye") is True
    assert len(payload["allowed_rule_ids"])<=5
    assert state==before



def test_small_planner_draft_maps_to_canonical_plan_and_records_observable_timings():
    from src.agents.planner import PlannerDraft
    payload={"actions":[{"related_rule_id":"R003","title":"Confirm your current location and passport entry","reason":"Location and entry declarations are unknown","action_type":"REQUEST_USER_CONFIRMATION"}],"summary":"Confirm location"}
    state={"relevant_requirements":evaluate_requirements({})}
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)),eval_duration=1234,prompt_eval_count=100)
        plan=create_action_plan(state)
    assert isinstance(plan,ActionPlan) and plan.actions[0].requires_approval
    assert plan.actions[0].available_in_prototype
    assert plan.actions[0].step_id=="step_1" and plan.actions[0].priority==1
    assert plan._planner_diagnostics["eval_duration"]==1234
    assert plan._planner_diagnostics["json_validation_seconds"]>=0
    assert len(json.dumps(PlannerDraft.model_json_schema()))<len(json.dumps(ActionPlan.model_json_schema()))



def test_planner_timeout_is_environment_configurable():
    import os,subprocess,sys
    environment={**os.environ,"OLLAMA_TIMEOUT":"12.5","PLANNER_TIMEOUT":"45","PLANNER_MAX_ACTIONS":"2"}
    result=subprocess.run([sys.executable,"-c","from config import OLLAMA_TIMEOUT,PLANNER_MAX_ACTIONS;print(OLLAMA_TIMEOUT,PLANNER_MAX_ACTIONS)"],env=environment,capture_output=True,text=True,check=True,timeout=15)
    assert result.stdout.strip()=="12.5 2"



def test_llm_diagnostics_forward_request_and_response_without_sensitive_data(capsys):
    from unittest.mock import Mock
    from src.core.llm_diagnostics import logged_chat
    result=object();client=Mock();client.chat.return_value=result
    request={"model":"configured-test-model","messages":[{"role":"user","content":"PRIVATE DOCUMENT CONTENT"}],"format":{"type":"object"},"think":False,"stream":False}
    with patch("src.core.llm_diagnostics.perf_counter",side_effect=[10.0,18.4]):
        assert logged_chat(client,component="planner",**request) is result
    client.chat.assert_called_once_with(**request)
    captured=capsys.readouterr()
    assert captured.out==""
    assert captured.err=="[LLM] planner model=configured-test-model started\n[LLM] planner input_chars=24\n[LLM] planner output_chars=0\n[LLM] planner model=configured-test-model success elapsed=8.4s\n"
    assert "PRIVATE" not in captured.err


@pytest.mark.parametrize("error,outcome",[(TimeoutError("PRIVATE timeout details"),"timeout"),(ConnectionError("PRIVATE connection details"),"failure")])
def test_llm_diagnostics_preserve_original_exception_without_logging_details(error,outcome,capsys):
    from unittest.mock import Mock
    from src.core.llm_diagnostics import logged_chat
    client=Mock();client.chat.side_effect=error
    with patch("src.core.llm_diagnostics.perf_counter",side_effect=[1.0,3.0]),pytest.raises(type(error)) as caught:
        logged_chat(client,component="document_extraction",model="configured-test-model",messages=[])
    assert caught.value is error
    captured=capsys.readouterr()
    assert captured.out=="" and "PRIVATE" not in captured.err
    assert f"[LLM] document_extraction model=configured-test-model {outcome} elapsed=2.0s" in captured.err


@pytest.mark.parametrize("state,component",[({},"planner"),({"run_status":"REPAIRING","retry_count":1},"reflection_replan")])
def test_planner_and_replan_calls_have_distinct_runtime_labels(state,component,capsys):
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response('{"actions":[],"summary":"No actions"}')
        plan=create_action_plan(state)
    assert plan.actions==[]
    captured=capsys.readouterr()
    assert f"[LLM] {component} model={OLLAMA_MODEL} started" in captured.err
    assert f"[LLM] {component} model={OLLAMA_MODEL} success elapsed=" in captured.err
    assert "No actions" not in captured.err
    assert captured.out==""


def test_planner_timeout_logs_fallback_activation_without_changing_result(capsys):
    import httpx
    from src.agents.planner import planner
    state={"relevant_requirements":evaluate_requirements({})}
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=httpx.ReadTimeout("PRIVATE request content")
        result=planner(state)
    assert result["plan_origin"]=="deterministic_fallback" and result["candidate_plan"] is not None
    captured=capsys.readouterr()
    assert f"[LLM] planner model={OLLAMA_MODEL} timeout elapsed=" in captured.err
    assert "[FALLBACK] deterministic planner activated" in captured.err
    assert "PRIVATE" not in captured.err and captured.out==""


def test_unwritable_diagnostics_stream_does_not_change_llm_result():
    from unittest.mock import Mock
    from src.core.llm_diagnostics import logged_chat
    stream=Mock();stream.write.side_effect=OSError("closed")
    client=Mock();expected=object();client.chat.return_value=expected
    with patch("src.core.llm_diagnostics.sys.stderr",stream):
        assert logged_chat(client,component="planner",model="configured-test-model") is expected



def test_missing_stderr_does_not_redirect_llm_diagnostics_to_stdout(capsys):
    from unittest.mock import Mock
    from src.core.llm_diagnostics import logged_chat
    client=Mock();expected=object();client.chat.return_value=expected
    with patch("src.core.llm_diagnostics.sys.stderr",None):
        assert logged_chat(client,component="planner",model="configured-test-model") is expected
    assert capsys.readouterr().out==""



@pytest.mark.parametrize("state",[{}, {"run_status":"REPAIRING","retry_count":1}])
@pytest.mark.parametrize("configured_timeout",[120.0,93.5])
def test_planner_and_reflection_use_central_ollama_timeout(state,configured_timeout):
    from config import OLLAMA_HOST
    with patch("src.agents.planner.OLLAMA_TIMEOUT",configured_timeout),patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=response('{"actions":[],"summary":"No actions"}')
        plan=create_action_plan(state)
    client.assert_called_once_with(host=OLLAMA_HOST,timeout=configured_timeout)
    assert plan._planner_diagnostics["timeout_seconds"]==configured_timeout


def test_default_shared_timeout_ignores_legacy_planner_override():
    import os,subprocess,sys
    environment=dict(os.environ)
    environment.pop("OLLAMA_TIMEOUT",None)
    environment["PLANNER_TIMEOUT"]="45"
    script="import config; import src.agents.planner as p; print(config.OLLAMA_TIMEOUT, p.OLLAMA_TIMEOUT, config.DOCUMENT_OLLAMA_TIMEOUT); assert not hasattr(config, 'PLANNER_TIMEOUT')"
    result=subprocess.run([sys.executable,"-c",script],env=environment,capture_output=True,text=True,check=True,timeout=30)
    assert result.stdout.strip()=="120.0 120.0 180.0"
