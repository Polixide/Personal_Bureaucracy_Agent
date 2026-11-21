"""Minimal V2 tool targets through the existing canonical planner and Gate 1."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.types import Command

from src.agents.planner import (PlannerDraft, _ground_draft, compact_planner_context, create_action_plan,
    PlannerError, PlannerUnavailable, actionable_needs)
from src.agents.planning_policy import canonical_tool_candidates, fallback_plan
from src.core.procedure import advance_procedure_if_ready
from src.core.state import ActionPlan, PlanStep, ProcedureState, UserEvent, ValidationIssue, new_run
from src.guardrails.gate_1 import validate_plan
from src.tools.registry import execute_tool
from src.tools.schemas import ToolCall
from src.workers.verification import MILESTONE_SCOPE
from test_procedure import ready_bundle

NOW = datetime(2026,11,19,12,tzinfo=timezone.utc)
TITLES = {"simulate_fill_eikamet_field":"Fill the accepted passport number in the local form",
    "search_official_sources":"Review official guidance for the preparation step",
    "prepare_iro_email":"Prepare an IRO review email draft",
    "create_calendar_reminder":"Review the exact saved reminder proposal"}


@pytest.fixture
def prepared(ready_bundle):
    state=deepcopy(ready_bundle[2])
    state.update(advance_procedure_if_ready(state,recorded_at=NOW))
    state["event"]=UserEvent(kind="generate_plan",text="Include official guidance and an IRO support email")
    return state


def draft_for(payload, ids=None):
    candidates=[c for c in payload["canonical_candidates"] if c.get("tool_call") or c.get("issue")=="supported_tool_action"]
    if ids is not None:candidates=[c for c in candidates if (c["candidate_id"] in ids or c.get("canonical_ref") in ids)]
    return PlannerDraft(actions=[{"candidate_ids":[c["candidate_id"]],"title":TITLES.get(c.get("tool_call",{}).get("name"),c["message"]),"reason":c["message"]} for c in candidates],
                        summary="Proposed preparation tools; no official action executed")


def grounded_plan(state):
    context=compact_planner_context(state)
    return _ground_draft(draft_for(context),state,context)


def test_v2_candidates_include_safe_fill_lookup_and_draft(prepared):
    before=deepcopy(prepared)
    first=canonical_tool_candidates(prepared);second=canonical_tool_candidates(prepared)
    assert first==second and prepared==before
    assert {c["tool_call"]["name"] for c in first}=={"simulate_fill_eikamet_field","search_official_sources","prepare_iro_email"}
    assert first[0]["tool_call"]["arguments"]=={"field_id":"passport_number","value":None}
    assert all(c["tool_call"]["call_id"].startswith("plan_") for c in first)
    assert not any(c["tool_call"]["name"] in {"create_calendar_reminder","send_iro_email"} for c in first)


@pytest.mark.parametrize("procedure", [None,ProcedureState(),ProcedureState(current_stage="PAYMENT")])
def test_legacy_collection_and_later_stage_candidate_behavior_is_unchanged(prepared,procedure):
    state={**prepared,"procedure":procedure}
    assert canonical_tool_candidates(state)==[]
    assert not any(c.get("tool_call") for c in compact_planner_context(state)["canonical_candidates"])


def test_llm_presentation_copies_exact_metadata_and_gate_accepts(prepared):
    payload=compact_planner_context(prepared)
    assert payload["workflow_stage"]=="EIKAMET_PREPARATION"
    draft=draft_for(payload)
    response=SimpleNamespace(message=SimpleNamespace(content=draft.model_dump_json()))
    with patch("src.agents.planner.Client") as client,patch("src.tools.registry.execute_tool") as execute:
        client.return_value.chat.return_value=response
        plan=create_action_plan(prepared)
    execute.assert_not_called()
    expected={c.get("canonical_ref",c["candidate_id"]):c["tool_call"] for c in payload["canonical_candidates"] if c.get("tool_call")}
    assert all(action.tool_call==expected[action.grounding_ids[0]] for action in plan.actions)
    assert all(action.action_type=="PROVIDE_GUIDANCE" and action.related_rule_id is None for action in plan.actions)
    assert not validate_plan(prepared,plan)


@pytest.mark.parametrize("change", ["name","arguments","call_id","approved","missing_grounding"])
def test_gate_rejects_unbound_or_tampered_tool_metadata(prepared,change):
    plan=grounded_plan(prepared)
    action=plan.actions[0]
    if change=="name":action.tool_call["name"]="unregistered_tool"
    elif change=="arguments":action.tool_call["arguments"]["field_id"]="nationality"
    elif change=="call_id":action.tool_call["call_id"]="different-call"
    elif change=="approved":action.tool_call["arguments"]["approved"]=True
    else:action.grounding_ids=[]
    codes={error.code for error in validate_plan(prepared,plan)}
    assert codes & {"UNSUPPORTED_GROUNDING","TOOL_SCHEMA_ERROR"}


def test_gate_rechecks_stage_and_current_accepted_fields(prepared):
    plan=grounded_plan(prepared)
    state={**prepared,"procedure":ProcedureState()}
    assert "UNSUPPORTED_GROUNDING" in {e.code for e in validate_plan(state,plan)}
    no_passport={**prepared,"documents":[d for d in prepared["documents"] if d.doc_type!="passport_copy"]}
    assert "UNSUPPORTED_GROUNDING" in {e.code for e in validate_plan(no_passport,plan)}


def test_tool_target_does_not_bypass_existing_titles_approval_or_official_claim_checks(prepared):
    plan=grounded_plan(prepared)
    plan.actions[0].title=plan.actions[0].action="information"
    plan.actions[0].requires_approval=False
    plan.summary="Your residence permit is officially approved"
    codes={error.code for error in validate_plan(prepared,plan)}
    assert {"VAGUE_TITLE","APPROVAL_REQUIRED","UNSUPPORTED_APPROVAL_CLAIM"}<=codes


def test_different_tool_targets_cannot_be_grouped_or_silently_dropped(prepared):
    payload=compact_planner_context(prepared)
    ids=[c["candidate_id"] for c in payload["canonical_candidates"] if c.get("tool_call")]
    grouped=PlannerDraft(actions=[{"candidate_ids":ids[:2],"title":"Review supported preparation actions","reason":"Prepare safely"}],summary="Proposed tools")
    with pytest.raises(PlannerError):_ground_draft(grouped,prepared,payload)
    with pytest.raises(PlannerError):_ground_draft(PlannerDraft(actions=[],summary="No actions"),prepared,payload)


def test_reflection_preserves_valid_tool_actions_and_exact_targets(prepared):
    original=grounded_plan(prepared)
    bad=original.actions[0].model_copy(deep=True,update={"title":"information","action":"information"})
    state={**prepared,"candidate_plan":[bad,*original.actions[1:]],"run_status":"REPAIRING",
        "repair_feedback":[ValidationIssue(code="VAGUE_TITLE",message="Use a specific title",source="gate_1",item=bad.step_id)],
        "repair_valid_actions":[a.model_copy(deep=True) for a in original.actions[1:]],"repair_rejected_action_ids":[bad.step_id]}
    payload=compact_planner_context(state)
    replacement=draft_for(payload,[bad.grounding_ids[0]])
    repaired=_ground_draft(replacement,state,payload)
    assert repaired.actions[1:]==original.actions[1:]
    assert repaired.actions[0].tool_call==original.actions[0].tool_call
    assert not validate_plan(state,repaired)
    with pytest.raises(PlannerError):_ground_draft(PlannerDraft(actions=[],summary="Dropped action"),state,payload)


def test_transport_fallback_uses_same_tool_targets(prepared):
    plan=fallback_plan(prepared,actionable_needs(prepared))
    assert plan.actions and all(a.tool_call for a in plan.actions)
    assert not validate_plan(prepared,plan)


def test_existing_calendar_proposal_is_preserved_without_inventing_arguments(prepared):
    call=ToolCall(name="create_calendar_reminder",arguments={"title":"Synthetic reminder","start_at":"2027-02-03T10:00:00+03:00"})
    observation=execute_tool(prepared,call)
    state={**prepared,"tool_proposals":[{"proposal_id":call.call_id,"tool_call":call.model_dump(mode="json"),"status":"APPROVAL_REQUIRED","preview":observation.diff}]}
    candidate=canonical_tool_candidates(state)[0]
    assert candidate["tool_call"]["call_id"]==call.call_id
    assert candidate["tool_call"]["arguments"]["start_at"]=="2027-02-03T10:00:00+03:00"
    plan=grounded_plan(state)
    assert plan.actions[0].tool_call==candidate["tool_call"] and not validate_plan(state,plan)


def test_plan_approval_does_not_execute_but_explicit_tool_event_does(ready_bundle):
    graph,config,state=ready_bundle
    graph.update_state(config,advance_procedure_if_ready(state,recorded_at=NOW),as_node="respond")
    def response(**kwargs):
        payload=json.loads(kwargs["messages"][1]["content"])
        return SimpleNamespace(message=SimpleNamespace(content=draft_for(payload).model_dump_json()))
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=response
        result=graph.invoke(new_run(UserEvent(kind="generate_plan",text="Include official guidance and an IRO support email"),"m1-user",scope=MILESTONE_SCOPE),config)
    assert result["__interrupt__"] and not result.get("sandbox_form")
    result=graph.invoke(Command(resume=True),config)
    assert result["approved_plan"].actions[0].tool_call and not result.get("sandbox_form") and not result.get("email_drafts")
    call=result["approved_plan"].actions[0].tool_call
    result=graph.invoke(new_run(UserEvent(kind="run_tool",tool_call=call),"m1-user",scope=MILESTONE_SCOPE),config)
    assert result["sandbox_form"]["fields"]["passport_number"]["value"]=="P1234567"
    assert result["procedure"].current_stage=="EIKAMET_PREPARATION"


def test_rule_independent_tool_target_cannot_assert_unknown_legal_failure(prepared):
    plan=grounded_plan(prepared)
    plan.actions[0].reason="Your passport is invalid"
    assert "UNSUPPORTED_TOOL_CLAIM" in {error.code for error in validate_plan(prepared,plan)}
