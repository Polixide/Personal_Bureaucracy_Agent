"""Grounded candidate metadata, model wording and bounded targeted repairs."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest
from src.core.state import PlanStep,ActionPlan,ResidencePermitContext,ALLOWED_ACTION_TYPES
from src.agents.planner import canonical_candidates,compact_planner_context,create_action_plan,PlannerError
from src.guardrails.gate_1 import validate_plan
from src.workers.rules_engine import evaluate_requirements
from src.workers.verification import MILESTONE_SCOPE
from src.memory.reflection import reflection


def state_for(ids=("R006","R008","R019"),context=None):
    state={"planning_scope":MILESTONE_SCOPE,"residence_permit_context":ResidencePermitContext(**(context or {}))}
    state["relevant_requirements"]=[r for r in evaluate_requirements(state) if r.rule_id in ids]
    return state


def model_plan(state,actions):
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=SimpleNamespace(message=SimpleNamespace(content=json.dumps({"actions":actions,"summary":"Proposed next steps for review"})))
        plan=create_action_plan(state)
        payload=json.loads(client.return_value.chat.call_args.kwargs["messages"][1]["content"])
    return plan,payload


@pytest.mark.parametrize("title",["information","prepare","information/prepare","process","validate"])
def test_vague_titles_rejected(title):
    state=state_for(("R006",))
    plan,_=model_plan(state,[{"title":title,"action_type":"REQUEST_DOCUMENT","reason":"Passport expiry evidence is missing","related_rule_id":"R006"}])
    assert "VAGUE_TITLE" in {e.code for e in validate_plan(state,plan)}


@pytest.mark.parametrize("rid,target",[("R006","passport_identity_page"),("R008","turkiye_entry_stamp"),("R019","declaration_control_document")])
def test_document_targets_grounded_before_model(rid,target):
    candidate=next(c for c in canonical_candidates(state_for()) if rid in c["rule_ids"])
    assert candidate["action_type"]=="REQUEST_DOCUMENT" and candidate["documents"]==[target]


def test_model_reorders_missing_documents_without_changing_targets():
    state=state_for();before=deepcopy(state)
    actions=[{"title":"Upload the Declaration Control Document","action_type":"REQUEST_DOCUMENT","reason":"R019 needs the declaration document","related_rule_id":"R019"},
             {"title":"Upload your passport identity page","action_type":"REQUEST_DOCUMENT","reason":"Passport expiry is required for R006","related_rule_id":"R006"},
             {"title":"Upload the latest Turkiye entry-stamp page","action_type":"REQUEST_DOCUMENT","reason":"Entry-page evidence is needed for R008","related_rule_id":"R008"}]
    plan,payload=model_plan(state,actions)
    assert [a.related_rule_id for a in plan.actions]==["R019","R006","R008"]
    assert [a.priority for a in plan.actions]==[1,2,3]
    assert all(a.title==a.action and a.requires_approval and a.grounding_ids for a in plan.actions)
    assert not validate_plan(state,plan) and state==before
    assert set(payload["allowed_rule_ids"])=={"R006","R008","R019"}


def test_insurance_failure_is_correction_with_grounded_dates():
    state=state_for(("R015",),{"insurance_valid_for_ikamet":True,"insurance_end_date":"2027-05-01","program_end_date":"2027-06-25","requested_permit_end_date":"2027-09-01"})
    candidate=canonical_candidates(state)[0]
    assert candidate["action_type"]=="CORRECT_INFORMATION" and candidate["target_document"]=="health_insurance"
    plan,payload=model_plan(state,[{"title":"Review and extend your health-insurance coverage","action_type":"CORRECT_INFORMATION","reason":"R015 coverage ends before both accepted end-date targets","related_rule_id":"R015"}])
    assert plan.actions[0].requested_fields==["insurance_end_date"]
    assert not validate_plan(state,plan)
    assert payload["relevant_context"]["insurance_end_date"]=="2027-05-01"


def test_compact_context_excludes_passed_later_and_raw_data():
    state=state_for(tuple(f"R{i:03}" for i in range(1,30)),{"exchange_student":True,"planned_stay_months":9,"permit_type":"short_term","is_in_turkiye":True})
    state.pop("planning_scope")
    state["extracted_facts"]={"sensitive":"PRIVATE SOURCE "*400}
    state["repair_history"]=[{"text":"PRIVATE HISTORY"}]*30
    payload=compact_planner_context(state)
    assert not {"R001","R002","R021","R022","R023","R027","R028","R029"}&set(payload["allowed_rule_ids"])
    encoded=json.dumps(payload)
    assert "PRIVATE" not in encoded and "extracted_facts" not in payload
    assert len(encoded)<len(json.dumps(state,default=lambda x:x.model_dump(mode="json")))*0.4


def test_cannot_invent_rule_type_or_canonical_fields():
    state=state_for(("R006",))
    for changes in ({"action_type":"PROVIDE_GUIDANCE"},{"requested_fields":["invented_expiry"]},{"related_rule_id":"R999"}):
        action={"candidate_ids":["R006"],"title":"Upload passport page","action_type":"REQUEST_DOCUMENT","reason":"Need expiry","related_rule_id":"R006",**changes}
        plan,_=model_plan(state,[action])
        assert plan.actions[0].related_rule_id=="R006" and plan.actions[0].action_type=="REQUEST_DOCUMENT"
        assert "invented_expiry" not in plan.actions[0].requested_fields
        assert not validate_plan(state,plan)


def test_reflection_repairs_only_rejected_action_and_preserves_valid_step():
    state=state_for()
    plan,_=model_plan(state,[{"title":"Upload your passport identity page","action_type":"REQUEST_DOCUMENT","reason":"R006 needs expiry evidence","related_rule_id":"R006"},{"title":"information/prepare","action_type":"REQUEST_DOCUMENT","reason":"R019 needs declaration evidence","related_rule_id":"R019"}])
    state.update(candidate_plan=plan.actions,validation_errors=validate_plan(state,plan),retry_count=0)
    state.update(reflection(state));valid=deepcopy(state["repair_valid_actions"][0])
    repaired,payload=model_plan(state,[{"repair_step_id":"step_2","title":"Upload the Declaration Control Document","action_type":"REQUEST_DOCUMENT","reason":"R019 needs the declaration document for review","related_rule_id":"R019"}])
    assert repaired.actions[0]==valid and repaired.actions[1].step_id=="step_2"
    assert not validate_plan(state,repaired)
    assert payload["repair"]["validation_errors"][0]["code"]=="VAGUE_TITLE"
    assert payload["repair"]["locked_actions"][0]["step_id"]=="step_1"
    assert set(payload["allowed_action_types"])==set(ALLOWED_ACTION_TYPES)
    assert state["candidate_plan"][1].title=="information/prepare"
    with pytest.raises(PlannerError):model_plan(state,[{"repair_step_id":"step_1","title":"Change already valid action","action_type":"REQUEST_DOCUMENT","reason":"Changed valid action","related_rule_id":"R006"}])


def test_gate_validates_type_rule_and_all_field_references():
    state=state_for(("R006",))
    step=PlanStep(step_id="bad",title="Review passport expiry",action_type="CORRECT_INFORMATION",reason="Review expiry",priority=1,related_rule_id="R006",requested_fields=["made_up_field"])
    assert "UNKNOWN_FIELD" in {e.code for e in validate_plan(state,ActionPlan(actions=[step],summary="Review evidence"))}
    step.action_type="information/prepare"
    with pytest.warns(UserWarning,match="serializer warnings"):
        assert "SCHEMA_ERROR" in {e.code for e in validate_plan(state,ActionPlan(actions=[step],summary="Review evidence"))}
    legacy=PlanStep(step_id="missing",action="Review passport expiry",priority=1,reason="Review evidence")
    assert "UNTARGETED_REQUEST" in {e.code for e in validate_plan(state,ActionPlan(actions=[legacy],summary="Review evidence"))}



def test_llm_can_group_compatible_candidates_without_losing_rules():
    state=state_for()
    candidates=canonical_candidates(state)
    keys=[c["candidate_id"] for c in candidates if c["rule_ids"][0] in {"R006","R008"}]
    plan,_=model_plan(state,[{"title":"Upload your passport identity and latest entry-stamp pages","action_type":"REQUEST_DOCUMENT","reason":"R006 needs expiry evidence and R008 needs entry-page evidence","related_rule_id":"R006","candidate_ids":keys}])
    assert plan.actions[0].related_rule_ids==["R006","R008"]
    assert set(plan.actions[0].requested_documents)=={"passport_identity_page","turkiye_entry_stamp"}
    assert not validate_plan(state,plan)


def test_repair_cannot_silently_drop_rejected_requirement():
    state=state_for(("R019",))
    plan,_=model_plan(state,[{"title":"information","action_type":"REQUEST_DOCUMENT","reason":"Need declaration","related_rule_id":"R019"}])
    state.update(candidate_plan=plan.actions,validation_errors=validate_plan(state,plan),retry_count=0)
    state.update(reflection(state))
    with pytest.raises(PlannerError,match="not silently remove"):
        model_plan(state,[])



def test_initial_model_ids_do_not_control_canonical_step_ids():
    state=state_for(("R006",))
    plan,_=model_plan(state,[{"title":"Upload your passport identity page","action_type":"REQUEST_DOCUMENT","reason":"Need expiry evidence","related_rule_id":"R006","repair_step_id":"model-chosen-id"}])
    assert plan.actions[0].step_id=="step_1" and not validate_plan(state,plan)


def test_failed_repair_generation_does_not_erase_locked_valid_actions():
    from src.agents.planner import planner
    from src.guardrails.gate_1 import gate_1
    state=state_for()
    plan,_=model_plan(state,[{"title":"Upload your passport identity page","action_type":"REQUEST_DOCUMENT","reason":"Need expiry evidence","related_rule_id":"R006"},{"title":"information","action_type":"REQUEST_DOCUMENT","reason":"Need declaration evidence","related_rule_id":"R019"}])
    state.update(candidate_plan=plan.actions,validation_errors=validate_plan(state,plan),retry_count=0)
    state.update(reflection(state));saved=deepcopy(state["repair_valid_actions"])
    with patch("src.agents.planner.create_action_plan",side_effect=PlannerError("Bad repair fields")):
        state.update(planner(state))
    state.update(gate_1(state));state.update(reflection(state))
    assert state["repair_valid_actions"]==saved
    assert state["candidate_plan"][0]==saved[0]



def test_failed_insurance_requires_a_corrective_user_title_not_another_generic_upload():
    state=state_for(("R015",),{"insurance_valid_for_ikamet":True,"insurance_end_date":"2027-05-01","program_end_date":"2027-06-25"})
    plan,_=model_plan(state,[{"title":"Upload your health insurance policy","action_type":"CORRECT_INFORMATION","reason":"Coverage is short","related_rule_id":"R015"}])
    assert "NON_CORRECTIVE_TITLE" in {e.code for e in validate_plan(state,plan)}



def test_fallback_uses_same_titled_action_categories_and_still_requires_gate_1():
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import fallback_plan
    state=state_for()
    plan=fallback_plan(state,actionable_needs(state))
    assert all(a.title and a.action_type in ALLOWED_ACTION_TYPES and a.requires_approval for a in plan.actions)
    assert not validate_plan(state,plan)



def draft_action(rid, title="Upload your passport identity page"):
    return {"title":title,"action_type":"REQUEST_DOCUMENT","reason":"Provide evidence for review","related_rule_id":rid}


def draft_response(actions):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps({"actions":actions,"summary":"Review proposed next steps"})))


def graph_request(state):
    from src.core.state import new_run,UserEvent
    return {**new_run(UserEvent(kind="generate_plan"),"offline",scope=MILESTONE_SCOPE),**state}


def test_latest_rejected_model_output_and_exact_feedback_reach_each_repair(capsys):
    from src.core.graph import build_graph
    from src.core.llm_diagnostics import plan_hash
    first=draft_response([draft_action("R020")]);second=draft_response([draft_action("R022")])
    valid=draft_response([draft_action("R006")])
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=[first,second,valid]
        state=build_graph().invoke(graph_request(state_for(("R006",))),{"configurable":{"thread_id":"latest-feedback"}})
        inputs=[json.loads(c.kwargs["messages"][1]["content"]) for c in client.return_value.chat.call_args_list]
    assert state["run_status"]=="VALIDATED" and state["__interrupt__"] and state["retry_count"]==2
    for i,rid in enumerate(("R020","R022"),1):
        repair=inputs[i]["repair"]
        assert repair["validation_errors"]==state["repair_history"][i-1]["errors"]
        assert repair["validation_errors"][0]["rule_id"]==rid
        assert repair["validation_errors"][0]["code"]=="RULE_NOT_ACTIONABLE"
        assert repair["previous_model_plan"]["actions"][0]["related_rule_id"]==rid
        assert repair["previous_plan_hash"]==plan_hash(repair["previous_model_plan"])
        assert {k:v for k,v in inputs[i].items() if k!="repair"}==inputs[0]
    assert inputs[1]["repair"]["previous_plan_hash"]!=inputs[2]["repair"]["previous_plan_hash"]
    captured=capsys.readouterr()
    assert '[PLANNER] allowed_rule_ids=["R006"]' in captured.err
    assert '[PLANNER] returned_rule_ids=["R020"]' in captured.err
    assert '[GATE1] rejected_rule_id=R020 rejection_reason=RULE_NOT_ACTIONABLE' in captured.err
    assert '[REFLECTION] previous_plan_hash=' in captured.err and '[REFLECTION] new_plan_hash=' in captured.err
    assert "Provide evidence for review" not in captured.err


def test_unchanged_invalid_structured_output_stops_without_three_redundant_repairs():
    from src.core.graph import build_graph
    bad=draft_response([draft_action("R020")])
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=bad
        state=build_graph().invoke(graph_request(state_for(("R006",))),{"configurable":{"thread_id":"unchanged-output"}})
    assert client.return_value.chat.call_count==2
    assert state["retry_count"]==1 and state["run_status"]=="HUMAN_REVIEW_REQUIRED"
    assert state["validation_errors"][0].code=="UNCHANGED_INVALID_PLAN"
    assert not state.get("approved_plan") and not state.get("__interrupt__")


def test_reflection_partitions_using_latest_gate_feedback_not_previous_round():
    from src.core.state import ValidationIssue
    plan,_=model_plan(state_for(),[draft_action("R006"),draft_action("R019","Upload the Declaration Control Document")])
    old=ValidationIssue(code="VAGUE_TITLE",message="Old error",item="step_1",source="gate_1")
    latest=ValidationIssue(code="UNKNOWN_AS_FAILURE",message="Latest error",item="step_2",source="gate_1")
    state={"candidate_plan":plan.actions,"validation_errors":[latest],"repair_feedback":[old],"retry_count":1}
    before=deepcopy(state);update=reflection(state)
    assert update["repair_feedback"]==[latest] and update["repair_rejected_action_ids"]==["step_2"]
    assert update["repair_valid_actions"]==[plan.actions[0]] and state==before


@pytest.mark.parametrize("problem",["missing_documents","missing_assessment","current_unknown","current_failure","conflict","wrong_scope"])
def test_completion_requires_real_checklist_current_assessment_and_no_conflicts(problem):
    from src.agents.planner import milestone_preparation_complete
    state=state_for(tuple(f"R{i:03}" for i in range(1,30)))
    state["relevant_requirements"]=[r.model_copy(update={"passed":True,"status":"passed","missing_fields":[]}) for r in state["relevant_requirements"]]
    if problem=="missing_assessment":state["relevant_requirements"]=None
    if problem in {"current_unknown","current_failure"}:
        rule=next(r for r in state["relevant_requirements"] if r.rule_id=="R001")
        rule.passed=False if problem=="current_failure" else None
        rule.status="failed" if problem=="current_failure" else "missing_information"
        rule.missing_fields=["exchange_student"]
    if problem=="conflict":state["verification_conflicts"]=["holder_name"]
    if problem=="wrong_scope":state.pop("planning_scope")
    if problem=="missing_documents":
        assert not milestone_preparation_complete(state)
    else:
        from src.workers.verification import milestone_checklist
        rows=[{**row,"status":"complete"} for row in milestone_checklist(state)]
        with patch("src.workers.verification.milestone_checklist",return_value=rows):
            assert not milestone_preparation_complete(state)


def test_repair_context_and_last_rejected_plan_survive_sqlite_recreation(tmp_path):
    from src.memory.checkpointer import sqlite_checkpointer
    from src.core.graph import build_graph
    from src.core.state import UserEvent,new_run
    config={"configurable":{"thread_id":"repair-persist"}}
    with sqlite_checkpointer(tmp_path/"repair.db") as saver,patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=[draft_response([draft_action("R020")]),draft_response([draft_action("R006")])]
        result=build_graph(saver).invoke(graph_request(state_for(("R006",))),config)
        assert result["__interrupt__"]
    with sqlite_checkpointer(tmp_path/"repair.db") as saver:
        restored=build_graph(saver).get_state(config).values
    assert restored["planning_context_snapshot"]["allowed_rule_ids"]==["R006"]
    assert restored["repair_history"][0]["rejected_model_plan"]["actions"][0]["related_rule_id"]=="R020"
    assert restored["last_model_plan_hash"]==result["last_model_plan_hash"]
    reset=new_run(UserEvent(kind="generate_plan"),"offline",scope=MILESTONE_SCOPE)
    assert reset["planning_context_snapshot"]=={} and reset["rejected_model_plan"] is None and reset["last_model_plan_hash"] is None



@pytest.mark.parametrize("scenario",["A","B"])
def test_successful_missing_document_and_insurance_scenarios_reach_human_approval(scenario):
    from src.core.graph import build_graph
    ids=("R006","R008","R019") if scenario=="A" else ("R006","R008","R015")
    context={} if scenario=="A" else {"insurance_valid_for_ikamet":True,"insurance_end_date":"2027-05-01","program_end_date":"2027-06-25"}
    actions=[draft_action("R006"),draft_action("R008","Upload your latest Turkiye entry-stamp page")]
    actions.append(draft_action("R019","Upload the Declaration Control Document") if scenario=="A" else {"title":"Review and extend your health-insurance coverage","action_type":"CORRECT_INFORMATION","reason":"R015 coverage ends before the accepted end-date target","related_rule_id":"R015"})
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=draft_response(actions)
        result=build_graph().invoke(graph_request(state_for(ids,context)),{"configurable":{"thread_id":"scenario-"+scenario}})
    assert result["run_status"]=="VALIDATED" and result["__interrupt__"]
    assert client.return_value.chat.call_count==1 and result["plan_origin"]=="ollama"
    assert result["retry_count"]==0 and not result["validation_errors"]
    assert {a.related_rule_id for a in result["candidate_plan"]}==set(ids)
    assert all(a.requires_approval for a in result["candidate_plan"])


def test_plan_hash_includes_summary_so_a_corrected_summary_is_not_unchanged():
    from src.core.llm_diagnostics import plan_hash
    actions=[draft_action("R006")]
    assert plan_hash({"actions":actions,"summary":"Unsupported claim"})!=plan_hash({"actions":actions,"summary":"Request information"})



def test_gate_rejected_summary_reaches_reflection_and_valid_actions_are_preserved():
    from src.core.graph import build_graph
    initial={"actions":[draft_action("R006")],"summary":"Your residence permit is officially approved"}
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(initial))),draft_response([])]
        state=build_graph().invoke(graph_request(state_for(("R006",))),{"configurable":{"thread_id":"summary-repair"}})
        repaired_input=json.loads(client.return_value.chat.call_args.kwargs["messages"][1]["content"])
    assert state["run_status"]=="VALIDATED" and state["retry_count"]==1 and state["__interrupt__"]
    assert repaired_input["repair"]["previous_model_plan"]["summary"]==initial["summary"]
    assert repaired_input["repair"]["validation_errors"][0]["code"]=="UNSUPPORTED_APPROVAL_CLAIM"
    assert len(state["candidate_plan"])==1
    locked=repaired_input["repair"]["locked_actions"][0]
    assert all(state["candidate_plan"][0].model_dump(mode="json")[key]==value for key,value in locked.items())



def test_live_schema_exposes_only_selection_and_wording_and_constrains_selectors():
    from src.agents.planner import planner_output_schema
    payload=compact_planner_context(state_for())
    schema=planner_output_schema(payload)
    fields=schema["$defs"]["DraftStep"]["properties"]
    assert set(fields)=={"candidate_ids","title","reason"}
    assert fields["candidate_ids"]["items"]["enum"]==[c["candidate_id"] for c in payload["canonical_candidates"]]
    assert fields["candidate_ids"]["minItems"]==1


def test_scenario_a_reconstructs_metadata_even_with_fabricated_model_structure():
    state=state_for();before=deepcopy(state)
    actions=[{"candidate_ids":[rid],"title":title,"reason":"Provide the named missing evidence for review",
              "related_rule_id":"<invalid_rule_id>","action_type":"official_application",
              "requested_fields":["made_up_field"],"requested_documents":["invented_document"],
              "target":"invented_target","target_document":"invented_document","fields":["invented_field"],"requirement_status":"passed","requires_approval":False,
              "step_id":"untrusted_step","repair_step_id":"untrusted_repair"}
             for rid,title in (("R008","Upload your latest Turkiye entry-stamp page"),("R019","Upload the Declaration Control Document"),("R006","Upload your passport identity page"))]
    plan,_=model_plan(state,actions)
    assert [a.related_rule_id for a in plan.actions]==["R008","R019","R006"]
    assert {a.related_rule_id for a in plan.actions}=={"R006","R008","R019"}
    assert [a.requested_documents for a in plan.actions]==[["turkiye_entry_stamp"],["declaration_control_document"],["passport_identity_page"]]
    assert all(a.action_type=="REQUEST_DOCUMENT" and a.requires_approval and a.execution_type=="information" for a in plan.actions)
    assert state==before and not validate_plan(state,plan)


def test_scenario_b_uses_canonical_insurance_correction_despite_model_mutation():
    from src.core.graph import build_graph
    state=state_for(("R006","R008","R015"),{"insurance_valid_for_ikamet":True,"insurance_end_date":"2027-05-01","program_end_date":"2027-06-25"})
    actions=[{"candidate_ids":[rid],"title":title,"reason":"Address the supplied requirement with the named evidence","related_rule_id":"R999","action_type":"REQUEST_DOCUMENT"}
             for rid,title in (("R015","Review and extend your health-insurance coverage"),("R006","Upload your passport identity page"),("R008","Upload your latest Turkiye entry-stamp page"))]
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=draft_response(actions)
        result=build_graph().invoke(graph_request(state),{"configurable":{"thread_id":"immutable-insurance"}})
    assert result["run_status"]=="VALIDATED" and result["__interrupt__"] and result["plan_origin"]=="ollama"
    insurance=next(a for a in result["candidate_plan"] if a.related_rule_id=="R015")
    assert insurance.action_type=="CORRECT_INFORMATION" and insurance.requested_fields==["insurance_end_date"]
    assert client.return_value.chat.call_count==1


def test_repair_after_pre_grounding_failure_ignores_invented_repair_step_id():
    from src.core.graph import build_graph
    first=draft_response([draft_action("<invalid_rule_id>")])
    repair=draft_response([{"candidate_ids":[rid],"title":title,"reason":"Review the missing document evidence",
                           "related_rule_id":"R999","repair_step_id":"step_999"}
                          for rid,title in (("R006","Upload your passport identity page"),("R008","Upload your latest Turkiye entry-stamp page"),("R019","Upload the Declaration Control Document"))])
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=[first,repair]
        result=build_graph().invoke(graph_request(state_for()),{"configurable":{"thread_id":"unmaterialized-plan-repair"}})
        payload=json.loads(client.return_value.chat.call_args.kwargs["messages"][1]["content"])
    assert not payload["repair"]["rejected_action_ids"] and not payload["repair"]["failed_candidate"]
    assert result["run_status"]=="VALIDATED" and result["__interrupt__"] and result["retry_count"]==1
    assert {a.related_rule_id for a in result["candidate_plan"]}=={"R006","R008","R019"}
    assert [a.step_id for a in result["candidate_plan"]]==["step_1","step_2","step_3"]


def test_schema_diagnostics_are_safe_and_exact_issue_reaches_reflection(capsys):
    from src.core.graph import build_graph
    initial=draft_response([{"candidate_ids":["R006"],"reason":"PRIVATE DOCUMENT CONTENT"}])
    repair=draft_response([{"candidate_ids":["R006"],"title":"Upload your passport identity page","reason":"Passport evidence is missing","related_rule_id":"<invalid_rule_id>"}])
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.side_effect=[initial,repair]
        result=build_graph().invoke(graph_request(state_for(("R006",))),{"configurable":{"thread_id":"safe-schema-feedback"}})
        payload=json.loads(client.return_value.chat.call_args.kwargs["messages"][1]["content"])
    error=payload["repair"]["validation_errors"][0]
    assert error["code"]=="PLANNER_SCHEMA_ERROR" and error["message"]=="Planner schema issue: missing_required_title"
    assert payload["repair"]["previous_model_plan"]["actions"][0]["reason"]=="PRIVATE DOCUMENT CONTENT"
    assert result["run_status"]=="VALIDATED" and result["__interrupt__"]
    captured=capsys.readouterr()
    assert "[GATE1] schema_issue=missing_required_title" in captured.err
    assert "PRIVATE" not in captured.err and captured.out==""


@pytest.mark.parametrize("changes,issue",[({"candidate_ids":["R999"]},"unknown_or_duplicate_candidate_reference"),({"title":None},"invalid_title"),({"reason":{}},"invalid_reason"),({"candidate_ids":"R006"},"invalid_candidate_reference"),({"candidate_ids":["R006","R006"]},"unknown_or_duplicate_candidate_reference")])
def test_invalid_presentation_or_unbound_candidate_is_rejected_with_safe_reason(changes,issue):
    action={"candidate_ids":["R006"],"title":"Upload passport identity page","reason":"Need passport evidence",**changes}
    with pytest.raises(PlannerError) as error:model_plan(state_for(("R006",)),[action])
    assert error.value.issue.code=="PLANNER_SCHEMA_ERROR"
    assert error.value.issue.message=="Planner schema issue: "+issue


def test_reflection_ignores_rule_and_target_mutation_and_preserves_locked_actions():
    state=state_for()
    plan,_=model_plan(state,[{"candidate_ids":["R006"],"title":"Upload your passport identity page","reason":"Passport evidence is missing"},{"candidate_ids":["R019"],"title":"information","reason":"Declaration evidence is missing"}])
    state.update(candidate_plan=plan.actions,validation_errors=validate_plan(state,plan),retry_count=0)
    state.update(reflection(state));locked=deepcopy(state["repair_valid_actions"][0])
    repaired,_=model_plan(state,[{"candidate_ids":["R019"],"title":"Upload the Declaration Control Document","reason":"Declaration evidence is needed for review","related_rule_id":"R999","requested_fields":["invented"],"repair_step_id":"step_999","action_type":"official_application"}])
    assert repaired.actions[0]==locked and repaired.actions[1].related_rule_id=="R019"
    assert repaired.actions[1].step_id=="step_2" and repaired.actions[1].requested_documents==["declaration_control_document"]
    assert not validate_plan(state,repaired)


def test_incompatible_groups_and_invented_rule_claims_are_not_accepted():
    state=state_for(("R006","R015"),{"insurance_valid_for_ikamet":True,"insurance_end_date":"2027-05-01","program_end_date":"2027-06-25"})
    for action in ({"candidate_ids":["R006","R015"],"title":"Review the documents","reason":"Review the evidence"},
                   {"candidate_ids":["R006"],"title":"Upload a criminal record under R999","reason":"New invented requirement"}):
        with pytest.raises(PlannerError):model_plan(state,[action])



def test_reflection_can_reprioritize_rejected_actions_without_owning_identifiers():
    state=state_for(("R006","R019"))
    initial,_=model_plan(state,[{"candidate_ids":[rid],"title":"information","reason":"Evidence is needed for review"} for rid in ("R006","R019")])
    state.update(candidate_plan=initial.actions,validation_errors=validate_plan(state,initial),retry_count=0)
    state.update(reflection(state))
    repaired,_=model_plan(state,[{"candidate_ids":["R019"],"title":"Upload the Declaration Control Document","reason":"Declaration evidence is needed"},{"candidate_ids":["R006"],"title":"Upload your passport identity page","reason":"Passport evidence is needed"}])
    assert [a.related_rule_id for a in repaired.actions]==["R019","R006"]
    assert [a.priority for a in repaired.actions]==[1,2]
    assert [a.step_id for a in repaired.actions]==["step_2","step_1"]
    assert not validate_plan(state,repaired)
