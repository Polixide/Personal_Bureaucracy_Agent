"""Offline seven-document M1 scope, evidence and canonical-state regressions."""
from copy import deepcopy
from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.types import Command
from src.core.graph import build_graph
from src.core.state import UserEvent,new_run,ActionPlan,PlanStep
from src.document_processing.processor import process_document
from src.memory.checkpointer import sqlite_checkpointer
from src.workers.verification import MILESTONE_SCOPE,milestone_checklist
from src.agents.planning_policy import workflow_stage,fallback_plan,selected_needs
from src.agents.planner import actionable_needs,PlannerUnavailable
from src.guardrails.gate_1 import validate_plan

SAMPLES={
    "passport_identity_page":"Passport identity page\nHolder name: Alex Example\nPassport number: P1234567\nNationality: Exampleland\nExpiry date: 2030-08-30",
    "turkiye_entry_stamp":"TURKIYE ENTRY STAMP\nEntry date: 2026-09-15",
    "declaration_control_document":"DECLARATION CONTROL DOCUMENT\nHolder name: Alex Example\nSigned: yes",
    "card_fee_receipt":"RESIDENCE PERMIT CARD FEE RECEIPT\nPayment date: 2026-10-14\nReceipt number: RECEIPT-123\nAmount: 1234.50 TRY\nTax ID: 99887766",
    "student_enrollment_certificate":"STUDENT CERTIFICATE\nHolder name: Alex Example\nStudent ID: S123\nInstitution: Sabanci University\nIssue date: 2026-10-14\nThe exchange student studies between September 15, 2026 - June 25, 2027 at Sabanci University",
    "health_insurance_certificate":"Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\nPolicy number: TEST-123\nCoverage start: 2026-09-01\nCoverage end: 2027-07-01",
    "ikamet_application_form":"Residence Permit Application Form\nDeclaration for Short Term Residence Permit Registration Form\nApplicant: Alex Example\nPassport number: P1234567\nRequested start date: 2026-09-15\nRequested end date: 2027-06-25",
}


def response(payload):return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))


def request(event):return new_run(event,"m1-user",scope=MILESTONE_SCOPE)


def analyze(graph,config,tmp_path,kind,text=None,payload=None):
    path=tmp_path/(kind+".txt");path.write_text(text or SAMPLES[kind],encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client,patch("src.agents.planner.create_action_plan") as planner:
        client.return_value.chat.side_effect=[response({"document_type":kind}),response(payload or {})]
        state=graph.invoke(request(UserEvent(kind="analyze_document",file_path=str(path))),config)
    planner.assert_not_called()
    return state


def confirm(graph,config,document_id,fields):
    with patch("src.agents.planner.create_action_plan") as planner:
        state=graph.invoke(request(UserEvent(kind="confirm_information",document_id=document_id,confirmed_fields=fields)),config)
    planner.assert_not_called();return state


def test_passport_profile_is_grounded_and_never_proves_entry(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"passport"}}
    state=analyze(graph,config,tmp_path,"passport_identity_page")
    info=state["documents"][0].info;c=state["residence_permit_context"]
    assert info.passport_number=="P1234567" and info.nationality=="Exampleland"
    assert info.expiry_date==date(2030,8,30) and info.holder_name=="Alex Example"
    assert c.passport_id_page_present is True and c.passport_expiry_date==info.expiry_date
    assert c.nationality=="Exampleland"
    assert c.entered_with_passport is c.latest_entry_date is c.legal_stay_expiry_date is c.is_in_turkiye is None
    assert info.source_pages["passport_number"]==1 and "P1234567" in info.source_snippets["passport_number"]


def test_entry_date_is_observation_until_explicitly_confirmed(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"entry"}}
    state=analyze(graph,config,tmp_path,"turkiye_entry_stamp")
    d=state["documents"][0];raw=deepcopy(d.info)
    assert d.info.latest_entry_date==date(2026,9,15)
    c=state["residence_permit_context"]
    assert c.entry_stamp_present is c.turkey_entry_page_present is True
    assert c.latest_entry_date is c.entered_with_passport is None
    assert next(r for r in milestone_checklist(state) if r["document_type"]=="turkiye_entry_stamp")["status"]=="needs confirmation"
    state=confirm(graph,config,d.doc_id,{"latest_entry_date":"2026-09-15"})
    assert state["residence_permit_context"].latest_entry_date==date(2026,9,15)
    assert state["verification_records"][d.doc_id]["latest_entry_date"].origin=="user_confirmed"
    assert state["documents"][0].info==raw


def test_ambiguous_entry_dates_require_user_resolution(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"ambiguous-entry"}}
    state=analyze(graph,config,tmp_path,"turkiye_entry_stamp",SAMPLES["turkiye_entry_stamp"]+"\nEntry date: 2026-10-01")
    doc=state["documents"][0]
    assert doc.info.latest_entry_date is None and "latest_entry_date" in doc.info.ambiguous_fields
    assert next(r for r in milestone_checklist(state) if r["document_type"]=="turkiye_entry_stamp")["status"]=="ambiguous"
    state=confirm(graph,config,doc.doc_id,{"latest_entry_date":"2026-10-01"})
    assert state["residence_permit_context"].latest_entry_date==date(2026,10,1)
    assert next(r for r in milestone_checklist(state) if r["document_type"]=="turkiye_entry_stamp")["status"]=="complete"


def test_declaration_signature_is_never_invented_or_visually_inferred(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"declaration"}}
    state=analyze(graph,config,tmp_path,"declaration_control_document","DECLARATION CONTROL DOCUMENT\nSignature: ______",{"signed":True})
    doc=state["documents"][0]
    assert doc.info.signed is None and state["residence_permit_context"].declaration_control_document_present is True
    assert state["residence_permit_context"].declaration_control_document_signed is None
    state=confirm(graph,config,doc.doc_id,{"signed":True})
    assert state["residence_permit_context"].declaration_control_document_signed is True
    assert next(r for r in state["relevant_requirements"] if r.rule_id=="R019").passed is True


def test_receipt_preserves_reference_and_amount_without_using_tax_id(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"receipt"}}
    state=analyze(graph,config,tmp_path,"card_fee_receipt")
    info=state["documents"][0].info;c=state["residence_permit_context"]
    assert info.reference_number=="RECEIPT-123" and info.amount=="1234.50 TRY"
    assert info.payment_date==date(2026,10,14) and c.fee_receipt_present is True
    assert c.receipt_reference_number=="RECEIPT-123" and "99887766" not in str(c.receipt_reference_number)
    assert next(r for r in state["relevant_requirements"] if r.rule_id=="R010").passed is True


def test_tax_identifier_alone_is_not_payment_evidence(tmp_path):
    path=tmp_path/"tax.txt";path.write_text("Turkish Tax ID certificate\nTax ID: 99887766",encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value=response({"document_type":"card_fee_receipt"})
        info=process_document(path)
    assert info.document_type=="tax_id_certificate"
    assert info.tax_identification_number is None


def test_legal_expiry_is_user_declared_not_inferred_from_nationality(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"legal-stay"}}
    state=analyze(graph,config,tmp_path,"passport_identity_page")
    state=confirm(graph,config,"__declarations__",{"latest_entry_date":"2026-09-15","legal_stay_expiry_date":"2026-12-14"})
    assert state["residence_permit_context"].legal_stay_expiry_date==date(2026,12,14)
    assert state["verification_records"]["__declarations__"]["legal_stay_expiry_date"].origin=="user_confirmed"
    assert "legal_stay_expiry_date" not in state["verification_records"].get("__derived__",{})


def test_missing_checklist_drives_preparation_plan_and_retains_all_rules(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"missing-documents"}}
    state=confirm(graph,config,"__declarations__",{"is_in_turkiye":True})
    assert len(state["relevant_requirements"])==24
    assert len(milestone_checklist(state))==7 and {r["status"] for r in milestone_checklist(state)}=={"missing"}
    assert workflow_stage(state)=="DOCUMENT_PREPARATION"
    needs=actionable_needs(state);by_id={n["rule_id"]:n for n in needs}
    assert by_id["R022"]["category"]=="future" and by_id["R017"]["category"]=="outside_scope"
    selected=selected_needs(needs)
    assert not {"R017","R021","R022","R023","R027","R028","R029"} & {n["rule_id"] for n in selected}
    plan=fallback_plan(state,needs)
    assert any("Passport identity page" in a.action for a in plan.actions)
    assert any("entry-stamp" in a.action for a in plan.actions)
    assert not validate_plan(state,plan)


def test_complete_seven_documents_persist_and_progress_without_faking_later_steps(tmp_path):
    db=tmp_path/"seven.db";config={"configurable":{"thread_id":"seven"}}
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        for kind in SAMPLES:state=analyze(graph,config,tmp_path,kind)
        stamp=next(d for d in state["documents"] if d.doc_type=="turkiye_entry_stamp")
        form=next(d for d in state["documents"] if d.doc_type=="ikamet_application_form")
        insurance=next(d for d in state["documents"] if d.doc_type=="health_insurance")
        state=confirm(graph,config,stamp.doc_id,{"latest_entry_date":"2026-09-15"})
        state=confirm(graph,config,form.doc_id,{"application_form_signed":True,"online_application_completed":True})
        state=confirm(graph,config,insurance.doc_id,{"insurance_valid_for_ikamet":True})
        state=confirm(graph,config,"__declarations__",{"insurance_evidence_type":"private_health_insurance","is_in_turkiye":True,"entered_with_passport":True,"application_date":"2026-10-14","legal_stay_expiry_date":"2026-12-14"})
        checklist=milestone_checklist(state)
        assert all(r["status"]=="complete" for r in checklist),checklist
        assert workflow_stage(state)=="APPLICATION_PREPARATION"
        c=state["residence_permit_context"]
        assert c.all_core_documents_present is c.gi_number_present is c.iro_preverification_confirmed is c.submitted_to_iro is None
        before=deepcopy(state["relevant_requirements"])
        with patch("src.agents.planner.create_action_plan",side_effect=PlannerUnavailable("offline")) as model:
            state=graph.invoke(request(UserEvent(kind="generate_plan")),config)
        model.assert_not_called()
        assert state["run_status"]=="VALIDATED" and state["__interrupt__"]
        assert state["plan_origin"]=="milestone_completion" and state["candidate_plan"]==[]
        assert state["retry_count"]==0 and not state["validation_errors"]
        assert "preparation is complete" in state["plan_summary"]
        from streamlit_app import unknown_requirement_groups
        groups=unknown_requirement_groups(state)
        assert not groups["immediate"] and any(groups.values())
        assert state["relevant_requirements"]==before
        state=graph.invoke(Command(resume=True),config)
        assert state["run_status"]=="APPROVED" and state["residence_permit_context"].submitted_to_iro is None
    with sqlite_checkpointer(db) as saver:
        state=build_graph(saver).get_state(config).values
        assert len(state["documents"])==7 and all(r["status"]=="complete" for r in milestone_checklist(state))
        assert state["residence_permit_context"].latest_entry_date==date(2026,9,15)
        assert state["planning_scope"]==MILESTONE_SCOPE



def test_outside_scope_failure_stays_failed_and_cannot_authorize_official_actions(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"outside-failure"}}
    initial=request(UserEvent(kind="reevaluate"))
    initial["residence_permit_context"]={"address_proof_type":"airbnb","gi_number_present":False}
    state=graph.invoke(initial,config)
    results={r.rule_id:r for r in state["relevant_requirements"]}
    assert results["R017"].status==results["R022"].status=="failed"
    needs=actionable_needs(state)
    assert not {"R017","R022"} & {n["rule_id"] for n in selected_needs(needs)}
    plan=fallback_plan(state,needs)
    assert not validate_plan(state,plan)
    official=ActionPlan(actions=[PlanStep(step_id="submit",priority=1,action="Submit application documents",reason="Attempt official submission",related_rule_id="R023",action_type="iro_submission")],summary="Attempt next step")
    assert {e.code for e in validate_plan(state,official)} & {"MISSING_PREREQUISITE","PREMATURE_SUBMISSION"}


def test_entry_confirmation_already_given_is_reused_without_a_loop(tmp_path):
    from src.workers.verification import document_review
    graph=build_graph();config={"configurable":{"thread_id":"entry-no-loop"}}
    confirm(graph,config,"__declarations__",{"latest_entry_date":"2026-09-15"})
    state=analyze(graph,config,tmp_path,"turkiye_entry_stamp")
    doc=state["documents"][0];review=document_review(state,doc)
    assert next(f for f in review["fields"] if f["name"]=="latest_entry_date")["status"]=="user_confirmed"
    assert state["residence_permit_context"].latest_entry_date==date(2026,9,15)


def test_product_ui_helpers_select_limited_scope_without_mutating_the_twin():
    from streamlit_app import ui_run
    state={"documents":[],"extracted_facts":{}}
    before=deepcopy(state)
    rows=milestone_checklist(state)
    assert len(rows)==7 and state==before
    assert ui_run(UserEvent(kind="generate_plan"),"ui-user")["planning_scope"]==MILESTONE_SCOPE



@pytest.mark.parametrize("text,expected", [
    ("TURKIYE\nGIRIS\n15.09.2026","2026-09-15"),
    ("TURKEY\nENTRY\n15.09.2026\n01.10.2026",None),
])
def test_stamp_country_entry_marker_and_dates_are_still_confirmation_only(tmp_path,text,expected):
    graph=build_graph();config={"configurable":{"thread_id":"printed-stamp"}}
    state=analyze(graph,config,tmp_path,"turkiye_entry_stamp",text)
    info=state["documents"][0].info
    assert info.document_type=="turkiye_entry_stamp" and info.entry_stamp_present is True
    assert (info.latest_entry_date.isoformat() if info.latest_entry_date else None)==expected
    assert state["residence_permit_context"].latest_entry_date is None
    assert state["residence_permit_context"].entered_with_passport is None



def test_gate_1_rejects_later_stage_document_requests_in_current_scope(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"scope-guard"}}
    state=confirm(graph,config,"__declarations__",{"is_in_turkiye":True})
    step=PlanStep(step_id="gi",priority=1,action="Upload GI evidence now",reason="GI is unknown",related_rule_id="R022",request_kind="REQUEST_DOCUMENT",requested_documents=["GI evidence"],action_type="information")
    assert "OUTSIDE_MILESTONE_SCOPE" in {e.code for e in validate_plan(state,ActionPlan(actions=[step],summary="Request GI"))}



def test_receipt_heading_and_tax_id_without_transaction_require_confirmation(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"blank-receipt"}}
    state=analyze(graph,config,tmp_path,"card_fee_receipt","RESIDENCE PERMIT CARD FEE RECEIPT\nTax ID: 99887766",{"receipt_present":True,"reference_number":"99887766"})
    assert state["documents"][0].info.receipt_present is None
    assert state["documents"][0].info.reference_number is None
    assert state["residence_permit_context"].fee_receipt_present is None
