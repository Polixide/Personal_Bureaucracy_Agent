"""Offline staged workflow, accepted-input provenance and SQLite continuity."""
from copy import deepcopy
from datetime import date
import json
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.types import Command

from src.core.graph import build_graph
from src.core.state import ActionPlan, PlanStep, UserEvent, new_run
from src.memory.checkpointer import sqlite_checkpointer
from src.workers.verification import document_review


STUDENT_TEXT = "Student Enrollment Certificate\nHolder name: Alex Example\nStudent ID: S123\nInstitution: Sabanc\u0131 University\nIssue date: 2026-11-01"
STUDENT = {"holder_name":"Alex Example", "student_id":"S123", "institution":"Sabanc\u0131 University", "issue_date":"2026-11-01"}


def response(payload):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))


def analyze(graph, config, path, payload=None, text=STUDENT_TEXT, kind="student_enrollment_certificate", context=None):
    path.write_text(text,encoding="utf-8")
    request=new_run(UserEvent(kind="analyze_document",file_path=str(path)),"review-user")
    if context is not None:
        request["residence_permit_context"]=context
    with patch("src.document_processing.processor.Client") as client, patch("src.agents.planner.create_action_plan",side_effect=AssertionError("Analysis must not plan")) as planner:
        client.return_value.chat.side_effect=[response({"document_type":kind})]+([] if kind=="unknown" else [response(STUDENT if payload is None else payload)])
        result=graph.invoke(request,config)
    planner.assert_not_called()
    assert client.return_value.chat.call_count in ((1,) if kind=="unknown" else (1,2))
    return result


def confirm(graph,config,document_id,fields):
    with patch("src.agents.planner.create_action_plan",side_effect=AssertionError("Confirmation must not plan")) as planner:
        result=graph.invoke(new_run(UserEvent(kind="confirm_information",document_id=document_id,confirmed_fields=fields),"review-user"),config)
    planner.assert_not_called()
    return result


def test_analysis_only_promotes_only_source_grounded_fields_without_planning(tmp_path):
    graph=build_graph(); config={"configurable":{"thread_id":"analysis"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    assert result["candidate_plan"] is None and result["retry_count"]==0
    assert result["run_status"]=="DOCUMENT_ANALYZED" and not result.get("__interrupt__")
    assert len(result["relevant_requirements"])==24
    document=result["documents"][0]
    entry=result["verification_records"][document.doc_id]["student_id"]
    assert entry.origin=="source_grounded" and entry.value=="S123"
    assert entry.source_page==1 and entry.source_snippet and entry.confirmed_by is None
    assert result["residence_permit_context"].student_certificate_present is True
    assert result["residence_permit_context"].all_core_documents_present is None
    assert result["approval_decision"] is None


def test_unconfirmed_field_remains_unknown_to_requirement_engine(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"unknown"}}
    result=analyze(graph,config,tmp_path/"student.txt",{**STUDENT,"student_id":None,"ambiguous_fields":["student_id"]})
    assert result["residence_permit_context"].student_id is None
    assert result["residence_permit_context"].student_certificate_present is None
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R014").status=="missing_information"
    assert result["run_status"]=="INFORMATION_NEEDS_CONFIRMATION"


def test_confirmation_is_distinct_and_persisted_without_changing_observations(tmp_path):
    db=tmp_path/"verification.db";config={"configurable":{"thread_id":"review"}}
    with sqlite_checkpointer(db) as saver:
        result=analyze(build_graph(saver),config,tmp_path/"student.txt",{**STUDENT,"student_id":None,"ambiguous_fields":["student_id"]})
        document=result["documents"][0]
        raw=document.info.model_dump(mode="json"); facts=deepcopy(result["extracted_facts"])
    with sqlite_checkpointer(db) as saver:
        result=confirm(build_graph(saver),config,document.doc_id,{"student_id":"CONFIRMED-777"})
    with sqlite_checkpointer(db) as saver:
        result=build_graph(saver).get_state(config).values
    entry=result["verification_records"][document.doc_id]["student_id"]
    assert entry.origin=="user_confirmed" and entry.confirmed_by=="review-user"
    assert entry.value=="CONFIRMED-777" and entry.source_page is None and entry.source_snippet is None
    assert result["documents"][0].info.model_dump(mode="json")==raw
    assert result["extracted_facts"]==facts
    assert result["residence_permit_context"].student_id=="CONFIRMED-777"
    assert result["residence_permit_context"].student_certificate_present is True
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R014").passed is True
    assert result["candidate_plan"] is None


def test_explicit_plan_event_calls_planner_and_still_requires_plan_approval(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"plan"}}
    analyze(graph,config,tmp_path/"student.txt")
    plan=ActionPlan(actions=[PlanStep(step_id="dates",priority=1,action="Provide legal-stay dates",reason="Dates remain unknown",related_rule_id="R004")],summary="Gather dates")
    with patch("src.agents.planner.create_action_plan",return_value=plan) as planner:
        result=graph.invoke(new_run(UserEvent(kind="generate_plan"),"review-user"),config)
    assert planner.call_count==1 and result["__interrupt__"]
    assert result["run_status"]=="VALIDATED" and result["approval_decision"] is None
    result=graph.invoke(Command(resume=True),config)
    assert result["run_status"]=="APPROVED" and not result.get("completed_steps")


def test_insurance_end_policy_does_not_claim_automatic_legal_acceptance(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"insurance"}}
    text="Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\nPolicy number: POLICY-123\nCoverage start: 2026-11-01\nCoverage end: 2027-06-01"
    result=analyze(graph,config,tmp_path/"policy.txt",{},text,"health_insurance_certificate",{"requested_start_date":"2026-11-01","requested_end_date":"2027-06-01"})
    context=result["residence_permit_context"]
    assert context.insurance_start_date==date(2026,11,1) and context.insurance_valid_for_ikamet is None
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R015").passed is True
    result=confirm(graph,config,result["documents"][0].doc_id,{"insurance_valid_for_ikamet":True})
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R015").passed is True
    assert result["residence_permit_context"].insurance_evidence_type is None


def test_invalid_user_dates_do_not_update_context_or_raw_document(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"invalid"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    document=result["documents"][0];before=deepcopy(result["verification_records"])
    result=confirm(graph,config,document.doc_id,{"issue_date":"1310-10-20"})
    assert result["run_status"]=="INVALID_INPUT"
    assert result["verification_records"]==before
    assert result["documents"][0].info.issue_date==date(2026,11,1)


def test_unsupported_confirmation_field_is_not_a_context_write(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"bad-field"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    result=confirm(graph,config,result["documents"][0].doc_id,{"all_core_documents_present":True})
    assert result["run_status"]=="INVALID_INPUT"
    assert result["residence_permit_context"].all_core_documents_present is None


def test_reanalysis_failure_revokes_derived_inputs(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"reprocess"}}
    path=tmp_path/"student.txt"
    result=analyze(graph,config,path)
    assert result["residence_permit_context"].student_certificate_present is True
    with patch("src.workers.document_analyzer.process_document",side_effect=ValueError("Unreadable document")), patch("src.agents.planner.create_action_plan") as planner:
        result=graph.invoke(new_run(UserEvent(kind="analyze_document",file_path=str(path)),"review-user"),config)
    planner.assert_not_called()
    assert result["documents"][0].status=="rejected"
    assert result["residence_permit_context"].student_certificate_present is None
    assert not result["verification_records"]


def test_explicit_null_revokes_confirmation_without_changing_raw_source(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"withdraw"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    document=result["documents"][0]
    result=confirm(graph,config,document.doc_id,{"student_id":None})
    assert result["documents"][0].info.student_id=="S123"
    assert result["residence_permit_context"].student_id is None
    assert result["residence_permit_context"].student_certificate_present is None
    assert document_review(result,result["documents"][0])["status"]=="needs_review"


def test_document_review_and_ui_card_are_compact_pure_views(tmp_path):
    from streamlit_app import document_card,user_action
    graph=build_graph();config={"configurable":{"thread_id":"display"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    before=deepcopy(result)
    card=document_card(result,result["documents"][0])
    assert card["status"]=="complete"
    assert not {"source_evidence","source_snippets","extracted_facts","residence_permit_context"} & card.keys()
    assert all("source_evidence" not in row for row in card["fields"])
    assert result==before
    step=PlanStep(step_id="ask",priority=1,action="Provide_requested_dates",reason="Unknown dates",related_rule_id="R004")
    assert user_action(step)=="Provide requested dates"
    assert step.action=="Provide_requested_dates" and step.related_rule_id=="R004"


def test_unknown_document_cannot_be_manually_promoted_to_certificate(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"unknown-doc"}}
    result=analyze(graph,config,tmp_path/"unknown.txt",{},"ciao sono marco","unknown")
    assert result["residence_permit_context"].student_certificate_present is None
    result=confirm(graph,config,result["documents"][0].doc_id,{"holder_name":"Alex Example"})
    assert result["run_status"]=="INVALID_INPUT"
    assert not result["verification_records"]


@pytest.mark.parametrize("first,second,conflict,projected", [
    ("NORA     FICTIVA", "NORA FICTIVA", False, "NORA FICTIVA"),
    ("Nora Fictiva", "NORA FICTIVA", False, "Nora Fictiva"),
    ("  NORA\tFICTIVA  ", "nora fictiva", False, "NORA FICTIVA"),
    ("Jos\u00e9 Fictiva", "JOSE\u0301 FICTIVA", False, "Jos\u00e9 Fictiva"),
    ("NORA FICTIVA", "ALICE FICTIVA", True, None),
])
def test_holder_comparison_normalizes_only_projection_not_verified_records(first,second,conflict,projected):
    from src.core.state import Document, VerifiedInput
    from src.document_processing.schemas import StudentCertificate
    from src.workers.verification import verification_updates
    documents=[Document(doc_id=key,file_path=f"{key}.txt",status="analyzed",doc_type="student_certificate",info=StudentCertificate()) for key in ("a","b")]
    records={key:{"holder_name":VerifiedInput(value=value,origin="user_confirmed",document_id=key,field_name="holder_name",confirmed_by="review-user",trace_id="original")} for key,value in zip(("a","b"),(first,second))}
    state={"documents":documents,"verification_records":records,"event":UserEvent(kind="generate_plan"),"trace_id":"next","user_id":"review-user"}
    before=deepcopy(state)
    updates=verification_updates(state)
    assert ("holder_name" in updates["verification_conflicts"]) is conflict
    assert updates["residence_permit_context"].holder_name==projected
    assert updates["verification_records"]==records
    assert state==before


@pytest.mark.parametrize("institution", ["Sabanc\u0131 University","Sabanci University"," SABANCI   UNIVERSITY "])
def test_scope_aliases_accept_current_inputs_and_legacy_typed_checkpoints(institution):
    from src.core.state import ResidencePermitContext
    from src.guardrails.gate_0 import gate_0
    fields=dict(institution=institution,applicant_type="Incoming Exchange",permit_type="short-term",application_kind=" FIRST ")
    context=ResidencePermitContext(**fields)
    assert context.institution=="Sabanc\u0131 University"
    assert (context.applicant_type,context.permit_type,context.application_kind)==("incoming_exchange","short_term","first")
    for value in (fields,ResidencePermitContext.model_construct(**fields)):
        result=gate_0({"event":UserEvent(kind="generate_plan"),"residence_permit_context":value})
        assert not result.get("validation_errors")
        assert result["residence_permit_context"]==context


@pytest.mark.parametrize("institution", ["Other University","Sabanci University International"])
def test_scope_aliases_do_not_allow_fuzzy_institution_matches(institution):
    from src.guardrails.gate_0 import gate_0
    result=gate_0({"event":UserEvent(kind="generate_plan"),"residence_permit_context":{"institution":institution}})
    assert result["run_status"]=="INVALID_INPUT"


def test_confirm_then_plan_reuses_assessment_through_gate_1_and_sqlite_restart(tmp_path):
    from streamlit_app import requirement_summary,status_label
    config={"configurable":{"thread_id":"confirm-plan"}}
    db=tmp_path/"confirm-plan.db"
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        result=analyze(graph,config,tmp_path/"student.txt")
        result=confirm(graph,config,result["documents"][0].doc_id,{"institution":"Sabanci University"})
        requirements=deepcopy(result["relevant_requirements"])
        blockers=deepcopy(result["blockers"])
        assert len(requirements)==24
        assert status_label(result["run_status"]) == "Document analyzed"
        assert result["residence_permit_context"].institution=="Sabanc\u0131 University"
        assert result["verification_records"][result["documents"][0].doc_id]["institution"].value=="Sabanci University"
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        request=new_run(UserEvent(kind="generate_plan"),"review-user")
        assert "relevant_requirements" not in request and "blockers" not in request
        plan=ActionPlan(actions=[PlanStep(step_id="dates",priority=1,action="Provide legal-stay dates",reason="Dates remain unknown",related_rule_id="R004")],summary="Gather dates")
        def propose(state):
            assert state["relevant_requirements"]==requirements
            assert state["blockers"]==blockers
            return plan
        with patch("src.agents.planner.create_action_plan",side_effect=propose) as planner, patch("src.workers.rules_engine.evaluate_requirements",side_effect=AssertionError("Unchanged Digital Twin must reuse assessment")):
            result=graph.invoke(request,config)
        assert planner.call_count==1 and result["run_status"]=="VALIDATED"
        assert result["__interrupt__"] and not result["validation_errors"]
        assert result["relevant_requirements"]==requirements
        assert requirement_summary(result["relevant_requirements"])==requirement_summary(requirements)


def test_invalid_generate_plan_keeps_last_valid_assessment_and_blockers(tmp_path):
    from streamlit_app import requirement_summary
    graph=build_graph();config={"configurable":{"thread_id":"invalid-plan"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    result=confirm(graph,config,result["documents"][0].doc_id,{"institution":"Other University"})
    requirements=deepcopy(result["relevant_requirements"]);blockers=deepcopy(result["blockers"])
    summary=requirement_summary(requirements)
    assert sum(summary.values())==24
    with patch("src.agents.planner.create_action_plan") as planner:
        result=graph.invoke(new_run(UserEvent(kind="generate_plan"),"review-user"),config)
    planner.assert_not_called()
    assert result["run_status"]=="INVALID_INPUT"
    assert result["relevant_requirements"]==requirements and result["blockers"]==blockers
    assert requirement_summary(result["relevant_requirements"])==summary


def test_analysis_and_confirmation_still_invalidate_requirement_assessment():
    for event in (UserEvent(kind="analyze_document",file_path="student.txt"),UserEvent(kind="confirm_information",document_id="student",confirmed_fields={"holder_name":"Nora Fictiva"})):
        assert new_run(event,"review-user")["relevant_requirements"] is None


def test_ui_separates_failed_and_missing_blocking_requirements_without_mutation():
    from streamlit_app import blocking_requirement_groups
    requirements=[{"rule_id":"R001","status":"failed","severity":"blocking"},
                  {"rule_id":"R002","status":"missing_information","severity":"blocking"},
                  {"rule_id":"R003","status":"passed","severity":"blocking"},
                  {"rule_id":"R004","status":"missing_information","severity":"advisory"}]
    before=deepcopy(requirements)
    groups=blocking_requirement_groups(requirements)
    assert [r["rule_id"] for r in groups["failed"]]==["R001"]
    assert [r["rule_id"] for r in groups["missing_information"]]==["R002"]
    assert requirements==before


ACADEMIC_TEXT = """STUDENT CERTIFICATE
Holder name: Alex Example
Student ID: S123
Institution: Sabanci University
Issue date: 2026-10-14
The student named above will be studying as an exchange student within the Exchange program
between September 15, 2026 - June 25, 2027 at Sabanci University
"""


def test_academic_fields_grounded_derived_persisted_and_rules_updated(tmp_path):
    config={"configurable":{"thread_id":"academic"}};db=tmp_path/"academic.db"
    with sqlite_checkpointer(db) as saver:
        result=analyze(build_graph(saver),config,tmp_path/"academic.txt",{},ACADEMIC_TEXT)
        c=result["residence_permit_context"]
        assert c.exchange_student is True and c.program_start_date==date(2026,9,15)
        assert c.program_end_date==date(2027,6,25) and c.planned_stay_months==9
        assert c.requested_start_date is None and c.requested_end_date is None
        results={r.rule_id:r.status for r in result["relevant_requirements"]}
        assert results["R001"]==results["R013"]=="passed"
        assert results["R002"]==results["R003"]=="missing_information"
        derived=result["verification_records"]["__derived__"]["planned_stay_months"]
        assert derived.origin=="derived" and derived.derived_from and derived.derivation_policy
        document=result["documents"][0]
        assert document.info.source_pages["program_start_date"]==1
        assert "September 15" in document.info.source_snippets["program_start_date"]
    with sqlite_checkpointer(db) as saver:
        restored=build_graph(saver).get_state(config).values
        assert restored["residence_permit_context"]==c
        assert restored["verification_records"]["__derived__"]["planned_stay_months"]==derived


def test_heading_requested_dates_and_contextual_answers_reach_rules(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"form-details"}}
    text="""Residence Permit Application Form
(Requested Start Date)
18 / 9 / 2026
18 / 9 / 2027
Talep Edilen Bitis Tarihi
(Requested End Date)
Applicant: Alex Example
Passport number: P1234567
Declaration for Short Term Residence Permit Registration Form
"""
    result=analyze(graph,config,tmp_path/"application.txt",{},text,"ikamet_application_form")
    c=result["residence_permit_context"]
    assert c.permit_type=="short_term" and c.requested_start_date==date(2026,9,18)
    assert c.requested_end_date==date(2027,9,18)
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R002").passed is True
    result=confirm(graph,config,"__declarations__",{"is_in_turkiye":True,"entered_with_passport":True})
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R003").passed is True
    assert result["verification_records"]["__declarations__"]["entered_with_passport"].origin=="user_confirmed"
    assert result["candidate_plan"] is None and result["retry_count"]==0


@pytest.mark.parametrize("identity,stamp,entry", [(True,False,None),(True,True,None),(None,None,None),(True,True,True)])
def test_identity_stamp_and_entry_declaration_remain_separate(tmp_path,identity,stamp,entry):
    graph=build_graph();config={"configurable":{"thread_id":"passport"}}
    # Passport evidence is explicitly reviewed by the user in M1, not automatically authenticated.
    result=confirm(graph,config,"__declarations__",{"passport_id_page_present":identity,"turkey_entry_page_present":stamp,"entered_with_passport":entry})
    c=result["residence_permit_context"]
    assert c.passport_id_page_present is identity and c.turkey_entry_page_present is stamp
    assert c.entered_with_passport is entry and c.is_in_turkiye is None
    assert next(r for r in result["relevant_requirements"] if r.rule_id=="R003").status=="missing_information"


def test_fabricated_academic_fields_without_source_are_discarded(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"fabricated-academic"}}
    result=analyze(graph,config,tmp_path/"student.txt",{**STUDENT,"exchange_student":True,"program_start_date":"2026-09-15","program_end_date":"2027-06-25"})
    assert result["residence_permit_context"].exchange_student is None
    assert result["residence_permit_context"].program_start_date is None
    assert result["residence_permit_context"].planned_stay_months is None


def test_new_evidence_does_not_silently_replace_confirmed_value(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"confirmed-conflict"}}
    result=analyze(graph,config,tmp_path/"student.txt")
    doc=result["documents"][0]
    result=confirm(graph,config,doc.doc_id,{"holder_name":"Different Person"})
    # A subsequent assessment compares new source evidence with the retained statement.
    result=confirm(graph,config,"__declarations__",{"is_in_turkiye":True})
    assert result["verification_records"][doc.doc_id]["holder_name"].value=="Different Person"
    assert result["documents"][0].info.holder_name=="Alex Example"
    assert "holder_name" not in result["verification_conflicts"]
    assert result["residence_permit_context"].holder_name=="Different Person"


def test_specific_planner_payload_and_generic_rejection(tmp_path):
    from src.agents.planner import actionable_needs
    from src.guardrails.gate_1 import validate_plan
    graph=build_graph();config={"configurable":{"thread_id":"specific-plan"}}
    result=analyze(graph,config,tmp_path/"academic.txt",{},ACADEMIC_TEXT)
    needs=actionable_needs(result)
    assert not any(n["rule_id"]=="R001" for n in needs)
    passport=next(n for n in needs if n["rule_id"]=="R008")
    assert "Turkish entry-stamp page" in passport["documents"]
    step=PlanStep(step_id="location",priority=1,action="Confirm whether you are currently in Turkiye",reason="R003 needs your location declaration",related_rule_id="R003",action_type="information",request_kind="REQUEST_USER_CONFIRMATION",requested_fields=["is_in_turkiye"],available_in_prototype=True)
    assert not validate_plan(result,ActionPlan(actions=[step],summary="Confirm your location"))
    bad=step.model_copy(update={"action":"gather required data"})
    assert "GENERIC_ACTION" in [e.code for e in validate_plan(result,ActionPlan(actions=[bad],summary="Next steps"))]
    assert "UNSUPPORTED_APPROVAL_CLAIM" in [e.code for e in validate_plan(result,ActionPlan(actions=[step],summary="The residence permit is officially approved"))]
    result=confirm(graph,config,"__declarations__",{"is_in_turkiye":True})
    assert "REDUNDANT_REQUEST" in [e.code for e in validate_plan(result,ActionPlan(actions=[step],summary="Next steps"))]


@pytest.mark.parametrize("text", [
    "The exchange student will study between September 15, 2026 - June 25, 2027",
    "Degisim ogrencisi 15 Eyl\u00fcl 2026 - 25 Haziran 2027 arasinda okuyacak",
])
def test_bilingual_period_evidence_keeps_original_source_values(text):
    from src.document_processing.evidence import source_candidates
    from src.document_processing.extractor import TextPage
    candidates,_=source_candidates([TextPage(3,text)])
    dates={c["field"]:c for c in candidates if c.get("field") in {"program_start_date","program_end_date"}}
    assert dates["program_start_date"]["normalized_value"]=="2026-09-15"
    assert dates["program_end_date"]["normalized_value"]=="2027-06-25"
    assert all(c["original_value"] in text and c["page_number"]==3 for c in dates.values())


def test_missing_data_specific_plan_does_not_trigger_reflection(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"no-unnecessary-repair"}}
    analyze(graph,config,tmp_path/"academic.txt",{},ACADEMIC_TEXT)
    step=PlanStep(step_id="passport-entry",priority=1,action="Provide the Turkish entry-stamp page for manual review",reason="R008 entry-page evidence remains unknown",related_rule_id="R008",action_type="information",request_kind="REQUEST_DOCUMENT",requested_documents=["Turkish entry-stamp page"],available_in_prototype=True)
    with patch("src.agents.planner.create_action_plan",return_value=ActionPlan(actions=[step],summary="Review entry evidence")):
        result=graph.invoke(new_run(UserEvent(kind="generate_plan"),"review-user"),config)
    assert result["run_status"]=="VALIDATED" and result["retry_count"]==0
    assert result["__interrupt__"] and result["approval_decision"] is None


@pytest.mark.parametrize("text,stamp", [
    ("PASSPORT IDENTITY PAGE\nName: Alex Example\nPassport number: P1234567",False),
    ("PASSPORT ENTRY STAMP\nTurkiye entry: 2026-09-15\nName: Alex Example",True),
    ("PASSPORT PAGE\nUnreadable or ambiguous stamp",None),
])
def test_passport_uploads_remain_manual_evidence_not_invented_entry(tmp_path,text,stamp):
    graph=build_graph();config={"configurable":{"thread_id":"passport-document"}}
    result=analyze(graph,config,tmp_path/"passport.txt",{},text,"unknown")
    assert result["documents"][0].doc_type=="unknown"
    assert result["residence_permit_context"].entered_with_passport is None
    assert result["residence_permit_context"].is_in_turkiye is None
    if stamp is not None:
        result=confirm(graph,config,"__declarations__",{"turkey_entry_page_present":stamp})
        assert result["residence_permit_context"].turkey_entry_page_present is stamp
        assert result["residence_permit_context"].entered_with_passport is None



def test_bilingual_form_reading_order_recovers_identity_and_period_without_llm_extraction(tmp_path):
    text="""Personal Information
Soyadi
EXAMPLE
Other Citizenship
Adi
ALEX
Nationality in Born
Travel Document Information
Belge Turu
Umuma Mahsus Pasaport
Belge No
P1234567
(Requested Start Date)
18 / 9 / 2026
18 / 9 / 2027
Talep Edilen Bitis Tarihi
(Requested End Date)
Registration
Number)
2026-61-1234567
Declaration for Short Term Residence Permit Registration Form
"""
    graph=build_graph();config={"configurable":{"thread_id":"bilingual-form"}}
    result=analyze(graph,config,tmp_path/"form.txt",{},text,"ikamet_application_form")
    info=result["documents"][0].info
    assert info.applicant_name=="ALEX EXAMPLE" and info.passport_or_foreign_id=="P1234567"
    assert info.application_number=="2026-61-1234567" and info.permit_type=="short_term"
    assert info.requested_end_date==date(2027,9,18)
    assert info.source_pages["applicant_name"]==1 and "Soyadi" in info.source_snippets["applicant_name"]
    assert result["residence_permit_context"].entered_with_passport is None



def test_transport_timeout_does_not_reflect_or_erase_assessment(tmp_path):
    from src.agents.planner import PlannerUnavailable
    graph=build_graph();config={"configurable":{"thread_id":"unavailable-plan"}}
    state=analyze(graph,config,tmp_path/"academic.txt",{},ACADEMIC_TEXT)
    requirements=deepcopy(state["relevant_requirements"])
    with patch("src.agents.planner.create_action_plan",side_effect=PlannerUnavailable("timed out")) as planner:
        state=graph.invoke(new_run(UserEvent(kind="generate_plan"),"review-user"),config)
    assert planner.call_count==1 and state["retry_count"]==0
    assert state["run_status"]=="VALIDATED" and state["plan_origin"]=="deterministic_fallback"
    assert state["relevant_requirements"]==requirements
    assert state.get("__interrupt__") and state["approved_plan"] is None



def test_gate_1_no_candidate_preserves_transport_error_without_schema_noise():
    from src.guardrails.gate_1 import gate_1
    from src.core.state import ValidationIssue
    error=ValidationIssue(code="PLANNER_UNAVAILABLE",message="offline",source="planner")
    result=gate_1({"candidate_plan":None,"validation_errors":[error],"run_status":"PLANNING_UNAVAILABLE"})
    assert result["validation_errors"]==[error]
    assert result["run_status"]=="PLANNING_UNAVAILABLE"
    assert gate_1({"candidate_plan":None})["validation_errors"][0].code=="NO_CANDIDATE_PLAN"


def test_stage_selector_defers_submission_and_guardrail_questions():
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import selected_needs,workflow_stage
    from src.workers.rules_engine import evaluate_requirements
    state={"relevant_requirements":evaluate_requirements({})}
    needs=actionable_needs(state);by_id={n["rule_id"]:n for n in needs}
    assert workflow_stage(state)=="INITIAL_INFORMATION"
    assert by_id["R021"]["category"]=="future" and "R020" in by_id["R021"]["waiting_for"]
    assert by_id["R028"]["category"]=="guardrail_only"
    assert by_id["R006"]["category"]==by_id["R017"]["category"]=="immediate"
    assert not {"R021","R027","R028"} & {n["rule_id"] for n in selected_needs(needs)}


def test_fallback_specific_missing_fields_and_documents_pass_gate_1():
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import fallback_plan
    from src.guardrails.gate_1 import validate_plan
    from src.workers.rules_engine import evaluate_requirements
    results=evaluate_requirements({})
    state={"relevant_requirements":[r for r in results if r.rule_id in {"R006","R008","R017"}]}
    plan=fallback_plan(state,actionable_needs(state))
    assert any("passport expiration date" in a.action for a in plan.actions)
    assert any("Turkish entry-stamp page" in a.action for a in plan.actions)
    assert any(a.related_rule_id=="R017" and a.requested_documents for a in plan.actions)
    assert not validate_plan(state,plan)
    assert all(a.requires_approval and not a.due_date and not a.required_documents for a in plan.actions)


@pytest.mark.parametrize("a,b,equal", [
    ("NORA FICTIVA","Nora Fictiva",True),("Nora  Fictiva","NORA\u00a0FICTIVA",True),
    ("Jean\u2011Luc O\u2019Neil","Jean-Luc O'Neil",True),("NORA\u200b FICTIVA","Nora Fictiva",True),
    ("Nora Fictiva","Luca Fictiva",False),("Nora Fictiva","Fictiva Nora",False),
])
def test_identity_comparison_formatting_without_fuzzy_matching(a,b,equal):
    from src.workers.verification import same_name
    assert same_name(a,b) is equal


def test_saved_identity_confirmation_correction_clears_conflict_in_same_event(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"same-event-fix"}}
    state=analyze(graph,config,tmp_path/"student.txt")
    doc=state["documents"][0]
    state=confirm(graph,config,doc.doc_id,{"holder_name":"Different Person"})
    assert "holder_name" not in state["verification_conflicts"]
    state=confirm(graph,config,doc.doc_id,{"holder_name":"ALEX  EXAMPLE"})
    assert "holder_name" not in state["verification_conflicts"]
    assert state["residence_permit_context"].holder_name=="ALEX EXAMPLE"
    assert state["verification_records"][doc.doc_id]["holder_name"].value=="ALEX  EXAMPLE"


def test_explicit_identity_resolution_preserves_evidence_and_survives_sqlite(tmp_path):
    from src.workers.verification import conflict_details
    config={"configurable":{"thread_id":"resolved"}};db=tmp_path/"resolved.db"
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        state=analyze(graph,config,tmp_path/"student.txt")
        document=state["documents"][0]
        state=confirm(graph,config,document.doc_id,{"holder_name":"Different Person"})
        insurance_text="Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\nPolicy number: TEST-123\nCoverage start: 2026-11-01\nCoverage end: 2027-07-01"
        state=analyze(graph,config,tmp_path/"independent-policy.txt",{},insurance_text,"health_insurance_certificate")
        details=conflict_details(state)["holder_name"]
        assert len(details["observations"])==2 and not details["resolved"]
        event=UserEvent(kind="resolve_conflict",confirmed_fields={"holder_name":"Alex Example"})
        with patch("src.agents.planner.create_action_plan") as planner:
            state=graph.invoke(new_run(event,"review-user"),config)
        planner.assert_not_called()
        assert "holder_name" not in state["verification_conflicts"]
        assert state["documents"][0].info.holder_name=="Alex Example"
        assert state["verification_records"][document.doc_id]["holder_name"].value=="Different Person"
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver);state=graph.get_state(config).values
        assert conflict_details(state)["holder_name"]["resolved"]
        state=confirm(graph,config,"__declarations__",{"is_in_turkiye":True})
        assert state["residence_permit_context"].holder_name=="Alex Example"


def test_pending_approval_can_be_rejected_then_declarations_edited_and_repeated(tmp_path):
    from streamlit_app import interaction_state
    config={"configurable":{"thread_id":"unlock"}};db=tmp_path/"unlock.db"
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver);analyze(graph,config,tmp_path/"student.txt")
        from src.agents.planner import PlannerUnavailable
        with patch("src.agents.planner.create_action_plan",side_effect=PlannerUnavailable("offline")):
            graph.invoke(new_run(UserEvent(kind="generate_plan"),"review-user"),config)
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver);snapshot=graph.get_state(config)
        assert interaction_state(snapshot)["approval_pending"]
        before=deepcopy(snapshot.values)
        assert interaction_state(snapshot)["editing_blocked"]
        assert snapshot.values==before  # Opening/reviewing performs no graph call.
        graph.invoke(Command(resume=False),config)
        assert not interaction_state(graph.get_state(config))["editing_blocked"]
        state=confirm(graph,config,"__declarations__",{"is_in_turkiye":True,"entered_with_passport":True})
        assert next(r for r in state["relevant_requirements"] if r.rule_id=="R003").passed is True
        assert state["candidate_plan"] is None
        state=confirm(graph,config,"__declarations__",{"is_in_turkiye":False})
        assert state["residence_permit_context"].is_in_turkiye is False


def test_approved_action_progress_is_only_a_declaration(tmp_path):
    from src.agents.planner import PlannerUnavailable
    graph=build_graph();config={"configurable":{"thread_id":"progress"}}
    analyze(graph,config,tmp_path/"student.txt")
    with patch("src.agents.planner.create_action_plan",side_effect=PlannerUnavailable("offline")):
        state=graph.invoke(new_run(UserEvent(kind="generate_plan"),"review-user"),config)
    state=graph.invoke(Command(resume=True),config)
    action=state["approved_plan"].actions[0]
    before=state["residence_permit_context"].model_dump()
    event=UserEvent(kind="record_progress",action_id=action.step_id)
    with patch("src.agents.planner.create_action_plan") as planner:
        state=graph.invoke(new_run(event,"review-user"),config)
    planner.assert_not_called()
    assert state["action_progress"] and not state.get("completed_steps")
    assert state["residence_permit_context"].model_dump()==before
    assert all(row["status"]=="declared_completed" for row in state["action_progress"].values())


def test_insurance_exemption_unknown_does_not_block_sufficient_declared_private_policy():
    from src.workers.rules_engine import evaluate_requirements
    state={"residence_permit_context":{"insurance_valid_for_ikamet":True,"insurance_evidence_type":"private_health_insurance","insurance_start_date":"2026-09-01","insurance_end_date":"2027-07-01","requested_start_date":"2026-09-15","requested_end_date":"2027-06-25"}}
    results={r.rule_id:r for r in evaluate_requirements(state)}
    assert results["R015"].passed is results["R016"].passed is True



def test_explicit_recheck_clears_stale_conflict_without_mutating_observations(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"stale-conflict"}}
    state=analyze(graph,config,tmp_path/"student.txt")
    raw=state["documents"][0].info.model_dump()
    graph.update_state(config,{"verification_conflicts":["holder_name"]})
    with patch("src.agents.planner.create_action_plan") as planner:
        state=graph.invoke(new_run(UserEvent(kind="reevaluate"),"review-user"),config)
    planner.assert_not_called()
    assert state["verification_conflicts"]==[] and state["documents"][0].info.model_dump()==raw


def test_progress_influences_next_requests_without_satisfying_requirements(tmp_path):
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import fallback_plan
    from src.workers.rules_engine import evaluate_requirements
    results=evaluate_requirements({})
    state={"relevant_requirements":[r for r in results if r.rule_id=="R006"],"action_progress":{"p":{"related_rule_id":"R006","status":"declared_completed"}}}
    needs=actionable_needs(state)
    assert needs[0]["declared_progress"]
    assert "supporting evidence" in fallback_plan(state,needs).actions[0].action
    assert state["relevant_requirements"][0].status=="missing_information"


def test_unknown_authority_is_not_requested_as_personal_input():
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import selected_needs
    from src.workers.rules_engine import evaluate_requirements
    state={"relevant_requirements":evaluate_requirements({})}
    assert not any("decision_authority" in n["missing_fields"] for n in selected_needs(actionable_needs(state)))
