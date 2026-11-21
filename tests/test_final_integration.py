"""December integration: explicit active policies, address evidence and legal arithmetic."""
from copy import deepcopy
from datetime import date,datetime,timezone
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from test_procedure import ready_bundle
from langgraph.types import Command
from src.core.state import Document,ProcedureState,ResidencePermitContext,UserEvent,new_run
from src.core.graph import build_graph
from src.core.legal_stay import assess_legal_stay,reminder_candidates,insurance_explanation
from src.core.procedure import advance_current_stage,assess_current_stage
from src.document_processing.extractor import TextPage
from src.document_processing.processor import process_document
from src.workers.verification import verification_updates
from src.workers.rules_engine import evaluate_requirements
from src.workers.rule_loader import load_rules,knowledge_snapshot
from src.tools.schemas import ToolCall
from src.tools.registry import execute_tool
from src.memory.checkpointer import sqlite_checkpointer,initialize_procedure_thread
from src.rag.monitoring import compare_snapshot

BILL="Utility Bill\nHolder name: Property Owner\nService address: Example Street 12\nBilling date: 2026-11-01\nProvider: Example Energy Ltd."
FORM="Address Form\nFirst name: Taylor\nSurname: Example\nPassport number: X1234567\nNationality: Italian\nBirth date: 01.02.2000\nPhone: +39 123456789\nDeclared address: Example Street 12\nSignature present: yes"
class LocalFixture:
    name="synthetic";model="offline";supports_images=False
    def __init__(self,kind):self.kind=kind
    def generate_json(self,*,stage,**kwargs):return {"document_type":self.kind} if stage=="classification" else {}

def observation(kind,text):
    return process_document("synthetic.txt",pages=[TextPage(1,text)],provider=LocalFixture(kind),mode="local-only")

def address_state(kinds=("utility_bill","address_form")):
    state={**new_run(UserEvent(kind="reevaluate"),"synthetic"),"procedure":ProcedureState()}
    state["documents"]=[Document(doc_id=kind,file_path="synthetic.txt",status="analyzed",doc_type=kind,info=observation(kind,BILL if kind=="utility_bill" else FORM)) for kind in kinds]
    return {**state,**verification_updates(state)}

def req(state,rid):return next(r for r in evaluate_requirements(state) if r.rule_id==rid)

@pytest.mark.parametrize("rid",["R012","R024","R025","R027","R029"])
def test_removed_rules_are_absent_from_active_runtime(rid):
    assert rid not in {r["rule_id"] for r in load_rules()}
    assert rid not in {r.rule_id for r in evaluate_requirements({})}

@pytest.mark.parametrize("values,expected",[({"insurance_exempt":True},True),({"insurance_end_date":"2027-07-20","program_end_date":"2027-06-20","requested_permit_end_date":"2027-09-20"},True),({"insurance_end_date":"2027-05-20","program_end_date":"2027-06-20","requested_permit_end_date":"2027-09-20"},False),({},None)])
def test_v2_insurance_policy(values,expected):
    assert req({"residence_permit_context":ResidencePermitContext(**values)},"R015").passed is expected

@pytest.mark.parametrize("kind",["sgk_provision","sgk_application","bilateral_social_security"])
def test_unimplemented_insurance_alternatives_need_manual_review(kind):
    result=req({"residence_permit_context":ResidencePermitContext(insurance_evidence_type=kind)},"R016")
    assert result.status=="missing_information" and "Manual review" in result.message

def test_address_profiles_and_combination_preserve_source_observations():
    state=address_state();before=deepcopy(state["documents"])
    assert req(state,"R017").passed is True
    assert state["residence_permit_context"].holder_name is None  # Payer is not inferred to be the student.
    assert state["residence_permit_context"].utility_service_address=="Example Street 12"
    assert not state["verification_conflicts"]
    assert state["documents"]==before
    for doc in state["documents"]:
        for field in doc.info.source_snippets:assert doc.info.source_pages[field]==1

@pytest.mark.parametrize("kind",["utility_bill","address_form"])
def test_one_address_piece_is_incomplete_not_pass(kind):
    result=req(address_state((kind,)),"R017")
    assert result.status=="missing_information" and "INCOMPLETE" in result.message

def test_blank_signature_is_not_inferred_from_model_or_label():
    info=observation("address_form",FORM.replace("Signature present: yes","Signature: ______"))
    assert info.signature_present is None and "signature_present" in info.missing_fields

def test_dropping_one_address_piece_clears_derived_acceptance():
    state=address_state();state["documents"]=[d for d in state["documents"] if d.doc_type!="address_form"]
    state.update(verification_updates(state))
    assert req(state,"R017").passed is not True

def stay(**extra):
    return {"residence_permit_context":ResidencePermitContext(nationality="ITA",passport_type="ordinary",visa_status="visa_exempt",latest_entry_date="2026-10-01",**extra)}

def test_unknown_prior_history_is_an_explicit_estimate_without_state_write():
    state=stay();before=deepcopy(state)
    result=assess_legal_stay(state,as_of=date(2026,11,20))
    assert result.status=="ESTIMATE" and result.estimated_or_calculated_end==date(2026,12,29)
    assert result.recommended_target==date(2026,11,29)
    assert any("unknown" in text for text in result.assumptions)
    assert state==before and state["residence_permit_context"].legal_stay_expiry_date is None

def test_known_prior_stays_reduce_rolling_allowance_and_overlap_counts_once():
    c=ResidencePermitContext(nationality="ITA",passport_type="ordinary",visa_status="visa_exempt",latest_entry_date="2026-11-01",previous_stay_history=json.dumps([{"start":"2026-10-01","end":"2026-10-30"}]*2))
    result=assess_legal_stay({"residence_permit_context":c},as_of=date(2026,11,20))
    assert result.confidence_type=="CALCULATED_FROM_DECLARED_HISTORY" and result.estimated_or_calculated_end==date(2026,12,30)

@pytest.mark.parametrize("changes",[{"nationality":"unsupported"},{"passport_type":None},{"visa_status":None}])
def test_no_universal_entry_plus_ninety_shortcut(changes):
    state=stay();state["residence_permit_context"]=state["residence_permit_context"].model_copy(update=changes)
    assert assess_legal_stay(state,as_of=date(2026,11,20)).estimated_or_calculated_end is None

def test_explicit_short_visa_and_validity_cap():
    c=ResidencePermitContext(nationality="Other",passport_type="ordinary",visa_status="visa",visa_stay_days=30,visa_valid_until="2026-11-20",latest_entry_date="2026-11-01",previous_stay_history="[]")
    assert assess_legal_stay({"residence_permit_context":c},as_of=date(2026,11,20)).estimated_or_calculated_end==date(2026,11,20)

def test_reminder_is_proposed_and_revalidated_after_source_change():
    state=stay();row=next(r for r in reminder_candidates(state,as_of=date(2026,11,20)) if r["candidate_id"]=="recommended_target")
    call=ToolCall(name="create_calendar_reminder",arguments={"title":row["title"],"start_at":row["date"]+"T09:00:00+03:00","source_candidate_id":row["candidate_id"]})
    with patch("src.core.legal_stay.local_today",return_value=date(2026,11,20)):
        assert execute_tool(state,call).status=="APPROVAL_REQUIRED"
        state["residence_permit_context"].legal_stay_expiry_date=date(2026,12,15)
        assert execute_tool(state,call,approved=True).status=="FAILED"
    assert not state.get("calendar_reminders")

def test_insurance_explanation_does_not_turn_academic_coverage_into_failure():
    state={"residence_permit_context":ResidencePermitContext(insurance_end_date="2027-07-20",program_end_date="2027-06-20",requested_permit_end_date="2027-09-20")}
    assert req(state,"R015").passed is True and "expires before" in insurance_explanation(state)

def test_rule_snapshot_is_stable_even_if_active_loader_changes():
    state={"knowledge_snapshot":knowledge_snapshot()}
    with patch("src.workers.rule_loader.load_rules",return_value=[]):
        # Engine consumes the host snapshot directly; no source refresh can replace it.
        from src.workers.rules_engine import load_rules as captured_loader
        assert len(captured_loader(state["knowledge_snapshot"]))==24
    assert len(evaluate_requirements(state))==24

def test_source_change_candidate_is_allow_listed_and_never_writes_rules():
    from src.rag.ingestion import build_corpus
    from config import KNOWLEDGE_DIR
    before=(KNOWLEDGE_DIR/"rules.json").read_bytes()
    source=next(r for r in build_corpus() if r.source_id=="SAB_EXCHANGE_CURRENT" and r.content_kind=="official_guidance")
    result=compare_snapshot(source.source_id,source.url,source.content+" Updated review arrangements.")
    assert result["status"]=="RULE_CHANGE_CANDIDATE"
    assert (KNOWLEDGE_DIR/"rules.json").read_bytes()==before
    with pytest.raises(ValueError):compare_snapshot(source.source_id,"https://untrusted.example",source.content)

def test_checklist_and_email_artifacts_require_explicit_approval(tmp_path):
    for name in ("build_checklist","prepare_iro_email"):
        call=ToolCall(name=name)
        preview=execute_tool({},call)
        assert preview.status=="APPROVAL_REQUIRED" and not preview.state_updates
        if name=="build_checklist":
            with patch("config.PROJECT_ROOT",tmp_path):result=execute_tool({},call,approved=True)
            assert result.status=="SUCCEEDED",result.data
            assert Path(result.data["artifact_path"]).is_file()

def test_rule_source_provenance_does_not_invent_an_official_url():
    from src.rag.ingestion import build_corpus
    row=next(r for r in build_corpus() if "R017" in r.rule_ids)
    assert row.authority=="sabanci_direct_instruction" and row.url is None

def test_combined_address_and_procedure_survive_sqlite_restart(tmp_path):
    config={"configurable":{"thread_id":"address"}};db=tmp_path/"twin.db"
    state=address_state()
    with sqlite_checkpointer(db) as saver:build_graph(saver).update_state(config,state,as_node="respond")
    with sqlite_checkpointer(db) as saver:
        restored=build_graph(saver).get_state(config).values
        assert req(restored,"R017").passed is True
        assert restored["documents"]==state["documents"]

def test_later_stage_never_advances_from_sandbox_or_plan_approval():
    state={"procedure":ProcedureState(current_stage="PAYMENT"),"sandbox_form":{"status":"SIMULATED_SUBMISSION"},"run_status":"APPROVED"}
    assert not assess_current_stage(state)["ready"]
    assert advance_current_stage(state,recorded_at=datetime.now(timezone.utc))=={}


def test_insurance_terms_mentioning_address_forms_do_not_reclassify_the_policy():
    from src.document_processing.processor import checked_classification
    pages=[TextPage(1,"HEALTH INSURANCE CERTIFICATE\nInsured: Taylor Example\nPolicy number: EX1234"),TextPage(2,"Terms: address declaration / adres beyan is mentioned in legal conditions.")]
    assert checked_classification("health_insurance_certificate",pages)=="health_insurance_certificate"


def test_complete_observed_stage_flow_requires_explicit_transitions(ready_bundle,tmp_path):
    from test_milestone_documents import analyze,confirm
    from test_procedure import RECORDED_AT
    graph,config,state=ready_bundle
    update=advance_current_stage(state,recorded_at=RECORDED_AT)
    graph.update_state(config,update,as_node="respond")
    state=analyze(graph,config,tmp_path,"ikamet_application_form")
    form=next(d for d in state["documents"] if d.doc_type=="ikamet_application_form")
    state=confirm(graph,config,form.doc_id,{"application_form_signed":True,"online_application_completed":True})
    assert state["procedure"].current_stage=="EIKAMET_PREPARATION"
    def advance(expected):
        state=graph.get_state(config).values
        assert assess_current_stage(state)["ready"],assess_current_stage(state)
        graph.update_state(config,advance_current_stage(state,recorded_at=RECORDED_AT),as_node="respond")
        assert graph.get_state(config).values["procedure"].current_stage==expected
    advance("EIKAMET_READY");advance("PAYMENT")
    analyze(graph,config,tmp_path,"card_fee_receipt")
    confirm(graph,config,"__declarations__",{"single_entry_visa_fee_required":False})
    advance("FINGERPRINT")
    confirm(graph,config,"__declarations__",{"fingerprint_step_completed":True,"gi_number_present":True})
    advance("IRO_REVIEW")
    analyze(graph,config,tmp_path,"declaration_control_document")
    confirm(graph,config,"__declarations__",{"scans_sent_to_iro":True,"iro_preverification_confirmed":True})
    advance("SUBMISSION")
    confirm(graph,config,"__declarations__",{"submitted_to_iro":True})
    advance("TRACKING")
    assert not assess_current_stage(graph.get_state(config).values)["ready"]
    confirm(graph,config,"__declarations__",{"official_decision_received":True,"approval_status":"rejected"})
    advance("COMPLETE")  # Closed on a reported authority decision, never agent permit approval.


def test_headingless_bilingual_address_declaration_preserves_clear_fields_only():
    text=("Adres bilgilerim asagida yer alir ve beyan ederim.\nTAYLOR\nAdi (First name):\nSoyadi (Last/Family name): &X AMPLE\nDogum Tarihi (Date of Birth): 26/06/2002\nTelefon (Mobile number): +901234567890\nImza (Signature):\nADRES BILGILERI (Address in Istanbul):\nExample Street 12\nDistrict Istanbul")
    info=observation("address_form",text)
    assert info.document_type=="address_form" and info.first_name=="TAYLOR"
    assert info.birth_date==date(2002,6,26) and info.phone=="+901234567890"
    assert info.last_name is info.passport_number is info.signature_present is None
    assert "Example Street" in info.declared_address
    assert info.source_pages["birth_date"]==1


def test_legacy_requested_end_alias_cannot_bypass_a_verified_context_conflict():
    state={"residence_permit_context":ResidencePermitContext(insurance_end_date="2027-07-20",requested_end_date="2027-06-20"),"verification_conflicts":["requested_permit_end_date"]}
    assert req(state,"R015").passed is None

def test_explicitly_invalid_private_policy_requires_review_without_requesting_a_positive_attestation():
    state={"residence_permit_context":ResidencePermitContext(insurance_evidence_type="private_health_insurance",insurance_valid_for_ikamet=False)}
    assert req(state,"R016").passed is False


def test_incomplete_address_pair_proposes_review_not_confirmation_of_derived_flags():
    from src.agents.planning_policy import canonical_candidates
    state=address_state(("utility_bill",))
    state["documents"].append(Document(doc_id="address_form",file_path="synthetic.txt",status="analyzed",doc_type="address_form",info=observation("address_form",FORM.replace("Signature present: yes","Signature: ______"))))
    state.update(verification_updates(state))
    state["relevant_requirements"]=evaluate_requirements(state)
    candidate=next(row for row in canonical_candidates(state) if row["candidate_id"]=="R017")
    assert candidate["action_type"]=="REVIEW_DOCUMENT"
    assert candidate["target_document"]=="address_form"
    assert candidate["fields"]==[]
    assert req(state,"R017").passed is None


def test_gate_1_uses_the_same_frozen_dependency_snapshot_as_the_engine():
    from src.core.state import ActionPlan
    from src.guardrails.gate_1 import validate_plan
    from src.workers.rule_loader import load_dependencies
    snapshot=knowledge_snapshot()
    with patch("src.guardrails.gate_1.load_dependencies",wraps=load_dependencies) as loader:
        validate_plan({"knowledge_snapshot":snapshot},ActionPlan(actions=[],summary="No proposed actions"))
    loader.assert_called_once_with(snapshot)


def test_unversioned_lesson_for_a_retired_rule_is_not_replayed():
    from config import TWIN_RECORD_PATH
    from src.core.state import ValidationIssue
    from src.memory.twin_store import TwinStore,Lesson
    from src.memory.twin_sync import relevant_lessons
    with TwinStore(TWIN_RECORD_PATH) as store:
        store.save_twin({"user_id":"retired-case","active_procedure":"ikamet"})
        store.add_lessons_once("retired-rule-trace",[Lesson(user_id="retired-case",procedure="ikamet",failure_type="VAGUE_TITLE",validator_error="Rejected title",corrective_action="Review a retired requirement",outcome="Former gate accepted",correction_validated=True,rule_or_source_id="R012",rule_version=None)])
    state={"user_id":"retired-case","active_procedure":"ikamet","knowledge_snapshot":knowledge_snapshot()}
    assert relevant_lessons(state,[ValidationIssue(code="VAGUE_TITLE",message="Use a concrete title",source="gate_1")])==[]
