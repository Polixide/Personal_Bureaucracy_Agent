from datetime import date
import pytest
from src.core.state import ActionPlan, PlanStep, ResidencePermitContext
from src.guardrails.gate_1 import validate_plan
from src.workers.rules_engine import evaluate_requirements


def action(**changes):
    data=dict(step_id="review",action="Review entry details",priority=1,reason="Information needed",related_rule_id="R003")
    data.update(changes)
    return PlanStep(**data)

def check(actions, values=None, summary="Proposed steps", **extra):
    state={"residence_permit_context":ResidencePermitContext(**(values or {}))}
    state["relevant_requirements"]=evaluate_requirements(state)
    state.update(extra)
    return {e.code for e in validate_plan(state,ActionPlan(actions=actions,summary=summary))}

def test_valid_information_plan(): assert not check([action()])
def test_unknown_rule(): assert "UNKNOWN_RULE_ID" in check([action(related_rule_id="R999")])
def test_failed_rule_ignored(): assert "FAILED_RULE_IGNORED" in check([],{"is_in_turkiye":False})
def test_unknown_not_failure(): assert "UNKNOWN_AS_FAILURE" in check([action(reason="R003 failed")])
def test_summary_unknown(): assert "UNKNOWN_AS_FAILURE" in check([],summary="R003 has failed")

@pytest.mark.parametrize("text",["Submit the application","Finalize the application","Submit documents to IRO"])
def test_premature_submission_unknown_blocks(text): assert "PREMATURE_SUBMISSION" in check([action(action=text)])

@pytest.mark.parametrize("text",["I submitted the application","Appointment has been booked","The fee is paid","Successfully renewed the passport"])
def test_completion_claim(text): assert "COMPLETION_CLAIM" in check([action(action=text)])
def test_summary_completion_claim(): assert "COMPLETION_CLAIM" in check([],summary="We submitted the application")
def test_duplicate_priorities(): assert "AMBIGUOUS_ORDER" in check([action(),action(step_id="other")])
def test_missing_dependency(): assert "MISSING_PREREQUISITE" in check([action(depends_on=["later"])])
def test_self_dependency(): assert "MISSING_PREREQUISITE" in check([action(depends_on=["review"])])
def test_missing_document(): assert "MISSING_DOCUMENT" in check([action(required_documents=["passport_copy"])])
def test_cannot_waive_approval(): assert "APPROVAL_REQUIRED" in check([action(requires_approval=False)])
def test_official_category_blocks_unknown_prerequisites(): assert "MISSING_PREREQUISITE" in check([action(action="Deliver file",action_type="iro_submission")])
def test_grounded_deadline():
    step=action(due_date=date(2026,11,15),deadline_key="submission")
    assert "UNSUPPORTED_DEADLINE" in check([step])
    assert "UNSUPPORTED_DEADLINE" not in check([step],deadlines={"submission":date(2026,11,15)})
def test_mutated_schema():
    step=action(); step.priority=0
    assert "SCHEMA_ERROR" in check([step])


def test_duplicate_step_ids():
    assert "DUPLICATE_STEP_ID" in check([action(),action(priority=2)])

def test_expired_document_is_rejected_for_scheduled_action():
    from src.core.state import Document
    from src.document_processing.schemas import HealthInsurance
    info=HealthInsurance(holder_name="Alex",insurer="Example",policy_number="P1",coverage_start_date="2026-01-01",coverage_end_date="2026-11-01")
    doc=Document(doc_id="insurance",file_path="test.txt",status="analyzed",doc_type="health_insurance",expiry_date=info.coverage_end_date,info=info)
    step=action(due_date=date(2026,11,15),deadline_key="review",required_documents=["health_insurance"])
    assert "DATE_CONFLICT" in check([step],documents=[doc],deadlines={"review":date(2026,11,15)})


def test_dependency_dates_must_be_ordered():
    first=action(due_date=date(2026,11,20),deadline_key="first")
    second=action(step_id="second",priority=2,depends_on=["review"],due_date=date(2026,11,15),deadline_key="second")
    assert "DATE_CONFLICT" in check([first,second],deadlines={"first":date(2026,11,20),"second":date(2026,11,15)})


def test_insurance_cannot_be_used_before_coverage_starts():
    from src.core.state import Document
    from src.document_processing.schemas import HealthInsurance
    info=HealthInsurance(holder_name="Alex",insurer="Example",policy_number="P1",coverage_start_date="2027-01-01",coverage_end_date="2027-12-31")
    doc=Document(doc_id="insurance",file_path="test.txt",status="analyzed",doc_type="health_insurance",expiry_date=info.coverage_end_date,info=info)
    step=action(due_date=date(2026,11,15),deadline_key="review",required_documents=["health_insurance"])
    assert "DATE_CONFLICT" in check([step],documents=[doc],deadlines={"review":date(2026,11,15)})
