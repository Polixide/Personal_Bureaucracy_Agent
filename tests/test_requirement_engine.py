import json
from datetime import date
import pytest
from config import KNOWLEDGE_DIR
from src.core.state import ResidencePermitContext
from src.workers.rules_engine import evaluate_requirements

CASES = json.loads((KNOWLEDGE_DIR / "test_cases.json").read_text(encoding="utf-8"))["tests"]

def evaluate(values):
    return {r.rule_id:r for r in evaluate_requirements({"residence_permit_context":ResidencePermitContext(**values)})}

@pytest.mark.parametrize("case", CASES, ids=lambda c:c["id"])
def test_current_knowledge_cases(case):
    assert evaluate(case["input"])[case["rule"]].passed is (case["expected"] == "PASS")

@pytest.mark.parametrize("number", range(1,30))
def test_unknown_is_preserved(number):
    r = evaluate({})[f"R{number:03}"]
    assert r.passed is None and r.status == "missing_information" and r.error_code is None

@pytest.mark.parametrize("expiry,expected", [("2027-07-31",True),("2027-07-30",False)])
def test_exact_sixty_day_boundary(expiry,expected):
    assert evaluate({"requested_permit_end_date":"2027-06-01", "passport_expiry_date":expiry})["R006"].passed is expected

def test_deadline_equal_is_allowed_by_root_condition():
    assert evaluate({"application_date":"2026-11-15", "legal_stay_expiry_date":"2026-11-15"})["R004"].passed is True

@pytest.mark.parametrize("values,rule", [({"insurance_exempt":True},"R015"),({"fee_payment_required":False},"R010"),({"single_entry_visa_fee_required":False},"R011"),({"submitted_to_iro":False},"R029")])
def test_conditional_alternatives_do_not_require_unneeded_inputs(values,rule):
    assert evaluate(values)[rule].passed is True

def test_known_negative_conjunction_is_false_even_with_other_unknowns():
    assert evaluate({"is_in_turkiye":False})["R003"].passed is False

def test_engine_does_not_mutate_state():
    context = ResidencePermitContext(is_in_turkiye=True)
    before = context.model_dump()
    evaluate_requirements({"residence_permit_context":context})
    assert context.model_dump() == before

# Each authoritative rule gets an explicit positive and negative branch.
RULE_BRANCHES = [
("R001", {"exchange_student":True,"planned_stay_months":3}, {"planned_stay_months":2}),
("R002", {"permit_type":"short_term"}, {"permit_type":"student"}),
("R003", {"is_in_turkiye":True,"entered_with_passport":True}, {"entered_with_passport":False}),
("R004", {"application_date":"2026-11-10","legal_stay_expiry_date":"2026-11-20"}, {"application_date":"2026-11-21"}),
("R005", {"application_channel":"https://e-ikamet.goc.gov.tr/"}, {"application_channel":"unofficial"}),
("R006", {"passport_expiry_date":"2027-08-01","requested_permit_end_date":"2027-06-01"}, {"passport_expiry_date":"2027-06-01"}),
("R007", {"application_form_generated":True,"application_form_signed":True}, {"application_form_signed":False}),
("R008", {"passport_id_page_present":True,"turkey_entry_page_present":True,"visa_required":False}, {"passport_id_page_present":False}),
("R009", {"photo_age_months":6,"biometric_photo":True,"icao_compliant":True}, {"photo_age_months":7}),
("R010", {"fee_payment_required":True,"fee_receipt_present":True}, {"fee_receipt_present":False}),
("R011", {"single_entry_visa_fee_required":True,"single_entry_visa_fee_receipt_present":True}, {"single_entry_visa_fee_receipt_present":False}),
("R012", {"financial_capacity_declared":True}, {"financial_capacity_declared":False}),
("R013", {"exchange_program_evidence_present":True}, {"exchange_program_evidence_present":False}),
("R014", {"student_certificate_present":True}, {"student_certificate_present":False}),
("R015", {"insurance_exempt":False,"insurance_valid_for_ikamet":True,"insurance_start_date":"2026-01-01","insurance_end_date":"2027-01-01","requested_start_date":"2026-02-01","requested_end_date":"2026-12-01"}, {"insurance_end_date":"2026-11-01"}),
("R016", {"insurance_evidence_type":"private_health_insurance"}, {"insurance_evidence_type":"travel_only"}),
("R017", {"address_proof_type":"dorm_certificate"}, {"address_proof_type":"airbnb"}),
("R018", {"turkey_address_present":True,"phone_present":True,"email_present":True}, {"phone_present":False}),
("R019", {"declaration_control_document_present":True,"declaration_control_document_signed":True}, {"declaration_control_document_signed":False}),
("R020", {"all_core_documents_present":True}, {"all_core_documents_present":False}),
("R021", {"scans_sent_to_iro":True,"iro_preverification_confirmed":True}, {"iro_preverification_confirmed":False}),
("R022", {"online_application_completed":True,"fingerprint_step_completed":True,"gi_number_present":True}, {"gi_number_present":False}),
("R023", {"online_application_completed":True,"iro_preverification_confirmed":True,"gi_number_present":True,"all_core_documents_present":True}, {"gi_number_present":False}),
("R024", {"current_date":"2026-11-11","iro_window_start":"2026-11-11","iro_window_end":"2026-11-12"}, {"current_date":"2026-11-13"}),
("R025", {"application_form_window_id":"nov","current_iro_window_id":"nov"}, {"application_form_window_id":"oct"}),
("R026", {"missing_required_documents_count":0}, {"missing_required_documents_count":1}),
("R027", {"approval_status":"pending"}, {"approval_status":"guaranteed"}),
("R028", {"decision_authority":"Migration Management"}, {"decision_authority":"Sabanc? IRO"}),
("R029", {"submitted_to_iro":True,"tracking_channel":"e_ikamet"}, {"tracking_channel":"unofficial"}),
]

@pytest.mark.parametrize("rule,positive,negative", RULE_BRANCHES, ids=[r[0] for r in RULE_BRANCHES])
def test_every_current_rule_pass_and_fail(rule,positive,negative):
    assert evaluate(positive)[rule].passed is True
    assert evaluate({**positive,**negative})[rule].passed is False


def test_missing_information_names_inputs():
    result=evaluate({})["R006"]
    assert set(result.missing_fields) == {"passport_expiry_date","requested_permit_end_date"}
