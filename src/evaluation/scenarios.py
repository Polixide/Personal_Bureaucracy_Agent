"""Curated Milestone-1 evaluation scenarios (moved from tests/test_evaluation.py).

Authored by Marco for the integrated prototype; shared by the test suite and
`python -m scripts.run_eval`. Each scenario fixes the residence-permit context,
the mocked model outputs (document extraction and planner proposals) and the
expected rule statuses, blockers, Gate 1 sequence and final run status.

APPROVED means the proposed next-action plan was explicitly approved, never that
the residence permit or an official action was approved/executed.
"""
import json
from types import SimpleNamespace

from src.core.state import ActionPlan, PlanStep

# Fixed dates and explicit inputs make every outcome reproducible. These are
# synthetic test facts, not current official submission dates or fee amounts.
READY_CONTEXT = dict(
    institution="Sabanc\u0131 University", applicant_type="incoming_exchange",
    application_kind="first", age=22, exchange_student=True, planned_stay_months=7,
    permit_type="short_term", is_in_turkiye=True, entered_with_passport=True,
    application_date="2026-11-10", legal_stay_expiry_date="2026-11-20",
    application_channel="https://e-ikamet.goc.gov.tr/",
    passport_expiry_date="2027-08-01", requested_permit_end_date="2027-06-01",
    application_form_generated=True, application_form_signed=True,
    passport_id_page_present=True, turkey_entry_page_present=True, visa_required=False,
    photo_age_months=2, biometric_photo=True, icao_compliant=True,
    fee_payment_required=True, fee_receipt_present=True,
    single_entry_visa_fee_required=False, financial_capacity_declared=True,
    exchange_program_evidence_present=True, student_certificate_present=True,
    insurance_exempt=False, insurance_valid_for_ikamet=True,
    insurance_start_date="2026-11-01", insurance_end_date="2027-06-30",
    requested_start_date="2026-11-01", requested_end_date="2027-06-01",
    insurance_evidence_type="private_health_insurance", address_proof_type="dorm_certificate",
    turkey_address_present=True, phone_present=True, email_present=True,
    declaration_control_document_present=True, declaration_control_document_signed=True,
    all_core_documents_present=True, scans_sent_to_iro=True, iro_preverification_confirmed=True,
    online_application_completed=True, fingerprint_step_completed=True, gi_number_present=True,
    current_date="2026-11-11", iro_window_start="2026-11-01", iro_window_end="2026-11-30",
    application_form_window_id="synthetic-nov", current_iro_window_id="synthetic-nov",
    missing_required_documents_count=0, approval_status="pending",
    decision_authority="Migration Management", submitted_to_iro=False,
)


SCENARIOS = [
    dict(id="01-happy-path", purpose="A ready file yields a gated, human-approved submission proposal.",
         changes={}, statuses={}, blockers=[], plan="submit", gates=[True], final="APPROVED"),
    dict(id="02-passport-buffer", purpose="Passport validity below the 60-day buffer remains a blocker.",
         changes={"passport_expiry_date": "2027-07-15"}, statuses={"R006": "failed"},
         blockers=["R006"], corrections=[("R006", "Review passport renewal options")],
         gates=[True], final="APPROVED"),
    dict(id="03-airbnb-address", purpose="Airbnb proof fails the scoped address rule; a corrective plan can pass.",
         changes={"address_proof_type": "airbnb"}, statuses={"R017": "failed"},
         blockers=["R017"], corrections=[("R017", "Obtain an accepted address document")],
         gates=[True], final="APPROVED"),
    dict(id="04-insurance-coverage", purpose="Coverage ending before the requested permit period is detected.",
         changes={"insurance_end_date": "2027-05-01"}, statuses={"R015": "failed"},
         blockers=["R015"], corrections=[("R015", "Review options to extend insurance coverage")],
         gates=[True], final="APPROVED"),
    dict(id="05-missing-gi", purpose="Missing GI blocks both the GI step and physical IRO submission.",
         changes={"gi_number_present": False}, statuses={"R022": "failed", "R023": "failed"},
         blockers=["R022", "R023"], corrections=[("R022", "Ask IRO how to obtain the GI number"),
                                                   ("R023", "Recheck IRO prerequisites after GI evidence is available")],
         gates=[True], final="APPROVED"),
    dict(id="06-expired-legal-stay", purpose="An application after the legal-stay deadline requires guidance, not submission.",
         changes={"application_date": "2026-11-21", "current_date": "2026-11-21"},
         statuses={"R004": "failed"}, blockers=["R004"],
         corrections=[("R004", "Ask IRO for guidance after the legal-stay deadline")],
         gates=[True], final="APPROVED"),
    dict(id="07-unknown-information", purpose="Unknown context remains missing_information rather than failure.",
         empty_context=True, changes={}, statuses={f"R{i:03}": "missing_information" for i in range(1, 30)},
         blockers=[f"R{i:03}" for i in range(1, 27)],
         corrections=[("R003", "Provide entry details and requested stay dates")], gates=[True], final="APPROVED"),
    dict(id="08-unknown-document", purpose="Unrelated uploaded text creates no administrative facts or legal readiness.",
         unknown_document=True,
         changes={"student_certificate_present": None, "all_core_documents_present": None,
                  "missing_required_documents_count": None, "iro_preverification_confirmed": None},
         statuses={r: "missing_information" for r in ["R014", "R020", "R021", "R023", "R026"]},
         blockers=["R014", "R020", "R021", "R023", "R026"],
         corrections=[("R014", "Provide a readable student certificate")], gates=[True], final="APPROVED"),
    dict(id="09-premature-submission", purpose="Proposed fixes do not satisfy prerequisites; a premature submission is repaired.",
         changes={"address_proof_type": "airbnb"}, statuses={"R017": "failed"}, blockers=["R017"],
         corrections=[("R017", "Obtain an accepted address document")], premature=True,
         gates=[False, True], error="PREMATURE_SUBMISSION", final="APPROVED"),
    dict(id="10-repair-exhaustion", purpose="Three unsuccessful repairs stop without approval or approved-plan write-back.",
         changes={"address_proof_type": "airbnb"}, statuses={"R017": "failed"}, blockers=["R017"],
         corrections=[("R017", "Obtain an accepted address document")], premature=True, exhaust=True,
         gates=[False, False, False, False], error="PREMATURE_SUBMISSION", final="HUMAN_REVIEW_REQUIRED"),
]


def candidate(scenario, *, submission=False):
    steps = [PlanStep(step_id=f"correct_{rule}", action=text, priority=i, reason="Address the supplied requirement result",
                      related_rule_id=rule, action_type="information")
             for i, (rule, text) in enumerate(scenario.get("corrections", []), 1)]
    if submission:
        steps.append(PlanStep(
            step_id="submit_to_iro", action="Submit the document set to IRO for review",
            priority=len(steps) + 1, reason="Proposed physical submission for user review",
            related_rule_id="R023", action_type="iro_submission",
            depends_on=[step.step_id for step in steps], required_documents=["student_certificate"],
        ))
    return ActionPlan(actions=steps, summary="Proposed next actions; no external execution")


def response(payload):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
