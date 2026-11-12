"""Ten curated M1 scenarios: real graph/SQLite/gates, mocked Ollama transport.

Run: python -m pytest tests/test_evaluation.py -v
APPROVED means the proposed next-action plan was explicitly approved, never that
the residence permit or an official action was approved/executed.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.types import Command

from src.core.graph import build_graph
from src.core.state import ActionPlan, PlanStep, ResidencePermitContext, UserEvent, new_run
from src.guardrails.gate_1 import gate_1 as real_gate_1
from src.memory.checkpointer import sqlite_checkpointer


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


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario["id"])
def test_curated_end_to_end_scenario(tmp_path, scenario):
    """Exercise extraction, rules, planner parsing, Gate 1, repairs, approval and persistence."""
    context = {} if scenario.get("empty_context") else READY_CONTEXT
    request = new_run(UserEvent(kind="upload", file_path=str(tmp_path / "synthetic.txt")), scenario["id"])
    request["residence_permit_context"] = ResidencePermitContext(**{**context, **scenario["changes"]})
    path = tmp_path / "synthetic.txt"
    if scenario.get("unknown_document"):
        path.write_text("ciao sono marco", encoding="utf-8")
        document_responses = [response({"document_type": "unknown"})]
    else:
        path.write_text("Student Enrollment Certificate\nHolder name: Alex Example\nStudent ID: SYN-001\nSabanc\u0131 University\nIssue date: 2026-11-01", encoding="utf-8")
        document_responses = [response({"document_type": "student_enrollment_certificate"}), response({
            "holder_name": "Alex Example", "student_id": "SYN-001",
            "institution": "Sabanc\u0131 University", "issue_date": "2026-11-01",
        })]
    good = candidate(scenario, submission=scenario.get("plan") == "submit")
    initial = candidate(scenario, submission=True) if scenario.get("premature") else good
    proposals = [initial]
    if scenario.get("premature"):
        proposals += [initial] * 3 if scenario.get("exhaust") else [good]
    gate_results = []

    def gate_1(state):
        output = real_gate_1(state)
        gate_results.append(output)
        return output

    db = tmp_path / "evaluation.db"
    config = {"configurable": {"thread_id": scenario["id"]}, "recursion_limit": 60}
    with patch("src.document_processing.processor.Client") as document_client, \
         patch("src.agents.planner.Client") as planner_client, \
         patch("src.core.graph.gate_1", gate_1):
        document_client.return_value.chat.side_effect = document_responses
        planner_client.return_value.chat.side_effect = [response(plan.model_dump(mode="json")) for plan in proposals]
        with sqlite_checkpointer(db) as saver:
            graph = build_graph(saver)
            result = graph.invoke(request, config)
            assert result["relevant_requirements"] is not None, result.get("final_response")
            expected_statuses = {f"R{i:03}": "passed" for i in range(1, 30)}
            expected_statuses.update(scenario["statuses"])
            assert {r.rule_id: r.status for r in result["relevant_requirements"]} == expected_statuses
            requirement_blockers = sorted(r.rule_id for r in result["relevant_requirements"]
                                          if r.severity == "blocking" and r.passed is not True)
            assert requirement_blockers == scenario["blockers"]
            assert [not output["validation_errors"] for output in gate_results] == scenario["gates"]
            if scenario.get("error"):
                assert scenario["error"] in {error.code for error in gate_results[0]["validation_errors"]}
            document = result["documents"][0]
            assert document.status == "analyzed"
            if scenario.get("unknown_document"):
                assert document.doc_type == "unknown"
                facts = result["extracted_facts"][document.doc_id]
                assert facts["holder_name"] is None
                assert not {"student_id", "policy_number", "application_number"} & facts.keys()
            else:
                assert document.doc_type == "student_certificate"
                assert document.info.student_id == "SYN-001"
            if scenario["gates"][-1]:
                assert result["__interrupt__"]
                assert result["approval_decision"] is None
                assert "approved_plan" not in result
                result = graph.invoke(Command(resume=True), config)
                assert result["approved_plan"].actions == good.actions
            else:
                assert not result.get("__interrupt__")
                assert "approved_plan" not in result
                assert result["approval_decision"] is None
                assert result["retry_count"] == 3
                assert len(result["repair_history"]) == 3
            assert result["run_status"] == scenario["final"]
            assert result.get("blockers", []) == scenario["blockers"]
            assert not result.get("completed_steps")
            expected_repairs = len(scenario["gates"]) - 1
            assert result["retry_count"] == expected_repairs
        assert planner_client.return_value.chat.call_count == len(scenario["gates"])
        assert document_client.return_value.chat.call_count == len(document_responses)

    # A fresh graph/SQLite connection must restore the complete final snapshot.
    with sqlite_checkpointer(db) as saver:
        snapshot = build_graph(saver).get_state(config)
        assert snapshot.next == ()
        assert snapshot.values == result
