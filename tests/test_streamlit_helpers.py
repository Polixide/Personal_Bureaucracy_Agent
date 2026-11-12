"""Pure presentation helpers only: no Streamlit sessions, server or browser."""
from datetime import date

import pytest

from src.core.state import Document, ResidencePermitContext
from streamlit_app import (
    friendly_error, jsonable, field, display_record, display_records,
    observation_view, status_label,
)


def test_state_display_serialization_does_not_mutate_canonical_models():
    context = ResidencePermitContext(legal_stay_expiry_date=date(2026, 11, 20))
    document = Document(doc_id="synthetic", file_path="synthetic.txt")
    state = {"residence_permit_context": context, "documents": [document],
             "extracted_facts": {"date": date(2026, 11, 11)}}
    rendered = jsonable(state)
    assert rendered["residence_permit_context"]["legal_stay_expiry_date"] == "2026-11-20"
    assert rendered["documents"][0]["doc_id"] == "synthetic"
    assert rendered["extracted_facts"]["date"] == "2026-11-11"
    assert state["residence_permit_context"] is context
    assert isinstance(context.legal_stay_expiry_date, date)
    assert state["documents"][0] is document


@pytest.mark.parametrize("message,expected", [
    ("Local Ollama timed out during extraction", "timed out"),
    ("ReadTimeout", "timed out"),
    ("OCR timed out on page 2", "Page OCR timed out"),
    ("RapidOCR initialization failed", "project OCR dependencies"),
    ("Local Ollama request failed: connection refused", "Start Ollama"),
    ("PDF contains no extractable text", "text-based PDF"),
    ("Cannot read PDF: corrupt.pdf", "could not be read"),
    ("Password-protected PDFs are not supported", "Password-protected"),
    ("Unsupported extension .exe", "Unsupported file"),
    ("The uploaded file is empty", "empty"),
    ("Unexpected processing exception", "Processing could not finish"),
])
def test_user_facing_error_messages(message, expected):
    assert expected in friendly_error(message)


def test_current_document_observations_display_without_mutation():
    from src.document_processing.schemas import StudentCertificate
    info = StudentCertificate(holder_name="Alex Example", student_id="S123",
                              institution="Example University", issue_date="2026-11-01")
    before = info.model_dump()
    rendered, legacy = observation_view(info)
    assert not legacy
    assert rendered["document_type"] == "student_enrollment_certificate"
    assert rendered["issue_date"] == "2026-11-01"
    rendered["missing_fields"].append("display-only")
    assert info.model_dump() == before
    assert field(info, "student_id") == "S123"


def test_legacy_observations_remain_labelled_and_detached():
    from copy import deepcopy
    info = {"document_type": "health_insurance_certificate", "student_id": "old-P123",
            "institution": "Old provider", "missing_fields": [], "source_pages": {"student_id": 1}}
    document = Document.model_construct(doc_id="old", file_path="old.pdf", info=info)
    before = deepcopy(info)
    rendered, legacy = observation_view(field(document, "info"))
    assert legacy
    assert rendered["student_id"] == "old-P123"  # Display, not silent schema migration.
    rendered["source_pages"]["student_id"] = 99
    rendered["missing_fields"].append("display-only")
    assert info == before
    assert document.info is info


@pytest.mark.parametrize("value", [None, "old extraction failure", ["unexpected shape"]])
def test_malformed_legacy_observations_can_be_displayed(value):
    rendered, legacy = observation_view(value)
    assert legacy
    assert "legacy_value" in rendered


def test_model_and_dict_access_for_requirement_plan_and_issue():
    from src.core.state import Requirement, PlanStep, ValidationIssue
    records = [
        Requirement(rule_id="R006", description="Passport buffer", passed=None,
                    status="missing_information", message="Need expiry date"),
        PlanStep(step_id="ask", action="Provide dates", priority=1, reason="Missing information",
                 related_rule_id="R006", due_date=date(2026,11,20)),
        ValidationIssue(code="DATE_CONFLICT", message="Review dates", source="gate_1"),
    ]
    for model in records:
        raw = model.model_dump(mode="json")
        assert display_record(model) == display_record(raw)
        for key, value in raw.items():
            assert jsonable(field(model, key)) == field(raw, key)
    assert display_records("legacy error") == [{"legacy_value": "legacy error"}]
    assert display_records(None) == []


@pytest.mark.parametrize("internal,label", [
    ("VALIDATED", "Plan validated"), ("APPROVED", "Proposed plan approved"),
    ("HUMAN_REVIEW_REQUIRED", "HUMAN_REVIEW_REQUIRED"), ("FUTURE_STATUS", "FUTURE_STATUS"),
])
def test_status_labels_are_presentation_only(internal, label):
    state = {"run_status": internal}
    assert status_label(state["run_status"]) == label
    assert state["run_status"] == internal


def test_requirement_summary_does_not_modify_canonical_state():
    from copy import deepcopy
    from src.core.state import Requirement
    from streamlit_app import requirement_summary
    requirement = Requirement(rule_id="R006", description="Passport buffer", passed=True,
                              status="passed", message="Checked")
    requirements = [requirement, {"rule_id": "R017", "status": "failed"},
                    {"rule_id": "R015", "status": "missing_information"}]
    state = {"relevant_requirements": requirements, "blockers": ["R017", "R015"]}
    before = deepcopy(state)
    assert requirement_summary(requirements) == {"PASS": 1, "FAIL": 1, "missing_information": 1}
    assert state == before
    assert state["relevant_requirements"][0] is requirement
    assert requirement_summary(None) == {"PASS": 0, "FAIL": 0, "missing_information": 0}
