"""Exercise the rules using the project's synthetic examples."""

import json
from pathlib import Path

import pytest

from src.requirement_engine import evaluate_requirements
from src.state import Document, ResidencePermitContext, State


CASES_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "test_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))["tests"]


def result_for(rule_id, inputs):
    """Move synthetic passport expiry into an uploaded Passport document."""
    values = dict(inputs)
    passport_expiry = values.pop("passport_expiry_date", None)
    documents = []
    if passport_expiry is not None:
        documents.append(Document(
            document_type="passport_copy", status="uploaded", expiry_date=passport_expiry
        ))
    state = State(
        user_goal="Obtain a residence permit",
        residence_permit_context=ResidencePermitContext(**values),
        uploaded_documents=documents,
    )
    return next(r for r in evaluate_requirements(state) if r.rule_id == rule_id)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["test_id"])
def test_synthetic_case(case):
    result = result_for(case["rule_id"], case["input"])
    assert result.passed is (case["expected"] == "PASS")
    assert result.status == ("passed" if result.passed else "failed")
    assert result.error_code == case.get("expected_error")


@pytest.mark.parametrize("rule_id", ["R001", "R002", "R003", "R005", "R006", "R007", "R008"])
def test_missing_information(rule_id):
    result = result_for(rule_id, {})
    assert result.passed is None
    assert result.status == "missing_information"
    assert result.error_code is None
    assert "missing information" in result.message


@pytest.mark.parametrize("rule_id, inputs, expected", [
    ("R001", {"is_in_turkiye": True, "entered_turkiye_with_passport": True}, True),
    ("R001", {"is_in_turkiye": False, "entered_turkiye_with_passport": True}, False),
    ("R006", {"online_application_completed": True, "application_form_signed": True}, True),
    ("R006", {"online_application_completed": True, "application_form_signed": False}, False),
    ("R002", {"application_date": "2026-11-15", "legal_stay_expiry_date": "2026-11-15"}, True),
    ("R003", {"requested_permit_end_date": "2027-06-01", "passport_expiry_date": "2027-07-31"}, True),
    ("R003", {"requested_permit_end_date": "2027-06-01", "passport_expiry_date": "2027-07-30"}, False),
])
def test_boolean_checks_and_date_boundaries(rule_id, inputs, expected):
    assert result_for(rule_id, inputs).passed is expected
