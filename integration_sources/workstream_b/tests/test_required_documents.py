"""Check the core document set using the categories in rules.json."""

import pytest

from src.requirement_engine import evaluate_requirements
from src.rule_loader import load_rules
from src.state import Document, State


REQUIRED_TYPES = next(
    rule["required_values"] for rule in load_rules() if rule["rule_id"] == "R004"
)


def document_result(document_types):
    state = State(
        user_goal="Obtain a residence permit",
        uploaded_documents=[
            Document(document_type=name, status="uploaded") for name in document_types
        ],
    )
    return next(result for result in evaluate_requirements(state) if result.rule_id == "R004")


def test_complete_document_set():
    result = document_result(REQUIRED_TYPES)
    assert result.passed is True
    assert result.status == "passed"
    assert result.error_code is None


@pytest.mark.parametrize("missing", [
    ["passport_copy"],
    ["passport_copy", "student_certificate", "valid_health_insurance"],
], ids=["one_missing", "multiple_missing"])
def test_missing_documents(missing):
    result = document_result([name for name in REQUIRED_TYPES if name not in missing])
    assert result.passed is False
    assert result.status == "failed"
    assert result.error_code == "MISSING_REQUIRED_DOCUMENTS"
    reported = result.message.removeprefix("Missing required documents: ").split(", ")
    assert set(reported) == set(missing)


def test_empty_document_list_is_unknown():
    result = document_result([])
    # UNKNOWN uses the engine's existing missing-information representation.
    assert result.passed is None
    assert result.status == "missing_information"
    assert result.error_code is None
    assert "available_document_types" in result.message
