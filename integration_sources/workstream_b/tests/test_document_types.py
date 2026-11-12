"""Canonical types keep passport checks and document-set checks consistent."""

from typing import get_args

import pytest
from pydantic import ValidationError

from src.requirement_engine import evaluate_requirements
from src.rule_loader import load_rules
from src.state import Document, DocumentType, ResidencePermitContext, State


def test_vocabulary_matches_rules():
    required = next(rule["required_values"] for rule in load_rules() if rule["rule_id"] == "R004")
    assert set(get_args(DocumentType)) == set(required)


@pytest.mark.parametrize("label", ["passport", "Passport", "passport_copy"])
def test_passport_labels_use_one_identifier(label):
    document = Document(document_type=label, status="uploaded", expiry_date="2027-08-01")
    assert document.document_type == "passport_copy"
    state = State(
        user_goal="Apply",
        uploaded_documents=[document],
        residence_permit_context=ResidencePermitContext(requested_permit_end_date="2027-06-01"),
    )
    results = {result.rule_id: result for result in evaluate_requirements(state)}
    assert results["R003"].passed is True
    assert "passport_copy" not in results["R004"].message


def test_health_insurance_alias():
    assert Document(document_type="Health insurance", status="uploaded").document_type == "valid_health_insurance"


def test_unrecognized_type_is_rejected():
    with pytest.raises(ValidationError):
        Document(document_type="invented_document", status="uploaded")
