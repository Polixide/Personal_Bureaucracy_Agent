"""Evaluate the prototype rules using explicit Python checks."""

from typing import Literal

from pydantic import BaseModel

from src.rule_loader import load_rules
from src.state import State


class RuleResult(BaseModel):
    """A missing input is distinct from a failed requirement."""

    rule_id: str
    passed: bool | None
    status: Literal["passed", "failed", "missing_information"]
    error_code: str | None
    message: str


def evaluate_requirements(state: State) -> list[RuleResult]:
    """Check each supported rule independently using known information."""
    context = state.residence_permit_context
    inputs = context.model_dump()
    passport = next(
        (doc for doc in state.uploaded_documents if doc.document_type == "passport_copy"),
        None,
    )
    inputs["passport_expiry_date"] = passport.expiry_date if passport else None
    # An empty list cannot distinguish unknown documents from none supplied yet.
    inputs["available_document_types"] = (
        {doc.document_type for doc in state.uploaded_documents}
        if state.uploaded_documents else None
    )

    # These checks implement the dataset conditions; strings are never executed.
    checks = {
        "R001": lambda rule: context.is_in_turkiye and context.entered_turkiye_with_passport,
        "R002": lambda rule: context.application_date <= context.legal_stay_expiry_date,
        "R003": lambda rule: (
            inputs["passport_expiry_date"] - context.requested_permit_end_date
        ).days >= 60,
        "R004": lambda rule: all(
            name in inputs["available_document_types"] for name in rule["required_values"]
        ),
        "R005": lambda rule: context.address_proof_type in rule["allowed_values"],
        "R006": lambda rule: context.online_application_completed and context.application_form_signed,
        "R007": lambda rule: context.fingerprint_completed and context.gi_number_present,
        "R008": lambda rule: (
            context.insurance_valid_for_residence_permit
            and context.insurance_start_date <= context.stay_start_date
            and context.insurance_end_date >= context.stay_end_date
        ),
    }

    results = []
    for rule in load_rules():
        rule_id = rule["rule_id"]
        if rule_id not in checks:
            continue
        # Only None means unknown; False is a known negative answer.
        missing = [name for name in rule["inputs"] if inputs.get(name) is None]
        if missing:
            results.append(RuleResult(
                rule_id=rule_id,
                passed=None,
                status="missing_information",
                error_code=None,
                message="Cannot evaluate; missing information: " + ", ".join(missing),
            ))
            continue

        passed = bool(checks[rule_id](rule))
        message = rule["pass_result"] if passed else rule["description"]
        if rule_id == "R004" and not passed:
            missing_documents = [
                name for name in rule["required_values"]
                if name not in inputs["available_document_types"]
            ]
            message = "Missing required documents: " + ", ".join(missing_documents)
        results.append(RuleResult(
            rule_id=rule_id,
            passed=passed,
            status="passed" if passed else "failed",
            error_code=None if passed else rule["fail_code"],
            message=message,
        ))
    return results
