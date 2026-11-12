"""Temporary manual demonstration of the existing R004 document check."""

from src.state import State, Document
from src.rule_loader import load_rules
from src.requirement_engine import evaluate_requirements


if __name__ == "__main__":
    # Use canonical document types; address_proof is omitted.
    document_types = [
        "passport_copy",
        "student_certificate",
        "signed_online_application_form",
        "card_fee_receipt",
        "biometric_photo",
        "fingerprint_gi_evidence",
        "valid_health_insurance",
    ]
    state = State(
        user_goal="Obtain a residence permit",
        uploaded_documents=[
            Document(document_type=name, status="uploaded") for name in document_types
        ],
    )
    rule = next(rule for rule in load_rules() if rule["rule_id"] == "R004")
    result = next(
        result for result in evaluate_requirements(state)
        if result.rule_id == rule["rule_id"]
    )
    print(f"{result.rule_id}: status={result.status}, error_code={result.error_code}")
    print(result.message)
