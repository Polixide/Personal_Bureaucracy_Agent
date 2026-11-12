"""Run with: python -m src.demo_planner (requires local Ollama)."""

from src.action_planner import PlannerError, create_action_plan
from src.requirement_engine import evaluate_requirements
from src.rule_loader import load_rules
from src.state import Document, ResidencePermitContext, State


if __name__ == "__main__":
    required = next(rule["required_values"] for rule in load_rules() if rule["rule_id"] == "R004")
    state = State(
        user_goal="Obtain a residence permit",
        current_procedure="First residence-permit application",
        uploaded_documents=[
            Document(document_type=name, status="uploaded",
                     expiry_date="2027-07-15" if name == "passport_copy" else None)
            for name in required if name != "address_proof"
        ],
        residence_permit_context=ResidencePermitContext(
            is_in_turkiye=True, entered_turkiye_with_passport=True,
            application_date="2026-11-01", legal_stay_expiry_date="2026-11-15",
            requested_permit_end_date="2027-06-01",
            stay_start_date="2026-09-20", stay_end_date="2027-06-01",
            address_proof_type="airbnb_booking",
            online_application_completed=True, application_form_signed=True,
            fingerprint_completed=True, gi_number_present=True,
            insurance_valid_for_residence_permit=True,
            insurance_start_date="2026-09-15", insurance_end_date="2027-06-30",
        ),
    )
    results = evaluate_requirements(state)
    print("Requirement results:", flush=True)
    for result in results:
        print(result.model_dump_json(), flush=True)
    try:
        plan = create_action_plan(state, results)
    except PlannerError as exc:
        raise SystemExit(f"Planner error: {exc}") from exc
    print("Generated structured plan:")
    print(plan.model_dump_json(indent=2))
