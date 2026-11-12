"""Manual end-to-end run: python -m src.demo_reflection (local Ollama)."""

from unittest.mock import patch

from src import reflection
from src.plan import ActionPlan
from src.reflection import plan_with_reflection
from src.requirement_engine import evaluate_requirements
from src.rule_loader import load_rules
from src.state import Document, ResidencePermitContext, State


def run_with_debug(state, results):
    """Temporarily trace actual model calls/parsing only during this manual run."""
    original_reflect = reflection.reflect_plan
    original_chat = reflection.ollama.chat
    original_parse = ActionPlan.model_validate_json
    attempt = 0

    def debug_chat(*args, **kwargs):
        response = original_chat(*args, **kwargs)
        print("DEBUG raw Ollama response (before parsing):", flush=True)
        print(response.message.content, flush=True)
        return response

    def debug_parse(*args, **kwargs):
        try:
            parsed = original_parse(*args, **kwargs)
        except Exception as exc:
            print("DEBUG parsing FAILED:", flush=True)
            print(f"{type(exc).__name__}: {exc}", flush=True)
            raise  # Preserve the normal error handling; do not accept bad output.
        print("DEBUG parsing SUCCEEDED", flush=True)
        return parsed

    def debug_reflect(state, results, plan, validation):
        nonlocal attempt
        attempt += 1
        print(f"\nDEBUG reflection attempt {attempt}", flush=True)
        print("DEBUG exact validator errors sent to Reflection:", flush=True)
        for error in validation.errors:
            print(error.model_dump_json(), flush=True)
        # Wrappers call the real implementation and do not change its outputs.
        with patch.object(reflection.ollama, "chat", side_effect=debug_chat), \
             patch.object(ActionPlan, "model_validate_json", side_effect=debug_parse):
            return original_reflect(state, results, plan, validation)

    with patch.object(reflection, "reflect_plan", side_effect=debug_reflect):
        return plan_with_reflection(state, results)


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
    print("1. Requirement results", flush=True)
    for result in results:
        print(result.model_dump_json(), flush=True)
    print("Calling local planner and validating (up to three repairs)...", flush=True)
    outcome = run_with_debug(state, results)
    if outcome.history:
        print("2. Original plan")
        print(outcome.history[0].plan.model_dump_json(indent=2))
        print("3. Original validator result")
        print(outcome.history[0].validation.model_dump_json(indent=2))
        for attempt, step in enumerate(outcome.history[1:], start=1):
            print(f"4. Repair attempt {attempt} (last candidate if model output was invalid)")
            print(step.plan.model_dump_json(indent=2))
            print(step.validation.model_dump_json(indent=2))
    else:
        print("2–4. No plan available: initial planner request failed.")
    print("5. Final validation result")
    print(outcome.validation.model_dump_json(indent=2))
    print(f"Final status: {outcome.status}")
    print(f"6. Reflection attempts: {outcome.attempts}")
