"""Record grounded repair context; the planner makes the next model call."""
from config import MAX_REFLECTION_ATTEMPTS


def reflection(state):
    attempts = state.get("retry_count", 0)
    if attempts >= MAX_REFLECTION_ATTEMPTS:
        return {"run_status": "HUMAN_REVIEW_REQUIRED"}
    feedback = list(state.get("validation_errors", []))
    record = {"attempt": attempts + 1, "errors": [e.model_dump() for e in feedback],
              "plan": [a.model_dump(mode="json") for a in state.get("candidate_plan") or []]}
    return {"retry_count": attempts + 1, "repair_feedback": feedback,
            "repair_history": [*state.get("repair_history", []), record], "run_status": "REPAIRING"}
