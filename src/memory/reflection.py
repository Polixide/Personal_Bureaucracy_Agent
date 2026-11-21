"""Record grounded repair context; the planner makes the next model call."""
from config import MAX_REFLECTION_ATTEMPTS
from src.core.llm_diagnostics import log_reflection, plan_hash


def repair_partition(state, feedback=None):
    if feedback is None:
        feedback=state.get("repair_feedback") or state.get("validation_errors") or []
    steps=state.get("candidate_plan") or []
    global_error=any(e.item is None and e.code not in {"FAILED_RULE_IGNORED","PLANNER_SCHEMA_ERROR"} for e in feedback)
    rejected={e.item for e in feedback if e.item}
    if global_error:rejected={a.step_id for a in steps}
    return [a.model_copy(deep=True) for a in steps if a.step_id not in rejected], [a.step_id for a in steps if a.step_id in rejected]


def reflection(state):
    attempts = state.get("retry_count", 0)
    if attempts >= MAX_REFLECTION_ATTEMPTS:
        return {"run_status": "HUMAN_REVIEW_REQUIRED"}
    feedback = list(state.get("validation_errors", []))
    from src.memory.twin_sync import relevant_lessons
    lessons=relevant_lessons(state,feedback)
    valid,rejected=repair_partition(state, feedback)
    previous_hash = state.get("last_model_plan_hash") or plan_hash(state.get("candidate_plan") or [])
    log_reflection(attempts + 1, feedback, previous_plan_hash=previous_hash)
    record = {"attempt": attempts + 1, "errors": [e.model_dump() for e in feedback],
              "plan": [a.model_dump(mode="json") for a in state.get("candidate_plan") or []],
              "rejected_model_plan": state.get("rejected_model_plan"), "previous_plan_hash": previous_hash}
    return {"repair_lessons":lessons,"repair_valid_actions":valid,"repair_rejected_action_ids":rejected,"retry_count": attempts + 1, "repair_feedback": feedback,
            "repair_history": [*state.get("repair_history", []), record], "run_status": "REPAIRING"}
