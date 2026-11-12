"""Deterministic approved-plan bookkeeping; never infer completed actions."""
from src.core.state import ActionPlan


def state_update(state):
    if state.get("approval_decision") is not True or state.get("validation_errors"):
        raise ValueError("State update requires an approved, validated plan")
    return {"approved_plan": ActionPlan(actions=state["candidate_plan"], summary=state.get("plan_summary", "")),
            "blockers": [r.rule_id for r in state.get("relevant_requirements") or [] if r.passed is not True and r.severity == "blocking"],
            "run_status": "APPROVED"}
