"""Explicit approval using an interrupt; no external actions are executed."""
from langgraph.types import interrupt
from src.core.state import ActionPlan, ValidationIssue


def human_approval(state):
    decision = interrupt({"kind": "approve_plan", "plan": ActionPlan(actions=state["candidate_plan"], summary=state.get("plan_summary", "")).model_dump(mode="json"),
                          "notice": "Approve this proposed plan? Approval does not execute official actions."})
    if decision is not True:
        return {"approval_decision": False, "run_status":"APPROVAL_DENIED", "validation_errors":[ValidationIssue(code="APPROVAL_DENIED", source="human_approval", message="Plan was not approved")]}
    return {"approval_decision": True}
