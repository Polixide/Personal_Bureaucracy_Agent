"""Explicit structural and English phrase checks, not semantic verification."""

import re

from pydantic import BaseModel, Field

from src.plan import ActionPlan
from src.requirement_engine import RuleResult
from src.rule_loader import load_rules
from src.state import State


class PlanValidationError(BaseModel):
    code: str
    message: str
    action_index: int | None = None  # Zero-based index; None refers to the plan.
    related_rule_id: str | None = None


class PlanValidationResult(BaseModel):
    valid: bool
    errors: list[PlanValidationError] = Field(default_factory=list)


# These intentionally small patterns define exactly what the text checks detect.
FAILURE_CLAIM = r"\b(failed|invalid|insufficient|not satisfied|not met|noncompliant)\b"
COMPLETION_CLAIM = (
    r"\b(?:already|successfully)\s+(?:been\s+)?(?:submitted|completed|paid|booked|renewed|uploaded)\b"
    r"|\b(?:has|have)\s+(?:been\s+)?(?:submitted|completed|paid|booked|renewed|uploaded)\b"
    r"|\b(?:was|were|is|are)\s+(?:submitted|completed|paid|booked|renewed|uploaded)\b"
    r"|\b(?:i|we)\s+(?:submitted|completed|paid|booked|renewed|uploaded)\b"
)
FINAL_ACTION = (
    r"\bsubmit\b.{0,60}\b(application|file|document set|documents|submission)\b"
    r"|\b(?:complete|finalize|finish)\b.{0,40}\b(application|procedure|process|submission)\b"
)


def validate_plan(state: State, results: list[RuleResult], plan: ActionPlan) -> PlanValidationResult:
    """Check references, blocker coverage, wording, deadlines, and ordering.

    Each failed rule needs its own action reference. All current failed rules
    block final submission; proposing a correction does not resolve the failure.
    Text checks are conservative phrase matching and cannot prove factual truth.
    """
    known_ids = {rule["rule_id"] for rule in load_rules()}
    failed = {result.rule_id for result in results if result.status == "failed"}
    unknown = {result.rule_id for result in results if result.status == "missing_information"}
    errors = []

    def add(code, message, index=None, rule_id=None):
        errors.append(PlanValidationError(
            code=code, message=message, action_index=index, related_rule_id=rule_id
        ))

    # Only explicit deadlines and the known appointment/legal-stay limit are due dates.
    known_deadlines = set(state.deadlines.values())
    known_deadlines.update(day for day in (
        state.appointment_date, state.residence_permit_context.legal_stay_expiry_date
    ) if day is not None)
    priorities = []
    for index, action in enumerate(plan.actions):
        rule_id = action.related_rule_id
        if rule_id is not None and rule_id not in known_ids:
            add("UNKNOWN_RULE_ID", "Use an existing rule ID.", index, rule_id)
        text = action.action + " " + action.reason
        if rule_id in unknown and re.search(FAILURE_CLAIM, text, re.I):
            add("UNKNOWN_AS_FAILURE", "Request missing information; do not assert failure.", index, rule_id)
        if re.search(COMPLETION_CLAIM, text, re.I):
            add("COMPLETION_CLAIM", "Propose actions without claiming external execution.", index, rule_id)
        if failed and re.search(FINAL_ACTION, action.action, re.I):
            add("PREMATURE_SUBMISSION", "Resolve and recheck failed requirements before final submission.", index, rule_id)
        if action.deadline is not None and action.deadline not in known_deadlines:
            add("UNSUPPORTED_DEADLINE", "Use an explicit State deadline or null.", index, rule_id)
        # Also protect against mutated or model_construct-created objects.
        if type(action.priority) is not int or action.priority <= 0:
            add("INVALID_PRIORITY", "Priority must be a positive integer.", index, rule_id)
        else:
            priorities.append(action.priority)

    if priorities != sorted(set(priorities)):
        add("AMBIGUOUS_ORDER", "List actions in increasing order with unique priorities.")
    covered = {action.related_rule_id for action in plan.actions}
    for rule_id in sorted(failed - covered):
        add("FAILED_RULE_IGNORED", "Include a corrective action for this failed requirement.", rule_id=rule_id)
    if re.search(COMPLETION_CLAIM, plan.summary, re.I):
        add("COMPLETION_CLAIM", "The summary must not claim external execution.")
    # A summary can explicitly assert a rule failed even without an action link.
    for rule_id in sorted(unknown):
        if re.search(rf"\b{re.escape(rule_id)}\b\s+(?:is\s+|has\s+)?{FAILURE_CLAIM}", plan.summary, re.I):
            add("UNKNOWN_AS_FAILURE", "The summary must preserve UNKNOWN status.", rule_id=rule_id)
    return PlanValidationResult(valid=not errors, errors=errors)
