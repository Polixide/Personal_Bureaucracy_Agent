"""Repair a proposed plan; only the deterministic validator can accept it."""

import json
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

from src import action_planner
from src.action_planner import PlannerError
from src.plan import ActionPlan
from src.plan_validator import PlanValidationError, PlanValidationResult, validate_plan
from src.requirement_engine import RuleResult
from src.rule_loader import load_rules
from src.state import State


MAX_REFLECTION_ATTEMPTS = 3


def reflect_plan(state: State, results: list[RuleResult], plan: ActionPlan,
                 validation: PlanValidationResult) -> ActionPlan:
    """Return a structured candidate, never a declaration of validity."""
    payload = {
        "state": state.model_dump(mode="json"),
        "requirement_results": [result.model_dump(mode="json") for result in results],
        "failed_plan": plan.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "rules": [{key: rule[key] for key in
                   ("rule_id", "description", "allowed_values", "required_values") if key in rule}
                  for rule in load_rules()],
        "output_schema": ActionPlan.model_json_schema(),
    }
    instructions = action_planner.SYSTEM_PROMPT + """
Repair only the supplied plan using the validator errors. Preserve valid actions
when possible. Do not invent rules. Include a separate corrective action linked
to each failed rule ID. UNKNOWN requires information gathering, not a failure claim.
Remove final submission actions while any failures remain. Do not claim external
actions were completed. Use unique positive priorities in increasing order.
Use only explicit State deadlines (or null), not a requested permit end date.
Apply each validator error code and message supplied in the user message:
- UNSUPPORTED_DEADLINE: set the action deadline to null unless that exact deadline
  exists as an explicit deadline in State (including appointment/legal-stay limit).
- PREMATURE_SUBMISSION: remove or replace final submission/completion actions
  while failed blocking requirements remain. Proposing fixes does not resolve them.
- Preserve actions that are already valid whenever possible.
- Do not return the original plan unchanged when validator errors are present.

Short repair example:
Invalid: "Submit documents" while blocking requirements are failed.
Corrected: "Re-check document completeness after blockers are resolved."

Return only the corrected ActionPlan JSON. Never declare your correction valid:
the deterministic validator will decide whether to accept it.
"""
    # Repeat every code/message explicitly next to the repair request.
    error_instructions = "\n".join(
        f"- {error.code}: {error.message} "
        f"(action_index={error.action_index}, related_rule_id={error.related_rule_id})"
        for error in validation.errors
    )
    user_message = (
        json.dumps(payload) + "\n\nValidator errors to repair:\n" + error_instructions
    )
    try:
        response = ollama.chat(
            model=action_planner.MODEL_NAME,
            messages=[{"role": "system", "content": instructions},
                      {"role": "user", "content": user_message}],
            format=ActionPlan.model_json_schema(), options={"temperature": 0}, think=False,
        )
    except (ollama.ResponseError, ConnectionError) as exc:
        raise PlannerError(f"Ollama reflection request failed: {exc}") from exc
    try:
        return ActionPlan.model_validate_json(response.message.content)
    except (ValidationError, TypeError, AttributeError) as exc:
        raise PlannerError("Ollama returned invalid reflection ActionPlan output.") from exc


class PlanReviewStep(BaseModel):
    plan: ActionPlan
    validation: PlanValidationResult


class PlanningOutcome(BaseModel):
    status: Literal["VALID", "HUMAN_REVIEW_REQUIRED"]
    plan: ActionPlan | None
    validation: PlanValidationResult
    attempts: int  # Reflection calls in this run, excluding initial planning.
    history: list[PlanReviewStep] = Field(default_factory=list)


def plan_with_reflection(state: State, results: list[RuleResult]) -> PlanningOutcome:
    """Plan, validate, and attempt at most three repairs; execute no actions."""
    try:
        plan = action_planner.create_action_plan(state, results)
    except PlannerError as exc:
        validation = PlanValidationResult(valid=False, errors=[
            PlanValidationError(code="PLANNER_ERROR", message=str(exc))
        ])
        return PlanningOutcome(status="HUMAN_REVIEW_REQUIRED", plan=None,
                               validation=validation, attempts=0)
    validation = validate_plan(state, results, plan)
    history = [PlanReviewStep(plan=plan, validation=validation)]
    attempts = 0
    while not validation.valid and attempts < MAX_REFLECTION_ATTEMPTS:
        attempts += 1
        state.iteration_count += 1  # Count attempted repairs, including invalid output.
        try:
            candidate = reflect_plan(state, results, plan, validation)
        except PlannerError as exc:
            validation = PlanValidationResult(valid=False, errors=validation.errors + [
                PlanValidationError(code="REFLECTION_ERROR", message=str(exc))
            ])
        else:
            plan = candidate
            validation = validate_plan(state, results, plan)
        history.append(PlanReviewStep(plan=plan, validation=validation))
    return PlanningOutcome(
        status="VALID" if validation.valid else "HUMAN_REVIEW_REQUIRED",
        plan=plan, validation=validation, attempts=attempts, history=history,
    )
