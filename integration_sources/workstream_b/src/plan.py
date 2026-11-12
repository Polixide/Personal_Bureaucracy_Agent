"""Structured output models for proposed next actions."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class PlanAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1)  # Proposed action, never a completion claim.
    priority: int = Field(ge=1, strict=True)  # Lower numbers come first.
    reason: str = Field(min_length=1)  # Explanation grounded in supplied facts.
    related_rule_id: str | None = None  # Relevant requirement, if applicable.
    deadline: date | None = None  # Only a known deadline; otherwise unknown.


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[PlanAction]  # An empty list is allowed when no action is useful.
    summary: str  # Brief explanation of the proposed plan.
