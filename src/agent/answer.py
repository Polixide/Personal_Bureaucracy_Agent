"""What the agent produces: decisions, worker findings, the final answer and the run result."""
from typing import Any, Literal

from pydantic import BaseModel, Field

from config import MAX_NEXT_ACTIONS
from src.models import Issue, RuleResult, RuleStatus

FINAL_ACTION = "final_answer"   # orchestrator's terminal action
FINISH_ACTION = "finish"        # worker's terminal action


class Decision(BaseModel):
    """One ReAct step: a one-sentence thought, an action (tool name or terminal action), its arguments."""
    thought: str = Field(default="", max_length=400)
    action: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


def decision_schema(actions: list[str]) -> dict:
    """JSON schema for Decision with `action` restricted to the allowed names (sent to Ollama)."""
    schema = Decision.model_json_schema()
    schema["properties"]["action"]["enum"] = actions
    return schema


class Claim(BaseModel):
    """A statement about one rule. Gate 1 checks it against the latest evaluation."""
    rule_id: str
    status: RuleStatus


class NextAction(BaseModel):
    """One concrete step for the user. `ref` must be a rule id, a finding id or a proposal id."""
    title: str = Field(min_length=3, max_length=240)
    ref: str = Field(min_length=1)


class FinalAnswer(BaseModel):
    """The orchestrator's answer to the user."""
    summary: str = Field(min_length=1, max_length=2000)
    claims: list[Claim] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list, max_length=MAX_NEXT_ACTIONS)


class WorkerFinish(BaseModel):
    """Arguments of the worker's `finish` action (the model's part of a finding)."""
    summary: str = Field(default="", max_length=800)
    suspected_mistakes: list[str] = Field(default_factory=list, max_length=5)


class MissingItem(BaseModel):
    """Information still needed, with the question to ask and the rules waiting for it."""
    field: str
    question: str
    rule_ids: list[str]


class Finding(BaseModel):
    """A worker's structured result for one section. Rule results always come from code."""
    finding_id: str
    section_id: str
    summary: str
    rule_results: list[RuleResult]
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    missing_items: list[MissingItem] = Field(default_factory=list)
    suspected_mistakes: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    steps: int = 0
    fallback_used: bool = False


class StepRecord(BaseModel):
    """One orchestrator step as it happened (for the UI and evaluation)."""
    step: int
    thought: str = ""
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error: str | None = None


AgentStatus = Literal["answered", "needs_user_input", "needs_human_review", "fallback"]


class AgentResult(BaseModel):
    """Everything about one run of the agent."""
    trace_id: str
    status: AgentStatus
    procedure_id: str | None = None
    answer: FinalAnswer | None = None
    question: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    gate_issues: list[Issue] = Field(default_factory=list)
    first_answer_rejected: bool = False
    steps: list[StepRecord] = Field(default_factory=list)
    llm_calls: int = 0
    repairs: int = 0
    fallback_used: bool = False
    needs_human_review: bool = False
    proposals: list[str] = Field(default_factory=list)
    loops_detected: int = 0
    invalid_calls: int = 0
