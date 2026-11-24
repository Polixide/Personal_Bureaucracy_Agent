"""Pydantic records stored in the Digital Twin (one model per table)."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models import FactValue, Issue

Provenance = Literal["document", "user_stated", "tool", "derived"]
DocumentStatus = Literal["pending", "analyzed", "rejected", "deleted"]
Direction = Literal["real_to_twin", "twin_to_real", "internal"]
ActionStatus = Literal["proposed", "executed", "denied", "blocked"]

# source_ref values that are not document ids
CHAT_REF = "chat"            # a statement the agent recorded from the conversation
APP_REF = "app"              # a value typed or confirmed in the app
RESOLUTION_REF = "resolution"  # the user's choice between conflicting values (wins over all)
OUTCOME_PREFIX = "outcome:"  # a result the user reported from the real world


class Fact(BaseModel):
    """One value for one field, with where it came from and when it was valid."""
    fact_id: int | None = None
    user_id: str
    procedure_id: str
    field: str
    value: FactValue | None
    provenance: Provenance
    source_ref: str
    page: int | None = None
    snippet: str | None = None
    confirmed: bool
    note: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class TwinDocument(BaseModel):
    """An uploaded document and what extraction observed in it."""
    doc_id: str
    file_path: str
    status: DocumentStatus
    doc_type: str | None = None
    observation: dict[str, Any] | None = None
    processing_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Event(BaseModel):
    """Something that crossed the real/twin boundary, or an internal agent event."""
    event_id: int | None = None
    user_id: str
    procedure_id: str | None = None
    kind: str
    direction: Direction
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ProposedAction(BaseModel):
    """An artifact the agent prepared. Nothing leaves the twin before the user approves it."""
    action_id: str
    user_id: str
    procedure_id: str
    tool: str
    title: str
    payload: dict[str, Any] = Field(default_factory=dict)
    refs: list[str] = Field(default_factory=list)
    requires_stage: str | None = None
    status: ActionStatus = "proposed"
    artifact_path: str | None = None
    note: str = ""
    created_at: datetime | None = None
    decided_at: datetime | None = None


class Simulation(BaseModel):
    """One mock-world attempt of a stage."""
    simulation_id: int | None = None
    user_id: str
    procedure_id: str
    stage_id: str
    accepted: bool
    facts_hash: str
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class Episode(BaseModel):
    """A validated repair lesson: what Gate 1 rejected and what fixed it. No chain-of-thought."""
    episode_id: int | None = None
    user_id: str
    procedure_id: str
    failure_type: str
    validator_error: str
    corrective_action: str
    outcome: str
    correction_validated: bool
    affected_item: str | None = None
    rule_or_source_id: str | None = None
    rule_version: str | None = None
    trace_id: str | None = None
    created_at: datetime | None = None


class RunRuleResult(BaseModel):
    """One rule verdict saved with a run."""
    rule_id: str
    status: Literal["pass", "fail", "unknown"]
    severity: str
    missing_fields: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """Summary of one agent run (used by evaluation)."""
    trace_id: str
    user_id: str
    procedure_id: str | None
    goal: str
    status: str
    llm_calls: int = 0
    steps: int = 0
    repairs: int = 0
    fallback_used: bool = False
    needs_human_review: bool = False
    summary: str = ""
    started_at: datetime
    finished_at: datetime
    issues: list[Issue] = Field(default_factory=list)
    results: list[RunRuleResult] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """One message of the conversation with the assistant. `details`: sources, recorded facts, run status."""
    user_id: str
    role: Literal["user", "assistant"]
    content: str
    procedure_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    message_id: int | None = None
    created_at: datetime | None = None
