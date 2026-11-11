"""Shared AgentState: the data contract between all LangGraph nodes.

The state has two parts:
  * Digital Twin - persistent facts about the student. They survive across
                   sessions once a checkpointer (SQLite) is attached.
  * Run memory   - working fields for the current event, reset by `new_run()`.

Pydantic models give every item an explicit, validated schema. The TypedDict is
what LangGraph uses to merge the partial updates returned by each node.
"""
from datetime import date
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel


# ------------------------------------------------------------------ schemas --
class UserEvent(BaseModel):
    """What triggers a run: a chat message or a document upload."""
    kind: Literal["message", "upload"]
    text: str = ""
    file_path: str | None = None


class Document(BaseModel):
    """A user document. Gate 0 registers it as 'pending'; the Document Analyzer fills the rest."""
    doc_id: str
    file_path: str
    status: Literal["pending", "analyzed", "rejected"] = "pending"
    doc_type: str | None = None          # e.g. "health_insurance", "student_certificate"
    issue_date: date | None = None
    expiry_date: date | None = None


class Requirement(BaseModel):
    """Result of applying one rule of the rule base to the current user state."""
    rule_id: str
    description: str
    satisfied: bool
    source_id: str | None = None         # provenance: where the rule comes from


class PlanStep(BaseModel):
    """One user-readable action of the candidate plan."""
    step_id: str
    action: str
    due_date: date | None = None
    depends_on: list[str] = []           # step_ids that must be completed first
    required_documents: list[str] = []   # doc_types that must be valid on due_date
    requires_approval: bool = False      # consequential action -> human confirmation


ErrorCode = Literal[
    "INVALID_INPUT",         # Gate 0: malformed or unsupported input
    "MISSING_PREREQUISITE",  # a step depends on something not done / not available
    "DATE_CONFLICT",         # e.g. a document expires before the step that needs it
    "SCHEMA_ERROR",          # a worker output does not match its schema
    "APPROVAL_DENIED",       # the user did not confirm the plan
]


class ValidationIssue(BaseModel):
    """Structured error: a machine-readable code plus an explanation, never free text only."""
    code: ErrorCode
    message: str
    source: str                          # node that raised it, e.g. "gate_1"
    item: str | None = None              # affected document / step id
    rule_id: str | None = None           # violated rule, for traceability


NextStep = Literal["document_analyzer", "rules_engine", "planner"]

# Schemas that may be stored in checkpoints. Pass them to the checkpointer's
# serializer, e.g. SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)).
CHECKPOINT_TYPES = [(__name__, cls.__name__)
                    for cls in (UserEvent, Document, Requirement, PlanStep, ValidationIssue)]


# ----------------------------------------------------------------- reducers --
def upsert_documents(current: list[Document], update: list[Document]) -> list[Document]:
    """Merge by doc_id: a node returns only new/changed documents and none is lost."""
    merged = {doc.doc_id: doc for doc in current or []}
    merged.update({doc.doc_id: doc for doc in update or []})
    return list(merged.values())


def merge_facts(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """New facts are added, existing keys are updated, nothing else is dropped."""
    return {**(current or {}), **(update or {})}


# -------------------------------------------------------------------- state --
class AgentState(TypedDict, total=False):
    # Digital Twin (persistent across sessions)
    user_id: str
    active_procedure: str                                     # e.g. "ikamet"
    documents: Annotated[list[Document], upsert_documents]
    extracted_facts: Annotated[dict[str, Any], merge_facts]   # e.g. {"appointment_date": date(...)}
    completed_steps: list[str]
    blockers: list[str]                                       # rule_ids still unsatisfied

    # Run memory (reset by new_run() at every new event)
    event: UserEvent
    trace_id: str
    next_step: NextStep                                       # last supervisor decision
    relevant_requirements: list[Requirement] | None           # None = not computed yet
    candidate_plan: list[PlanStep] | None                     # None = not computed yet
    validation_errors: list[ValidationIssue]                  # empty = PASS
    retry_count: int                                          # replan attempts (Milestone 2)
    final_response: str


def new_run(event: UserEvent, user_id: str, procedure: str = "ikamet") -> dict:
    """Input for one graph run.

    Only run-memory fields are reset, so the Digital Twin loaded from the
    checkpointer (documents, facts, completed steps, blockers) is preserved.
    """
    return {
        "user_id": user_id,
        "active_procedure": procedure,
        "event": event,
        "trace_id": uuid4().hex,
        "relevant_requirements": None,
        "candidate_plan": None,
        "validation_errors": [],
        "retry_count": 0,
        "final_response": "",
    }
