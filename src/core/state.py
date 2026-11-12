"""Shared AgentState: the data contract between all LangGraph nodes.

The state has two parts:
  * Digital Twin - persistent facts about the student. They survive across
                   sessions when a caller supplies a checkpointer.
  * Run memory   - working fields for the current event, reset by `new_run()`.

Pydantic models give every item an explicit, validated schema. The TypedDict is
what LangGraph uses to merge the partial updates returned by each node.
"""
from datetime import date
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.document_processing.schemas import (
    DocumentInfo, ExtractedDocument, StudentCertificate, HealthInsurance,
    IkametApplicationForm, UnknownDocument,
)


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
    doc_type: Literal["declaration_control_document", "address_proof", "passport_copy",
                      "biometric_photo", "student_certificate", "signed_online_application_form",
                      "card_fee_receipt", "fingerprint_gi_evidence", "health_insurance",
                      "ikamet_application_form", "unknown"] | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    info: ExtractedDocument | None = None
    processing_error: str | None = None

    @field_validator("doc_type", mode="before")
    @classmethod
    def normalize_type(cls, value):
        aliases = {"passport": "passport_copy", "health insurance": "health_insurance",
                   "valid_health_insurance": "health_insurance",
                   "health_insurance_certificate": "health_insurance",
                   "student_enrollment_certificate": "student_certificate"}
        if isinstance(value, str):
            value = value.strip().lower()
            return aliases.get(value, value)
        return value


class Requirement(BaseModel):
    rule_id: str
    description: str
    passed: bool | None
    status: Literal["passed", "failed", "missing_information"]
    message: str
    error_code: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    severity: str = "blocking"
    missing_fields: list[str] = Field(default_factory=list)

    @property
    def satisfied(self):
        return self.passed is True


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    priority: int = Field(ge=1, strict=True)
    reason: str = Field(min_length=1)
    related_rule_id: str | None = None
    due_date: date | None = None
    deadline_key: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    # All proposals pass explicit human approval; a model cannot waive it.
    requires_approval: bool = True
    action_type: Literal["information", "prepare", "official_application", "payment",
                         "iro_preverification", "fingerprint", "iro_submission"] = "prepare"


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[PlanStep]
    summary: str


class ValidationIssue(BaseModel):
    code: str
    message: str
    source: str
    item: str | None = None
    rule_id: str | None = None


NextStep = Literal["document_analyzer", "rules_engine", "planner"]

class ResidencePermitContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exchange_student: bool | None = None
    is_in_turkiye: bool | None = None
    entered_with_passport: bool | None = None
    application_form_generated: bool | None = None
    application_form_signed: bool | None = None
    passport_id_page_present: bool | None = None
    turkey_entry_page_present: bool | None = None
    visa_required: bool | None = None
    visa_page_present: bool | None = None
    biometric_photo: bool | None = None
    icao_compliant: bool | None = None
    fee_payment_required: bool | None = None
    fee_receipt_present: bool | None = None
    single_entry_visa_fee_required: bool | None = None
    single_entry_visa_fee_receipt_present: bool | None = None
    financial_capacity_declared: bool | None = None
    exchange_program_evidence_present: bool | None = None
    student_certificate_present: bool | None = None
    insurance_exempt: bool | None = None
    insurance_valid_for_ikamet: bool | None = None
    turkey_address_present: bool | None = None
    phone_present: bool | None = None
    email_present: bool | None = None
    declaration_control_document_present: bool | None = None
    declaration_control_document_signed: bool | None = None
    all_core_documents_present: bool | None = None
    scans_sent_to_iro: bool | None = None
    iro_preverification_confirmed: bool | None = None
    online_application_completed: bool | None = None
    fingerprint_step_completed: bool | None = None
    gi_number_present: bool | None = None
    submitted_to_iro: bool | None = None
    application_date: date | None = None
    legal_stay_expiry_date: date | None = None
    passport_expiry_date: date | None = None
    requested_permit_end_date: date | None = None
    insurance_start_date: date | None = None
    insurance_end_date: date | None = None
    requested_start_date: date | None = None
    requested_end_date: date | None = None
    current_date: date | None = None
    iro_window_start: date | None = None
    iro_window_end: date | None = None
    institution: str | None = None
    applicant_type: str | None = None
    permit_type: str | None = None
    application_kind: str | None = None
    application_channel: str | None = None
    insurance_evidence_type: str | None = None
    address_proof_type: str | None = None
    application_form_window_id: str | None = None
    current_iro_window_id: str | None = None
    approval_status: str | None = None
    decision_authority: str | None = None
    tracking_channel: str | None = None
    age: int | None = None
    planned_stay_months: int | None = None
    photo_age_months: int | None = None
    missing_required_documents_count: int | None = None


# Pydantic classes stored by the in-memory and SQLite checkpointers.
CHECKPOINT_TYPES = [(cls.__module__, cls.__name__) for cls in
                    (UserEvent, Document, DocumentInfo, StudentCertificate, HealthInsurance,
                     IkametApplicationForm, UnknownDocument, Requirement, PlanStep,
                     ActionPlan, ValidationIssue, ResidencePermitContext)]


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
    residence_permit_context: ResidencePermitContext
    deadlines: dict[str, date]
    approved_plan: ActionPlan | None
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
    retry_count: int                                          # repair attempts in this run
    final_response: str
    plan_summary: str
    repair_feedback: list[ValidationIssue]
    repair_history: list[dict[str, Any]]
    run_status: str
    approval_decision: bool | None


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
        "plan_summary": "",
        "repair_feedback": [],
        "repair_history": [],
        "run_status": "RUNNING",
        "approval_decision": None,
    }
