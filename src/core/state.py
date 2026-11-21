"""Shared AgentState: the data contract between all LangGraph nodes.

The state has two parts:
  * Digital Twin - persistent facts about the student. They survive across
                   sessions when a caller supplies a checkpointer.
  * Run memory   - working fields for the current event, reset by `new_run()`.

Pydantic models give every item an explicit, validated schema. The TypedDict is
what LangGraph uses to merge the partial updates returned by each node.
"""
from datetime import date, datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4
from unicodedata import normalize

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.document_processing.schemas import (
    DocumentInfo, ExtractedDocument, StudentCertificate, HealthInsurance,
    IkametApplicationForm, UnknownDocument, PassportIdentity, EntryStamp, DeclarationControl, CardFeeReceipt, UtilityBill, AddressForm,
)


# ------------------------------------------------------------------ schemas --
class UserEvent(BaseModel):
    """What triggers a run: a chat message or a document upload."""
    kind: Literal["message", "upload", "analyze_document", "confirm_information", "generate_plan", "resolve_conflict", "record_progress", "reevaluate", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction", "assistant_message", "run_tool"]
    text: str = ""
    file_path: str | None = None
    document_id: str | None = None
    confirmed: bool = False
    document_type_hint: str | None = None
    action_id: str | None = None
    tool_call: dict[str, Any] | None = None
    original_filename: str | None = None
    confirmed_fields: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A user document. Gate 0 registers it as 'pending'; the Document Analyzer fills the rest."""
    doc_id: str
    file_path: str
    status: Literal["pending", "analyzed", "rejected", "replacement_pending", "deleted"] = "pending"
    doc_type: Literal["declaration_control_document", "address_proof", "passport_copy",
                      "biometric_photo", "student_certificate", "signed_online_application_form",
                      "card_fee_receipt", "fingerprint_gi_evidence", "health_insurance",
                      "ikamet_application_form", "turkiye_entry_stamp", "utility_bill", "address_form", "unknown"] | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    info: ExtractedDocument | None = None
    processing_error: str | None = None
    original_filename: str | None = None

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


class VerifiedInput(BaseModel):
    """Accepted input, kept separate from immutable extraction observations."""
    value: str | bool | int | None
    origin: Literal["source_grounded", "user_confirmed", "derived"]
    document_id: str
    field_name: str
    source_page: int | None = None
    source_snippet: str | None = None
    confirmed_by: str | None = None
    trace_id: str
    derived_from: list[str] = Field(default_factory=list)
    derivation_policy: str | None = None


class Requirement(BaseModel):
    rule_id: str
    description: str
    passed: bool | None
    status: Literal["passed", "failed", "missing_information"]
    message: str
    error_code: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    severity: str = "blocking"
    assessment_policy: str | None = None
    missing_fields: list[str] = Field(default_factory=list)

    @property
    def satisfied(self):
        return self.passed is True


PlanActionType = Literal["REQUEST_DOCUMENT","REQUEST_USER_CONFIRMATION","CORRECT_INFORMATION","REVIEW_DOCUMENT","PROVIDE_GUIDANCE"]
ALLOWED_ACTION_TYPES=("REQUEST_DOCUMENT","REQUEST_USER_CONFIRMATION","CORRECT_INFORMATION","REVIEW_DOCUMENT","PROVIDE_GUIDANCE")
EXECUTION_TYPES=("information","prepare","official_application","payment","iro_preverification","fingerprint","iro_submission")


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1)
    action: str = ""  # Legacy human-readable alias for existing checkpoints/callers.
    title: str | None = None
    grounding_ids: list[str] = Field(default_factory=list)
    related_rule_ids: list[str] = Field(default_factory=list)
    execution_type: Literal["information","prepare","official_application","payment","iro_preverification","fingerprint","iro_submission"] | None = None
    priority: int = Field(ge=1, strict=True)
    reason: str = Field(min_length=1)
    tool_call: dict[str, Any] | None = None  # Exact host-grounded target; plan approval never executes it.
    request_kind: Literal["REQUEST_USER_CONFIRMATION", "REQUEST_DOCUMENT", "REVIEW_EXTRACTED_EVIDENCE", "RESOLVE_CONFLICT", "COMPLETE_APPLICATION_STEP", "PROVIDE_GUIDANCE"] | None = None
    requested_fields: list[str] = Field(default_factory=list)
    requested_documents: list[str] = Field(default_factory=list)
    available_in_prototype: bool = False
    related_rule_id: str | None = None
    due_date: date | None = None
    deadline_key: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    # All proposals pass explicit human approval; a model cannot waive it.
    requires_approval: bool = True
    action_type: PlanActionType | Literal["information", "prepare", "official_application", "payment",
                         "iro_preverification", "fingerprint", "iro_submission"] = "prepare"


    @model_validator(mode="after")
    def normalize_legacy_action(self):
        if self.action_type in EXECUTION_TYPES:
            self.execution_type=self.action_type
            self.action=self.action or self.title or ""
            self.title=self.action
        else:
            self.title=self.title or self.action
            if self.action and self.action!=self.title: raise ValueError("Action alias must match title")
            self.action=self.title or ""
            self.execution_type=self.execution_type or "information"
            kinds={"REQUEST_DOCUMENT":"REQUEST_DOCUMENT","REQUEST_USER_CONFIRMATION":"REQUEST_USER_CONFIRMATION","CORRECT_INFORMATION":"REVIEW_EXTRACTED_EVIDENCE","REVIEW_DOCUMENT":"REVIEW_EXTRACTED_EVIDENCE","PROVIDE_GUIDANCE":"PROVIDE_GUIDANCE"}
            self.request_kind=self.request_kind or kinds[self.action_type]
        if not self.action.strip(): raise ValueError("Action title must not be empty")
        return self


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


NextStep = Literal["document_analyzer", "rules_engine", "planner", "respond", "assistant_worker", "tool_dispatch"]

class ResidencePermitContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("institution", "applicant_type", "permit_type", "application_kind", mode="before")
    @classmethod
    def normalize_scope_value(cls, value, info):
        """Known aliases only: unknown values remain available for scope rejection."""
        if not isinstance(value, str):
            return value
        key = " ".join(normalize("NFKC", value).split()).casefold()
        aliases = {
            "institution": {"sabanci university": "Sabanc\u0131 University", "sabanc\u0131 university": "Sabanc\u0131 University"},
            "applicant_type": {alias: "incoming_exchange" for alias in ("incoming_exchange", "incoming exchange", "incoming-exchange")},
            "permit_type": {alias: "short_term" for alias in ("short_term", "short term", "short-term")},
            "application_kind": {"first": "first"},
        }
        return aliases[info.field_name].get(key, value)

    first_name: str | None = None
    last_name: str | None = None
    birth_date: date | None = None
    nationality: str | None = None
    latest_entry_date: date | None = None
    entry_stamp_present: bool | None = None
    receipt_payment_date: date | None = None
    receipt_reference_number: str | None = None
    receipt_amount: str | None = None
    program_start_date: date | None = None
    program_end_date: date | None = None
    holder_name: str | None = None
    student_id: str | None = None
    student_certificate_issue_date: date | None = None
    insurance_insurer: str | None = None
    insurance_policy_number: str | None = None
    passport_or_foreign_id: str | None = None
    application_number: str | None = None
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
    official_decision_received: bool | None = None
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
    utility_bill_accepted: bool | None = None
    address_form_accepted: bool | None = None
    utility_service_address: str | None = None
    residential_address: str | None = None
    passport_type: str | None = None
    gender: Literal["female","male","unspecified"] | None = None
    passport_issue_date: date | None = None
    passport_issuing_country: str | None = None
    passport_issuing_authority: str | None = None
    email: str | None = None
    turkish_phone: str | None = None
    reason_of_stay: str | None = None
    gi_number: str | None = None
    father_name: str | None = None
    mother_name: str | None = None
    marital_status: Literal["single","married","divorced","widowed"] | None = None
    birth_country: str | None = None
    contact_method: Literal["email","turkish_phone"] | None = None
    address_province: str | None = None
    address_district: str | None = None
    address_locality: str | None = None
    form_current_date: date | None = None
    simulation_declaration_acknowledged: bool | None = None
    visa_status: str | None = None
    visa_stay_days: int | None = None
    visa_valid_until: date | None = None
    previous_stay_history: str | None = None
    existing_permit_until: date | None = None
    application_form_window_id: str | None = None
    current_iro_window_id: str | None = None
    approval_status: str | None = None
    decision_authority: str | None = None
    tracking_channel: str | None = None
    age: int | None = None
    planned_stay_months: int | None = None
    photo_age_months: int | None = None
    missing_required_documents_count: int | None = None


    @field_validator("email")
    @classmethod
    def contact_email(cls,value):
        import re
        if value is not None and not re.fullmatch(r"[^\s@\r\n]+@[^\s@\r\n]+\.[^\s@\r\n]+",value):
            raise ValueError("Enter one valid email address")
        return value

    @field_validator("turkish_phone")
    @classmethod
    def contact_phone(cls,value):
        import re
        if value is not None and not re.fullmatch(r"[+0-9() -]{6,30}",value):raise ValueError("Enter a valid contact phone number")
        return value

    @field_validator("first_name","last_name","nationality","passport_issuing_country","passport_issuing_authority","residential_address","reason_of_stay","gi_number","father_name","mother_name","birth_country","address_province","address_district","address_locality")
    @classmethod
    def bounded_form_text(cls,value):
        if value is not None and (not value.strip() or len(value)>500 or any(ord(c)<32 and c not in "\t" for c in value)):
            raise ValueError("Enter a nonempty value of at most 500 characters")
        return value


ProcedureStage = Literal[
    "DOCUMENT_COLLECTION", "EIKAMET_PREPARATION", "EIKAMET_READY", "PAYMENT",
    "FINGERPRINT", "IRO_REVIEW", "SUBMISSION", "TRACKING", "COMPLETE",
]
StageStatus = Literal[
    "NOT_STARTED", "IN_PROGRESS", "BLOCKED", "READY", "WAITING_CONFIRMATION",
    "COMPLETED", "NOT_APPLICABLE",
]


class StageHistoryEntry(BaseModel):
    """Recorded procedure change; recorded_at is not an official-event timestamp."""
    model_config = ConfigDict(extra="forbid")

    stage: ProcedureStage
    status: StageStatus
    trace_id: str
    recorded_at: datetime
    reason_refs: list[str] = Field(default_factory=list)


EIkametFieldStatus = Literal["MISSING", "EXTRACTED", "VERIFIED", "USER_CONFIRMED", "CONFLICT"]
EIkametFieldSource = Literal["DOCUMENT", "USER", "DERIVED"]


class EIkametFieldState(BaseModel):
    """Preparation reference, never a second fact/provenance record.

    source_ref is a JSON pointer into canonical verification_records. Values,
    source snippets, confidence and user/trace metadata stay in their owners.
    EXTRACTED is reserved for future review; it is not accepted form input.
    """
    model_config = ConfigDict(extra="forbid")

    field_id: str = Field(min_length=1)
    status: EIkametFieldStatus = "MISSING"
    source_type: EIkametFieldSource | None = None
    source_ref: str | None = None
    reason_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def accepted_fields_need_references(self):
        if self.status in {"VERIFIED", "USER_CONFIRMED"}:
            if not self.source_ref or not self.source_ref.startswith("/verification_records/") or not self.source_type:
                raise ValueError("Accepted preparation fields require a verification-record reference")
        elif self.status in {"MISSING", "CONFLICT"} and (self.source_ref or self.source_type):
            raise ValueError("Missing/conflicting fields cannot select a source")
        return self


class EIkametPreparationState(BaseModel):
    """Initial supported-field projection; does not certify a complete official form."""
    model_config = ConfigDict(extra="forbid")

    fields: dict[str, EIkametFieldState] = Field(default_factory=dict)

    @model_validator(mode="after")
    def field_keys_match(self):
        if any(key != value.field_id for key, value in self.fields.items()):
            raise ValueError("Preparation field keys must match field_id")
        return self

    @property
    def prepared_field_count(self):
        return sum(item.status in {"VERIFIED", "USER_CONFIRMED"} for item in self.fields.values())

    @property
    def missing_field_ids(self):
        return [key for key, item in self.fields.items() if item.status in {"MISSING", "EXTRACTED"}]

    @property
    def conflict_field_ids(self):
        return [key for key, item in self.fields.items() if item.status == "CONFLICT"]


class ProcedureEntryIntent(BaseModel):
    """Onboarding declaration only; never evidence of a completed official step."""
    model_config = ConfigDict(extra="forbid")
    mode: Literal["START_FROM_SCRATCH", "HAVE_DOCUMENTS", "EIKAMET_COMPLETED", "RESUME"]
    user_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    recorded_at: datetime


class ProcedureState(BaseModel):
    """Persistent lifecycle and optional form-preparation references."""
    model_config = ConfigDict(extra="forbid")

    current_stage: ProcedureStage = "DOCUMENT_COLLECTION"
    stage_status: StageStatus = "NOT_STARTED"  # Current stage only.
    completed_stages: list[ProcedureStage] = Field(default_factory=list)  # Confirmed stages, not plan steps.
    stage_blockers: list[str] = Field(default_factory=list)  # References, not facts/results.
    stage_history: list[StageHistoryEntry] = Field(default_factory=list)
    eikamet_preparation: EIkametPreparationState | None = None
    eikamet_path: Literal["UPLOADED_FORM","ASSISTED"] | None = None
    reviewed_application_document_id: str | None = None


# Pydantic classes stored by the in-memory and SQLite checkpointers.
CHECKPOINT_TYPES = [(cls.__module__, cls.__name__) for cls in
                    (UserEvent, Document, VerifiedInput, DocumentInfo, StudentCertificate, HealthInsurance,
                     IkametApplicationForm, UnknownDocument, PassportIdentity, EntryStamp, DeclarationControl, CardFeeReceipt, UtilityBill, AddressForm, Requirement, PlanStep,
                     ActionPlan, ValidationIssue, ResidencePermitContext, StageHistoryEntry, ProcedureState, EIkametFieldState, EIkametPreparationState, ProcedureEntryIntent)]


# ----------------------------------------------------------------- reducers --
def upsert_documents(current: list[Document], update: list[Document]) -> list[Document]:
    """Merge by doc_id: a node returns only new/changed documents and none is lost."""
    merged = {doc.doc_id: doc for doc in current or []}
    for doc in update or []:
        if doc.status=="deleted": merged.pop(doc.doc_id,None)
        else: merged[doc.doc_id]=doc
    return list(merged.values())


def merge_facts(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """New facts are added, existing keys are updated, nothing else is dropped."""
    result=dict(current or {})
    for key,value in (update or {}).items():
        if value is None: result.pop(key,None)
        else: result[key]=value
    return result


# -------------------------------------------------------------------- state --
class AgentState(TypedDict, total=False):
    # Digital Twin (persistent across sessions)
    user_id: str
    verification_records: dict[str, dict[str, VerifiedInput]]
    verified_context_fields: list[str]
    verification_conflicts: list[str]
    identity_resolutions: dict[str, Any]
    action_progress: dict[str, dict[str, Any]]
    residence_permit_context: ResidencePermitContext
    deadlines: dict[str, date]
    approved_plan_origin: str
    approved_plan: ActionPlan | None
    active_procedure: str                                     # e.g. "ikamet"
    procedure: ProcedureState | None                          # New V2 threads initialize at the interaction boundary; legacy absence is preserved.
    procedure_entry_intent: ProcedureEntryIntent | None
    documents: Annotated[list[Document], upsert_documents]
    extracted_facts: Annotated[dict[str, Any], merge_facts]   # e.g. {"appointment_date": date(...)}
    completed_steps: list[str]
    blockers: list[str]                                       # rule_ids still unsatisfied

    # V2 episodic interactions and a separate simulated environment. These
    # dictionaries contain typed-boundary tool records, never a second fact store.
    assistant_messages: list[dict[str, Any]]
    assistant_events: list[dict[str, Any]]
    assistant_diagnostics: dict[str, Any]
    tool_proposals: list[dict[str, Any]]
    tool_outcomes: list[dict[str, Any]]
    pending_tool_proposal: dict[str, Any] | None
    tool_approval_decision: bool | None
    sandbox_form: dict[str, Any]
    calendar_reminders: list[dict[str, Any]]
    email_drafts: list[dict[str, Any]]
    last_tool_outcome: dict[str, Any] | None

    # Run memory (reset by new_run() at every new event)
    event: UserEvent
    trace_id: str
    next_step: NextStep                                       # last supervisor decision
    relevant_requirements: list[Requirement] | None           # None = not computed yet
    candidate_plan: list[PlanStep] | None                     # None = not computed yet
    validation_errors: list[ValidationIssue]                  # empty = PASS
    retry_count: int                                          # repair attempts in this run
    final_response: str
    plan_origin: str
    planner_diagnostics: dict[str, Any]
    planning_context_snapshot: dict[str, Any]
    rejected_model_plan: dict[str, Any] | None
    last_model_plan_hash: str | None
    planning_scope: str
    workflow_stage: str
    plan_summary: str
    repair_valid_actions: list[PlanStep]
    repair_rejected_action_ids: list[str]
    repair_feedback: list[ValidationIssue]
    knowledge_snapshot: dict[str, Any]
    requirements_knowledge_version: str
    repair_lessons: list[dict[str, Any]]
    repair_history: list[dict[str, Any]]
    run_status: str
    approval_decision: bool | None


def new_run(event: UserEvent, user_id: str, procedure: str = "ikamet", *, scope: str | None = None) -> dict:
    """Input for one graph run.

    Only run-memory fields are reset, so the Digital Twin loaded from the
    checkpointer (documents, facts, completed steps, blockers) is preserved.
    The procedure argument names active_procedure; the optional ProcedureState
    component is initialized separately and omitted from these reset updates.
    """
    from src.workers.rule_loader import knowledge_snapshot
    snapshot=knowledge_snapshot()
    if event.kind in {"assistant_message", "run_tool"}:
        # Chat/tool requests do not destroy assessed facts or the last plan.
        return {"knowledge_snapshot":snapshot,"user_id":user_id, "active_procedure":procedure, "event":event,
            "trace_id":uuid4().hex, "validation_errors":[], "final_response":"",
            "run_status":"RUNNING", "tool_approval_decision":None}
    result = {
        "knowledge_snapshot":snapshot,
        "repair_lessons":[],
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
        "plan_origin": "",
        "planner_diagnostics": {},
        "planning_context_snapshot": {},
        "rejected_model_plan": None,
        "last_model_plan_hash": None,
        "repair_valid_actions": [],
        "repair_rejected_action_ids": [],
        "repair_feedback": [],
        "repair_history": [],
        "run_status": "RUNNING",
        "approval_decision": None,
    }

    if scope is not None:
        result["planning_scope"] = scope
    # Planning consumes the last assessment of the unchanged Digital Twin.
    # Omission retains it in LangGraph, including when Gate 0 rejects this run.
    if event.kind == "generate_plan":
        result.pop("relevant_requirements")
    if event.kind in {"analyze_document", "confirm_information", "generate_plan", "resolve_conflict", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction", "reevaluate"}:
        result["approved_plan"] = None
    return result
