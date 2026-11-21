"""Typed tool boundary; model intent never confers host authorization."""
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ToolCall(StrictModel):
    call_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

class ToolSpec(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    risk_tier: Literal["TIER_1", "TIER_2"]
    effect: Literal["READ_ONLY", "SANDBOX_WRITE", "EXTERNAL_MUTATION", "ARTIFACT_MUTATION"]

class ToolObservation(StrictModel):
    call_id: str
    name: str
    status: Literal["SUCCEEDED", "FAILED", "APPROVAL_REQUIRED", "REJECTED"]
    data: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, Any] = Field(default_factory=dict)
    diff: dict[str, Any] = Field(default_factory=dict)
    failure_category: str | None = None
    elapsed_seconds: float = 0.0
    state_updates: dict[str, Any] = Field(default_factory=dict)
    replayed: bool = False
    fingerprint: str | None = None

class EmptyInput(StrictModel):
    pass

class SearchInput(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=3, ge=1, le=5, strict=True)

class FillFieldInput(StrictModel):
    field_id: str = Field(min_length=1, max_length=100)
    value: str | None = None

class ValidateFieldInput(StrictModel):
    field_id: str = Field(min_length=1, max_length=100)

class CalendarInput(StrictModel):
    delivery: Literal["configured","calendar_app"] = "configured"
    source_candidate_id: str | None = None
    title: str = Field(min_length=1, max_length=200)
    start_at: datetime
    duration_minutes: int = Field(default=30, ge=5, le=1440, strict=True)
    description: str = Field(default="Personal preparation reminder; not a legal deadline.", max_length=2000)

    @field_validator("start_at", mode="before")
    @classmethod
    def explicit_datetime_input(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError("Provide an explicit ISO date/time with timezone")
        return value

    @field_validator("start_at")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("An explicit timezone is required")
        return value

class EmailDraftInput(StrictModel):
    purpose: Literal["support","package_review"] = "support"
    recipient: str | None = None
    subject: str = Field(default="Residence permit preparation review request", min_length=1, max_length=200)
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("subject")
    @classmethod
    def single_header_line(cls, value):
        if "\r" in value or "\n" in value:
            raise ValueError("Email subject must be one line")
        return value

    @field_validator("recipient")
    @classmethod
    def recipient_format(cls, value):
        if value is not None:
            import re
            if not re.fullmatch(r"[^\s@\r\n]+@[^\s@\r\n]+\.[^\s@\r\n]+", value):
                raise ValueError("Provide one explicit email address")
        return value

class SendEmailInput(StrictModel):
    draft_id: str = Field(min_length=1)

class SandboxField(StrictModel):
    value: str
    source_ref: str
    source_status: Literal["VERIFIED", "USER_CONFIRMED"]

class SandboxAnswersInput(StrictModel):
    answers: dict[str,str | bool | None] = Field(default_factory=dict,max_length=40)

class SandboxForm(StrictModel):
    # Draft answers may be invalid/unconfirmed. They never project legal facts.
    draft_answers: dict[str,str | bool | None] = Field(default_factory=dict)
    fields: dict[str, SandboxField] = Field(default_factory=dict)
    revision: int = 0
    status: Literal["DRAFT", "VALIDATED", "SIMULATED_SUBMISSION"] = "DRAFT"
    validated_revision: int | None = None
    validation_issues: list[str] = Field(default_factory=list)
    simulated_application_ready: bool = False
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    artifact_revision: int | None = None

class SandboxResult(StrictModel):
    form: SandboxForm
    valid: bool = False
    missing_field_ids: list[str] = Field(default_factory=list)
    conflict_field_ids: list[str] = Field(default_factory=list)
    stale_field_ids: list[str] = Field(default_factory=list)
    invalid_field_ids: list[str] = Field(default_factory=list)
    message: str = "Local simulation only; no official portal action."

class ProcedureResult(StrictModel):
    procedure: dict[str, Any] | None

class MissingDocumentsResult(StrictModel):
    ready: bool
    prerequisites: dict[str, str]
    blockers: list[str]

class SearchResult(StrictModel):
    results: list[dict[str, Any]]

class CalendarResult(StrictModel):
    reminder_id: str
    mode: Literal["LOCAL_ICS", "GOOGLE_CALENDAR"]
    title: str
    start_at: str
    end_at: str
    description: str
    ics: str
    external_id: str | None = None

class EmailResult(StrictModel):
    purpose: Literal["support","package_review"] = "support"
    draft_id: str
    recipient: str | None
    recipient_source_ref: str | None = None
    subject: str
    body: str
    attachments: list[dict[str, str]]
    status: Literal["DRAFT", "SIMULATED_SEND", "SENT"] = "DRAFT"
    delivery_mode: Literal["LOCAL_SIMULATION", "GMAIL"] = "LOCAL_SIMULATION"
    external_id: str | None = None


class ReadResult(StrictModel):
    data: dict[str, Any]


class SnapshotInput(StrictModel):
    source_id: str = Field(min_length=1,max_length=100)
    source_url: str = Field(min_length=1,max_length=1000)
    content: str = Field(min_length=1,max_length=20000)


class ArtifactResult(StrictModel):
    title: str
    content: str
    artifact_path: str | None = None
    status: str = "PROPOSED"


class SimulatedApplicationResult(StrictModel):
    title: str = "SIMULATED E-IKAMET APPLICATION"
    notice: str = "Educational sandbox artifact. NOT AN OFFICIAL GOVERNMENT FORM. NOT SUBMITTED."
    prepared_values: dict[str,str]
    form_revision: int
    artifact_path: str | None = None
    artifact_sha256: str | None = None
