"""Shared state models for the Personal Bureaucracy Digital Twin."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Canonical identifiers match R004's required_values in knowledge/rules.json.
DocumentType = Literal[
    "declaration_control_document", "address_proof", "passport_copy",
    "biometric_photo", "student_certificate", "signed_online_application_form",
    "card_fee_receipt", "fingerprint_gi_evidence", "valid_health_insurance",
]


class Document(BaseModel):
    """Basic information about an uploaded document."""

    # Canonical category used consistently by the requirement engine.
    document_type: DocumentType
    # Current document status, such as uploaded, valid, or expired.
    status: str
    # Date the document was issued, if known.
    issue_date: date | None = None
    # Date the document expires, if known.
    expiry_date: date | None = None

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, value):
        """Accept the earlier prototype labels, storing only canonical values."""
        aliases = {
            "passport": "passport_copy",
            "health insurance": "valid_health_insurance",
        }
        if isinstance(value, str):
            value = value.strip().lower()
            return aliases.get(value, value)
        return value


class ResidencePermitContext(BaseModel):
    """Known facts for the prototype rules; None means not yet known."""

    # Whether the applicant is currently in Türkiye.
    is_in_turkiye: bool | None = None
    # Whether the applicant entered using a passport.
    entered_turkiye_with_passport: bool | None = None
    # Date of the residence-permit application.
    application_date: date | None = None
    # Last day of the applicant's legal stay.
    legal_stay_expiry_date: date | None = None
    # Requested final day of the permit.
    requested_permit_end_date: date | None = None
    # First and last days of the exchange stay.
    stay_start_date: date | None = None
    stay_end_date: date | None = None
    # Address-proof category as defined in rules.json.
    address_proof_type: str | None = None
    # Whether the online application has been completed.
    online_application_completed: bool | None = None
    # Whether the application form has been signed.
    application_form_signed: bool | None = None
    # Whether fingerprint processing has been completed.
    fingerprint_completed: bool | None = None
    # Whether a GI number is available.
    gi_number_present: bool | None = None
    # Whether the insurance is valid for residence-permit purposes.
    insurance_valid_for_residence_permit: bool | None = None
    # First and last days covered by the insurance.
    insurance_start_date: date | None = None
    insurance_end_date: date | None = None


class State(BaseModel):
    """Information shared while working toward a user's goal."""

    # Outcome the user wants to achieve.
    user_goal: str
    # Known residence-permit facts, initially all unknown.
    residence_permit_context: ResidencePermitContext = Field(
        default_factory=ResidencePermitContext
    )
    # Procedure currently being handled, once identified.
    current_procedure: str | None = None
    # Documents the user has uploaded.
    uploaded_documents: list[Document] = Field(default_factory=list)
    # Scheduled appointment date, if one has been booked.
    appointment_date: date | None = None
    # Named deadlines and their due dates.
    deadlines: dict[str, date] = Field(default_factory=dict)
    # Requirements that still need to be satisfied.
    missing_prerequisites: list[str] = Field(default_factory=list)
    # Ordered steps in the current plan.
    current_plan: list[str] = Field(default_factory=list)
    # Problems found when validating the procedure or documents.
    validation_errors: list[str] = Field(default_factory=list)
    # Number of completed planning or validation iterations.
    iteration_count: int = Field(default=0, ge=0)
