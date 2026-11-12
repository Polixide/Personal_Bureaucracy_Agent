"""Document-specific observations and deterministic quality checks, never legal decisions."""
from datetime import date
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal["student_enrollment_certificate", "health_insurance_certificate",
                       "ikamet_application_form", "unknown"]


class DocumentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_type: DocumentType


class DocumentInfo(BaseModel):
    """Small common base. Use a concrete subtype for administrative documents."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    document_type: DocumentType = "unknown"
    holder_name: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    ambiguous_fields: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)
    source_pages: dict[str, int] = Field(default_factory=dict)
    selected_pages: list[int] = Field(default_factory=list)

    REQUIRED: ClassVar[tuple[str, ...]] = ()
    DATES: ClassVar[tuple[str, ...]] = ()
    PERIOD: ClassVar[tuple[str, str] | None] = None

    @classmethod
    def observation_fields(cls):
        return set(cls.model_fields) - {"document_type", "missing_fields", "ambiguous_fields",
                                      "validation_issues", "source_pages", "selected_pages"}

    @model_validator(mode="before")
    @classmethod
    def normalize_values(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        issues = list(values.get("validation_issues", []))
        if cls is DocumentInfo and values.get("document_type", "unknown") != "unknown":
            # Old SQLite checkpoints used this base as a catch-all extraction.
            # Keep them readable, but require reprocessing under a concrete schema.
            values = {key: value for key, value in values.items() if key in cls.model_fields}
            values["document_type"] = "unknown"
            issues.append("Legacy extraction schema; reprocess this document")
        for name in cls.observation_fields():
            value = values.get(name)
            if isinstance(value, str):
                value = value.strip() or None
                values[name] = value
            if name in cls.DATES and value is not None:
                try:
                    parsed = date.fromisoformat(value) if isinstance(value, str) else value
                    if not isinstance(parsed, date) or (isinstance(value, str) and parsed.isoformat() != value):
                        raise ValueError("expected YYYY-MM-DD")
                    # Broad data-quality bounds, not a legal validity decision.
                    if not 1900 <= parsed.year <= 2100:
                        raise ValueError("implausible year; expected 1900..2100")
                except (ValueError, TypeError):
                    values[name] = None
                    issues.append(f"{name}: invalid or implausible date {value!r}; expected YYYY-MM-DD, year 1900..2100")
        values["validation_issues"] = issues
        return values

    @model_validator(mode="after")
    def check_quality(self) -> Self:
        fields = type(self).observation_fields()
        self.ambiguous_fields = list(dict.fromkeys(self.ambiguous_fields))
        for name in self.ambiguous_fields:
            if name not in fields:
                raise ValueError(f"Invalid ambiguous field for {self.document_type}: {name}")
            setattr(self, name, None)
            self.validation_issues.append(f"{name}: ambiguous; manual review needed")
        if self.document_type == "unknown":
            if self.holder_name is not None:
                self.holder_name = None
                self.validation_issues.append("Unknown document: discarded holder_name")
            self.validation_issues.append("Document type is unknown; manual review needed")
        if self.PERIOD:
            start, end = self.PERIOD
            if getattr(self, start) and getattr(self, end) and getattr(self, end) < getattr(self, start):
                setattr(self, end, None)
                self.validation_issues.append(f"{end} precedes {start}; end date discarded")
        self.missing_fields = [name for name in self.REQUIRED if getattr(self, name) is None]
        for name in list(self.source_pages):
            if name not in fields or getattr(self, name) is None or self.source_pages[name] < 1:
                del self.source_pages[name]
                self.validation_issues.append(f"{name}: invalid source page discarded")
        self.validation_issues = list(dict.fromkeys(self.validation_issues))
        return self


class StudentCertificate(DocumentInfo):
    document_type: Literal["student_enrollment_certificate"] = "student_enrollment_certificate"
    student_id: str | None = None
    institution: str | None = None
    issue_date: date | None = None
    REQUIRED = ("holder_name", "student_id", "institution", "issue_date")
    DATES = ("issue_date",)


class HealthInsurance(DocumentInfo):
    document_type: Literal["health_insurance_certificate"] = "health_insurance_certificate"
    policy_number: str | None = None
    insurer: str | None = None
    passport_or_foreign_id: str | None = None
    coverage_start_date: date | None = None
    coverage_end_date: date | None = None
    REQUIRED = ("holder_name", "policy_number", "insurer", "coverage_start_date", "coverage_end_date")
    DATES = ("coverage_start_date", "coverage_end_date")
    PERIOD = ("coverage_start_date", "coverage_end_date")


class IkametApplicationForm(DocumentInfo):
    document_type: Literal["ikamet_application_form"] = "ikamet_application_form"
    application_number: str | None = None
    applicant_name: str | None = None
    passport_or_foreign_id: str | None = None
    requested_start_date: date | None = None
    requested_end_date: date | None = None
    REQUIRED = ("applicant_name",)
    DATES = ("requested_start_date", "requested_end_date")
    PERIOD = ("requested_start_date", "requested_end_date")


class UnknownDocument(DocumentInfo):
    document_type: Literal["unknown"] = "unknown"


ExtractedDocument = Annotated[
    StudentCertificate | HealthInsurance | IkametApplicationForm | UnknownDocument,
    Field(discriminator="document_type"),
]
DOCUMENT_SCHEMAS = {
    "student_enrollment_certificate": StudentCertificate,
    "health_insurance_certificate": HealthInsurance,
    "ikamet_application_form": IkametApplicationForm,
    "unknown": UnknownDocument,
}
