"""Structured document data and deterministic quality checks."""

from datetime import date
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal["student_enrollment_certificate", "health_insurance_certificate", "unknown"]
FieldName = Literal["holder_name", "student_id", "institution", "policy_number", "issue_date", "expiry_date",
                    "coverage_start_date", "coverage_end_date"]
DATE_FIELDS = ("issue_date", "expiry_date", "coverage_start_date", "coverage_end_date")

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "student_enrollment_certificate": ("holder_name", "student_id", "institution", "issue_date"),
    "health_insurance_certificate": ("holder_name", "institution", "policy_number", "issue_date", "expiry_date"),
    "unknown": (),
}


class SourceEvidence(BaseModel):
    """Deterministic source candidates; TXT has line references but no PDF page."""

    field: str | None
    original_value: str
    normalized_value: str | None = None
    snippet: str
    line_number: int
    page_number: int | None = None


class DocumentInfo(BaseModel):
    """Extracted facts; quality flags are informational, not legal decisions."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_type: DocumentType = "unknown"
    holder_name: str | None = None
    student_id: str | None = None
    institution: str | None = Field(default=None, description="University or insurance provider")
    policy_number: str | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    coverage_start_date: date | None = None
    coverage_end_date: date | None = None
    source_evidence: list[SourceEvidence] = Field(default_factory=list)
    correction_notes: list[str] = Field(default_factory=list)
    source_snippets: dict[FieldName, str] = Field(
        default_factory=dict, description="Exact source quotes supporting extracted fields, when available",
        json_schema_extra={
            "properties": {name: {"type": "string"} for name in get_args(FieldName)},
            "additionalProperties": False,
        })
    ambiguous_fields: list[FieldName] = Field(default_factory=list)
    missing_fields: list[FieldName] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_values(cls, values: object) -> object:
        """Keep invalid dates as null and preserve an explicit quality flag."""
        if not isinstance(values, dict):
            return values
        values = dict(values)
        raw_issues = values.get("validation_issues", [])
        if not isinstance(raw_issues, list):
            raise ValueError("validation_issues must be a list")
        issues = list(raw_issues)
        for name in ("holder_name", "student_id", "institution", "policy_number", *DATE_FIELDS):
            value = values.get(name)
            if name in DATE_FIELDS and isinstance(value, (int, float, bool)):
                raise ValueError(f"{name}: numeric and boolean values are not valid dates")
            if isinstance(value, str):
                value = value.strip() or None
                values[name] = value
            if name in DATE_FIELDS and isinstance(value, str):
                try:
                    parsed = date.fromisoformat(value)
                    if parsed.isoformat() != value:
                        raise ValueError("Expected YYYY-MM-DD")
                except ValueError:
                    values[name] = None
                    issues.append(f"{name}: invalid date {value!r}; expected YYYY-MM-DD")
        values["validation_issues"] = issues
        return values

    @model_validator(mode="after")
    def check_quality(self) -> Self:
        self.ambiguous_fields = list(dict.fromkeys(self.ambiguous_fields))
        for name in self.ambiguous_fields:
            setattr(self, name, None)
            if not any(issue.startswith(f"{name}:") for issue in self.validation_issues):
                self.validation_issues.append(f"{name}: ambiguous; manual review needed")
        # Maintain the existing insurance expiry field for downstream consumers.
        if (self.document_type == "health_insurance_certificate" and
                self.expiry_date is None and "expiry_date" not in self.ambiguous_fields):
            self.expiry_date = self.coverage_end_date
        self.missing_fields = [
            name for name in REQUIRED_FIELDS[self.document_type] if getattr(self, name) is None
        ]
        if self.document_type == "unknown":
            self.validation_issues.append(
                "Document type is unknown; required fields cannot be determined; manual review needed")
        if self.issue_date and self.expiry_date and self.expiry_date < self.issue_date:
            self.validation_issues.append("expiry_date precedes issue_date")
        if (self.coverage_start_date and self.coverage_end_date and
                self.coverage_end_date < self.coverage_start_date):
            self.validation_issues.append("coverage_end_date precedes coverage_start_date")
        if (self.document_type == "health_insurance_certificate" and self.expiry_date and
                self.coverage_end_date and self.expiry_date != self.coverage_end_date):
            self.validation_issues.append("expiry_date conflicts with coverage_end_date")
        self.validation_issues = list(dict.fromkeys(self.validation_issues))
        return self
