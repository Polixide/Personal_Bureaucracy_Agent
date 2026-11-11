"""Structured document data and deterministic quality checks."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal["student_enrollment_certificate", "health_insurance_certificate", "unknown"]
FieldName = Literal["holder_name", "student_id", "institution", "policy_number", "issue_date", "expiry_date"]

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "student_enrollment_certificate": ("holder_name", "student_id", "institution", "issue_date"),
    "health_insurance_certificate": ("holder_name", "institution", "policy_number", "issue_date", "expiry_date"),
    "unknown": (),
}


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
        issues = list(values.get("validation_issues", []))
        for name in ("holder_name", "student_id", "institution", "policy_number", "issue_date", "expiry_date"):
            value = values.get(name)
            if name in ("issue_date", "expiry_date") and isinstance(value, (int, float, bool)):
                raise ValueError(f"{name}: numeric and boolean values are not valid dates")
            if isinstance(value, str):
                value = value.strip() or None
                values[name] = value
            if name in ("issue_date", "expiry_date") and isinstance(value, str):
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
            self.validation_issues.append(f"{name}: ambiguous; manual review needed")
        self.missing_fields = [
            name for name in REQUIRED_FIELDS[self.document_type] if getattr(self, name) is None
        ]
        if self.document_type == "unknown":
            self.validation_issues.append("Document type is unknown; manual review needed")
        if self.issue_date and self.expiry_date and self.expiry_date < self.issue_date:
            self.validation_issues.append("expiry_date precedes issue_date")
        self.validation_issues = list(dict.fromkeys(self.validation_issues))
        return self
