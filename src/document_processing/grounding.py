"""Which extracted values are backed by source evidence. Generic over all typed document kinds.

A value counts as an observation of the document only if it survived validation, is not marked
ambiguous, and has a source page and quote that match a deterministic evidence record.
Moved unchanged in behaviour from src/workers/verification.py:grounded_values (milestone 1).
"""
from typing import Any

from pydantic import BaseModel

from src.document_processing.evidence import normalized_snippet
from src.document_processing.schemas import DocumentInfo

NAME_FIELDS = ("holder_name", "applicant_name")


class GroundedValue(BaseModel):
    """One document field with the page and exact quote that support it."""
    name: str
    value: Any
    page: int | None = None
    snippet: str | None = None


def _expected_evidence_value(value: Any) -> Any:
    """Evidence stores booleans as 'true'/'false' text."""
    return str(value).lower() if isinstance(value, bool) else value


def _has_matching_evidence(info: DocumentInfo, name: str, value: Any, page: int, quote: str) -> bool:
    roles = {name, "name"} if name in NAME_FIELDS else {name}
    expected = _expected_evidence_value(value)
    for evidence in info.source_evidence:
        if (evidence.field in roles and evidence.normalized_value == expected
                and evidence.page_number == page
                and normalized_snippet(evidence.snippet) == normalized_snippet(quote)):
            return True
    return False


def grounded_values(info: DocumentInfo) -> dict[str, GroundedValue]:
    """Accept only field-specific, matching source evidence (values in JSON form: dates as text)."""
    if info.document_type == "unknown":
        return {}
    values = info.model_dump(mode="json")
    accepted: dict[str, GroundedValue] = {}
    for name in sorted(info.observation_fields()):
        value = values.get(name)
        if value is None or name in info.ambiguous_fields:
            continue
        if any(issue.startswith(name) for issue in info.validation_issues):
            continue
        quote, page = info.source_snippets.get(name), info.source_pages.get(name)
        if quote and page and _has_matching_evidence(info, name, value, page, quote):
            accepted[name] = GroundedValue(name=name, value=value, page=page, snippet=quote)
    return accepted
