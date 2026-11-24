"""Result models of document extraction, shared by the extract tool and the Twin."""
from typing import Any

from pydantic import BaseModel, Field

from src.document_processing.grounding import GroundedValue
from src.models import FactValue


class FactProposal(BaseModel):
    """A fact the document supports. Not stored as confirmed when requires_confirmation is set."""
    field: str
    value: FactValue
    from_field: str
    page: int | None = None
    snippet: str | None = None
    requires_confirmation: bool = False


class DocumentObservation(BaseModel):
    """What one logical document showed (a PDF bundle can hold two)."""
    doc_type: str
    grounded: dict[str, GroundedValue] = Field(default_factory=dict)
    ambiguous_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    facts: list[FactProposal] = Field(default_factory=list)
    info: dict[str, Any] | None = None
    pages: list[int] = Field(default_factory=list)        # the pages of the upload this document is on
    found_by: str = ""                                    # how its type was found: user, mrz, title, keywords, model
    notes: list[str] = Field(default_factory=list)        # what went right (shown next to issues, not as a warning)


class ExtractionResult(BaseModel):
    """All observations of one file, or an error message."""
    observations: list[DocumentObservation] = Field(default_factory=list)
    error: str | None = None
