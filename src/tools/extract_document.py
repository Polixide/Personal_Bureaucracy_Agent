"""Document -> grounded observations -> proposed facts for the active Procedure Pack.

Typed path: the milestone-1 processor (classification, typed schema, deterministic evidence,
RapidOCR for scans) for the seven document kinds it knows. Generic path: pack document types
without a typed schema (src/document_processing/generic.py). In both paths only source-grounded
values become fact proposals; `requires_confirmation` values wait for the user.

`extract_file` is pure (no Twin). The registered tool (registry.py) stores the result in the Twin.
"""
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.document_processing.extractor import extract_pages
from src.document_processing.generic import extract_generic
from src.document_processing.grounding import GroundedValue, grounded_values
from src.document_processing.processor import process_document, process_passport_pages
from src.document_processing.schemas import DOCUMENT_SCHEMAS, DocumentInfo
from src.llm import Backend
from src.models import FactValue, Pack
from src.pack import coerce_fact

PASSPORT_BUNDLE_TYPES = ("passport_identity_page", "turkiye_entry_stamp")
EXTRACTION_ERRORS = (OSError, ValueError, RuntimeError)


class ExtractDocumentInput(BaseModel):
    """Tool arguments: which uploaded document, and optionally its type if the user knows it."""
    doc_id: str = Field(description="Id of an uploaded document in the Twin")
    doc_type: str | None = Field(default=None, description="Document type from the procedure, if known")


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


class ExtractionResult(BaseModel):
    """All observations of one file, or an error message."""
    observations: list[DocumentObservation] = Field(default_factory=list)
    error: str | None = None


def facts_from_grounded(pack: Pack, doc_type: str, grounded: dict[str, GroundedValue]) -> tuple[list[FactProposal], list[str]]:
    """Map document fields to pack facts using procedure.json `documents[].extracts`."""
    spec = pack.document(doc_type)
    if spec is None:
        return [], [f"{doc_type} is not a document of procedure {pack.id}"]
    facts, issues = [], []
    for extract in spec.extracts:
        found = grounded.get(extract.source_field())
        if found is None:
            continue
        try:
            value = coerce_fact(pack, extract.field, found.value)
        except ValueError as error:
            issues.append(f"{extract.field}: {error}")
            continue
        facts.append(FactProposal(field=extract.field, value=value, from_field=found.name, page=found.page,
                                  snippet=found.snippet, requires_confirmation=extract.requires_confirmation))
    return facts, issues


def observe_typed(pack: Pack, info: DocumentInfo) -> DocumentObservation:
    """Observation of one typed extraction."""
    grounded = grounded_values(info)
    facts, issues = facts_from_grounded(pack, info.document_type, grounded) if info.document_type != "unknown" else ([], [])
    return DocumentObservation(doc_type=info.document_type, grounded=grounded,
                               ambiguous_fields=list(info.ambiguous_fields),
                               issues=list(info.validation_issues) + issues, facts=facts,
                               info=info.model_dump(mode="json"))


def _typed_infos(path: Path, doc_type: str | None, processor: Callable, bundle: Callable) -> list[DocumentInfo]:
    info = processor(path, document_type_hint=doc_type) if doc_type else processor(path)
    if info.document_type in PASSPORT_BUNDLE_TYPES and path.suffix.lower() == ".pdf":
        return bundle(path, info)
    return [info]


def extract_file(path: str | Path, pack: Pack, doc_type: str | None = None, backend: Backend | None = None,
                 processor: Callable = process_document, bundle: Callable = process_passport_pages) -> ExtractionResult:
    """Extract one file. Never raises for document or model problems: returns `error` instead."""
    path = Path(path)
    try:
        if doc_type and doc_type not in DOCUMENT_SCHEMAS:
            return _extract_generic_file(path, pack, doc_type, backend)
        infos = _typed_infos(path, doc_type, processor, bundle)
    except EXTRACTION_ERRORS as error:
        return ExtractionResult(error=str(error))
    return ExtractionResult(observations=[observe_typed(pack, info) for info in infos])


def _extract_generic_file(path: Path, pack: Pack, doc_type: str, backend: Backend | None) -> ExtractionResult:
    spec = pack.document(doc_type)
    if spec is None:
        return ExtractionResult(error=f"Unknown document type {doc_type!r} for procedure {pack.id}")
    if backend is None:
        return ExtractionResult(error="No language model available for this document type")
    generic = extract_generic(extract_pages(path), spec, pack, backend)
    facts, issues = facts_from_grounded(pack, doc_type, generic.grounded)
    return ExtractionResult(observations=[DocumentObservation(
        doc_type=doc_type, grounded=generic.grounded, issues=generic.issues + issues, facts=facts)])
