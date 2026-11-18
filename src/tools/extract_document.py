"""extract_document tool: document -> grounded observations -> proposed facts in the Twin.

Typed path: the milestone-1 processor (classification, typed schema, deterministic evidence,
RapidOCR for scans) for the seven document kinds it knows. Generic path: pack document types
without a typed schema (src/document_processing/generic.py). In both paths only source-grounded
values become fact proposals; `requires_confirmation` values wait for the user.

`extract_file` is pure (no Twin). `run` is the registered tool: it stores the result in the Twin.
"""
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from src.document_processing import processor as document_processor
from src.document_processing.extractor import extract_pages
from src.document_processing.generic import extract_generic
from src.document_processing.grounding import GroundedValue, grounded_values
from src.document_processing.observations import DocumentObservation, ExtractionResult, FactProposal
from src.document_processing.schemas import DOCUMENT_SCHEMAS, DocumentInfo
from src.llm import Backend
from src.models import Pack
from src.pack import coerce_fact
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.twin import service

__all__ = ["DocumentObservation", "ExtractionResult", "FactProposal", "extract_file", "run", "SPEC"]

PASSPORT_BUNDLE_TYPES = ("passport_identity_page", "turkiye_entry_stamp")
EXTRACTION_ERRORS = (OSError, ValueError, RuntimeError)


class ExtractDocumentInput(BaseModel):
    """Tool arguments: which uploaded document, and optionally its type if the user knows it."""
    doc_id: str = Field(description="Id of an uploaded document in the Twin")
    doc_type: str | None = Field(default=None, description="Document type from the procedure, if known")


class ExtractedFactRow(BaseModel):
    field: str
    value: str
    needs_confirmation: bool


class ExtractedDocumentRow(BaseModel):
    doc_id: str
    doc_type: str | None
    status: str
    facts: list[ExtractedFactRow] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ExtractDocumentOutput(BaseModel):
    """What the agent sees: per logical document, the facts it now supports."""
    documents: list[ExtractedDocumentRow]
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
    facts, issues = ([], []) if info.document_type == "unknown" else facts_from_grounded(pack, info.document_type, grounded)
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
                 processor: Callable | None = None, bundle: Callable | None = None) -> ExtractionResult:
    """Extract one file. Never raises for document or model problems: returns `error` instead."""
    path = Path(path)
    processor = processor or document_processor.process_document
    bundle = bundle or document_processor.process_passport_pages
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


def _row(document, observation: DocumentObservation | None) -> ExtractedDocumentRow:
    facts = [ExtractedFactRow(field=f.field, value=str(f.value), needs_confirmation=f.requires_confirmation)
             for f in (observation.facts if observation else [])]
    issues = (observation.issues if observation else [document.processing_error or ""])[:5]
    return ExtractedDocumentRow(doc_id=document.doc_id, doc_type=document.doc_type, status=document.status,
                                facts=facts, issues=[issue for issue in issues if issue])


def run(args: ExtractDocumentInput, ctx: ToolContext) -> ExtractDocumentOutput:
    """Registered tool: extract an uploaded document and store its observations and facts."""
    pack = require_pack(ctx)
    document = ctx.store.get_document(ctx.user_id, args.doc_id)
    if document is None or document.status == "deleted":
        raise ToolError(f"No uploaded document {args.doc_id!r}")
    result = extract_file(document.file_path, pack, args.doc_type, backend=ctx.backend)
    saved = service.store_extraction(ctx.store, pack, ctx.user_id, args.doc_id, result)
    observations = result.observations or [None]
    rows = [_row(doc, obs) for doc, obs in zip(saved, observations)]
    return ExtractDocumentOutput(documents=rows, error=result.error)


SPEC = ToolSpec(
    name="extract_document",
    description="Read an uploaded document and store the values it really shows (with page and quote) as facts. "
                "Use it for documents with status 'pending' in read_twin.",
    input_model=ExtractDocumentInput, output_model=ExtractDocumentOutput, fn=run, kind="read")
