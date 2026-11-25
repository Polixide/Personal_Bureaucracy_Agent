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

from config import VISION_ENABLED

from src.document_processing import processor as document_processor
from src.document_processing.extractor import TextPage, extract_pages
from src.document_processing.generic import extract_generic
from src.document_processing.grounding import GroundedValue, grounded_values
from src.document_processing.observations import DocumentObservation, ExtractionResult, FactProposal
from src.document_processing.segment import Segment, segment_pages
from src.document_processing.vision import read_image
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
    pages: list[int] | None = Field(default=None, description="Only these pages of the file (one part of a split file)")


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


def _typed_infos(path: Path, pages: list[TextPage], doc_type: str | None, processor: Callable,
                 bundle: Callable | None) -> list[DocumentInfo]:
    """The milestone-1 typed processor on these pages (its own classification when doc_type is None).

    `bundle` splits a passport PDF into identity and entry-stamp pages; it is used only when the whole file is
    read as one document (a split file already has one segment per document).
    """
    if doc_type:
        info = processor(path, pages=pages, document_type_hint=doc_type)
    else:
        info = processor(path, pages=pages)
    if bundle is not None and info.document_type in PASSPORT_BUNDLE_TYPES and path.suffix.lower() == ".pdf":
        return bundle(path, info)
    return [info]


def unknown_observation(pages: list[int], tied: list[str]) -> DocumentObservation:
    """Pages nothing could be recognised on: no facts; the user can choose the type and read them again."""
    issue = ("Could not tell which document this is" + (f" (it could be: {', '.join(tied)})" if tied else "")
             + ". Choose its type and read it again.")
    return DocumentObservation(doc_type="unknown", issues=[issue], pages=pages, found_by="none")


def generic_observation(pack: Pack, spec_type: str, pages: list[TextPage], backend: Backend | None,
                        found_by: str) -> DocumentObservation:
    """A pack document without a typed schema: MRZ and checked model proposals (src/document_processing/generic.py)."""
    generic = extract_generic(pages, pack.document(spec_type), pack, backend)
    facts, issues = facts_from_grounded(pack, spec_type, generic.grounded)
    if backend is None:
        issues.append("No language model available: only machine-readable values were read")
    return DocumentObservation(doc_type=spec_type, grounded=generic.grounded, issues=generic.issues + issues,
                               facts=facts, pages=sorted({page.number for page in pages}), found_by=found_by,
                               notes=generic.notes)


def read_part(path: Path, pages: list[TextPage], pack: Pack, segment: Segment, backend: Backend | None,
              processor: Callable, bundle: Callable | None, typed_pack: bool) -> list[DocumentObservation]:
    """Read one document of the upload with the right path for its type.

    `bundle` is None when the file was split: each part is then read with the type its evidence showed.
    When the whole file is one document, a typed document keeps the typed processor's own classification
    (milestone-1 behaviour, the regression oracle), unless the user chose the type.
    """
    numbers = [page.number for page in pages]
    if segment.doc_type is not None and segment.doc_type not in DOCUMENT_SCHEMAS:
        return [generic_observation(pack, segment.doc_type, pages, backend, segment.how)]
    if not typed_pack:
        return [unknown_observation(numbers, segment.tied)]
    trusted = segment.how == "user" or (bundle is None and segment.how in ("mrz", "title", "model"))
    hint = segment.doc_type if trusted else None
    observations = []
    for info in _typed_infos(path, pages, hint, processor, bundle):
        observation = observe_typed(pack, info)
        observation.pages = sorted(info.selected_pages) if getattr(info, "selected_pages", None) else numbers
        observation.found_by = segment.how if hint else "model"
        observations.append(complete_typed(observation, pack, pages, backend))
    return observations


def complete_typed(observation: DocumentObservation, pack: Pack, pages: list[TextPage],
                   backend: Backend | None) -> DocumentObservation:
    """Second pass for values the typed processor left out (not those it found ambiguous).

    The generic reader proposes them with a quote that code finds on the page; they are stored as values the
    user must confirm, because the typed processor's stricter label checks did not accept them.
    """
    spec = pack.document(observation.doc_type)
    if spec is None or backend is None:
        return observation
    have = {fact.from_field for fact in observation.facts} | set(observation.ambiguous_fields)
    wanted = [extract.source_field() for extract in spec.extracts if extract.source_field() not in have]
    if not wanted:
        return observation
    generic = extract_generic(pages, spec, pack, backend, only=wanted)
    extra, issues = facts_from_grounded(pack, spec.doc_type, generic.grounded)
    known = {fact.field for fact in observation.facts}
    extra = [fact.model_copy(update={"requires_confirmation": True}) for fact in extra if fact.field not in known]
    if extra:
        observation.notes.append(f"{len(extra)} more values read with a quote from the page; please confirm them.")
    observation.notes += [note for note in generic.notes if note.startswith("Read again")]   # the reflection round
    observation.facts += extra
    observation.grounded.update({name: value for name, value in generic.grounded.items() if name not in observation.grounded})
    observation.issues += issues
    return observation


def vision_ready(backend: Backend | None) -> bool:
    """The vision model is switched on and the backend can show it images."""
    return VISION_ENABLED and backend is not None and hasattr(backend, "see")


def name_from_image(path: Path, pages: list[TextPage], pack: Pack, backend: Backend | None,
                    unknown: DocumentObservation) -> DocumentObservation:
    """A picture nothing printed on it could name: the vision model chooses among the pack's documents.

    Every value of a document named this way waits for the user's confirmation.
    """
    pictures = [page for page in pages if page.ocr]
    if not pictures or not vision_ready(backend):
        return unknown
    read = read_image(backend, pack, path, pictures[0], pack.procedure.documents)
    if read.doc_type is None:
        unknown.issues += read.issues
        return unknown
    facts, issues = facts_from_grounded(pack, read.doc_type, {**read.to_confirm, **read.checked})
    facts = [fact.model_copy(update={"requires_confirmation": True}) for fact in facts]
    return DocumentObservation(doc_type=read.doc_type, grounded={**read.to_confirm, **read.checked},
                               issues=read.issues + issues, facts=facts, pages=[page.number for page in pages],
                               found_by="vision",
                               notes=["Recognised from the image by the vision model: check the type and confirm the values."])


def fill_from_image(observation: DocumentObservation, path: Path, pages: list[TextPage], pack: Pack,
                    backend: Backend | None) -> DocumentObservation:
    """Values still missing on a picture: the vision model reads them, code compares them with the OCR text."""
    spec = pack.document(observation.doc_type)
    pictures = [page for page in pages if page.ocr]
    if spec is None or not pictures or not vision_ready(backend):
        return observation
    have = {fact.from_field for fact in observation.facts} | set(observation.ambiguous_fields)
    wanted = [extract.source_field() for extract in spec.extracts if extract.source_field() not in have]
    if not wanted:
        return observation
    read = read_image(backend, pack, path, pictures[0], [spec], wanted, known_type=spec.doc_type)
    checked, issues = facts_from_grounded(pack, spec.doc_type, read.checked)
    unsure, more = facts_from_grounded(pack, spec.doc_type, read.to_confirm)
    unsure = [fact.model_copy(update={"requires_confirmation": True}) for fact in unsure]
    known = {fact.field for fact in observation.facts}
    added = [fact for fact in checked + unsure if fact.field not in known]
    if added:
        observation.notes.append(f"{len(added)} values read from the image by the vision model"
                                 + (f"; {len(unsure)} to confirm." if unsure else ", matching the OCR text."))
    observation.facts += added
    observation.issues += read.issues + issues + more
    return observation


def look_at_pictures(observation: DocumentObservation, path: Path, pages: list[TextPage], pack: Pack,
                     backend: Backend | None) -> DocumentObservation:
    """The vision model's part: name an unrecognised picture, or complete a recognised one."""
    if observation.doc_type == "unknown":
        return name_from_image(path, pages, pack, backend, observation)
    return fill_from_image(observation, path, pages, pack, backend)


def find_documents(pages: list[TextPage], pack: Pack, doc_type: str | None, backend: Backend | None) -> list[Segment]:
    """The user's choice (the whole file is that document), or the documents the pages show."""
    if doc_type is not None:
        return [Segment(doc_type=doc_type, pages=[page.number for page in pages], how="user")]
    return segment_pages(pages, pack, backend)


def extract_file(path: str | Path, pack: Pack, doc_type: str | None = None, backend: Backend | None = None,
                 processor: Callable | None = None, bundle: Callable | None = None,
                 only_pages: list[int] | None = None) -> ExtractionResult:
    """Extract one file: find the documents in it, then read each one. Never raises for document or model problems."""
    path = Path(path)
    processor = processor or document_processor.process_document
    bundle = bundle or document_processor.process_passport_pages
    typed_pack = any(spec.doc_type in DOCUMENT_SCHEMAS for spec in pack.procedure.documents)
    if doc_type is not None and pack.document(doc_type) is None:
        return ExtractionResult(error=f"Unknown document type {doc_type!r} for procedure {pack.id}")
    try:
        pages = [page for page in extract_pages(path) if not only_pages or page.number in only_pages]
        if not pages:
            return ExtractionResult(error=f"The file has none of the pages {only_pages}")
        segments = find_documents(pages, pack, doc_type, backend)
        split = len(segments) > 1
        observations = []
        for segment in segments:
            part = [page for page in pages if page.number in segment.pages]
            for observation in read_part(path, part, pack, segment, backend, processor, None if split else bundle, typed_pack):
                observations.append(look_at_pictures(observation, path, part, pack, backend))
    except EXTRACTION_ERRORS as error:
        return ExtractionResult(error=str(error))
    return ExtractionResult(observations=observations)


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
    result = extract_file(document.file_path, pack, args.doc_type, backend=ctx.backend, only_pages=args.pages)
    saved = service.store_extraction(ctx.store, pack, ctx.user_id, args.doc_id, result)
    observations = result.observations or [None]
    rows = [_row(doc, obs) for doc, obs in zip(saved, observations)]
    return ExtractDocumentOutput(documents=rows, error=result.error)


SPEC = ToolSpec(
    name="extract_document",
    description="Read an uploaded document and store the values it really shows (with page and quote) as facts. "
                "Use it for documents with status 'pending' in read_twin.",
    input_model=ExtractDocumentInput, output_model=ExtractDocumentOutput, fn=run, kind="read")
