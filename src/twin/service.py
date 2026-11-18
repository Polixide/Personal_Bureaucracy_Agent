"""Twin operations used by tools and the app. Real -> twin paths live here.

Real -> twin: uploads and their extraction, user statements and confirmations, conflict
resolutions, outcomes reported from the real world, and rule updates. Twin -> real happens only
through approved proposals (src/sandbox/approval.py).
"""
from datetime import date
from typing import Any

from src.models import Pack
from src.pack import coerce_fact
from src.document_processing.observations import ExtractionResult
from src.twin.facts import FactView, build_view
from src.twin.models import (APP_REF, CHAT_REF, OUTCOME_PREFIX, RESOLUTION_REF, Event, Fact, TwinDocument)
from src.twin.store import TwinStore


def twin_view(store: TwinStore, pack: Pack, user_id: str, today: date) -> FactView:
    """The current fact view of one user for one procedure."""
    return build_view(pack, store.current_facts(user_id, pack.id), store.list_documents(user_id), today)


def record_statement(store: TwinStore, pack: Pack, user_id: str, field: str, raw: Any, *, confirmed: bool,
                     source_ref: str = CHAT_REF, note: str = "") -> Fact:
    """Store something the user said. Chat statements stay unconfirmed until the user confirms them."""
    value = coerce_fact(pack, field, raw)
    fact = Fact(user_id=user_id, procedure_id=pack.id, field=field, value=value, provenance="user_stated",
                source_ref=source_ref, confirmed=confirmed, note=note)
    return store.add_fact(fact)


def correct_document_field(store: TwinStore, pack: Pack, user_id: str, doc_id: str, field: str, raw: Any) -> Fact:
    """The user corrects (or confirms) one value of one document; the observation itself is kept."""
    if store.get_document(user_id, doc_id) is None:
        raise ValueError(f"Unknown document {doc_id!r}")
    return record_statement(store, pack, user_id, field, raw, confirmed=True, source_ref=doc_id,
                            note="user correction of the document observation")


def resolve_conflict(store: TwinStore, pack: Pack, user_id: str, field: str, raw: Any) -> Fact:
    """The user explicitly chooses the value of a field whose sources disagree."""
    return record_statement(store, pack, user_id, field, raw, confirmed=True, source_ref=RESOLUTION_REF,
                            note="explicit conflict resolution")


def register_upload(store: TwinStore, user_id: str, doc_id: str, file_path: str, procedure_id: str | None = None) -> TwinDocument:
    """A new upload enters the twin as a pending document."""
    with store.transaction():
        document = store.save_document(user_id, TwinDocument(doc_id=doc_id, file_path=file_path, status="pending"))
        store.add_event(Event(user_id=user_id, procedure_id=procedure_id, kind="document_uploaded",
                              direction="real_to_twin", payload={"doc_id": doc_id}))
    return document


def _part_id(doc_id: str, index: int) -> str:
    """Logical id of the n-th document found in one file (a passport PDF can hold two)."""
    return doc_id if index == 0 else f"{doc_id}_part{index + 1}"


def store_extraction(store: TwinStore, pack: Pack, user_id: str, doc_id: str, result: ExtractionResult) -> list[TwinDocument]:
    """Save the observations of one file and replace that document's previous document facts."""
    original = store.get_document(user_id, doc_id)
    if original is None:
        raise ValueError(f"Unknown document {doc_id!r}")
    with store.transaction():
        if result.error:
            store.close_source(user_id, pack.id, doc_id, provenance="document")
            saved = store.save_document(user_id, original.model_copy(update={"status": "rejected", "processing_error": result.error}))
            store.add_event(Event(user_id=user_id, procedure_id=pack.id, kind="document_rejected",
                                  direction="real_to_twin", payload={"doc_id": doc_id}))
            return [saved]
        return [_store_observation(store, pack, user_id, original, index, observation)
                for index, observation in enumerate(result.observations)]


def _store_observation(store: TwinStore, pack: Pack, user_id: str, original: TwinDocument, index: int, observation) -> TwinDocument:
    part_id = _part_id(original.doc_id, index)
    store.close_source(user_id, pack.id, part_id, provenance="document")
    saved = store.save_document(user_id, TwinDocument(
        doc_id=part_id, file_path=original.file_path, status="analyzed", doc_type=observation.doc_type,
        observation=observation.model_dump(mode="json"), processing_error=None))
    for proposal in observation.facts:
        store.add_fact(Fact(user_id=user_id, procedure_id=pack.id, field=proposal.field, value=proposal.value,
                            provenance="document", source_ref=part_id, page=proposal.page, snippet=proposal.snippet,
                            confirmed=not proposal.requires_confirmation, note=f"from {proposal.from_field}"))
    store.add_event(Event(user_id=user_id, procedure_id=pack.id, kind="document_analyzed", direction="real_to_twin",
                          payload={"doc_id": part_id, "doc_type": observation.doc_type,
                                   "facts": len(observation.facts), "issues": len(observation.issues)}))
    return saved


def delete_document(store: TwinStore, user_id: str, procedure_id: str, doc_id: str) -> None:
    """Remove a document from the current state; its facts (and corrections) stop counting."""
    document = store.get_document(user_id, doc_id)
    if document is None:
        raise ValueError(f"Unknown document {doc_id!r}")
    with store.transaction():
        store.close_source(user_id, procedure_id, doc_id)
        store.save_document(user_id, document.model_copy(update={"status": "deleted"}))
        store.add_event(Event(user_id=user_id, procedure_id=procedure_id, kind="document_deleted",
                              direction="real_to_twin", payload={"doc_id": doc_id}))


def record_outcome(store: TwinStore, pack: Pack, user_id: str, kind: str, description: str,
                   fields: dict[str, Any] | None = None) -> list[Fact]:
    """Real -> twin: the user reports what happened (e.g. "IRO confirmed my scans").

    Every field is validated before anything is written; the reported values are confirmed facts.
    """
    typed = {name: coerce_fact(pack, name, raw) for name, raw in (fields or {}).items()}
    with store.transaction():
        event = store.add_event(Event(user_id=user_id, procedure_id=pack.id, kind="outcome", direction="real_to_twin",
                                      payload={"kind": kind, "description": description,
                                               "fields": {k: str(v) for k, v in typed.items()}}))
        source = f"{OUTCOME_PREFIX}{event.event_id}"
        return [store.add_fact(Fact(user_id=user_id, procedure_id=pack.id, field=name, value=value,
                                    provenance="user_stated", source_ref=source, confirmed=True, note=description))
                for name, value in typed.items()]


def record_rule_update(store: TwinStore, user_id: str, procedure_id: str, rule_id: str, new_version: str, note: str) -> int:
    """Real -> twin: an official rule changed. Lessons tied to it are retired. Returns how many."""
    with store.transaction():
        store.add_event(Event(user_id=user_id, procedure_id=procedure_id, kind="rule_update", direction="real_to_twin",
                              payload={"rule_id": rule_id, "version": new_version, "note": note}))
        return store.invalidate_episodes(rule_id)


def record_app_value(store: TwinStore, pack: Pack, user_id: str, field: str, raw: Any) -> Fact:
    """A value the user typed in the app form: explicit, so it is confirmed."""
    return record_statement(store, pack, user_id, field, raw, confirmed=True, source_ref=APP_REF,
                            note="entered in the app")
