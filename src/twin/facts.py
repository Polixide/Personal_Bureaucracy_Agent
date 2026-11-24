"""From stored facts to the values the rule engine may use. Deterministic and pack-driven.

Rules (the provenance guarantees of milestone 1, now generic):
  1. Only confirmed facts count. Unconfirmed chat statements and observations that need
     confirmation are listed separately and never reach the rules.
  2. A resolution (the user's explicit choice) wins over every other value of that field.
  3. Within one source, the user's correction wins over the document's observation.
  4. If different sources disagree, the value is withheld (unknown) and the conflict is reported.
  5. Derived fields (procedure.json `derived`) are computed by code from the resulting values,
     unless the field already has an explicit value.
"""
import hashlib
import json
import unicodedata
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from src.models import DerivedField, FactMap, FactValue, Pack
from src.twin.models import RESOLUTION_REF, Fact, TwinDocument


class FactView(BaseModel):
    """What the Twin currently knows, ready for evaluation and display."""
    values: dict[str, FactValue] = Field(default_factory=dict)
    sources: dict[str, list[Fact]] = Field(default_factory=dict)
    conflicts: dict[str, list[FactValue]] = Field(default_factory=dict)
    unconfirmed: list[Fact] = Field(default_factory=list)
    derived: dict[str, FactValue] = Field(default_factory=dict)

    def fact_map(self) -> FactMap:
        """Values for src.rules (unknown fields are simply absent)."""
        return dict(self.values)


def comparison_key(value: Any) -> Any:
    """Equal-for-conflict form: text is NFKC-casefolded with punctuation-insensitive spacing."""
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFKC", value).casefold().replace(",", " ")
    return " ".join(text.split())


def _value_of_source(facts: list[Fact]) -> Fact:
    """One source's current value: a user correction beats the document; otherwise the newest."""
    corrections = [fact for fact in facts if fact.provenance == "user_stated"]
    chosen = corrections or facts
    return max(chosen, key=lambda fact: fact.fact_id or 0)


def _resolve_field(facts: list[Fact]) -> tuple[FactValue | None, list[FactValue]]:
    """Value of one field and, if sources disagree, the competing values."""
    resolutions = [fact for fact in facts if fact.source_ref == RESOLUTION_REF]
    if resolutions:
        return max(resolutions, key=lambda fact: fact.fact_id or 0).value, []
    by_source: dict[str, list[Fact]] = {}
    for fact in facts:
        by_source.setdefault(fact.source_ref, []).append(fact)
    candidates = [_value_of_source(group).value for group in by_source.values()]
    candidates = [value for value in candidates if value is not None]
    distinct: dict[Any, FactValue] = {}
    for value in candidates:
        distinct.setdefault(comparison_key(value), value)
    if len(distinct) == 1:
        return next(iter(distinct.values())), []
    return None, list(distinct.values())


def months_between(start: date, end: date) -> int | None:
    """Whole calendar months from start to end (minus one if the end day is earlier)."""
    if not isinstance(start, date) or not isinstance(end, date) or end < start:
        return None
    return (end.year - start.year) * 12 + end.month - start.month - (1 if end.day < start.day else 0)


def _document_fields_present(pack: Pack, document: TwinDocument, confirmed: list[Fact]) -> set[str]:
    """Document fields backed by evidence, plus fields the user corrected for that document."""
    observation = document.observation or {}
    present = set((observation.get("grounded") or {}).keys())
    spec = pack.document(document.doc_type or "")
    corrected = {fact.field for fact in confirmed if fact.source_ref == document.doc_id}
    if spec is not None:
        present |= {extract.source_field() for extract in spec.extracts if extract.field in corrected}
    return present


def _derive(pack: Pack, rule: DerivedField, values: dict[str, FactValue], documents: list[TwinDocument],
            confirmed: list[Fact], today: date) -> FactValue | None:
    if rule.op == "today":
        return today
    if rule.op == "months_between":
        start, end = (values.get(name) for name in rule.args)
        return months_between(start, end) if start is not None and end is not None else None
    if rule.op == "true_if_all":
        args = [values.get(name) for name in rule.args]
        return True if all(arg is not None and arg is not False for arg in args) else None
    if rule.op == "document_complete":
        for document in documents:
            if document.status == "analyzed" and document.doc_type == rule.doc_type:
                if set(rule.required) <= _document_fields_present(pack, document, confirmed):
                    return True
        return None
    raise ValueError(f"Unknown derivation {rule.op!r}")


def build_view(pack: Pack, facts: list[Fact], documents: list[TwinDocument], today: date) -> FactView:
    """Apply the provenance rules (module docstring) to the current facts of one procedure."""
    view = FactView()
    confirmed = [fact for fact in facts if fact.confirmed and fact.valid_to is None]
    view.unconfirmed = [fact for fact in facts if not fact.confirmed and fact.valid_to is None]
    by_field: dict[str, list[Fact]] = {}
    for fact in confirmed:
        by_field.setdefault(fact.field, []).append(fact)
    for field, field_facts in by_field.items():
        value, competing = _resolve_field(field_facts)
        view.sources[field] = field_facts
        if competing:
            view.conflicts[field] = competing
        elif value is not None:
            view.values[field] = value
    for rule in pack.procedure.derived:
        if rule.field in view.values or rule.field in view.conflicts:
            continue
        value = _derive(pack, rule, view.values, documents, confirmed, today)
        if value is not None:
            view.values[rule.field] = value
            view.derived[rule.field] = value
    return view


def facts_hash(values: FactMap) -> str:
    """Stable fingerprint of a fact map (sandbox results are tied to it)."""
    text = json.dumps({key: values[key] for key in sorted(values)}, default=str, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
