"""Pydantic models that cross module boundaries: Procedure Packs, rule results and issues.

A Procedure Pack is data (procedures/<id>/*.json). These models are its contract: the engine
reads only these fields, so adding a procedure never changes engine code.
"""
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FieldType = Literal["bool", "int", "str", "date"]
FactValue = bool | int | date | str
RuleStatus = Literal["pass", "fail", "unknown"]
Severity = Literal["blocking", "guardrail", "advisory"]
VerificationStatus = Literal["verified", "unverified"]
Operator = Literal["all", "any", "not", "eq", "in", "lt", "lte", "gt", "gte", "is_true"]
GROUP_OPS = ("all", "any")
COMPARISON_OPS = ("eq", "in", "lt", "lte", "gt", "gte")


class StrictModel(BaseModel):
    """Base for pack files: unknown keys are errors, so typos are caught by validation."""
    model_config = ConfigDict(extra="forbid")


class Issue(StrictModel):
    """A structured error: machine-readable code, short message, affected item."""
    code: str
    message: str
    item: str | None = None


# ------------------------------------------------------------- procedure --
class FieldSpec(StrictModel):
    """One fact the procedure knows about. `aliases` map typed-in text to a canonical value."""
    name: str
    type: FieldType
    question: str = ""
    aliases: dict[str, str] = Field(default_factory=dict)


class ExtractSpec(StrictModel):
    """A document field that becomes a fact. `from_field` is the document's own field name."""
    field: str
    from_field: str = ""
    requires_confirmation: bool = False

    def source_field(self) -> str:
        """The document field to read (defaults to the fact name)."""
        return self.from_field or self.field


class DocumentSpec(StrictModel):
    """A document the user can upload and the facts it provides."""
    doc_type: str
    description: str
    extracts: list[ExtractSpec]
    # Phrases printed on this document (any language, compared without accents or case), read by code to tell
    # which pages of an upload are which document (src/document_processing/segment.py).
    # `titles`: printed at the top of its first page; a page with one of them on top starts this document.
    # `keywords`: found anywhere in it; they name a document that has no title.
    titles: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # "passport" or "id_card" when the document has a machine-readable zone (read and checked by code).
    mrz: Literal["passport", "id_card"] | None = None

    @field_validator("extracts", mode="before")
    @classmethod
    def accept_plain_names(cls, value: Any) -> Any:
        """Allow `"holder_name"` as a short form of `{"field": "holder_name"}`."""
        if not isinstance(value, list):
            return value
        return [{"field": item} if isinstance(item, str) else item for item in value]


class Section(StrictModel):
    """A group of rules; the orchestrator delegates one section to one worker."""
    id: str
    title: str
    order: int = 0
    delegate: bool = True


class Stage(StrictModel):
    """An official step (e.g. submission). The sandbox accepts it only if `requires` all pass."""
    id: str
    title: str
    requires: list[str] = Field(default_factory=list)


class DerivedField(StrictModel):
    """A fact computed by code from other facts (never by the LLM).

    ops: months_between(start, end), true_if_all(fields...), document_complete(doc_type, required),
    today() (the evaluation date).
    """
    field: str
    op: Literal["months_between", "true_if_all", "document_complete", "today"]
    args: list[str] = Field(default_factory=list)
    doc_type: str = ""
    required: list[str] = Field(default_factory=list)


class Contact(StrictModel):
    """Who the user may write to. `email` stays None unless a source gives it."""
    role: str
    name: str
    email: str | None = None


class FormSpec(StrictModel):
    """A fillable PDF template in the pack: PDF field name -> fact name."""
    form_id: str
    title: str
    template: str
    fields: dict[str, str]
    stage_id: str = ""


class Condition(StrictModel):
    """A JSON condition tree node, evaluated with three-valued logic by src/rules.py."""
    op: Operator
    args: list["Condition"] = Field(default_factory=list)
    arg: "Condition | None" = None
    field: str | None = None
    value: Any = None
    other_field: str | None = None
    offset_days: int = 0

    @model_validator(mode="after")
    def check_shape(self) -> "Condition":
        """Each operator needs a specific shape; reject anything else early."""
        if self.op in GROUP_OPS and not self.args:
            raise ValueError(f"'{self.op}' needs a non-empty 'args' list")
        if self.op == "not" and self.arg is None:
            raise ValueError("'not' needs 'arg'")
        if self.op in COMPARISON_OPS + ("is_true",) and not self.field:
            raise ValueError(f"'{self.op}' needs 'field'")
        if self.op in COMPARISON_OPS:
            has_value = "value" in self.model_fields_set
            if has_value == (self.other_field is not None):
                raise ValueError(f"'{self.op}' needs exactly one of 'value' or 'other_field'")
        if self.op == "in" and not isinstance(self.value, list):
            raise ValueError("'in' needs a list 'value'")
        return self

    def nodes(self) -> list["Condition"]:
        """This node and every node below it."""
        found = [self]
        for child in list(self.args) + ([self.arg] if self.arg else []):
            found.extend(child.nodes())
        return found

    def leaf_fields(self) -> list[str]:
        """Every field name this condition reads (used by validation and workers)."""
        names: list[str] = []
        if self.field:
            names.append(self.field)
        if self.other_field:
            names.append(self.other_field)
        children = list(self.args) + ([self.arg] if self.arg else [])
        for child in children:
            names.extend(child.leaf_fields())
        return names


class Rule(StrictModel):
    """One requirement with its sources. `status: unverified` must be disclosed in answers."""
    rule_id: str
    section: str
    description: str
    severity: Severity
    condition: Condition
    depends_on: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    version: str
    status: VerificationStatus
    name: str = ""
    category: str = ""
    volatility: str = ""
    notes: str = ""
    legacy_condition: str = ""


class Source(StrictModel):
    """An official page a rule is based on. URLs and dates are never invented."""
    source_id: str
    title: str
    url: str
    retrieved_at: str
    authority: str
    organization: str = ""
    notes: str = ""
    # Hash of the saved text in sources_text/ (written by `python -m src.rag fetch`); "" = never fetched.
    content_sha256: str = ""


class Procedure(StrictModel):
    """procedure.json: what the procedure is, its sections, fields, documents and stages."""
    id: str
    title: str
    status: VerificationStatus
    version: str = "1.0"
    aliases: list[str] = Field(default_factory=list)
    # Words that tell this pack apart from a sibling ("lost", "first", "renew"). They only add to the score
    # when an alias matched too, so "I lost my passport" does not select an ID-card pack.
    qualifiers: list[str] = Field(default_factory=list)
    scope: str
    excluded: list[str] = Field(default_factory=list)
    scope_conditions: list[Condition] = Field(default_factory=list)
    sections: list[Section]
    stages: list[Stage] = Field(default_factory=list)
    fields: list[FieldSpec]
    documents: list[DocumentSpec] = Field(default_factory=list)
    derived: list[DerivedField] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    forms: list[FormSpec] = Field(default_factory=list)


class Pack(BaseModel):
    """A loaded Procedure Pack: procedure + rules + sources, with small lookup helpers."""
    procedure: Procedure
    rules: list[Rule]
    sources: list[Source]
    root: Path

    @property
    def id(self) -> str:
        return self.procedure.id

    def field(self, name: str) -> FieldSpec | None:
        """The field spec called `name`, or None."""
        return next((f for f in self.procedure.fields if f.name == name), None)

    def rule(self, rule_id: str) -> Rule | None:
        """The rule called `rule_id`, or None."""
        return next((r for r in self.rules if r.rule_id == rule_id), None)

    def source(self, source_id: str) -> Source | None:
        """The source called `source_id`, or None."""
        return next((s for s in self.sources if s.source_id == source_id), None)

    def section(self, section_id: str) -> Section | None:
        """The section called `section_id`, or None."""
        return next((s for s in self.procedure.sections if s.id == section_id), None)

    def stage(self, stage_id: str) -> Stage | None:
        """The stage called `stage_id`, or None."""
        return next((s for s in self.procedure.stages if s.id == stage_id), None)

    def document(self, doc_type: str) -> DocumentSpec | None:
        """The document spec for `doc_type`, or None."""
        return next((d for d in self.procedure.documents if d.doc_type == doc_type), None)

    def rules_in_section(self, section_id: str) -> list[Rule]:
        """Rules of one section, in file order."""
        return [r for r in self.rules if r.section == section_id]

    def rule_ids(self) -> list[str]:
        """All rule ids, in file order."""
        return [r.rule_id for r in self.rules]


# ----------------------------------------------------------- evaluation --
class RuleResult(StrictModel):
    """The deterministic verdict of one rule on the current facts."""
    rule_id: str
    section: str
    description: str
    status: RuleStatus
    severity: Severity
    missing_fields: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    rule_status: VerificationStatus
    blocked_by: list[str] = Field(default_factory=list)


FactMap = dict[str, FactValue | None]
