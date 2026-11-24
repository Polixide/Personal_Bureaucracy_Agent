"""Where the user is and what to do next, computed by code from the Twin (no model).

The briefing gathers what the app's views show one by one (documents, requirements, open questions,
values to confirm, actions waiting for approval) into one ordered plan:
  1. decide   actions the agent prepared and the user must approve or deny;
  2. fix      blocking requirements that fail;
  3. confirm  values read or stated that wait for the user's check, and documents that disagree;
  4. upload   documents that would answer open questions;
  5. answer   the remaining questions;
  6. improve  advisory requirements that are not met;
  7. try      when nothing blocks: the official step in the sandbox.
The assistant (src/agent/chat.py) explains this plan in plain words and answers questions about it. Without
the model, `briefing_markdown` is the answer, so the user always has the plan.
"""
from typing import Literal

from pydantic import BaseModel, Field

from src.models import Pack, RuleResult, Section
from src.twin.facts import FactView
from src.twin.models import ProposedAction, TwinDocument

OPEN_PROPOSALS = ("proposed", "blocked")
ORIGIN_WORDS = {"document": "a document", "user_stated": "you", "tool": "a tool", "derived": "computed"}
NOTE_CHARS = 220          # how much of a rule's source quote the model sees

StepKind = Literal["decide", "fix", "confirm", "upload", "answer", "improve", "try"]


# ------------------------------------------------------- shared helpers --
def readable(name: str) -> str:
    """'passport_expiry_date' -> 'passport expiry date'."""
    return name.replace("_", " ")


def field_question(pack: Pack, field: str) -> str:
    """The pack's question for a field, or its name as a short question when the pack has none."""
    spec = pack.field(field)
    if spec is not None and spec.question:
        return spec.question
    label = readable(field).capitalize()
    return f"{label}?" if spec is not None and spec.type == "bool" else label


def user_sections(pack: Pack) -> list[Section]:
    """Sections about the user's file, in order. Sections with delegate=false are the agent's guardrails."""
    ordered = sorted(pack.procedure.sections, key=lambda section: section.order)
    return [section for section in ordered if section.delegate]


def in_section(results: list[RuleResult], section_id: str) -> list[RuleResult]:
    return [result for result in results if result.section == section_id]


def user_results(pack: Pack, results: list[RuleResult]) -> list[RuleResult]:
    """Verdicts of the user sections only."""
    wanted = {section.id for section in user_sections(pack)}
    return [result for result in results if result.section in wanted]


def count_status(results: list[RuleResult]) -> dict[str, int]:
    return {status: sum(result.status == status for result in results) for status in ("pass", "fail", "unknown")}


def missing_fields(results: list[RuleResult]) -> list[str]:
    """Fields that unknown rules wait for, without duplicates, in rule order."""
    fields: list[str] = []
    for result in results:
        if result.status != "unknown":
            continue
        for field in result.missing_fields:
            if field not in fields:
                fields.append(field)
    return fields


def askable_fields(pack: Pack, missing: list[str]) -> list[str]:
    """Fields the user can answer. A derived field is replaced by the fields it is computed from."""
    derived = {spec.field: spec for spec in pack.procedure.derived}
    asked: list[str] = []
    for field in missing:
        inputs = derived[field].args if field in derived else [field]
        for name in inputs:
            if name not in asked and name not in derived and pack.field(name) is not None:
                asked.append(name)
    return asked


def documents_for(pack: Pack, missing: list[str]) -> list[str]:
    """Descriptions of the documents whose upload would fill some of the missing fields."""
    return [pack.document(doc_type).description for doc_type in document_types_for(pack, missing)]


def document_types_for(pack: Pack, missing: list[str]) -> list[str]:
    """Document types whose upload would fill some of the missing fields (directly or via a derived field)."""
    derived = {spec.field: spec for spec in pack.procedure.derived}
    wanted, doc_types = set(missing), set()
    for field in missing:
        if field in derived:
            wanted.update(derived[field].args)
            if derived[field].doc_type:
                doc_types.add(derived[field].doc_type)
    return [document.doc_type for document in pack.procedure.documents
            if document.doc_type in doc_types or {extract.field for extract in document.extracts} & wanted]


def document_status(pack: Pack, documents: list[TwinDocument], view: FactView, doc_type: str) -> str:
    """complete / needs confirmation / missing / ambiguous, for one document type of the pack."""
    mine = [document for document in documents if document.doc_type == doc_type]
    analyzed = [document for document in mine if document.status == "analyzed" and not document.processing_error]
    if not mine:
        return "missing"
    if not analyzed:
        return "ambiguous"
    ids = {document.doc_id for document in analyzed}
    fields = {extract.field for extract in pack.document(doc_type).extracts}
    waiting = any(fact.source_ref in ids for fact in view.unconfirmed) or bool(fields & set(view.conflicts))
    return "needs confirmation" if waiting else "complete"


def checklist_rows(pack: Pack, documents: list[TwinDocument], view: FactView) -> list[dict]:
    """One row per document of the pack: its label and status."""
    return [{"doc_type": spec.doc_type, "label": spec.description,
             "status": document_status(pack, documents, view, spec.doc_type)} for spec in pack.procedure.documents]


# -------------------------------------------------------------- models --
class DocumentLine(BaseModel):
    doc_type: str
    label: str
    status: str


class SectionLine(BaseModel):
    section_id: str
    title: str
    passed: int
    failed: int
    unknown: int


class Requirement(BaseModel):
    """A requirement that is not met, with where it comes from."""
    rule_id: str
    description: str
    severity: str
    verified: bool
    source_ids: list[str]
    quote: str = ""


class Question(BaseModel):
    field: str
    question: str
    rule_ids: list[str]
    documents: list[str] = Field(default_factory=list)   # doc types whose upload answers it


class ValueToCheck(BaseModel):
    field: str
    value: str
    origin: str
    fact_id: int | None = None


class Step(BaseModel):
    """One thing to do, in plan order. `ref` points at what it is about (rule, field, document, action, stage)."""
    kind: StepKind
    title: str
    ref: str
    detail: str = ""


class Briefing(BaseModel):
    """Everything the user needs to know right now about one procedure."""
    procedure_id: str
    title: str
    unverified: bool
    where: str
    checks_passed: int
    checks_total: int
    documents_ready: int
    documents_total: int
    documents: list[DocumentLine]
    sections: list[SectionLine]
    problems: list[Requirement]
    advice: list[Requirement]
    questions: list[Question]
    to_check: list[ValueToCheck]
    conflicts: dict[str, list[str]]
    pending: list[dict]
    steps: list[Step]
    rule_status: dict[str, str] = Field(default_factory=dict)   # every user rule id -> pass / fail / unknown

    def refs(self) -> set[str]:
        """Ids an answer about this briefing may point to."""
        return {step.ref for step in self.steps} | {q.field for q in self.questions} | {d.doc_type for d in self.documents}


# ------------------------------------------------------------- build --
def requirement(pack: Pack, result: RuleResult) -> Requirement:
    rule = pack.rule(result.rule_id)
    quote = (rule.notes or "")[:NOTE_CHARS] if rule is not None else ""
    return Requirement(rule_id=result.rule_id, description=result.description, severity=result.severity,
                       verified=result.rule_status != "unverified", source_ids=result.source_ids, quote=quote)


def open_questions(pack: Pack, results: list[RuleResult], waiting: set[str] | None = None) -> list[Question]:
    """One question per askable field, with the rules waiting for it and the documents that would answer it."""
    mine = user_results(pack, results)
    questions = []
    for field in askable_fields(pack, missing_fields(mine)):
        if field in (waiting or set()):      # a value already waits for the user's check: the confirm step asks it
            continue
        waiting = [r.rule_id for r in mine if r.status == "unknown" and (field in r.missing_fields or
                   any(field in spec.args for spec in pack.procedure.derived if spec.field in r.missing_fields))]
        questions.append(Question(field=field, question=field_question(pack, field), rule_ids=waiting,
                                  documents=document_types_for(pack, [field])))
    return questions


def values_to_check(view: FactView) -> list[ValueToCheck]:
    return [ValueToCheck(field=fact.field, value=str(fact.value), origin=ORIGIN_WORDS.get(fact.provenance, fact.provenance),
                         fact_id=fact.fact_id) for fact in view.unconfirmed]


def where_label(problems: list, pending: list, questions: list, to_check: list, conflicts: dict) -> str:
    """The user's position in one sentence."""
    if pending:
        return "Waiting for your decision on actions the assistant prepared"
    if problems:
        return "Some requirements are not met yet"
    if questions:
        return "Collecting documents and information"
    if to_check or conflicts:
        return "Almost there: a few values need your check"
    return "Ready for the official step (nothing blocking right now)"


def build_briefing(pack: Pack, documents: list[TwinDocument], proposals: list[ProposedAction], view: FactView,
                   results: list[RuleResult]) -> Briefing:
    """The briefing of one procedure from the Twin's current state (pure: no store, no model)."""
    mine = user_results(pack, results)
    counts = count_status(mine)
    rows = [DocumentLine(**row) for row in checklist_rows(pack, documents, view)]
    problems = [requirement(pack, r) for r in mine if r.status == "fail" and r.severity == "blocking"]
    advice = [requirement(pack, r) for r in mine if r.status == "fail" and r.severity != "blocking"]
    questions = open_questions(pack, results, {fact.field for fact in view.unconfirmed})
    to_check = values_to_check(view)
    conflicts = {field: [str(v) for v in values] for field, values in view.conflicts.items()}
    pending = [{"action_id": p.action_id, "title": p.title, "tool": p.tool} for p in proposals
               if p.status in OPEN_PROPOSALS and p.procedure_id == pack.id]
    sections = [SectionLine(section_id=s.id, title=s.title, passed=c["pass"], failed=c["fail"], unknown=c["unknown"])
                for s in user_sections(pack) for c in [count_status(in_section(results, s.id))]]
    briefing = Briefing(
        procedure_id=pack.id, title=pack.procedure.title,
        unverified=pack.procedure.status == "unverified" or any(r.rule_status == "unverified" for r in mine),
        where=where_label(problems, pending, questions, to_check, conflicts), checks_passed=counts["pass"],
        checks_total=len(mine), documents_ready=sum(row.status == "complete" for row in rows), documents_total=len(rows),
        documents=rows, sections=sections, problems=problems, advice=advice, questions=questions, to_check=to_check,
        conflicts=conflicts, pending=pending, steps=[], rule_status={r.rule_id: r.status for r in mine})
    briefing.steps = plan_steps(pack, briefing)
    return briefing


# -------------------------------------------------------------- plan --
def plan_steps(pack: Pack, briefing: Briefing) -> list[Step]:
    """The ordered plan (see the module docstring)."""
    steps = [Step(kind="decide", title=f"Approve or deny: {item['title']}", ref=item["action_id"]) for item in briefing.pending]
    steps += [Step(kind="fix", title=f"Fix: {p.description}", ref=p.rule_id, detail=_source_note(p)) for p in briefing.problems]
    steps += [Step(kind="confirm", title=f"Check the {readable(v.field)}: {v.value} (from {v.origin})", ref=v.field)
              for v in briefing.to_check]
    steps += [Step(kind="confirm", title=f"Your documents disagree on the {readable(field)}: choose the right value", ref=field)
              for field in briefing.conflicts]
    steps += upload_steps(pack, briefing)
    covered = {question.field for step in steps if step.kind == "upload" for question in briefing.questions
               if step.ref in question.documents}
    steps += [Step(kind="answer", title=q.question, ref=q.field) for q in briefing.questions if q.field not in covered]
    steps += [Step(kind="improve", title=f"Recommended: {a.description}", ref=a.rule_id, detail=_source_note(a))
              for a in briefing.advice]
    if not briefing.problems and not briefing.questions:
        steps += [Step(kind="try", title=f"Try '{stage.title}' in the sandbox, then do it at the official office",
                       ref=stage.id) for stage in pack.procedure.stages[:1]]
    return steps


def upload_steps(pack: Pack, briefing: Briefing) -> list[Step]:
    """Documents not uploaded yet that would answer open questions, the most useful first."""
    status = {line.doc_type: line.status for line in briefing.documents}
    answered: dict[str, list[str]] = {}
    for question in briefing.questions:
        for doc_type in question.documents:
            if status.get(doc_type) in ("missing", "ambiguous"):
                answered.setdefault(doc_type, []).append(readable(question.field))
    ordered = sorted(answered, key=lambda doc_type: -len(answered[doc_type]))
    return [Step(kind="upload", title=f"Upload, if you have it: {pack.document(doc_type).description}", ref=doc_type,
                 detail="It answers: " + ", ".join(answered[doc_type])) for doc_type in ordered]


def _source_note(requirement_: Requirement) -> str:
    sources = ", ".join(requirement_.source_ids)
    label = "" if requirement_.verified else " (unverified rule)"
    return f"Source: {sources}{label}" if sources else label.strip()


# ------------------------------------------------------------- render --
STEP_ICONS = {"decide": "🗳️", "fix": "❌", "confirm": "🔎", "upload": "📄", "answer": "✍️", "improve": "💡", "try": "🧪"}


def briefing_markdown(briefing: Briefing, limit: int = 8) -> str:
    """The plan in plain words: the answer when the model cannot be used, and the chat's first message."""
    lines = [f"**{briefing.title}** · {briefing.where}.",
             f"{briefing.checks_passed} of {briefing.checks_total} requirements are met; "
             f"{briefing.documents_ready} of {briefing.documents_total} documents are ready."]
    if briefing.steps:
        lines.append("")
        lines.append("**What to do next:**")
        for number, step in enumerate(briefing.steps[:limit], start=1):
            detail = f" — {step.detail}" if step.detail else ""
            lines.append(f"{number}. {STEP_ICONS[step.kind]} {step.title}{detail}")
        if len(briefing.steps) > limit:
            lines.append(f"…and {len(briefing.steps) - limit} more (see the other tabs).")
    if briefing.unverified:
        lines.append("")
        lines.append("_Some rules of this procedure are unverified: check every result with the official office._")
    return "\n".join(lines)


MODEL_STEPS = 10        # plan steps the model sees (the rest are in the app's tabs)
MODEL_QUESTIONS = 10
MODEL_QUOTE_CHARS = 160


def state_for_model(briefing: Briefing) -> dict:
    """The briefing as short data for the model's prompt, without empty parts.

    Short on purpose: a 4,600-token prompt made the first chat call take about a minute on the 4 GB GPU.
    """
    data = {
        "procedure": briefing.title, "procedure_id": briefing.procedure_id, "where": briefing.where,
        "rules_unverified": briefing.unverified,
        "progress": f"{briefing.checks_passed} of {briefing.checks_total} requirements met; "
                    f"{briefing.documents_ready} of {briefing.documents_total} documents ready",
        "documents": {line.label: line.status for line in briefing.documents},
        "sections": [{"id": s.section_id, "title": s.title,
                      "status": f"{s.passed} met, {s.failed} not met, {s.unknown} waiting for information"}
                     for s in briefing.sections],
        "rule_status": briefing.rule_status,
        "not_met": [{"rule_id": p.rule_id, "requirement": p.description, "severity": p.severity,
                     "sources": p.source_ids, "quote": p.quote[:MODEL_QUOTE_CHARS]} for p in briefing.problems + briefing.advice],
        "questions": [{"field": q.field, "question": q.question} for q in briefing.questions[:MODEL_QUESTIONS]],
        "values_to_check": [f"{v.field} = {v.value} (from {v.origin})" for v in briefing.to_check],
        "conflicts": briefing.conflicts, "waiting_for_approval": [item["title"] for item in briefing.pending],
        "steps": [{"do": step.title + (f" ({step.detail})" if step.detail else ""), "kind": step.kind, "ref": step.ref}
                  for step in briefing.steps[:MODEL_STEPS]],
    }
    if len(briefing.questions) > MODEL_QUESTIONS:
        data["more_questions"] = len(briefing.questions) - MODEL_QUESTIONS
    return {key: value for key, value in data.items() if value not in ([], {}, "")}
