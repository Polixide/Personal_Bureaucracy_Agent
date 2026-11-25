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
What the pack marks as later (sections about steps after the application, fields asked "later") is kept apart in
`later`, and fields the pack never asks (computed, read from documents only, published by an office) are not
questions. A yes/no question that a missing document answers ("Do you have a copy of your passport page?") becomes
an upload step instead.
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


def now_results(pack: Pack, results: list[RuleResult]) -> list[RuleResult]:
    """Verdicts of the user sections that are about today (not the later steps)."""
    wanted = {section.id for section in user_sections(pack) if not section.later}
    return [result for result in results if result.section in wanted]


def later_results(pack: Pack, results: list[RuleResult]) -> list[RuleResult]:
    """Verdicts of the user sections about steps after the application."""
    wanted = {section.id for section in user_sections(pack) if section.later}
    return [result for result in results if result.section in wanted]


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


def askable_fields(pack: Pack, missing: list[str], when: str = "now") -> list[str]:
    """Fields the user can answer now (or `when`="later"). A derived field is replaced by the fields it comes from.

    Fields the pack marks ask="never" (computed, read from documents only, published by an office) are left out.
    """
    derived = {spec.field: spec for spec in pack.procedure.derived}
    asked: list[str] = []
    for field in missing:
        inputs = derived[field].args if field in derived else [field]
        for name in inputs:
            spec = pack.field(name)
            if name not in asked and name not in derived and spec is not None and spec.ask == when:
                asked.append(name)
    return asked


def field_label(field: str) -> str:
    """'passport_id_page_present' -> 'passport id page' (what a document gives, in words)."""
    return readable(field).removesuffix(" present")


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
    """One row per document of the pack: its label, status, and whether it is needed only in some cases."""
    return [{"doc_type": spec.doc_type, "label": spec.description, "optional": spec.optional,
             "status": document_status(pack, documents, view, spec.doc_type)} for spec in pack.procedure.documents]


def optional_only_fields(pack: Pack) -> set[str]:
    """Fields that only optional documents give (e.g. utility_bill_present)."""
    required = {e.field for spec in pack.procedure.documents if not spec.optional for e in spec.extracts}
    return {e.field for spec in pack.procedure.documents if spec.optional for e in spec.extracts} - required


def wanted_fields(pack: Pack, results: list[RuleResult]) -> list[str]:
    """What these rules wait for, for the upload steps.

    A field that only optional documents give counts only when its rule waits for nothing else (the utility bill
    is suggested once you said you live in your own property, not to everybody).
    """
    optional = optional_only_fields(pack)
    fields: list[str] = []
    for result in results:
        if result.status != "unknown":
            continue
        for field in result.missing_fields:
            only_optional = field in optional and not set(result.missing_fields) <= optional
            if not only_optional and field not in fields:
                fields.append(field)
    return fields


# -------------------------------------------------------------- models --
class DocumentLine(BaseModel):
    doc_type: str
    label: str
    status: str
    optional: bool = False


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
    upload_instead: bool = False   # a yes/no about having a document that is not uploaded: the upload answers it
    choices: dict[str, str] = Field(default_factory=dict)   # text shown -> value stored


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
    later: list[Question] = Field(default_factory=list)   # questions about steps after the application

    def refs(self) -> set[str]:
        """Ids an answer about this briefing may point to."""
        return {step.ref for step in self.steps} | {q.field for q in self.questions} | {d.doc_type for d in self.documents}


# ------------------------------------------------------------- build --
def requirement(pack: Pack, result: RuleResult) -> Requirement:
    rule = pack.rule(result.rule_id)
    quote = (rule.notes or "")[:NOTE_CHARS] if rule is not None else ""
    return Requirement(rule_id=result.rule_id, description=result.description, severity=result.severity,
                       verified=result.rule_status != "unverified", source_ids=result.source_ids, quote=quote)


def open_questions(pack: Pack, results: list[RuleResult], waiting: set[str] | None = None,
                   documents: list[TwinDocument] | None = None, when: str = "now") -> list[Question]:
    """One question per askable field, with the rules waiting for it and the documents that would answer it.

    when="now": today's sections; when="later": fields asked later, and today's fields of the later sections.
    """
    if when == "now":
        mine, fields = now_results(pack, results), askable_fields(pack, missing_fields(now_results(pack, results)))
    else:
        mine = user_results(pack, results)
        fields = askable_fields(pack, missing_fields(mine), "later")
        fields += [f for f in askable_fields(pack, missing_fields(later_results(pack, results))) if f not in fields]
    read = {document.doc_type for document in documents or [] if document.status == "analyzed"}
    questions = []
    for field in fields:
        if field in (waiting or set()):      # a value already waits for the user's check: the confirm step asks it
            continue
        rule_ids = [r.rule_id for r in mine if r.status == "unknown" and (field in r.missing_fields or
                    any(field in spec.args for spec in pack.procedure.derived if spec.field in r.missing_fields))]
        doc_types = document_types_for(pack, [field])
        spec = pack.field(field)
        instead = spec.type == "bool" and bool(doc_types) and not read & set(doc_types)
        questions.append(Question(field=field, question=field_question(pack, field), rule_ids=rule_ids,
                                  documents=doc_types, upload_instead=instead, choices=spec.choices))
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
    today = now_results(pack, results)
    counts = count_status(mine)
    rows = [DocumentLine(**row) for row in checklist_rows(pack, documents, view)]
    problems = [requirement(pack, r) for r in today if r.status == "fail" and r.severity == "blocking"]
    advice = [requirement(pack, r) for r in today if r.status == "fail" and r.severity != "blocking"]
    waiting = {fact.field for fact in view.unconfirmed}
    questions = open_questions(pack, results, waiting, documents)
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
        checks_total=len(mine), documents_ready=sum(row.status == "complete" for row in rows if not row.optional),
        documents_total=sum(not row.optional for row in rows),
        documents=rows, sections=sections, problems=problems, advice=advice, questions=questions, to_check=to_check,
        conflicts=conflicts, pending=pending, steps=[], rule_status={r.rule_id: r.status for r in mine},
        later=open_questions(pack, results, waiting, documents, "later"))
    briefing.steps = plan_steps(pack, briefing, wanted_fields(pack, today))
    return briefing


# -------------------------------------------------------------- plan --
def plan_steps(pack: Pack, briefing: Briefing, missing: list[str] | None = None) -> list[Step]:
    """The ordered plan (see the module docstring). `missing`: what today's rules wait for (for the uploads)."""
    steps = [Step(kind="decide", title=f"Approve or deny: {item['title']}", ref=item["action_id"]) for item in briefing.pending]
    steps += [Step(kind="fix", title=f"Fix: {p.description}", ref=p.rule_id, detail=_source_note(p)) for p in briefing.problems]
    steps += [Step(kind="confirm", title=f"Check the {readable(v.field)}: {v.value} (from {v.origin})", ref=v.field)
              for v in briefing.to_check]
    steps += [Step(kind="confirm", title=f"Your documents disagree on the {readable(field)}: choose the right value", ref=field)
              for field in briefing.conflicts]
    steps += upload_steps(pack, briefing, missing)
    covered = {question.field for step in steps if step.kind == "upload" for question in briefing.questions
               if step.ref in question.documents}
    steps += [Step(kind="answer", title=q.question, ref=q.field) for q in briefing.questions
              if q.field not in covered and not q.upload_instead]
    steps += [Step(kind="improve", title=f"Recommended: {a.description}", ref=a.rule_id, detail=_source_note(a))
              for a in briefing.advice]
    if not briefing.problems and not briefing.questions:
        steps += [Step(kind="try", title=f"Try '{stage.title}' in the sandbox, then do it at the official office",
                       ref=stage.id) for stage in pack.procedure.stages[:1]]
    return steps


def upload_steps(pack: Pack, briefing: Briefing, missing: list[str] | None = None) -> list[Step]:
    """Documents not uploaded yet that give what today's rules wait for, the most useful first.

    `missing` covers every field the rules wait for, also those never asked as questions (read from documents
    only); without it, the open questions are used.
    """
    status = {line.doc_type: line.status for line in briefing.documents}
    fields = missing if missing is not None else [question.field for question in briefing.questions]
    answered: dict[str, list[str]] = {}
    optional_only = optional_only_fields(pack)
    for field in fields:
        for doc_type in document_types_for(pack, [field]):
            if pack.document(doc_type).optional and field not in optional_only:
                continue      # a required document gives this field too: no need for the optional one
            if status.get(doc_type) in ("missing", "ambiguous") and field_label(field) not in answered.get(doc_type, []):
                answered.setdefault(doc_type, []).append(field_label(field))
    ordered = sorted(answered, key=lambda doc_type: -len(answered[doc_type]))
    return [Step(kind="upload", title=f"Upload, if you have it: {pack.document(doc_type).description}", ref=doc_type,
                 detail="It gives: " + short_list(answered[doc_type])) for doc_type in ordered]


def short_list(items: list[str], shown: int = 3) -> str:
    """'a, b, c and 2 more'."""
    more = len(items) - shown
    return ", ".join(items[:shown]) + (f" and {more} more" if more > 0 else "")


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
MODEL_LATER = 6         # questions about later steps the model sees


def state_for_model(briefing: Briefing) -> dict:
    """The briefing as short data for the model's prompt, without empty parts.

    Short on purpose: a 4,600-token prompt made the first chat call take about a minute on the 4 GB GPU.
    """
    data = {
        "procedure": briefing.title, "procedure_id": briefing.procedure_id, "where": briefing.where,
        "rules_unverified": briefing.unverified,
        "progress": f"{briefing.checks_passed} of {briefing.checks_total} requirements met; "
                    f"{briefing.documents_ready} of {briefing.documents_total} documents ready",
        "documents": {line.label: line.status if not (line.optional and line.status == "missing") else "only if needed"
                      for line in briefing.documents},
        "sections": [{"id": s.section_id, "title": s.title,
                      "status": f"{s.passed} met, {s.failed} not met, {s.unknown} waiting for information"}
                     for s in briefing.sections],
        "rule_status": briefing.rule_status,
        "not_met": [{"rule_id": p.rule_id, "requirement": p.description, "severity": p.severity,
                     "sources": p.source_ids, "quote": p.quote[:MODEL_QUOTE_CHARS]} for p in briefing.problems + briefing.advice],
        "questions": [{"field": q.field, "question": q.question} | ({"choices": q.choices} if q.choices else {})
                      for q in [q for q in briefing.questions if not q.upload_instead][:MODEL_QUESTIONS]],
        "later_steps": [q.question for q in briefing.later[:MODEL_LATER]],
        "values_to_check": [f"{v.field} = {v.value} (from {v.origin})" for v in briefing.to_check],
        "conflicts": briefing.conflicts, "waiting_for_approval": [item["title"] for item in briefing.pending],
        "steps": [{"do": step.title + (f" ({step.detail})" if step.detail else ""), "kind": step.kind, "ref": step.ref}
                  for step in briefing.steps[:MODEL_STEPS]],
    }
    if len(briefing.questions) > MODEL_QUESTIONS:
        data["more_questions"] = len(briefing.questions) - MODEL_QUESTIONS
    return {key: value for key, value in data.items() if value not in ([], {}, "")}
