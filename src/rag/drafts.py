"""Draft rules from a pack's official texts with RAG, for a person to review.

For each topic (by default: every field that has a question) the most similar passages are retrieved, the
model proposes the requirements those passages state, each with a verbatim quote, and the code then checks
every proposal: the quote must really be in the saved source text, the fields must exist in procedure.json.
The drafts go to procedures/<id>/rule_drafts.json with status "unverified". rules.json is never touched:
turning a draft into a rule (with its condition) is a person's job (add-rule skill).
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import RAG_TOP_K
from src.llm import Backend, ask_structured, to_json
from src.models import Pack
from src.document_processing.evidence import normalized
from src.rag.index import Hit, search
from src.rag.ingest import stored_text
from src.twin.store import TwinStore

DRAFTS_FILE = "rule_drafts.json"
MAX_PROPOSALS = 4
MIN_QUOTE_CHARS = 12      # shorter quotes prove nothing
MAX_DRAFT_FIELDS = 3      # a requirement decided by more facts is probably several requirements
SYSTEM_PROMPT = (
    "You extract the requirements of an official procedure from passages of official texts. Use ONLY the "
    "passages. For each requirement about the TOPIC that a passage states, give: a one-sentence description "
    "in English of what the applicant must have or do (not the topic question); the exact quote, copied "
    "character by character from one passage (at most 300 characters, in the passage's language); that "
    "passage's source_id; the 1 to 3 FIELDS the requirement is decided by (only names from the list, only "
    "the ones it really needs); severity 'blocking' if the application cannot go on without it, otherwise "
    "'advisory'. "
    "If no passage states a requirement about the topic, return an empty list. Never invent a requirement, "
    "a number, a fee or a date.")


class DraftProposal(BaseModel):
    """One requirement as the model proposes it."""
    description: str = Field(max_length=300)
    quote: str = Field(max_length=600)
    source_id: str
    fields: list[str] = Field(default_factory=list, max_length=6)
    severity: Literal["blocking", "advisory"] = "blocking"


class DraftReply(BaseModel):
    requirements: list[DraftProposal] = Field(default_factory=list, max_length=MAX_PROPOSALS)


class RuleDraft(BaseModel):
    """A checked proposal, waiting for a person. `problems` lists what the code could not confirm."""
    draft_id: str
    topics: list[str]
    description: str
    source_id: str
    quote: str
    fields: list[str]
    severity: str
    quote_found: bool
    problems: list[str] = Field(default_factory=list)
    similar_rules: list[str] = Field(default_factory=list)
    status: Literal["unverified"] = "unverified"


# --------------------------------------------------------------- topics --
def default_topics(pack: Pack) -> list[str]:
    """One topic per field the user is asked about: its question, which says what the field means."""
    return [spec.question for spec in pack.procedure.fields if spec.question]


def field_list(pack: Pack) -> list[str]:
    """'name (type): question' for every field, so the model can name the facts a requirement needs."""
    return [f"{spec.name} ({spec.type}): {spec.question}" for spec in pack.procedure.fields]


def draft_messages(pack: Pack, topic: str, hits: list[Hit]) -> list[dict]:
    """The prompt: topic, the field names and the retrieved passages with their source ids."""
    passages = [{"source_id": hit.source_id, "text": hit.text} for hit in hits]
    content = (f"PROCEDURE: {pack.procedure.title}\nTOPIC: {topic}\nFIELDS:\n" + "\n".join(field_list(pack))
               + f"\nPASSAGES: {to_json(passages)}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


# ---------------------------------------------------------------- check --
def comparable(text: str) -> str:
    """Comparison form: lower case without accents (Turkish I/ı/İ folded), single spaces."""
    return " ".join(normalized(text).split())


def quote_in_source(pack: Pack, source_id: str, quote: str) -> bool:
    """True when the quote (ignoring case, accents and spacing) is part of the saved source text."""
    wanted = comparable(quote)
    text = stored_text(pack.root / "sources_text" / f"{source_id}.txt")
    return len(wanted) >= MIN_QUOTE_CHARS and wanted in comparable(text)


def proposal_problems(pack: Pack, proposal: DraftProposal, retrieved: set[str]) -> list[str]:
    """What a reviewer must know: unknown source or fields, quote not found, no field named."""
    problems = []
    if pack.source(proposal.source_id) is None:
        problems.append(f"unknown source {proposal.source_id!r}")
    elif proposal.source_id not in retrieved:
        problems.append(f"source {proposal.source_id} was not among the retrieved passages")
    elif not quote_in_source(pack, proposal.source_id, proposal.quote):
        problems.append("quote not found in the saved source text")
    unknown = [name for name in proposal.fields if pack.field(name) is None]
    if unknown:
        problems.append("unknown fields: " + ", ".join(unknown))
    if not proposal.fields:
        problems.append("no field named: decide which fact this requirement needs")
    elif len(proposal.fields) > MAX_DRAFT_FIELDS:
        problems.append(f"{len(proposal.fields)} fields named: keep only the facts this requirement really needs")
    return problems


def description_problems(topic: str, proposal: DraftProposal) -> list[str]:
    """A description that only repeats the topic question says nothing about the requirement."""
    if comparable(proposal.description).rstrip("?. ") == comparable(topic).rstrip("?. "):
        return ["the description repeats the topic: write the requirement in your own words"]
    return []


def rule_inputs(pack: Pack, fields: list[str]) -> set[str]:
    """The fields a rule reads, plus the inputs of the derived ones (photo_age_months -> photo_date)."""
    derived = {spec.field: spec.args for spec in pack.procedure.derived}
    inputs = set(fields)
    for name in fields:
        inputs.update(derived.get(name, []))
    return inputs


def similar_rules(pack: Pack, proposal: DraftProposal) -> list[str]:
    """Existing rules on the same source that read one of the same fields (probably already covered)."""
    wanted = set(proposal.fields)
    return [rule.rule_id for rule in pack.rules if proposal.source_id in rule.source_ids
            and wanted & rule_inputs(pack, rule.condition.leaf_fields())]


def to_draft(pack: Pack, topic: str, proposal: DraftProposal, retrieved: set[str], number: int) -> RuleDraft:
    """Turn one model proposal into a checked draft."""
    problems = proposal_problems(pack, proposal, retrieved) + description_problems(topic, proposal)
    found = not any(problem.startswith(("quote", "unknown source", "source ")) for problem in problems)
    return RuleDraft(draft_id=f"D{number:03d}", topics=[topic], description=proposal.description,
                     source_id=proposal.source_id, quote=proposal.quote.strip(),
                     fields=[name for name in proposal.fields if pack.field(name) is not None],
                     severity=proposal.severity, quote_found=found, problems=problems,
                     similar_rules=similar_rules(pack, proposal))


def merge(drafts: list[RuleDraft], draft: RuleDraft) -> bool:
    """Add the topic to an existing draft with the same quote; True when merged."""
    for existing in drafts:
        if existing.source_id == draft.source_id and comparable(existing.quote) == comparable(draft.quote):
            existing.topics.extend(topic for topic in draft.topics if topic not in existing.topics)
            return True
    return False


# ------------------------------------------------------------------ run --
def draft_rules(store: TwinStore, pack: Pack, backend: Backend, topics: list[str] | None = None,
                k: int = RAG_TOP_K + 1) -> list[RuleDraft]:
    """Retrieve, propose and check, topic by topic. A topic whose model call fails is skipped."""
    drafts: list[RuleDraft] = []
    for topic in topics or default_topics(pack):
        hits = search(store, pack, backend, topic, k)
        if not hits:
            continue
        reply = ask_structured(backend, draft_messages(pack, topic, hits), DraftReply, "rule_drafts")
        if not reply.ok:
            continue
        retrieved = {hit.source_id for hit in hits}
        for proposal in reply.value.requirements:
            draft = to_draft(pack, topic, proposal, retrieved, len(drafts) + 1)
            if not merge(drafts, draft):
                drafts.append(draft)
    return drafts


def save_drafts(pack: Pack, drafts: list[RuleDraft]) -> Path:
    """Write procedures/<id>/rule_drafts.json (replaced on every run)."""
    path = pack.root / DRAFTS_FILE
    path.write_text(json.dumps([draft.model_dump() for draft in drafts], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path
