"""search_procedures tool: which Procedure Pack matches the user's goal.

Deterministic: a pack scores one point per word of each alias/title phrase found in the goal.
A single best match becomes the active procedure; a tie means "ask the user"; no match means
"no procedure is available" (the agent must say so instead of improvising).
"""
from pydantic import BaseModel, Field

from src.document_processing.evidence import normalized
from src.models import Pack
from src.tools.common import ToolContext, ToolSpec
from src.twin.models import Event


class SearchProceduresInput(BaseModel):
    query: str = Field(min_length=2, description="The user's goal in their own words")


class ProcedureMatch(BaseModel):
    procedure_id: str
    title: str
    status: str
    score: int


class SearchProceduresOutput(BaseModel):
    matches: list[ProcedureMatch]
    selected: str | None
    message: str


def plain(text: str) -> str:
    """Lower case, no accents, Turkish dotless i folded, punctuation to spaces."""
    folded = normalized(text)
    return " ".join("".join(c if c.isalnum() else " " for c in folded).split())


def score(pack: Pack, query: str) -> int:
    """Words of every alias (and the id) that appear as a whole phrase in the query."""
    words = f" {plain(query)} "
    phrases = pack.procedure.aliases + [pack.id.replace("_", " ")]
    return sum(len(plain(p).split()) for p in phrases if plain(p) and f" {plain(p)} " in words)


def rank(packs: dict[str, Pack], query: str) -> list[ProcedureMatch]:
    """Matching packs, best first (ties keep id order)."""
    matches = [ProcedureMatch(procedure_id=p.id, title=p.procedure.title, status=p.procedure.status, score=score(p, query))
               for p in packs.values()]
    return sorted([m for m in matches if m.score > 0], key=lambda m: (-m.score, m.procedure_id))


def run(args: SearchProceduresInput, ctx: ToolContext) -> SearchProceduresOutput:
    matches = rank(ctx.packs, args.query)
    if not matches:
        return SearchProceduresOutput(matches=[], selected=None,
                                      message="No procedure matches. Say clearly that no verified procedure is available.")
    if len(matches) > 1 and matches[0].score == matches[1].score:
        return SearchProceduresOutput(matches=matches, selected=None,
                                      message="Several procedures match equally. Ask the user which one they mean.")
    chosen = matches[0]
    ctx.procedure_id = chosen.procedure_id
    ctx.store.set_active_procedure(ctx.user_id, chosen.procedure_id)
    ctx.store.add_event(Event(user_id=ctx.user_id, procedure_id=chosen.procedure_id, kind="procedure_selected",
                              direction="internal", payload={"query_words": len(args.query.split())}))
    note = "" if chosen.status == "verified" else " Its rules are UNVERIFIED: say so in the answer."
    return SearchProceduresOutput(matches=matches, selected=chosen.procedure_id,
                                  message=f"Active procedure: {chosen.procedure_id}.{note}")


SPEC = ToolSpec(
    name="search_procedures",
    description="Find the procedure that matches the user's goal and make it active. Tells you when nothing "
                "matches or when the goal is ambiguous.",
    input_model=SearchProceduresInput, output_model=SearchProceduresOutput, fn=run, kind="read",
    roles=("orchestrator",))
