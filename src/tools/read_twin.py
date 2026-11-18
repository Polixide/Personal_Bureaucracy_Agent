"""read_twin tool: a compact snapshot of what the Digital Twin knows right now."""
from pydantic import BaseModel, Field

from src.tools.common import ToolContext, ToolSpec
from src.twin.facts import FactView

MAX_ISSUES = 3


class ReadTwinInput(BaseModel):
    fields: list[str] = Field(default_factory=list, description="Only these facts (empty means all known facts)")


class DocumentRow(BaseModel):
    doc_id: str
    doc_type: str | None
    status: str
    issues: list[str] = Field(default_factory=list)


class FactRow(BaseModel):
    field: str
    value: str
    provenance: str
    source: str


class UnconfirmedRow(BaseModel):
    fact_id: int
    field: str
    value: str
    source: str


class ProposalRow(BaseModel):
    action_id: str
    tool: str
    title: str
    status: str


class ReadTwinOutput(BaseModel):
    procedure_id: str | None
    procedure_title: str | None = None
    procedure_status: str | None = None
    documents: list[DocumentRow] = Field(default_factory=list)
    facts: list[FactRow] = Field(default_factory=list)
    unconfirmed: list[UnconfirmedRow] = Field(default_factory=list)
    conflicts: dict[str, list[str]] = Field(default_factory=dict)
    proposals: list[ProposalRow] = Field(default_factory=list)
    simulations: list[dict] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)


def _document_rows(ctx: ToolContext) -> list[DocumentRow]:
    rows = []
    for document in ctx.store.list_documents(ctx.user_id):
        issues = (document.observation or {}).get("issues", [])
        if document.processing_error:
            issues = [document.processing_error] + issues
        rows.append(DocumentRow(doc_id=document.doc_id, doc_type=document.doc_type, status=document.status,
                                issues=issues[:MAX_ISSUES]))
    return rows


def _fact_rows(view: FactView, wanted: list[str]) -> list[FactRow]:
    rows = []
    for field in sorted(view.values):
        if wanted and field not in wanted:
            continue
        if field in view.derived:
            rows.append(FactRow(field=field, value=str(view.values[field]), provenance="derived", source="computed"))
            continue
        newest = max(view.sources[field], key=lambda fact: fact.fact_id or 0)
        rows.append(FactRow(field=field, value=str(view.values[field]), provenance=newest.provenance,
                            source=newest.source_ref))
    return rows


def run(args: ReadTwinInput, ctx: ToolContext) -> ReadTwinOutput:
    """Documents, confirmed facts, unconfirmed statements, conflicts, proposals and simulations."""
    pack = ctx.pack()
    output = ReadTwinOutput(procedure_id=ctx.procedure_id, documents=_document_rows(ctx))
    output.proposals = [ProposalRow(action_id=p.action_id, tool=p.tool, title=p.title, status=p.status)
                        for p in ctx.store.list_proposals(ctx.user_id)]
    output.outcomes = [e.payload.get("description", "") for e in ctx.store.list_events(ctx.user_id, kind="outcome")]
    if pack is None:
        return output
    view = ctx.view()
    output.procedure_title = pack.procedure.title
    output.procedure_status = pack.procedure.status
    output.facts = _fact_rows(view, args.fields)
    output.unconfirmed = [UnconfirmedRow(fact_id=f.fact_id, field=f.field, value=str(f.value), source=f.source_ref)
                          for f in view.unconfirmed]
    output.conflicts = {field: [str(v) for v in values] for field, values in view.conflicts.items()}
    output.simulations = [{"stage_id": s.stage_id, "accepted": s.accepted}
                          for s in ctx.store.list_simulations(ctx.user_id, pack.id)[-5:]]
    return output


SPEC = ToolSpec(
    name="read_twin",
    description="Show what the Digital Twin knows: documents, confirmed facts, statements waiting for "
                "confirmation, conflicts, proposed actions and sandbox results.",
    input_model=ReadTwinInput, output_model=ReadTwinOutput, fn=run, kind="read")
