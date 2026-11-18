"""build_checklist tool (artifact): propose a Markdown checklist built from the rule results (no LLM text)."""
from pydantic import BaseModel, Field

from src.agent.findings import missing_items
from src.models import Pack, RuleResult
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.tools.evaluate_rules import current_results
from src.tools.proposals import ProposalOutput, propose

MARKS = {"pass": "[x]", "fail": "[ ]", "unknown": "[?]"}


class BuildChecklistInput(BaseModel):
    section_ids: list[str] = Field(default_factory=list, description="Sections to include (empty means all)")


def checklist_markdown(pack: Pack, results: list[RuleResult], documents: list, today) -> str:
    """Deterministic checklist text."""
    lines = [f"# Checklist: {pack.procedure.title}", "",
             f"Prepared on {today.isoformat()} by the Personal Bureaucracy Agent. This is not an official document; "
             "nothing has been submitted.", "",
             "Legend: [x] passes, [ ] fails, [?] information missing.", ""]
    for section in sorted(pack.procedure.sections, key=lambda s: s.order):
        rows = [r for r in results if r.section == section.id]
        if not rows:
            continue
        lines.append(f"## {section.title}")
        for row in rows:
            note = " (unverified rule)" if row.rule_status == "unverified" else ""
            missing = f" (missing: {', '.join(row.missing_fields)})" if row.missing_fields else ""
            lines.append(f"- {MARKS[row.status]} {row.rule_id}: {row.description}{missing}{note}")
        lines.append("")
    needed = missing_items(pack, results)
    if needed:
        lines += ["## Still needed"] + [f"- {item.question} ({', '.join(item.rule_ids)})" for item in needed] + [""]
    if documents:
        lines += ["## Documents in your Twin"] + [f"- {d.doc_id}: {d.doc_type or 'unknown type'} ({d.status})" for d in documents]
    return "\n".join(lines).rstrip() + "\n"


def run(args: BuildChecklistInput, ctx: ToolContext) -> ProposalOutput:
    pack = require_pack(ctx)
    unknown = [s for s in args.section_ids if pack.section(s) is None]
    if unknown:
        raise ToolError(f"Unknown sections: {', '.join(unknown)}")
    results = [r for r in current_results(ctx) if not args.section_ids or r.section in args.section_ids]
    text = checklist_markdown(pack, results, ctx.store.list_documents(ctx.user_id), ctx.today)
    counts = {status: sum(r.status == status for r in results) for status in MARKS}
    preview = f"{counts['pass']} pass, {counts['fail']} fail, {counts['unknown']} missing information"
    return propose(ctx, pack, "build_checklist", f"Checklist: {pack.procedure.title}", {"markdown": text},
                   [], None, preview)


SPEC = ToolSpec(
    name="build_checklist",
    description="Propose a checklist file (Markdown) of every rule with its status and what is still needed.",
    input_model=BuildChecklistInput, output_model=ProposalOutput, fn=run, kind="artifact", roles=("orchestrator",))
