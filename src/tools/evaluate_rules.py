"""evaluate_rules tool: the deterministic verdicts. The LLM never decides a rule; it calls this."""
from pydantic import BaseModel, Field

from src.models import RuleResult
from src.rules import evaluate_rules as evaluate
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack


class EvaluateRulesInput(BaseModel):
    section_id: str | None = Field(default=None, description="One section id, or empty for all rules")


class EvaluateRulesOutput(BaseModel):
    procedure_id: str
    section_id: str | None
    results: list[RuleResult]
    counts: dict[str, int]
    unverified_rules: list[str] = Field(default_factory=list)


def current_results(ctx: ToolContext, section_id: str | None = None) -> list[RuleResult]:
    """Verdicts on the Twin's current facts (also used by Gate 1, workers and the fallback)."""
    pack = require_pack(ctx)
    return evaluate(pack, ctx.view().fact_map(), section_id)


def run(args: EvaluateRulesInput, ctx: ToolContext) -> EvaluateRulesOutput:
    pack = require_pack(ctx)
    if args.section_id is not None and pack.section(args.section_id) is None:
        known = ", ".join(s.id for s in pack.procedure.sections)
        raise ToolError(f"Unknown section {args.section_id!r}. Sections: {known}")
    results = current_results(ctx, args.section_id)
    counts = {status: sum(r.status == status for r in results) for status in ("pass", "fail", "unknown")}
    unverified = [r.rule_id for r in results if r.rule_status == "unverified"]
    return EvaluateRulesOutput(procedure_id=pack.id, section_id=args.section_id, results=results,
                               counts=counts, unverified_rules=unverified)


SPEC = ToolSpec(
    name="evaluate_rules",
    description="Evaluate the procedure's rules on the confirmed facts: pass, fail or unknown, with the missing "
                "fields and source ids. Always use this before saying anything about a rule.",
    input_model=EvaluateRulesInput, output_model=EvaluateRulesOutput, fn=run, kind="read")
