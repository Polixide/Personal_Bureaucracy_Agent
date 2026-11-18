"""evaluate_rules tool: the deterministic verdicts. The LLM never decides a rule; it calls this.

The observation is grouped by status (compact enough for a small model's context): passing ids,
failing ids with their sources, unknown ids with the missing fields.
"""
from pydantic import BaseModel, Field

from src.models import RuleResult
from src.rules import evaluate_rules as evaluate
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack


class EvaluateRulesInput(BaseModel):
    section_id: str | None = Field(default=None, description="One section id, or empty for all rules")


class EvaluateRulesOutput(BaseModel):
    procedure_id: str
    section_id: str | None
    counts: dict[str, int]
    passing: list[str] = Field(default_factory=list)
    failing: dict[str, list[str]] = Field(default_factory=dict, description="rule id -> source ids")
    unknown: dict[str, list[str]] = Field(default_factory=dict, description="rule id -> missing fields")
    blocked_by: dict[str, list[str]] = Field(default_factory=dict)
    not_blocking: list[str] = Field(default_factory=list)
    unverified_rules: list[str] = Field(default_factory=list)


def current_results(ctx: ToolContext, section_id: str | None = None) -> list[RuleResult]:
    """Verdicts on the Twin's current facts (also used by Gate 1, workers and the fallback)."""
    pack = require_pack(ctx)
    return evaluate(pack, ctx.view().fact_map(), section_id)


def summarize(procedure_id: str, section_id: str | None, results: list[RuleResult]) -> EvaluateRulesOutput:
    """Group verdicts by status."""
    output = EvaluateRulesOutput(procedure_id=procedure_id, section_id=section_id,
                                 counts={s: sum(r.status == s for r in results) for s in ("pass", "fail", "unknown")})
    for result in results:
        if result.status == "pass":
            output.passing.append(result.rule_id)
        elif result.status == "fail":
            output.failing[result.rule_id] = result.source_ids
        else:
            output.unknown[result.rule_id] = result.missing_fields
        if result.blocked_by:
            output.blocked_by[result.rule_id] = result.blocked_by
        if result.severity != "blocking":
            output.not_blocking.append(result.rule_id)
        if result.rule_status == "unverified":
            output.unverified_rules.append(result.rule_id)
    return output


def run(args: EvaluateRulesInput, ctx: ToolContext) -> EvaluateRulesOutput:
    pack = require_pack(ctx)
    if args.section_id is not None and pack.section(args.section_id) is None:
        known = ", ".join(s.id for s in pack.procedure.sections)
        raise ToolError(f"Unknown section {args.section_id!r}. Sections: {known}")
    return summarize(pack.id, args.section_id, current_results(ctx, args.section_id))


SPEC = ToolSpec(
    name="evaluate_rules",
    description="Evaluate the procedure's rules on the confirmed facts: passing, failing (with sources) and unknown "
                "(with missing fields). Always use this before saying anything about a rule.",
    input_model=EvaluateRulesInput, output_model=EvaluateRulesOutput, fn=run, kind="read")
