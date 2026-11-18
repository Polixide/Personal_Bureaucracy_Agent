"""Deterministic workflow: the answer when the model cannot be used (the app never dies).

Evaluate every section, build a Finding for each section that is not complete, and turn the
verdicts into a checklist answer. No model call; it passes Gate 1 by construction.
"""
from config import MAX_NEXT_ACTIONS
from src.agent.answer import Claim, FinalAnswer, Finding, NextAction
from src.agent.findings import build_finding, question_for
from src.models import Pack, RuleResult
from src.tools.common import ToolContext
from src.tools.evaluate_rules import current_results


def no_procedure_answer(ctx: ToolContext) -> FinalAnswer:
    titles = "; ".join(pack.procedure.title for pack in ctx.packs.values()) or "none"
    return FinalAnswer(summary="I could not identify a verified procedure for this request, so I cannot check any "
                               f"rules. Procedures I can help with: {titles}. Tell me which one you need.")


def _actions(pack: Pack, results: list[RuleResult]) -> list[NextAction]:
    """Blocking failures first, then the first missing information of blocking unknown rules."""
    actions = []
    for result in results:
        if result.status == "fail" and result.severity == "blocking":
            name = pack.rule(result.rule_id).name.replace("_", " ") or result.rule_id
            actions.append(NextAction(title=f"Correct the problem with {name} ({result.rule_id})", ref=result.rule_id))
    for result in results:
        if result.status == "unknown" and result.severity == "blocking" and result.missing_fields:
            actions.append(NextAction(title=question_for(pack, result.missing_fields[0])[:240], ref=result.rule_id))
    return actions[:MAX_NEXT_ACTIONS]


def _summary(pack: Pack, results: list[RuleResult]) -> str:
    counts = {status: sum(r.status == status for r in results) for status in ("pass", "fail", "unknown")}
    text = (f"Checklist for {pack.procedure.title}: {counts['pass']} rules pass, {counts['fail']} fail and "
            f"{counts['unknown']} need more information. This answer was built from the rules alone, without the "
            f"language model.")
    if pack.procedure.status == "unverified" or any(r.rule_status == "unverified" for r in results):
        text += " Some rules are unverified: check them with the official office."
    return text


def deterministic_answer(ctx: ToolContext) -> tuple[FinalAnswer, list[Finding]]:
    """The fallback answer and the findings it is based on."""
    pack = ctx.pack()
    if pack is None:
        return no_procedure_answer(ctx), []
    results = current_results(ctx)
    findings = []
    for section in sorted(pack.procedure.sections, key=lambda s: s.order):
        if section.delegate and any(r.status != "pass" for r in results if r.section == section.id):
            findings.append(build_finding(ctx, section.id, None))
    claims = [Claim(rule_id=r.rule_id, status=r.status) for r in results if r.status != "pass"]
    answer = FinalAnswer(summary=_summary(pack, results), claims=claims, next_actions=_actions(pack, results))
    return answer, findings
