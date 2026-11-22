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
from src.tools.search_procedures import SearchProceduresInput
from src.tools.search_procedures import run as search


def no_procedure_answer(ctx: ToolContext) -> FinalAnswer:
    titles = "; ".join(pack.procedure.title for pack in ctx.packs.values()) or "none"
    return FinalAnswer(summary="I could not identify a verified procedure for this request, so I cannot check any "
                               f"rules. Procedures I can help with: {titles}. Tell me which one you need.")


def _actions(pack: Pack, results: list[RuleResult]) -> list[NextAction]:
    """Blocking failures first, then the first missing information of blocking unknown rules.

    A question that several rules wait for (for example the reason of the application) is asked once.
    """
    actions = []
    for result in results:
        if result.status == "fail" and result.severity == "blocking":
            name = pack.rule(result.rule_id).name.replace("_", " ") or result.rule_id
            actions.append(NextAction(title=f"Correct the problem with {name} ({result.rule_id})", ref=result.rule_id))
    asked: set[str] = set()
    for result in results:
        if result.status == "unknown" and result.severity == "blocking" and result.missing_fields:
            field = next((name for name in result.missing_fields if name not in asked), None)
            if field is None:
                continue
            asked.add(field)
            actions.append(NextAction(title=question_for(pack, field)[:240], ref=result.rule_id))
    return actions[:MAX_NEXT_ACTIONS]


def _summary(pack: Pack, results: list[RuleResult]) -> str:
    counts = {status: sum(r.status == status for r in results) for status in ("pass", "fail", "unknown")}
    text = (f"Checklist for {pack.procedure.title}: {counts['pass']} rules pass, {counts['fail']} fail and "
            f"{counts['unknown']} need more information. This answer was built from the rules alone, without the "
            f"language model.")
    if pack.procedure.status == "unverified" or any(r.rule_status == "unverified" for r in results):
        text += " Some rules are unverified: check them with the official office."
    return text


def follow_goal(ctx: ToolContext, goal: str) -> None:
    """Without the model, the goal still decides the procedure when search_procedures finds exactly one best match.

    Otherwise (no match, a tie) the active procedure stays. search_procedures is deterministic, so this is code
    doing what the orchestrator's first step would have done.
    """
    if len(goal.strip()) < 2:
        return
    search(SearchProceduresInput(query=goal), ctx)


def deterministic_answer(ctx: ToolContext, goal: str = "") -> tuple[FinalAnswer, list[Finding]]:
    """The fallback answer and the findings it is based on."""
    follow_goal(ctx, goal)
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
