"""Deterministic parts of a worker Finding: verdicts, missing items, suspected mistakes.

The model only adds a summary and extra suspected mistakes; everything checkable comes from code.
"""
from src.agent.answer import Finding, MissingItem, WorkerFinish
from src.models import Pack, RuleResult
from src.tools.common import ToolContext
from src.tools.evaluate_rules import current_results
from src.twin.facts import FactView

MAX_DOC_ISSUES = 2


def finding_id(section_id: str) -> str:
    return f"F-{section_id}"


def section_fields(pack: Pack, section_id: str) -> set[str]:
    """Every field read by the section's rules."""
    return {name for rule in pack.rules_in_section(section_id) for name in rule.condition.leaf_fields()}


def question_for(pack: Pack, field: str) -> str:
    spec = pack.field(field)
    if spec is not None and spec.question:
        return spec.question
    return "Please provide: " + field.replace("_", " ")


def missing_items(pack: Pack, results: list[RuleResult]) -> list[MissingItem]:
    """One item per missing field, listing every unknown rule that needs it."""
    by_field: dict[str, list[str]] = {}
    for result in results:
        for field in result.missing_fields:
            by_field.setdefault(field, []).append(result.rule_id)
    return [MissingItem(field=field, question=question_for(pack, field), rule_ids=rule_ids)
            for field, rule_ids in by_field.items()]


def deterministic_mistakes(pack: Pack, section_id: str, results: list[RuleResult], ctx: ToolContext, view: FactView) -> list[str]:
    """Failures, conflicts, unconfirmed statements and document issues that touch this section."""
    fields = section_fields(pack, section_id)
    mistakes = [f"{r.rule_id} fails: {r.description}" for r in results if r.status == "fail"]
    mistakes += [f"Sources disagree on {name}: {', '.join(str(v) for v in values)}"
                 for name, values in view.conflicts.items() if name in fields]
    mistakes += [f"{fact.field} = {fact.value} was stated but is not confirmed yet"
                 for fact in view.unconfirmed if fact.field in fields]
    for document in ctx.store.list_documents(ctx.user_id):
        observation = document.observation or {}
        touched = {fact.get("field") for fact in observation.get("facts", [])} & fields
        if touched and observation.get("issues"):
            mistakes += [f"{document.doc_id}: {issue}" for issue in observation["issues"][:MAX_DOC_ISSUES]]
        if document.status == "rejected":
            mistakes.append(f"{document.doc_id} could not be read: {document.processing_error}")
    return mistakes


def deterministic_summary(pack: Pack, section_id: str, results: list[RuleResult]) -> str:
    counts = {status: sum(r.status == status for r in results) for status in ("pass", "fail", "unknown")}
    title = pack.section(section_id).title
    return (f"{title}: {counts['pass']} rules pass, {counts['fail']} fail, "
            f"{counts['unknown']} need information.")


def build_finding(ctx: ToolContext, section_id: str, finish: WorkerFinish | None,
                  llm_calls: int = 0, steps: int = 0) -> Finding:
    """The Finding of one section. `finish` is the model's part (None = deterministic only)."""
    pack = ctx.pack()
    results = current_results(ctx, section_id)
    view = ctx.view()
    mistakes = deterministic_mistakes(pack, section_id, results, ctx, view)
    summary = deterministic_summary(pack, section_id, results)
    if finish is not None:
        mistakes += [m for m in finish.suspected_mistakes if m not in mistakes]
        summary = finish.summary or summary
    return Finding(finding_id=finding_id(section_id), section_id=section_id, summary=summary, rule_results=results,
                   evidence={r.rule_id: r.source_ids for r in results}, missing_items=missing_items(pack, results),
                   suspected_mistakes=mistakes, llm_calls=llm_calls, steps=steps, fallback_used=finish is None)
