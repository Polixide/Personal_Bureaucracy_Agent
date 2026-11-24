"""Worker: a small bounded ReAct loop for ONE section, with a restricted tool set.

The loop: the model returns a Decision; the code validates and runs the tool; the model sees the
observation; repeat up to WORKER_STEPS. Two invalid steps in a row, a repeated call or running out of
steps all end the loop, and the Finding is then built from code alone. Rule results in the Finding
always come from evaluate_rules, never from the model.
"""
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from config import MAX_CONSECUTIVE_FAILURES, WORKER_STEPS
from src.agent.answer import FINISH_ACTION, Decision, Finding, MissingItem, WorkerFinish, decision_schema
from src.agent.findings import build_finding
from src.agent.loop import (ask_decision, assistant_message, call_key, error_message, observation_message)
from src.agent.prompts import WORKER_SYSTEM, worker_intro
from src.llm import Backend, short_validation_error
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.tools.registry import run_tool, tools_for


@dataclass
class WorkerLoop:
    """Mutable state of one worker run."""
    messages: list[dict]
    calls: int = 0
    steps: int = 0
    failures: int = 0
    seen: set[str] = field(default_factory=set)
    finish: WorkerFinish | None = None


def _handle(decision: Decision, loop: WorkerLoop, tools: dict[str, ToolSpec], ctx: ToolContext) -> None:
    """Run one valid decision: finish, refuse a repeat, or execute the tool."""
    loop.messages.append(assistant_message(decision))
    if decision.action == FINISH_ACTION:
        try:
            loop.finish = WorkerFinish.model_validate(decision.args)
        except ValidationError as error:
            loop.failures += 1
            loop.messages.append(error_message(f"finish args invalid: {short_validation_error(error)}"))
        return
    key = call_key(decision)
    if key in loop.seen:
        loop.failures += 1
        loop.messages.append(error_message(f"You already called {decision.action} with these arguments. Use the result or finish."))
        return
    loop.seen.add(key)
    observation = run_tool(tools, decision.action, decision.args, ctx, "worker")
    loop.messages.append(observation_message(observation))
    loop.failures = 0 if observation.ok else loop.failures + 1


def run_worker(section_id: str, focus: str, ctx: ToolContext, backend: Backend, registry: dict[str, ToolSpec]) -> Finding:
    """Check one section and return its Finding (never raises for model problems)."""
    pack = require_pack(ctx)
    if pack.section(section_id) is None:
        raise ToolError(f"Unknown section {section_id!r}. Sections: {', '.join(s.id for s in pack.procedure.sections)}")
    tools = tools_for(registry, "worker")
    first = run_tool(tools, "evaluate_rules", {"section_id": section_id}, ctx, "worker")
    loop = WorkerLoop(messages=[{"role": "system", "content": WORKER_SYSTEM},
                                {"role": "user", "content": worker_intro(ctx, section_id, focus, tools, first.output or {})}])
    schema = decision_schema(sorted(tools) + [FINISH_ACTION])
    while loop.steps < WORKER_STEPS and loop.finish is None and loop.failures < MAX_CONSECUTIVE_FAILURES:
        loop.steps += 1
        result = ask_decision(backend, loop.messages, schema, "worker")
        loop.calls += result.attempts
        if not result.ok:
            loop.failures += 1
            loop.messages.append(error_message(f"Invalid decision: {result.error}"))
            continue
        _handle(result.value, loop, tools, ctx)
    return build_finding(ctx, section_id, loop.finish, llm_calls=loop.calls, steps=loop.steps)


# ------------------------------------------------------ delegate_section --
class DelegateSectionInput(BaseModel):
    section_id: str = Field(description="Id of the section to check")
    focus: str = Field(default="", max_length=300, description="What the worker should look at in particular")


class DelegateSectionOutput(BaseModel):
    """The finding, compact: the orchestrator copies statuses from here."""
    finding_id: str
    section_id: str
    statuses: dict[str, str]
    missing_items: list[MissingItem]
    suspected_mistakes: list[str]
    summary: str


def run_delegate(args: DelegateSectionInput, ctx: ToolContext) -> DelegateSectionOutput:
    if ctx.run_worker is None:
        raise ToolError("Delegation is not available in this context")
    finding = ctx.run_worker(args.section_id, args.focus)
    ctx.findings[finding.finding_id] = finding
    return DelegateSectionOutput(finding_id=finding.finding_id, section_id=finding.section_id,
                                 statuses={r.rule_id: r.status for r in finding.rule_results},
                                 missing_items=finding.missing_items, suspected_mistakes=finding.suspected_mistakes,
                                 summary=finding.summary)


DELEGATE_SPEC = ToolSpec(
    name="delegate_section",
    description="Give one section to a worker. Returns a finding: rule statuses, missing information (with the "
                "questions to ask) and suspected mistakes, with a finding_id you can use as a next-action ref.",
    input_model=DelegateSectionInput, output_model=DelegateSectionOutput, fn=run_delegate, kind="orchestrator",
    roles=("orchestrator",))
