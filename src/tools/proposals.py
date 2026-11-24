"""Shared by the artifact tools: create a PROPOSED action in the Twin (nothing is written or sent)."""
from uuid import uuid4

from pydantic import BaseModel

from src.agent.gate1 import build_gate_context, verify_text
from src.models import Pack
from src.tools.common import ToolContext, ToolError
from src.twin.models import Event, ProposedAction

PROPOSAL_NOTE = "Proposed only. It is written to your outbox after you approve it; nothing is sent anywhere."


class ProposalOutput(BaseModel):
    """What the model sees after an artifact tool: the proposal id to use as a next-action ref."""
    action_id: str
    tool: str
    title: str
    status: str
    requires_stage: str | None
    preview: str
    note: str = PROPOSAL_NOTE


def check_rule_ids(pack: Pack, rule_ids: list[str]) -> None:
    unknown = [rule_id for rule_id in rule_ids if pack.rule(rule_id) is None]
    if unknown:
        raise ToolError(f"Unknown rule ids: {', '.join(unknown)}")


def check_stage(pack: Pack, stage_id: str | None) -> None:
    if stage_id and pack.stage(stage_id) is None:
        raise ToolError(f"Unknown stage {stage_id!r}. Stages: {', '.join(s.id for s in pack.procedure.stages)}")


def check_text(ctx: ToolContext, text: str) -> None:
    """Gate 1 text checks on text that would leave the Twin (no completion claims, no invented dates)."""
    issues = verify_text(text, build_gate_context(ctx))
    if issues:
        raise ToolError("Text refused: " + "; ".join(f"[{i.code}] {i.message}" for i in issues))


def propose(ctx: ToolContext, pack: Pack, tool: str, title: str, payload: dict, refs: list[str],
            requires_stage: str | None, preview: str) -> ProposalOutput:
    """Store the proposal, remember it for this run, and describe it to the model."""
    action_id = f"A-{uuid4().hex[:8]}"
    with ctx.store.transaction():
        ctx.store.add_proposal(ProposedAction(action_id=action_id, user_id=ctx.user_id, procedure_id=pack.id, tool=tool,
                                              title=title, payload=payload, refs=refs, requires_stage=requires_stage))
        ctx.store.add_event(Event(user_id=ctx.user_id, procedure_id=pack.id, kind="action_proposed", direction="internal",
                                  payload={"action_id": action_id, "tool": tool}))
    ctx.created_proposals.append(action_id)
    return ProposalOutput(action_id=action_id, tool=tool, title=title, status="proposed",
                          requires_stage=requires_stage, preview=preview[:300])
