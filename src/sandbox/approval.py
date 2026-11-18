"""Approval flow: the only path from the Twin to the real world.

approve_action: check owner and status -> promotion gate (accepted simulation on current facts, for
stage-bound actions) -> write the artifact to the user's local outbox -> mark executed -> log a
twin_to_real event. "Executed" means a FILE was written for the user; nothing is ever sent, booked,
paid or submitted by the agent.
"""
import re
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from config import OUTBOX_DIR
from src.models import Pack
from src.sandbox.mock_world import promotion_check
from src.tools.fill_pdf import fill_form
from src.twin.models import Event, ProposedAction
from src.twin.store import TwinStore

APPROVABLE = ("proposed", "blocked")
EXECUTED_NOTE = "Written to your outbox. Nothing was sent; you decide what to do with it."


class ApprovalResult(BaseModel):
    action_id: str
    status: str
    artifact_path: str | None = None
    message: str


def user_outbox(outbox_dir: Path, user_id: str) -> Path:
    """Per-user outbox folder with a file-system-safe name."""
    return Path(outbox_dir) / re.sub(r"[^A-Za-z0-9_-]", "_", user_id)


def _eml_text(payload: dict) -> str:
    return (f"To: {payload['to_name']} <{payload['to_email']}>\nSubject: {payload['subject']}\n"
            "X-Draft-Note: prepared by the Personal Bureaucracy Agent; NOT sent\n\n" + payload["body"] + "\n")


def write_artifact(proposal: ProposedAction, pack: Pack, folder: Path) -> Path:
    """Write the approved artifact file and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = proposal.payload
    if proposal.tool == "create_ics":
        path = folder / f"{proposal.action_id}.ics"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload["ics"])
        return path
    if proposal.tool == "draft_email":
        path = folder / f"{proposal.action_id}.eml"
        path.write_text(_eml_text(payload), encoding="utf-8")
        return path
    if proposal.tool == "build_checklist":
        path = folder / f"{proposal.action_id}.md"
        path.write_text(payload["markdown"], encoding="utf-8")
        return path
    if proposal.tool == "fill_pdf":
        return fill_form(pack.root / payload["template"], payload["values"], folder / f"{proposal.action_id}.pdf")
    raise ValueError(f"No writer for tool {proposal.tool!r}")


def _load(store: TwinStore, user_id: str, action_id: str) -> ProposedAction:
    proposal = store.get_proposal(action_id)
    if proposal is None or proposal.user_id != user_id:
        raise ValueError(f"No proposal {action_id!r} for this user")
    return proposal


def approve_action(store: TwinStore, packs: dict[str, Pack], user_id: str, action_id: str, today: date,
                   outbox_dir: Path = OUTBOX_DIR) -> ApprovalResult:
    """The user approved: run the promotion gate, then write the artifact."""
    proposal = _load(store, user_id, action_id)
    if proposal.status not in APPROVABLE:
        return ApprovalResult(action_id=action_id, status=proposal.status, message=f"Already {proposal.status}")
    pack = packs[proposal.procedure_id]
    allowed, reason = promotion_check(store, pack, user_id, today, proposal.requires_stage)
    if not allowed:
        store.decide_proposal(action_id, "blocked", note=reason)
        store.add_event(Event(user_id=user_id, procedure_id=pack.id, kind="approval_blocked", direction="internal",
                              payload={"action_id": action_id, "reason": reason}))
        return ApprovalResult(action_id=action_id, status="blocked", message=reason)
    path = write_artifact(proposal, pack, user_outbox(outbox_dir, user_id))
    with store.transaction():
        store.decide_proposal(action_id, "executed", artifact_path=str(path), note=EXECUTED_NOTE)
        store.add_event(Event(user_id=user_id, procedure_id=pack.id, kind="artifact_created", direction="twin_to_real",
                              payload={"action_id": action_id, "tool": proposal.tool, "file": path.name}))
    return ApprovalResult(action_id=action_id, status="executed", artifact_path=str(path), message=EXECUTED_NOTE)


def deny_action(store: TwinStore, user_id: str, action_id: str) -> ApprovalResult:
    """The user said no: the proposal is closed and nothing is written."""
    proposal = _load(store, user_id, action_id)
    if proposal.status not in APPROVABLE:
        return ApprovalResult(action_id=action_id, status=proposal.status, message=f"Already {proposal.status}")
    with store.transaction():
        store.decide_proposal(action_id, "denied", note="denied by the user")
        store.add_event(Event(user_id=user_id, procedure_id=proposal.procedure_id, kind="action_denied",
                              direction="internal", payload={"action_id": action_id}))
    return ApprovalResult(action_id=action_id, status="denied", message="Denied; nothing was written.")
