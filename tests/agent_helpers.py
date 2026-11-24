"""Shared helpers for agent tests: decisions, seeded Twins, runs with a scripted model."""
from datetime import date

from fakes import ScriptedBackend
from src.agent.orchestrator import run_agent
from src.pack import load_all_packs
from src.twin import service

USER = "synthetic_student_01"
TODAY = date(2026, 11, 18)
GOAL = "I want to apply for İkamet"


def d(action: str, args: dict | None = None, thought: str = "Next step.") -> dict:
    """A scripted Decision reply."""
    return {"thought": thought, "action": action, "args": args or {}}


def seed(store, facts: dict, procedure_id: str = "ikamet_first", active: bool = True) -> None:
    """Confirmed facts typed in the app, and optionally the active procedure."""
    pack = load_all_packs()[procedure_id]
    for name, value in facts.items():
        service.record_app_value(store, pack, USER, name, value)
    if active:
        store.set_active_procedure(USER, procedure_id)


def run(store, replies, goal: str = GOAL, today: date = TODAY, tmp_path=None, **kwargs):
    """Run the agent with a scripted backend; returns (result, backend)."""
    backend = ScriptedBackend(replies)
    outbox = (tmp_path / "outbox") if tmp_path is not None else None
    extra = {"outbox_dir": outbox} if outbox is not None else {}
    result = run_agent(goal, store, load_all_packs(), USER, today, backend, **extra, **kwargs)
    return result, backend


PASSPORT_TOO_SHORT = {"passport_expiry_date": "2027-07-15", "requested_permit_end_date": "2027-06-01"}


def correct_final(extra_actions: list[dict] | None = None) -> dict:
    """A final answer that matches the PASSPORT_TOO_SHORT facts."""
    return d("final_answer", {
        "summary": "Your passport validity is too short for the requested permit period (R006).",
        "claims": [{"rule_id": "R006", "status": "fail"}],
        "next_actions": [{"title": "Renew your passport or ask IRO about a shorter permit period", "ref": "R006"}]
        + (extra_actions or [])})
