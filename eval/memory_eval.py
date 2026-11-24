"""Memory across sessions: the Twin is closed and reopened between steps, like a user coming back.

Ported from Neda's milestone-1 harness (scenarios M1-M3) to our architecture. Each scenario uses a real
SQLite file, the real agent (with the offline rule-following policy), the approval gate and the outbox,
and collects every expectation as a named check instead of stopping at the first failure.

M1 three sessions: answer + agent run -> restart -> approve the checklist -> restart -> everything is still there.
M2 deny after restart: a proposal made before a restart is denied after it; nothing is written.
M3 reset one profile: deleting one user removes all of their data and nothing of another user.
"""
from collections.abc import Callable
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from eval.policy import RuleFollowingPolicy
from src.agent.orchestrator import run_agent
from src.models import Pack
from src.rules import evaluate_rules
from src.sandbox.approval import approve_action, deny_action
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool
from src.twin import service
from src.twin.facts import build_view
from src.twin.store import TwinStore

PROCEDURE = "ikamet_first"
TODAY = date(2026, 11, 18)
FACTS = {"exchange_student": True, "institution": "Sabancı University", "passport_expiry_date": "2027-09-30"}
GOAL = "I want to apply for İkamet"


class Check(BaseModel):
    scenario: str
    name: str
    ok: bool


def verdicts(store: TwinStore, pack: Pack, user_id: str) -> dict[str, str]:
    view = build_view(pack, store.current_facts(user_id, pack.id), store.list_documents(user_id), TODAY)
    return {result.rule_id: result.status for result in evaluate_rules(pack, view.fact_map())}


def propose_checklist(store: TwinStore, packs: dict[str, Pack], user_id: str, outbox: Path) -> str:
    """The same proposal the app's 'Prepare my checklist' button makes; returns its id."""
    ctx = ToolContext(store=store, packs=packs, user_id=user_id, today=TODAY, procedure_id=PROCEDURE,
                      outbox_dir=outbox)
    return run_tool(build_registry(), "build_checklist", {}, ctx, "orchestrator").output["action_id"]


def session_one(path: Path, packs: dict[str, Pack], user_id: str, outbox: Path) -> tuple[dict, str]:
    """Typed answers, one agent run and a proposed checklist; returns (verdicts, proposal id)."""
    pack = packs[PROCEDURE]
    with TwinStore(path) as store:
        store.set_active_procedure(user_id, PROCEDURE)
        for field, value in FACTS.items():
            service.record_app_value(store, pack, user_id, field, value)
        run_agent(GOAL, store, packs, user_id, TODAY, RuleFollowingPolicy(), outbox_dir=outbox)
        action_id = propose_checklist(store, packs, user_id, outbox)
        return verdicts(store, pack, user_id), action_id


def m1_three_sessions(folder: Path, packs: dict[str, Pack]) -> list[Check]:
    """Answers and a proposal, restart, approval, restart: everything is still there."""
    path, outbox, user = folder / "m1.sqlite", folder / "outbox_m1", "m1_user"
    before, action_id = session_one(path, packs, user, outbox)
    checks = []
    with TwinStore(path) as store:                                   # session 2: after a restart
        checks.append(Check(scenario="M1", name="facts survive a restart", ok=verdicts(store, packs[PROCEDURE], user) == before))
        checks.append(Check(scenario="M1", name="proposal still waits", ok=store.get_proposal(action_id).status == "proposed"))
        result = approve_action(store, packs, user, action_id, TODAY, outbox)
        checks.append(Check(scenario="M1", name="approval after restart writes the file",
                            ok=result.status == "executed" and any(outbox.rglob("*"))))
    with TwinStore(path) as store:                                   # session 3
        runs = store.list_runs(user)
        checks.append(Check(scenario="M1", name="approval is remembered", ok=store.get_proposal(action_id).status == "executed"))
        checks.append(Check(scenario="M1", name="run log kept with its rule verdicts",
                            ok=len(runs) == 1 and {r.rule_id: r.status for r in runs[0].results} == before))
        checks.append(Check(scenario="M1", name="active procedure remembered", ok=store.active_procedure(user) == PROCEDURE))
    return checks


def m2_deny_after_restart(folder: Path, packs: dict[str, Pack]) -> list[Check]:
    """A proposal made before a restart is denied after it; nothing is written."""
    path, outbox, user = folder / "m2.sqlite", folder / "outbox_m2", "m2_user"
    _, action_id = session_one(path, packs, user, outbox)
    with TwinStore(path) as store:
        result = deny_action(store, user, action_id)
    with TwinStore(path) as store:
        proposal = store.get_proposal(action_id)
    return [Check(scenario="M2", name="denied after restart", ok=result.status == "denied" and proposal.status == "denied"),
            Check(scenario="M2", name="nothing written for a denied action", ok=not any(outbox.rglob("*.*")))]


def m3_reset_one_profile(folder: Path, packs: dict[str, Pack]) -> list[Check]:
    """Resetting one profile removes all of its data and none of another's."""
    path, outbox = folder / "m3.sqlite", folder / "outbox_m3"
    session_one(path, packs, "keep_me", outbox)
    session_one(path, packs, "reset_me", outbox)
    with TwinStore(path) as store:
        store.delete_user("reset_me")
    with TwinStore(path) as store:
        gone = not (store.user_exists("reset_me") or store.current_facts("reset_me", PROCEDURE)
                    or store.list_runs("reset_me") or store.list_proposals("reset_me"))
        kept = bool(store.current_facts("keep_me", PROCEDURE) and store.list_runs("keep_me")
                    and store.list_proposals("keep_me"))
    return [Check(scenario="M3", name="reset removes every record of that profile", ok=gone),
            Check(scenario="M3", name="another profile is untouched", ok=kept)]


SCENARIOS: list[Callable[[Path, dict[str, Pack]], list[Check]]] = [m1_three_sessions, m2_deny_after_restart,
                                                                     m3_reset_one_profile]


def run_memory_checks(folder: Path, packs: dict[str, Pack]) -> list[Check]:
    """Every check of every scenario (the folder holds the scenario databases and outboxes)."""
    checks: list[Check] = []
    for scenario in SCENARIOS:
        checks.extend(scenario(Path(folder), packs))
    return checks
