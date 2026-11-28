"""What the agent can prepare (src/agent/actions.py): built by code from the pack and the Twin.

Each suggestion runs an artifact tool (a proposal the user approves) or, for e-mails, asks the assistant to write
the draft. Suggestions already waiting for approval are marked, so the same file is not proposed twice.
"""
from datetime import date

from agent_helpers import USER, seed
from src.agent.actions import suggestions
from src.pack import load_all_packs, load_pack
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool
from src.twin import service

PACK = load_pack("ikamet_first")
TODAY = date(2026, 11, 18)


def found(store):
    view = service.twin_view(store, PACK, USER, TODAY)
    return suggestions(PACK, view, TODAY, store.list_proposals(USER))


def test_an_empty_file_can_still_get_a_checklist_a_form_and_e_mails(store):
    seed(store, {})
    tools = [s.tool for s in found(store)]
    assert tools[0] == "build_checklist" and "fill_pdf" in tools and "draft_email" in tools
    assert "create_ics" not in tools                       # no date known yet: nothing to remind


def test_upcoming_known_dates_become_reminders_nearest_first_and_past_ones_do_not(store):
    seed(store, {"passport_expiry_date": "2030-01-01", "legal_stay_expiry_date": "2026-12-20",
                 "program_start_date": "2026-09-01"})
    reminders = [s for s in found(store) if s.tool == "create_ics"]
    assert [s.args["date_field"] for s in reminders] == ["legal_stay_expiry_date", "passport_expiry_date"]


def test_a_reminder_runs_the_real_tool_and_is_then_marked_as_waiting(store):
    seed(store, {"legal_stay_expiry_date": "2026-12-20"})
    reminder = next(s for s in found(store) if s.tool == "create_ics")
    ctx = ToolContext(store=store, packs=load_all_packs(), user_id=USER, today=TODAY, procedure_id=PACK.id)
    assert run_tool(build_registry(), reminder.tool, reminder.args, ctx, "orchestrator").ok
    assert next(s for s in found(store) if s.tool == "create_ics").pending


def test_e_mails_are_written_by_the_assistant_not_by_code(store):
    seed(store, {})
    email = next(s for s in found(store) if s.tool == "draft_email")
    assert email.args == {} and email.ask_assistant.startswith("Draft an e-mail to IRO")


def test_an_approved_reminder_is_not_offered_again(store, tmp_path):
    from src.sandbox.approval import approve_action
    seed(store, {"legal_stay_expiry_date": "2026-12-20"})
    reminder = next(s for s in found(store) if s.tool == "create_ics")
    ctx = ToolContext(store=store, packs=load_all_packs(), user_id=USER, today=TODAY, procedure_id=PACK.id)
    action_id = run_tool(build_registry(), reminder.tool, reminder.args, ctx, "orchestrator").output["action_id"]
    assert approve_action(store, load_all_packs(), USER, action_id, TODAY, tmp_path).status == "executed"
    assert not any(s.tool == "create_ics" for s in found(store))
