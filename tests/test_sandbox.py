"""Mock world, artifact tools, promotion gate and approval flow."""
from datetime import date

import pymupdf
import pytest

from src.pack import load_all_packs
from src.sandbox.approval import approve_action, deny_action
from src.sandbox.mock_world import simulate
from src.tools.common import ToolContext
from src.tools.create_ics import fold_line, ics_escape
from src.tools.registry import build_registry, run_tool
from src.twin import service

USER = "synthetic_student_01"
TODAY = date(2026, 11, 18)
PACKS = load_all_packs()
PACK = PACKS["ikamet_first"]
APPLICATION_READY = {"exchange_student": True, "planned_stay_months": 7, "permit_type": "short_term",
                     "is_in_turkiye": True, "entered_with_passport": True, "application_date": "2026-11-10",
                     "legal_stay_expiry_date": "2026-11-20", "application_channel": "https://e-ikamet.goc.gov.tr/",
                     "passport_expiry_date": "2027-08-01", "requested_permit_end_date": "2027-06-01"}


@pytest.fixture
def ctx(store, tmp_path):
    return ToolContext(store=store, packs=PACKS, user_id=USER, today=TODAY, procedure_id="ikamet_first",
                       outbox_dir=tmp_path / "outbox")


def seed(ctx, facts):
    for name, value in facts.items():
        service.record_app_value(ctx.store, PACK, USER, name, value)


def call(ctx, name, args):
    return run_tool(build_registry(), name, args, ctx, "orchestrator")


def approve(ctx, action_id):
    return approve_action(ctx.store, PACKS, USER, action_id, TODAY, ctx.outbox_dir)


# ------------------------------------------------------------- mock world --
def test_stage_is_accepted_only_when_all_required_rules_pass():
    from src.pack import coerce_fact
    facts = {k: coerce_fact(PACK, k, v) for k, v in APPLICATION_READY.items()}
    assert simulate(PACK, facts, "official_application").accepted
    facts.pop("application_channel")
    outcome = simulate(PACK, facts, "official_application")
    assert not outcome.accepted and outcome.unknown == {"R005": ["application_channel"]}
    assert "nothing was submitted" in outcome.message


def test_stage_requirements_include_prerequisites():
    outcome = simulate(PACK, {}, "iro_preverification")
    assert {"R020", "R007", "R005"} <= set(outcome.required)


def test_simulate_tool_stores_the_attempt_and_rejects_unknown_stages(ctx):
    seed(ctx, APPLICATION_READY)
    output = call(ctx, "simulate_submission", {"stage_id": "official_application"}).output
    assert output["accepted"] and ctx.store.latest_simulation(USER, PACK.id, "official_application").accepted
    assert "Unknown stage" in call(ctx, "simulate_submission", {"stage_id": "moon_landing"}).error


# ----------------------------------------------------------------- ics --
def test_create_ics_uses_the_fact_date_and_only_proposes(ctx):
    seed(ctx, {"legal_stay_expiry_date": "2026-11-20"})
    output = call(ctx, "create_ics", {"title": "Legal stay ends", "date_field": "legal_stay_expiry_date",
                                      "rule_ids": ["R004"]}).output
    assert output["status"] == "proposed" and "2026-11-20" in output["title"]
    assert not ctx.outbox_dir.exists()
    proposal = ctx.store.get_proposal(output["action_id"])
    text = proposal.payload["ics"]
    assert text.startswith("BEGIN:VCALENDAR\r\n") and text.endswith("END:VCALENDAR\r\n")
    assert "DTSTART;VALUE=DATE:20261120\r\n" in text and "DTEND;VALUE=DATE:20261121\r\n" in text
    assert output["action_id"] in ctx.created_proposals


@pytest.mark.parametrize("args,message", [
    ({"title": "Legal stay ends", "date_field": "legal_stay_expiry_date"}, "not known yet"),
    ({"title": "Wrong", "date_field": "age"}, "not a date field"),
    ({"title": "Legal stay ends", "date_field": "application_date", "description": "Go on 2026-12-24"}, "UNSUPPORTED_DATE"),
])
def test_create_ics_errors(ctx, args, message):
    seed(ctx, {"application_date": "2026-11-10", "age": 22})
    observation = call(ctx, "create_ics", args)
    assert not observation.ok and message in observation.error


def test_ics_escaping_and_folding():
    assert ics_escape("a,b;c\\d\ne") == "a\\,b\\;c\\\\d\\ne"
    folded = fold_line("DESCRIPTION:" + "ş" * 80)
    assert all(len(part.encode("utf-8")) <= 75 for part in folded.split("\r\n"))
    assert folded.split("\r\n")[1].startswith(" ")


# --------------------------------------------------------------- email --
def test_draft_email_is_a_draft_with_no_invented_address(ctx):
    # Migration Management has no address in the sources: the draft says so instead of inventing one.
    output = call(ctx, "draft_email", {"to_role": "Migration Management", "subject": "Question about my permit",
                                       "body": "Hello, could you tell me which documents are still missing?"}).output
    payload = ctx.store.get_proposal(output["action_id"]).payload
    assert "fill it in yourself" in payload["to_email"] and "Migration" in payload["to_name"]


def test_draft_email_uses_the_address_the_official_source_gives(ctx):
    # IRO's address is on the official Sabancı page SAB_EXCHANGE_CURRENT ("send ... to irolegal@sabanciuniv.edu").
    output = call(ctx, "draft_email", {"to_role": "IRO", "subject": "Question about my residence permit",
                                       "body": "Dear IRO, could you tell me which documents are still missing?"}).output
    payload = ctx.store.get_proposal(output["action_id"]).payload
    assert payload["to_email"] == "irolegal@sabanciuniv.edu" and "Sabancı" in payload["to_name"]


@pytest.mark.parametrize("args,message", [
    ({"to_role": "Embassy", "subject": "Hello there", "body": "Some polite body text."}, "Unknown contact role"),
    ({"to_role": "IRO", "subject": "Done", "body": "I have submitted my online application already."}, "COMPLETION_CLAIM"),
    ({"to_role": "IRO", "subject": "Visit", "body": "Can I come on 2026-12-01 to bring my file?"}, "UNSUPPORTED_DATE"),
    ({"to_role": "IRO", "subject": "Visit", "body": "Some polite body text.", "stage_id": "nope"}, "Unknown stage"),
])
def test_draft_email_errors(ctx, args, message):
    observation = call(ctx, "draft_email", args)
    assert not observation.ok and message in observation.error


# ----------------------------------------------------------- checklist --
def test_checklist_is_deterministic_markdown(ctx):
    seed(ctx, {"passport_expiry_date": "2027-07-15", "requested_permit_end_date": "2027-06-01"})
    output = call(ctx, "build_checklist", {"section_ids": ["passport_photo"]}).output
    text = ctx.store.get_proposal(output["action_id"]).payload["markdown"]
    assert "- [ ] R006:" in text and "- [?] R008:" in text and "nothing has been submitted" in text
    assert "R001" not in text
    assert not call(ctx, "build_checklist", {"section_ids": ["moon"]}).ok


# ------------------------------------------------------------------ pdf --
def test_fill_pdf_fills_known_fields_and_lists_missing(ctx):
    seed(ctx, {"holder_name": "Alex Example", "passport_expiry_date": "2027-08-01"})
    output = call(ctx, "fill_pdf", {"form_id": "preparation_sheet"}).output
    payload = ctx.store.get_proposal(output["action_id"]).payload
    assert payload["values"] == {"holder_name": "Alex Example", "passport_expiry": "2027-08-01"}
    assert "insurance_end_date" in payload["missing"]
    result = approve(ctx, output["action_id"])
    with pymupdf.open(result.artifact_path) as document:
        values = {w.field_name: w.field_value for page in document for w in page.widgets()}
    assert values["holder_name"] == "Alex Example" and values["passport_expiry"] == "2027-08-01"
    assert not call(ctx, "fill_pdf", {"form_id": "visa_form"}).ok


# ------------------------------------------------------------- approval --
def test_approval_writes_only_inside_the_user_outbox_and_logs_twin_to_real(ctx):
    seed(ctx, {"legal_stay_expiry_date": "2026-11-20"})
    action_id = call(ctx, "create_ics", {"title": "Legal stay ends", "date_field": "legal_stay_expiry_date"}).output["action_id"]
    result = approve(ctx, action_id)
    assert result.status == "executed" and result.artifact_path.endswith(".ics")
    assert (ctx.outbox_dir / USER).resolve() in (ctx.outbox_dir / USER / "x").resolve().parents
    assert open(result.artifact_path, encoding="utf-8", newline="").read().count("\r\n") > 5
    event = ctx.store.list_events(USER, kind="artifact_created")[0]
    assert event.direction == "twin_to_real" and approve(ctx, action_id).message == "Already executed"


def test_deny_writes_nothing(ctx):
    action_id = call(ctx, "build_checklist", {}).output["action_id"]
    assert deny_action(ctx.store, USER, action_id).status == "denied"
    assert approve(ctx, action_id).status == "denied" and not ctx.outbox_dir.exists()


def test_other_users_cannot_approve(ctx):
    action_id = call(ctx, "build_checklist", {}).output["action_id"]
    with pytest.raises(ValueError):
        approve_action(ctx.store, PACKS, "someone_else", action_id, TODAY, ctx.outbox_dir)


def test_stage_bound_action_needs_an_accepted_simulation_on_current_facts(ctx):
    seed(ctx, APPLICATION_READY)
    action_id = call(ctx, "draft_email", {"to_role": "IRO", "subject": "Starting my application",
                                          "body": "Dear IRO, I am about to start my online application.",
                                          "stage_id": "official_application"}).output["action_id"]
    blocked = approve(ctx, action_id)
    assert blocked.status == "blocked" and "simulate_submission" in blocked.message
    call(ctx, "simulate_submission", {"stage_id": "official_application"})
    service.record_app_value(ctx.store, PACK, USER, "application_date", "2026-11-12")
    assert "Facts changed" in approve(ctx, action_id).message
    call(ctx, "simulate_submission", {"stage_id": "official_application"})
    executed = approve(ctx, action_id)
    assert executed.status == "executed" and "NOT sent" in open(executed.artifact_path, encoding="utf-8").read()
