"""End-to-end two-way flow between the real world and the Digital Twin.

real -> twin : upload + extraction, user confirmation, an outcome reported later
twin         : agent run (scripted model) proposes artifacts and tries the stage in the sandbox
twin -> real : only approved proposals become files in the outbox; a refused simulation blocks one
"""
from datetime import date

from agent_helpers import USER, d, run, seed
from src.document_processing.schemas import PassportIdentity, SourceEvidence
from src.pack import load_all_packs
from src.sandbox.approval import approve_action
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool
from src.twin import service

TODAY = date(2026, 11, 18)
PACKS = load_all_packs()
PACK = PACKS["ikamet_first"]
KNOWN = {"exchange_student": True, "planned_stay_months": 7, "permit_type": "short_term", "is_in_turkiye": True,
         "entered_with_passport": True, "application_date": "2026-11-10", "legal_stay_expiry_date": "2026-11-20",
         "requested_permit_end_date": "2027-06-01"}


def passport_info():
    quote = "Date of expiry: 01.08.2027"
    return PassportIdentity(expiry_date="2027-08-01", source_pages={"expiry_date": 1}, source_snippets={"expiry_date": quote},
                            source_evidence=[SourceEvidence(field="expiry_date", original_value="01.08.2027",
                                                            normalized_value="2027-08-01", snippet=quote,
                                                            line_number=1, page_number=1)])


def test_documents_in_artifacts_out_and_outcomes_back_in(store, tmp_path, monkeypatch):
    # real -> twin: the passport upload is extracted; the other facts are confirmed in the app
    seed(store, KNOWN)
    path = tmp_path / "passport.txt"
    path.write_text("SYNTHETIC PASSPORT", encoding="utf-8")
    monkeypatch.setattr("src.document_processing.processor.process_document", lambda *a, **k: passport_info())
    service.register_upload(store, USER, "passport", str(path), PACK.id)
    ctx = ToolContext(store=store, packs=PACKS, user_id=USER, today=TODAY, procedure_id=PACK.id)
    assert run_tool(build_registry(), "extract_document", {"doc_id": "passport"}, ctx, "orchestrator").ok

    # twin: the agent simulates the stage (refused: the official channel is unknown) and proposes artifacts
    replies = [d("simulate_submission", {"stage_id": "official_application"}),
               d("create_ics", {"title": "Legal stay ends", "date_field": "legal_stay_expiry_date", "rule_ids": ["R004"]}),
               d("draft_email", {"to_role": "IRO", "subject": "Starting my online application",
                                 "body": "Dear IRO, I will start my online application soon. Is anything missing?",
                                 "stage_id": "official_application"}),
               d("final_answer", {"summary": "Your file is almost ready; the official application channel is not confirmed yet.",
                                  "claims": [{"rule_id": "R005", "status": "unknown"}],
                                  "next_actions": [{"title": "Confirm the official e-İkamet channel", "ref": "R005"}]})]
    result, _ = run(store, replies, today=TODAY, tmp_path=tmp_path)
    assert result.status == "answered" and len(result.proposals) == 2
    calendar_id, email_id = result.proposals
    outbox = tmp_path / "outbox"

    # twin -> real: the calendar file may leave; the stage-bound e-mail is blocked by the failed simulation
    assert approve_action(store, PACKS, USER, calendar_id, TODAY, outbox).status == "executed"
    assert approve_action(store, PACKS, USER, email_id, TODAY, outbox).status == "blocked"

    # real -> twin: the user reports the outcome; the sandbox now accepts and the e-mail can be approved
    service.record_outcome(store, PACK, USER, "portal_checked", "I opened the official e-İkamet portal",
                           {"application_channel": "https://e-ikamet.goc.gov.tr/"})
    assert run_tool(build_registry(), "simulate_submission", {"stage_id": "official_application"}, ctx, "orchestrator").output["accepted"]
    executed = approve_action(store, PACKS, USER, email_id, TODAY, outbox)
    assert executed.status == "executed"

    directions = [e.direction for e in store.list_events(USER)]
    assert directions.count("twin_to_real") == 2 and "real_to_twin" in directions
    assert sorted(p.name for p in (outbox / USER).iterdir()) == sorted([f"{calendar_id}.ics", f"{email_id}.eml"])
