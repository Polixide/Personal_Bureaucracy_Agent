"""Deterministic readiness, targeted corrections and revalidation in the real workflow."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

import config as settings
from scripts.benchmark_address_vision import EXPECTED, PAGES, TERRA, make_pdf, mocked_openai_client
from src.agents.planner import actionable_needs
from src.agents.planning_policy import canonical_candidates
from src.core.graph import build_graph
from src.core.model_provider import AIOptions, use_ai_options
from src.core.procedure import (
    address_proof_status, advance_procedure_if_ready, assess_document_collection, stage_document_checklist,
)
from src.core.state import Document, ProcedureState, UserEvent, new_run
from src.document_processing.extractor import TextPage
from src.document_processing.processor import get_document_provider, process_document
from src.document_processing.providers import OpenAIProvider
from src.evaluation.scenarios import SYNTHETIC, ScriptedDocumentProvider
from src.memory.checkpointer import sqlite_checkpointer
from src.ui.components import primary_action, workflow_summary
from src.workers.document_analyzer import TYPE_MAP
from src.workers.verification import MILESTONE_SCOPE, accepted_sources, conflict_details, document_review
from test_address_proof import SAMPLES as ACCOMMODATION
from test_address_proof import add_document, case as address_case
from src.evaluation.scenarios import confirm as confirm_address
from test_milestone_documents import SAMPLES, analyze, confirm
from test_ui_workspaces import choose, entered, ui_case


ADDRESS = SYNTHETIC["address_form"].replace("Evaluation", "Alex").replace("Student", "Example").replace("X0000001", "P1234567")
STAMP = SAMPLES["turkiye_entry_stamp"].replace("Entry date", "Latest entry date")
INSURANCE = SAMPLES["health_insurance_certificate"].replace("Coverage start: 2026-09-01\n", "") + "\nPrivate health insurance"


def fresh_state(*, guided=False):
    state = new_run(UserEvent(kind="reevaluate"), "demo-user", scope=MILESTONE_SCOPE)
    return {**state, "procedure": ProcedureState()} if guided else state


def relevant(review):
    return {row["name"]: row for row in review["fields"] if row["issue"]}


@pytest.mark.parametrize(("kind", "text", "declarations"), [
    ("passport_identity_page", SAMPLES["passport_identity_page"], {}),
    ("turkiye_entry_stamp", STAMP, {}),
    ("student_enrollment_certificate", SAMPLES["student_enrollment_certificate"], {}),
    ("health_insurance_certificate", INSURANCE, {}),
    ("ikamet_application_form", SAMPLES["ikamet_application_form"],
     {"application_form_signed": True, "online_application_completed": True}),
    ("declaration_control_document", "Declaration Control Document\nSigned: yes", {}),
    ("card_fee_receipt", SAMPLES["card_fee_receipt"], {}),
    ("utility_bill", ACCOMMODATION["utility_bill"], {}),
    ("address_form", ADDRESS, {}),
    ("dorm_certificate", ACCOMMODATION["dorm_certificate"], {"dorm_certificate_format": True}),
    ("rental_agreement", ACCOMMODATION["rental_agreement"], {"notary_certification": True}),
    ("notarized_commitment", ACCOMMODATION["notarized_commitment"],
     {"notary_certification": True, "host_married": False, "host_is_turkish": False}),
    ("host_support", ACCOMMODATION["host_support"], {"host_identity_copy": True}),
    ("utility_subscription", ACCOMMODATION["utility_subscription"], {"host_utility_current": True}),
])
def test_supported_categories_use_valid_readings_and_only_explicit_declarations(tmp_path, kind, text, declarations):
    graph = build_graph()
    scope = {"configurable": {"thread_id": "synthetic-category"}}
    state = analyze(graph, scope, tmp_path, kind, text=text, payload={"holder_name": "Alex Example", "student_id": "S123"})
    document = state["documents"][0]
    assert document.doc_type == TYPE_MAP[kind]
    review = document_review(state, document)
    expected = set(declarations)
    if kind == "host_support":
        expected.add("host_address_certificate")  # The file may supply either recorded supporting page.
    assert set(relevant(review)) == expected
    if declarations:
        state = confirm(graph, scope, document.doc_id, declarations)
    assert document_review(state, state["documents"][0])["status"] == "complete"
    records = state["verification_records"][document.doc_id]
    assert {name for name, record in records.items() if record.origin == "user_confirmed"} == declarations.keys()


@pytest.mark.parametrize(("changed", "issue"), [
    ("", "missing"),
    ("Expiry date: 31.02.2030", "invalid"),
])
def test_only_missing_or_invalid_mandatory_date_needs_review(tmp_path, changed, issue):
    graph = build_graph()
    scope = {"configurable": {"thread_id": "date-issue"}}
    text = SAMPLES["passport_identity_page"].replace("Expiry date: 2030-08-30", changed)
    state = analyze(graph, scope, tmp_path, "passport_identity_page", text=text)
    review = document_review(state, state["documents"][0])
    assert review["status"] == "needs_review"
    assert {name: row["issue"] for name, row in relevant(review).items()} == {"expiry_date": issue}
    if issue == "invalid":
        assert relevant(review)["expiry_date"]["message"]
    assert "birth_date" not in relevant(review)
    assert state["residence_permit_context"].passport_or_foreign_id == "P1234567"


def test_local_ocr_flags_only_the_low_confidence_field_and_avoids_false_conflicts():
    wrong = "Passport number: P123456I"
    text = ADDRESS.replace("Passport number: P1234567", wrong)
    page = TextPage(1, text, extraction_method="rapidocr", ocr_used=True, ocr_confidence=0.92,
                    low_confidence_lines=(wrong,))
    address_info = process_document("address.txt", pages=[page])
    address = Document(doc_type="address_form", status="analyzed", doc_id="address",
                       file_path="address.txt", info=address_info)
    passport_info = process_document("passport.txt", pages=[TextPage(1, SAMPLES["passport_identity_page"])],
                                     provider=ScriptedDocumentProvider("passport_identity_page"))
    passport = Document(doc_type="passport_copy", status="analyzed", doc_id="passport",
                        file_path="passport.txt", info=passport_info)
    state = {**fresh_state(), "documents": [address, passport]}
    assert address_info.passport_number == "P123456I"
    assert address_info.extraction_diagnostics["ocr_uncertain_fields"] == ["passport_number"]
    assert set(relevant(document_review(state, address))) == {"passport_number"}
    assert relevant(document_review(state, address))["passport_number"]["suggestion"] == "P123456I"
    assert {entry["document_id"] for entry in accepted_sources(state)["passport_or_foreign_id"]} == {"passport"}
    assert not conflict_details(state)


def test_complete_application_collection_has_no_document_confirmations(tmp_path):
    graph = build_graph()
    scope = {"configurable": {"thread_id": "zero-document-confirmations"}}
    graph.update_state(scope, fresh_state(guided=True), as_node="respond")
    confirm(graph, scope, "__declarations__", {"address_proof_type": "utility_bill_and_address_form",
                                               "biometric_photo": True,
                                               "requested_permit_end_date": "2027-06-25"})
    for kind, text in (
        ("passport_identity_page", SAMPLES["passport_identity_page"]),
        ("turkiye_entry_stamp", STAMP),
        ("student_enrollment_certificate", SAMPLES["student_enrollment_certificate"]),
        ("health_insurance_certificate", INSURANCE),
        ("address_form", ADDRESS),
        ("utility_bill", ACCOMMODATION["utility_bill"]),
    ):
        state = analyze(graph, scope, tmp_path, kind, text=text,
                        payload={"holder_name": "Alex Example", "student_id": "S123"})
    assert assess_document_collection(state).ready
    assert state["procedure"].stage_status == "READY"
    assert all(row["status"] == "complete" for row in stage_document_checklist(state))
    assert primary_action(state)["kind"] == "start_application"
    assert all(document_review(state, doc)["status"] == "complete" for doc in state["documents"])
    for owner, fields in state["verification_records"].items():
        if not owner.startswith("__"):
            assert all(record.origin == "source_grounded" for record in fields.values())
    assert not any(need.get("prototype_confirmation_fields") for need in actionable_needs(state)
                   if need.get("document_type") in {"passport_copy", "turkiye_entry_stamp", "student_certificate", "health_insurance"})
    transition = advance_procedure_if_ready(state, recorded_at=datetime(2026, 11, 26, tzinfo=timezone.utc))
    assert transition["procedure"].current_stage == "EIKAMET_PREPARATION"
    assert "verification_records" not in transition

    older = deepcopy(state)
    address = next(doc for doc in older["documents"] if doc.doc_type == "address_form")
    del older["verification_records"][address.doc_id]
    restored = advance_procedure_if_ready(older, recorded_at=datetime(2026, 11, 26, tzinfo=timezone.utc))
    assert restored["verification_records"][address.doc_id]["declared_address"].origin == "source_grounded"
    assert restored["procedure"].eikamet_preparation.fields["turkiye_address"].status == "VERIFIED"
    assert address.doc_id not in older["verification_records"]


def test_explicit_latest_and_ordinary_stamp_declaration_are_distinct(tmp_path):
    graph = build_graph()
    scope = {"configurable": {"thread_id": "ordinary-entry"}}
    state = analyze(graph, scope, tmp_path, "turkiye_entry_stamp")
    review = document_review(state, state["documents"][0])
    row = relevant(review)["latest_entry_date"]
    assert row["issue"] == "declaration"
    assert row["suggestion"] == "2026-09-15"
    assert state["residence_permit_context"].latest_entry_date is None


def test_later_upload_conflict_correction_and_restart_use_the_same_state(ui_case, tmp_path):
    ui_case.seed(entered(fresh_state(guided=True)))
    with sqlite_checkpointer(tmp_path / "ui.db") as checkpointer:
        graph = build_graph(checkpointer)
        scope = ui_case.config
        confirm(graph, scope, "__declarations__", {"address_proof_type": "utility_bill_and_address_form"})
        initial = analyze(graph, scope, tmp_path, "address_form", text=ADDRESS)
        address = initial["documents"][0]
        assert document_review(initial, address)["status"] == "complete"
        assert not conflict_details(initial)
        passport_text = SAMPLES["passport_identity_page"].replace("P1234567", "P7654321")
        conflict = analyze(graph, scope, tmp_path, "passport_identity_page", text=passport_text)
        passport = next(doc for doc in conflict["documents"] if doc.doc_type == "passport_copy")
        assert conflict["verification_conflicts"] == ["passport_or_foreign_id"]
        assert document_review(conflict, address)["status"] == document_review(conflict, passport)["status"] == "conflict"
        assert assess_document_collection(conflict).prerequisites["passport_identity"] == "CONFLICT"
        assert address_proof_status(conflict)["parts"][0]["status"] == "conflict"
        assert next(row for row in workflow_summary(conflict)["items"] if row["key"] == "passport_identity")["status"] == "conflict"
        candidate = next(row for row in canonical_candidates(conflict) if row["target_document"] == "address_form")
        assert candidate["action_type"] == "CORRECT_INFORMATION"
        assert candidate["fields"] == ["passport_or_foreign_id"]
    ui_case.nav("documents")
    choose(ui_case.app, "selectbox", "Saved document", address.doc_id)
    ui_case.app.run()
    assert "Conflict detected" in ui_case.text()
    assert "P1234567" in ui_case.text() and "P7654321" in ui_case.text()
    assert [widget.label for widget in ui_case.app.text_input] == ["Passport number"]
    choose(ui_case.app, "text_input", "Passport number", "p765 4321")
    ui_case.click("Save and confirm")
    saved = ui_case.read()
    assert not saved["verification_conflicts"]
    assert saved["verification_records"][address.doc_id]["passport_number"].value == "p765 4321"
    assert next(doc for doc in saved["documents"] if doc.doc_id == address.doc_id).info.passport_number == "P1234567"
    assert assess_document_collection(saved).prerequisites["passport_identity"] == "READY"
    assert address_proof_status(saved)["parts"][0]["status"] == "complete"
    assert not any(button.label == "Save and confirm" for button in ui_case.app.button)

    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reopened.session_state["workspace_profile_id"] = "demo-user"
    reopened.session_state[f"view_{ui_case.key}"] = "documents"
    reopened.run()
    choose(reopened, "selectbox", "Saved document", address.doc_id)
    reopened.run()
    assert not reopened.exception
    assert not any(button.label == "Save and confirm" for button in reopened.button)
    assert ui_case.read()["trace_id"] == saved["trace_id"]


def test_complete_vision_extraction_has_no_mandatory_form_or_repeat_requests(ui_case, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_REASONING_EFFORT", "low")
    path = tmp_path / "synthetic-signed-address.pdf"
    make_pdf(path)
    with pymupdf.open(path) as pdf:
        pdf[1].insert_text((70, 310), "Signature present: Yes")
        pdf.saveIncr()
    text = (*PAGES[1], "Signature present: Yes")
    wire = []
    with mocked_openai_client(wire, transcribed_pages=(PAGES[0], text)) as client:
        monkeypatch.setattr("src.document_processing.processor.get_document_provider",
                            lambda name, **kwargs: OpenAIProvider(model=TERRA, client=client)
                            if name == "openai" else get_document_provider(name, **kwargs))
        ui_case.seed(entered(fresh_state(guided=True)))
        event = UserEvent(kind="analyze_document", file_path=str(path))
        with sqlite_checkpointer(tmp_path / "ui.db") as checkpointer, use_ai_options(
            AIOptions(cloud_document=(event.kind, event.file_path))
        ):
            state = build_graph(checkpointer).invoke(new_run(event, "demo-user", scope=MILESTONE_SCOPE), ui_case.config)
        document = state["documents"][0]
        assert document.info.model_dump(mode="json", include=set(EXPECTED)) == EXPECTED
        assert document.info.signature_present is True
        assert document_review(state, document)["status"] == "complete"
        assert state["verification_records"][document.doc_id]["last_name"].origin == "source_grounded"
        assert state["run_status"] == "DOCUMENT_ANALYZED"
        assert document.analysis_diagnostics["image_count"] == 1
        assert len(wire) == 1
        ui_case.nav("documents")
        choose(ui_case.app, "selectbox", "Saved document", document.doc_id)
        ui_case.app.run()
        assert not ui_case.app.text_input
        assert not any(button.label == "Save and confirm" for button in ui_case.app.button)
        assert "Ready" in ui_case.text()
        ui_case.app.run()
        ui_case.healthy()
        assert len(wire) == 1
        assert ui_case.read()["trace_id"] == state["trace_id"]

def test_hosted_evidence_points_to_mismatched_fields_and_accepts_a_matching_alternative():
    state = confirm_address(address_case(), "__declarations__", {"address_proof_type": "notarized_commitment"})
    host = add_document(state, "notarized_commitment")
    state = confirm_address(state, host.doc_id, {
        "notary_certification": True, "host_married": False, "host_is_turkish": True,
        "host_identity_copy": True, "host_population_record": True,
    })
    bill = add_document(state, "utility_bill", text=ACCOMMODATION["utility_bill"].replace("Example Host", "Different Host"))
    state = confirm_address(state, bill.doc_id, {"host_utility_current": True})
    support = add_document(state, "host_support", text=ACCOMMODATION["host_support"].replace("Street 12", "Street 15"))
    state = confirm_address(state, support.doc_id, {"host_address_certificate": True})
    assert set(state["verification_conflicts"]) == {"host_name", "host_address"}
    assert set(relevant(document_review(state, bill))) == {"holder_name"}
    assert set(relevant(document_review(state, support))) == {"declared_address"}
    assert {row["document_id"] for row in conflict_details(state)["host_name"]["observations"]} == {host.doc_id, bill.doc_id}
    assert address_proof_status(state)["status"] == "conflict"

    alternative = add_document(state, "utility_subscription")
    state = confirm_address(state, alternative.doc_id, {"host_utility_current": True})
    assert state["verification_conflicts"] == ["host_address"]
    state = confirm_address(state, support.doc_id, {"declared_address": "Synthetic Street 12"})
    assert state["verification_conflicts"] == []
    assert address_proof_status(state)["ready"]
