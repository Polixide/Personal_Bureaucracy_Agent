"""The İkamet pack after the review of its questions: address documents, when to ask, choices, later steps.

- The address in Türkiye is read as text (from the address proof or a utility bill), not asked as yes/no.
- R030 (own property: title deed + utility bill) comes from METU's list and stays unverified.
- Fields are asked now, later (after the online application) or never (computed, office data, documents only).
- A yes/no about having a document that is not uploaded becomes an upload step; the number of missing documents is
  computed by code once the first document is read; later-step failures are not "problems to fix today".
"""
from datetime import date

import pytest

from agent_helpers import USER, seed
from src.agent.briefing import build_briefing, open_questions
from src.agent.gate1 import GateContext, check_failed_rules_addressed
from src.agent.answer import FinalAnswer
from src.pack import coerce_fact, load_pack
from src.rules import evaluate_rules
from src.twin import service
from src.twin.facts import build_view
from src.twin.models import TwinDocument

PACK = load_pack("ikamet_first")
TODAY = date(2026, 11, 18)


def status(facts: dict, rule_id: str) -> str:
    typed = {name: coerce_fact(PACK, name, value) for name, value in facts.items()}
    return next(r.status for r in evaluate_rules(PACK, typed) if r.rule_id == rule_id)


def analyzed(doc_type: str) -> TwinDocument:
    return TwinDocument(doc_id=f"d_{doc_type}", file_path="x.pdf", status="analyzed", doc_type=doc_type)


# --------------------------------------------------------------- R030 --
@pytest.mark.parametrize("facts,expected", [
    ({"address_proof_type": "rental_agreement"}, "pass"),
    ({"address_proof_type": "title_deed", "utility_bill_present": True}, "pass"),
    ({"address_proof_type": "title_deed", "utility_bill_present": False}, "fail"),
    ({"address_proof_type": "title_deed"}, "unknown"),
    ({}, "unknown")])
def test_own_property_needs_a_utility_bill(facts, expected):
    assert status(facts, "R030") == expected


def test_the_utility_bill_rule_is_unverified_and_cites_its_source():
    rule = PACK.rule("R030")
    assert rule.status == "unverified" and rule.source_ids == ["METU_RESIDENCE_2026"]
    assert "title deed and a recent utility bill" in rule.notes


def test_the_address_documents_give_the_address_as_text():
    for doc_type in ("address_proof_document", "utility_bill"):
        assert "turkey_address" in {extract.field for extract in PACK.document(doc_type).extracts}
    assert PACK.field("turkey_address").type == "str"


def test_a_typed_address_and_phone_complete_the_contact_requirement():
    facts = {"turkey_address": "Orta Mah. Üniversite Cad. No:27, Tuzla, İstanbul", "turkey_phone": "+90 555 000 00 00",
             "contact_email": "student@example.org"}
    with_view = derived_values(facts)
    assert with_view["turkey_address_present"] is True and with_view["phone_present"] is True
    assert status({**facts, "turkey_address_present": True, "phone_present": True, "email_present": True}, "R018") == "pass"


def derived_values(facts: dict) -> dict:
    from src.twin.store import TwinStore
    with TwinStore(":memory:") as store:
        for name, value in facts.items():
            service.record_app_value(store, PACK, USER, name, value)
        return build_view(PACK, store.current_facts(USER, PACK.id), [], TODAY).values


# ------------------------------------------------------------- choices --
def test_a_choice_label_is_stored_as_the_value_the_rules_compare():
    assert coerce_fact(PACK, "address_proof_type", "Rental agreement in my name (kira sözleşmesi)") == "rental_agreement"
    assert coerce_fact(PACK, "address_proof_type", "kira sözleşmesi") == "rental_agreement"        # alias from documents
    assert coerce_fact(PACK, "application_channel", "Online at e-ikamet.goc.gov.tr (the official site)") == \
        "https://e-ikamet.goc.gov.tr/"


# ------------------------------------------------------ when to ask --
def test_office_data_and_computed_values_are_never_asked():
    asked = {q.field for q in open_questions(PACK, evaluate_rules(PACK, {}))}
    assert not asked & {"iro_window_start", "current_iro_window_id", "missing_required_documents_count",
                        "all_core_documents_present", "phone_present"}
    assert {"turkey_address", "turkey_phone", "permit_type"} <= asked


def test_steps_after_the_application_are_asked_later():
    later = {q.field for q in open_questions(PACK, evaluate_rules(PACK, {}), when="later")}
    assert {"fingerprint_step_completed", "gi_number_present"} <= later
    assert "fingerprint_step_completed" not in {q.field for q in open_questions(PACK, evaluate_rules(PACK, {}))}


def test_questions_do_not_mention_rule_ids():
    assert all("R0" not in field.question for field in PACK.procedure.fields)


def test_a_yes_no_about_a_missing_document_becomes_an_upload_step():
    view = build_view(PACK, [], [], TODAY)
    briefing = build_briefing(PACK, [], [], view, evaluate_rules(PACK, view.fact_map()))
    passport = next(q for q in briefing.questions if q.field == "passport_id_page_present")
    assert passport.upload_instead
    titles = [step.title for step in briefing.steps]
    assert not any("readable copy of your passport" in title for title in titles)
    assert any(step.kind == "upload" and step.ref == "passport_identity_page" for step in briefing.steps)
    assert any(step.kind == "upload" and step.ref == "address_proof_document" for step in briefing.steps)


# ----------------------------------------------------- document counts --
def test_missing_documents_are_counted_by_code_after_the_first_upload():
    assert "missing_required_documents_count" not in build_view(PACK, [], [], TODAY).values
    values = build_view(PACK, [], [analyzed("passport_identity_page")], TODAY).values
    assert values["missing_required_documents_count"] == 7 and values["all_core_documents_present"] is False


def test_a_later_step_failure_is_not_a_problem_to_fix_today():
    view = build_view(PACK, [], [analyzed("passport_identity_page")], TODAY)
    results = evaluate_rules(PACK, view.fact_map())
    assert next(r.status for r in results if r.rule_id == "R026") == "fail"
    briefing = build_briefing(PACK, [analyzed("passport_identity_page")], [], view, results)
    assert "R026" not in {p.rule_id for p in briefing.problems}
    gate = GateContext(pack=PACK, results=results)
    assert check_failed_rules_addressed(FinalAnswer(summary="Upload the rest."), gate) == []


# ------------------------------------------------- correcting a value --
def test_a_corrected_value_counts_and_the_unsure_one_is_closed(store):
    seed(store, {})
    unsure = service.record_statement(store, PACK, USER, "passport_expiry_date", "2027-01-01", confirmed=False)
    service.correct_unconfirmed(store, PACK, USER, unsure.fact_id, "2028-01-01")
    view = service.twin_view(store, PACK, USER, TODAY)
    assert view.values["passport_expiry_date"] == date(2028, 1, 1) and view.unconfirmed == []


# ------------------------------------------- reading the address whole --
def test_an_address_over_two_lines_is_read_whole_from_the_page():
    from src.document_processing.generic import continued
    page = "Address of the rented property:\nOrnek Mahallesi, Deneme Sokak No: 12\n34956 Tuzla / Istanbul\nMonthly rent: 15.000 TL"
    value, quote = continued(page, "Ornek Mahallesi, Deneme Sokak No: 12", "Ornek Mahallesi, Deneme Sokak No: 12", 3)
    assert value == "Ornek Mahallesi, Deneme Sokak No: 12, 34956 Tuzla / Istanbul"
    assert "Monthly rent" not in value and quote.endswith("34956 Tuzla / Istanbul")


def test_a_printed_document_title_is_mapped_to_the_code_the_rules_compare():
    assert coerce_fact(PACK, "address_proof_type", "KİRA SÖZLEŞMESİ / RENTAL AGREEMENT") == "rental_agreement"
    assert coerce_fact(PACK, "address_proof_type", "Noter onaylı TAAHHÜTNAME") == "notarized_commitment"
    assert coerce_fact(PACK, "address_proof_type", "something unrelated") == "something unrelated"


def test_the_utility_bill_is_optional_and_suggested_only_for_a_title_deed():
    def upload_refs(facts):
        view = build_view(PACK, [], [], TODAY)
        typed = {**view.fact_map(), **{k: coerce_fact(PACK, k, v) for k, v in facts.items()}}
        briefing = build_briefing(PACK, [], [], view, evaluate_rules(PACK, typed))
        return briefing, {step.ref for step in briefing.steps if step.kind == "upload"}
    briefing, refs = upload_refs({})
    assert "utility_bill" not in refs and briefing.documents_total == 8
    assert "utility_bill" in upload_refs({"address_proof_type": "title_deed"})[1]
