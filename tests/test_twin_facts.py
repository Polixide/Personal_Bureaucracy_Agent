"""src/twin/facts.py + service.py: the provenance guarantees of milestone 1, now generic.

Ported behaviours (tests/test_verification.py, test_document_lifecycle.py, test_milestone_documents.py):
unconfirmed stays unknown, a confirmation never overwrites the observation, a same-document correction
wins, independent documents conflict until resolved, deleting a document removes its facts, derived
values are labelled, an entry date needs explicit confirmation.
"""
from datetime import date, datetime, timezone

import pytest

from src.pack import load_pack
from src.rules import evaluate_rules
from src.tools.extract_document import DocumentObservation, ExtractionResult, FactProposal
from src.twin import service
from src.twin.facts import build_view, months_between
from src.twin.models import Fact, TwinDocument

PACK = load_pack("ikamet_first")
USER = "synthetic_student_01"
TODAY = date(2026, 11, 18)


def view(store):
    return service.twin_view(store, PACK, USER, TODAY)


def status(store, rule_id):
    return next(r for r in evaluate_rules(PACK, view(store).fact_map()) if r.rule_id == rule_id).status


def upload(store, doc_id, doc_type, facts, path="data/uploads/x.pdf"):
    service.register_upload(store, USER, doc_id, path, PACK.id)
    proposals = [FactProposal(field=f, value=v, from_field=f, page=1, snippet=f"{f}: {v}",
                              requires_confirmation=(f == "latest_entry_date")) for f, v in facts.items()]
    observation = DocumentObservation(doc_type=doc_type, facts=proposals,
                                      grounded={f: {"name": f, "value": str(v)} for f, v in facts.items()})
    return service.store_extraction(store, PACK, USER, doc_id, ExtractionResult(observations=[observation]))


def test_unconfirmed_chat_statement_stays_unknown_for_the_rules(store):
    stated = service.record_statement(store, PACK, USER, "is_in_turkiye", "yes", confirmed=False)
    service.record_statement(store, PACK, USER, "entered_with_passport", True, confirmed=True)
    assert status(store, "R003") == "unknown"
    assert [f.field for f in view(store).unconfirmed] == ["is_in_turkiye"]
    store.confirm_fact(USER, stated.fact_id)
    assert status(store, "R003") == "pass"


def test_document_facts_count_but_entry_date_waits_for_confirmation(store):
    upload(store, "stamp", "turkiye_entry_stamp", {"turkey_entry_page_present": True, "latest_entry_date": date(2026, 9, 15)})
    current = view(store)
    assert current.values["turkey_entry_page_present"] is True
    assert "latest_entry_date" not in current.values
    assert [f.field for f in current.unconfirmed] == ["latest_entry_date"]


def test_same_document_correction_wins_and_the_observation_is_kept(store):
    upload(store, "ins", "health_insurance_certificate", {"insurance_end_date": date(2027, 5, 31)})
    service.correct_document_field(store, PACK, USER, "ins", "insurance_end_date", "2027-06-30")
    assert view(store).values["insurance_end_date"] == date(2027, 6, 30)
    history = store.fact_history(USER, PACK.id, "insurance_end_date")
    assert {f.provenance for f in history} == {"document", "user_stated"}
    assert any(f.value == date(2027, 5, 31) and f.valid_to is None for f in history)


def test_independent_documents_conflict_until_the_user_resolves(store):
    upload(store, "passport", "passport_identity_page", {"holder_name": "Alex Example"})
    upload(store, "insurance", "health_insurance_certificate", {"holder_name": "Alexa Sample"})
    current = view(store)
    assert "holder_name" not in current.values
    assert set(current.conflicts["holder_name"]) == {"Alex Example", "Alexa Sample"}
    service.resolve_conflict(store, PACK, USER, "holder_name", "Alex Example")
    assert view(store).values["holder_name"] == "Alex Example" and not view(store).conflicts


def test_name_formatting_differences_are_not_conflicts(store):
    upload(store, "passport", "passport_identity_page", {"holder_name": "ALEX  EXAMPLE"})
    upload(store, "insurance", "health_insurance_certificate", {"holder_name": "Alex Example"})
    assert "holder_name" in view(store).values and not view(store).conflicts


def test_deleting_a_document_removes_its_facts_and_corrections(store):
    upload(store, "ins", "health_insurance_certificate", {"insurance_end_date": date(2027, 5, 31)})
    service.correct_document_field(store, PACK, USER, "ins", "insurance_end_date", "2027-06-30")
    service.delete_document(store, USER, PACK.id, "ins")
    assert "insurance_end_date" not in view(store).values
    assert store.list_documents(USER) == []
    assert store.fact_history(USER, PACK.id, "insurance_end_date")


def test_reanalysis_replaces_the_previous_document_facts(store):
    upload(store, "ins", "health_insurance_certificate", {"insurance_end_date": date(2027, 5, 31)})
    upload(store, "ins", "health_insurance_certificate", {"insurance_start_date": date(2026, 9, 1)})
    current = view(store)
    assert "insurance_end_date" not in current.values and current.values["insurance_start_date"] == date(2026, 9, 1)


def test_failed_extraction_rejects_the_document_and_clears_old_facts(store):
    upload(store, "ins", "health_insurance_certificate", {"insurance_end_date": date(2027, 5, 31)})
    service.store_extraction(store, PACK, USER, "ins", ExtractionResult(error="Local Ollama timed out"))
    assert store.get_document(USER, "ins").status == "rejected"
    assert "insurance_end_date" not in view(store).values


def test_passport_bundle_parts_get_their_own_document_ids(store):
    service.register_upload(store, USER, "bundle", "data/uploads/p.pdf", PACK.id)
    result = ExtractionResult(observations=[DocumentObservation(doc_type="passport_identity_page"),
                                            DocumentObservation(doc_type="turkiye_entry_stamp")])
    saved = service.store_extraction(store, PACK, USER, "bundle", result)
    assert [d.doc_id for d in saved] == ["bundle", "bundle_part2"]


# -------------------------------------------------------------- derived --
def test_months_between_counts_whole_calendar_months():
    assert months_between(date(2026, 9, 15), date(2027, 1, 14)) == 3
    assert months_between(date(2026, 9, 15), date(2027, 1, 15)) == 4
    assert months_between(date(2027, 1, 1), date(2026, 1, 1)) is None


def test_derived_values_follow_the_pack_and_are_labelled(store):
    upload(store, "cert", "student_enrollment_certificate",
           {"exchange_student": True, "program_start_date": date(2026, 9, 1), "program_end_date": date(2027, 2, 1)})
    current = view(store)
    assert current.derived["planned_stay_months"] == 5
    assert current.derived["exchange_program_evidence_present"] is True
    assert current.derived["current_date"] == TODAY
    assert "student_certificate_present" not in current.values


def test_document_complete_needs_every_required_field(store):
    fields = {"holder_name": "Alex Example", "student_id": "S1", "institution": "Sabancı University",
              "student_certificate_issue_date": date(2026, 11, 1)}
    service.register_upload(store, USER, "cert", "x.pdf", PACK.id)
    grounded = {name: {"name": name, "value": "x"} for name in ("holder_name", "student_id", "institution", "issue_date")}
    proposals = [FactProposal(field=f, value=v, from_field=f) for f, v in fields.items()]
    service.store_extraction(store, PACK, USER, "cert", ExtractionResult(observations=[DocumentObservation(
        doc_type="student_enrollment_certificate", grounded=grounded, facts=proposals)]))
    assert view(store).values["student_certificate_present"] is True
    assert status(store, "R014") == "pass"


def test_explicit_value_beats_derived(store):
    service.record_app_value(store, PACK, USER, "planned_stay_months", 2)
    upload(store, "cert", "student_enrollment_certificate",
           {"program_start_date": date(2026, 9, 1), "program_end_date": date(2027, 6, 1)})
    assert view(store).values["planned_stay_months"] == 2 and "planned_stay_months" not in view(store).derived


def test_conflicting_inputs_block_derivation(store):
    upload(store, "c1", "student_enrollment_certificate", {"program_start_date": date(2026, 9, 1), "program_end_date": date(2027, 6, 1)})
    upload(store, "c2", "student_enrollment_certificate", {"program_start_date": date(2026, 10, 1)})
    assert "program_start_date" in view(store).conflicts and "planned_stay_months" not in view(store).values


# ------------------------------------------------------- real -> twin --
def test_reported_outcome_becomes_confirmed_facts_and_an_event(store):
    facts = service.record_outcome(store, PACK, USER, "iro_reply", "IRO confirmed my scans",
                                   {"scans_sent_to_iro": "yes", "iro_preverification_confirmed": "yes"})
    assert all(f.confirmed and f.source_ref.startswith("outcome:") for f in facts)
    assert status(store, "R021") == "pass"
    event = store.list_events(USER, kind="outcome")[0]
    assert event.direction == "real_to_twin" and event.payload["kind"] == "iro_reply"


def test_outcome_with_a_bad_value_writes_nothing(store):
    with pytest.raises(ValueError):
        service.record_outcome(store, PACK, USER, "x", "y", {"scans_sent_to_iro": "yes", "age": "old"})
    assert store.current_facts(USER, PACK.id) == [] and store.list_events(USER) == []


def test_rule_update_retires_related_lessons(store):
    from src.twin.models import Episode
    store.add_episode(Episode(user_id=USER, procedure_id=PACK.id, failure_type="CLAIM_MISMATCH", validator_error="e",
                              corrective_action="c", outcome="o", correction_validated=True, rule_or_source_id="R015"))
    assert service.record_rule_update(store, USER, PACK.id, "R015", "1.1", "coverage wording changed") == 1
    assert store.list_events(USER, kind="rule_update")[0].payload["rule_id"] == "R015"


def test_unknown_field_statement_is_refused(store):
    with pytest.raises(ValueError):
        service.record_statement(store, PACK, USER, "favourite_colour", "blue", confirmed=False)


def test_build_view_ignores_closed_facts():
    closed = Fact(user_id=USER, procedure_id=PACK.id, field="age", value=30, provenance="user_stated", source_ref="app",
                  confirmed=True, fact_id=1, valid_to=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "age" not in build_view(PACK, [closed], [TwinDocument(doc_id="d", file_path="x", status="pending")], TODAY).values
