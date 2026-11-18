"""Offline first-stage V2 readiness; reuse real grounding and graph confirmations."""
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.core.graph import build_graph
from src.core.procedure import (
    V2_PRE_EIKAMET_INSURANCE_READINESS, assess_document_collection,
    advance_procedure_if_ready, document_collection_update,
)
from src.core.state import ProcedureState, UserEvent, new_run
from src.memory.checkpointer import sqlite_checkpointer
from src.memory.state_update import state_update
from src.workers.rules_engine import evaluate_requirements
from src.workers.verification import MILESTONE_SCOPE, milestone_checklist, verification_updates
from test_milestone_documents import analyze, confirm

RECORDED_AT = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)


@pytest.fixture
def ready_bundle(tmp_path):
    graph = build_graph()
    config = {"configurable": {"thread_id": "v2-collection"}}
    graph.update_state(config, {"procedure": ProcedureState()})
    for kind in ("passport_identity_page", "turkiye_entry_stamp", "student_enrollment_certificate", "health_insurance_certificate"):
        state = analyze(graph, config, tmp_path, kind)
    stamp = next(d for d in state["documents"] if d.doc_type == "turkiye_entry_stamp")
    confirm(graph, config, stamp.doc_id, {"latest_entry_date": "2026-09-15"})
    state = confirm(graph, config, "__declarations__", {
        "address_proof_type": "dorm_certificate", "biometric_photo": True,
        "insurance_evidence_type": "private_health_insurance",
        "requested_permit_end_date": "2027-06-25",
    })
    assert state["procedure"].stage_status == "READY"
    return graph, config, state


def assess(state):
    before = deepcopy(state)
    result = assess_document_collection(state)
    assert state == before  # Every assessment is read-only, including provenance.
    return result


def drop_document(state, kind):
    state = deepcopy(state)
    state["documents"] = [d for d in state["documents"] if d.doc_type != kind]
    return state


def override(state, owner, field, value):
    """Use the existing validated confirmation boundary, not bare context flags."""
    state = deepcopy(state)
    state.update(new_run(UserEvent(kind="confirm_information", document_id=owner,
                                  confirmed_fields={field: value}), "test-user", scope=MILESTONE_SCOPE))
    return {**state, **verification_updates(state)}


def test_fresh_procedure_assessment_is_not_started():
    state = {"procedure": ProcedureState()}
    result = assess(state)
    assert state["procedure"].current_stage == "DOCUMENT_COLLECTION"
    assert not result.ready and result.status == "NOT_STARTED"
    assert "missing:biometric_photo" in result.blockers


@pytest.mark.parametrize("kind,blocker", [
    ("passport_copy", "missing:passport_identity_evidence"),
    ("turkiye_entry_stamp", "missing:latest_turkiye_entry_evidence"),
    ("student_certificate", "R014"),
    ("health_insurance", "missing:health_insurance"),
])
def test_missing_document_blocks_even_with_stale_verified_context(ready_bundle, kind, blocker):
    state = drop_document(ready_bundle[2], kind)
    result = assess(state)
    assert not result.ready and result.status == "BLOCKED"
    assert blocker in result.blockers


@pytest.mark.parametrize("field,blocker", [
    ("address_proof_type", "missing:address_proof"),
    ("biometric_photo", "missing:biometric_photo"),
])
def test_missing_confirmed_prerequisite_blocks(ready_bundle, field, blocker):
    state = override(ready_bundle[2], "__declarations__", field, None)
    assert blocker in assess(state).blockers


@pytest.mark.parametrize("proof", ["dorm_certificate", "rental_agreement", "notarized_commitment"])
def test_reuses_accepted_address_categories(ready_bundle, proof):
    state = override(ready_bundle[2], "__declarations__", "address_proof_type", proof)
    assert assess(state).ready


def test_invalid_address_is_not_accepted(ready_bundle):
    state = override(ready_bundle[2], "__declarations__", "address_proof_type", "airbnb")
    assert "R017" in assess(state).blockers


@pytest.mark.parametrize("requested,end,ready", [
    ("2027-06-25", "2027-06-25", True),  # Requested boundary, inclusive.
    ("2027-09-01", "2027-06-25", True),  # Program-end fallback, inclusive.
    ("2027-05-01", "2027-05-01", True),  # Requested end alone is sufficient.
    ("2027-09-01", "2027-06-24", False),
])
def test_named_v2_insurance_end_policy(ready_bundle, requested, end, ready):
    state = override(ready_bundle[2], "__declarations__", "requested_permit_end_date", requested)
    doc = next(d for d in state["documents"] if d.doc_type == "health_insurance")
    state = override(state, doc.doc_id, "coverage_end_date", end)
    result = assess(state)
    assert result.ready is ready
    assert ("R015" in result.blockers) is not ready
    if ready:
        assert V2_PRE_EIKAMET_INSURANCE_READINESS in result.reason_refs


def test_insurance_start_not_required_and_general_R015_unchanged(ready_bundle):
    state = deepcopy(ready_bundle[2])
    doc = next(d for d in state["documents"] if d.doc_type == "health_insurance")
    state = override(state, doc.doc_id, "coverage_start_date", None)
    state = override(state, doc.doc_id, "insurance_valid_for_ikamet", False)
    state = override(state, "__declarations__", "insurance_exempt", False)
    # General/legal evaluation keeps its existing meaning; V2 is not a rewrite.
    general = {**state, "planning_scope": None}
    assert next(r for r in evaluate_requirements(general) if r.rule_id == "R015").passed is False
    assert assess(state).ready


def test_exemption_needs_no_policy_category_or_period_target(ready_bundle):
    state = drop_document(ready_bundle[2], "health_insurance")
    state = override(state, "__declarations__", "insurance_exempt", True)
    state = override(state, "__declarations__", "insurance_evidence_type", None)
    state = override(state, "__declarations__", "requested_permit_end_date", None)
    student = next(d for d in state["documents"] if d.doc_type == "student_certificate")
    state = override(state, student.doc_id, "program_end_date", None)
    assert assess(state).ready


def test_missing_insurance_category_blocks_nonexempt(ready_bundle):
    state = override(ready_bundle[2], "__declarations__", "insurance_evidence_type", None)
    assert "R016" in assess(state).blockers


def test_missing_insurance_target_does_not_guess_dates(ready_bundle):
    state = override(ready_bundle[2], "__declarations__", "requested_permit_end_date", None)
    student = next(d for d in state["documents"] if d.doc_type == "student_certificate")
    state = override(state, student.doc_id, "program_end_date", None)
    assert "missing:insurance_end_target" in assess(state).blockers


def test_entry_date_observation_still_requires_explicit_confirmation(ready_bundle):
    state = deepcopy(ready_bundle[2])
    stamp = next(d for d in state["documents"] if d.doc_type == "turkiye_entry_stamp")
    state["verification_records"][stamp.doc_id].pop("latest_entry_date")
    result = assess(state)
    assert not result.ready and "confirmation:latest_entry_date" in result.blockers


def test_relevant_conflict_blocks_but_later_unknowns_do_not(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state["verification_conflicts"] = ["holder_name"]
    assert "conflict:holder_name" in assess(state).blockers
    state["verification_conflicts"] = ["application_number", "g_number", "insurance_start_date"]
    assert assess(state).ready


def test_recomputes_cross_document_conflicts(ready_bundle):
    state = deepcopy(ready_bundle[2])
    student = next(d for d in state["documents"] if d.doc_type == "student_certificate")
    state = override(state, student.doc_id, "holder_name", "Different Person")
    state["verification_conflicts"] = []  # A stale cached list cannot hide a conflict.
    assert "conflict:holder_name" in assess(state).blockers


def test_raw_flags_approval_and_progress_cannot_establish_readiness(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state.update(documents=[], verification_records={}, verification_conflicts=[],
                 approval_decision=True, run_status="APPROVED",
                 action_progress={"done": {"status": "declared_completed"}})
    # Retained raw residence context and previous passed requirements are insufficient.
    result = assess(state)
    assert not result.ready and "missing:passport_identity_evidence" in result.blockers
    assert "missing:biometric_photo" in result.blockers


def test_unreviewed_observations_and_stale_source_records_do_not_count(ready_bundle):
    state = deepcopy(ready_bundle[2])
    for doc in state["documents"]:
        doc.info.source_evidence = []
        doc.info.source_pages = {}
        doc.info.source_snippets = {}
    result = assess(state)
    assert not result.ready and "missing:passport_identity_evidence" in result.blockers


def test_confirmation_provenance_required(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state["verification_records"]["__declarations__"]["biometric_photo"].confirmed_by = None
    assert "missing:biometric_photo" in assess(state).blockers


def test_complete_collection_uses_source_evidence_and_explicit_confirmations(ready_bundle):
    state = ready_bundle[2]
    result = assess(state)
    assert result.ready and result.status == "READY" and not result.blockers
    passport = next(d for d in state["documents"] if d.doc_type == "passport_copy")
    assert state["verification_records"][passport.doc_id]["passport_number"].origin == "source_grounded"
    assert state["verification_records"]["__declarations__"]["biometric_photo"].origin == "user_confirmed"
    assert state["procedure"].current_stage == "DOCUMENT_COLLECTION"
    assert not state["procedure"].stage_history
    assert len(milestone_checklist(state)) == 7  # Legacy checklist remains unchanged.
    missing = {r["document_type"] for r in milestone_checklist(state) if r["status"] == "missing"}
    assert missing == {"ikamet_application_form", "card_fee_receipt", "declaration_control_document"}
    assert all(getattr(state["residence_permit_context"], f) is None for f in (
        "application_form_generated", "fee_receipt_present", "gi_number_present",
        "iro_preverification_confirmed", "submitted_to_iro"))


def test_explicit_transition_only_updates_procedure_and_preserves_audit(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state["procedure"].stage_blockers = ["R008", "missing:biometric_photo", "external:manual_review"]
    before = deepcopy(state)
    update = advance_procedure_if_ready(state, recorded_at=RECORDED_AT)
    assert state == before and set(update) == {"procedure"}
    procedure = update["procedure"]
    assert procedure.current_stage == "EIKAMET_PREPARATION"
    assert procedure.stage_status == "NOT_STARTED"
    assert procedure.completed_stages == ["DOCUMENT_COLLECTION"]
    assert procedure.stage_blockers == ["external:manual_review"]
    entry = procedure.stage_history[-1]
    assert entry.stage == "EIKAMET_PREPARATION" and entry.recorded_at == RECORDED_AT
    assert entry.trace_id == state["trace_id"] and "R017" in entry.reason_refs
    assert "completed:DOCUMENT_COLLECTION" in entry.reason_refs
    assert not advance_procedure_if_ready({**state, **update}, recorded_at=RECORDED_AT)


def test_cached_ready_cannot_force_transition_when_evidence_is_removed(ready_bundle):
    state = drop_document(ready_bundle[2], "passport_copy")
    assert state["procedure"].stage_status == "READY"
    update = advance_procedure_if_ready(state, recorded_at=RECORDED_AT)
    assert update["procedure"].current_stage == "DOCUMENT_COLLECTION"
    assert update["procedure"].stage_status == "BLOCKED"
    assert not update["procedure"].stage_history


def test_graph_factual_updates_refresh_status_without_advancing(ready_bundle):
    graph, config, state = ready_bundle
    state = confirm(graph, config, "__declarations__", {"biometric_photo": False})
    assert state["procedure"].stage_status == "BLOCKED"
    state = confirm(graph, config, "__declarations__", {"biometric_photo": True})
    assert state["procedure"].stage_status == "READY"
    assert state["procedure"].current_stage == "DOCUMENT_COLLECTION"
    passport = next(d for d in state["documents"] if d.doc_type == "passport_copy")
    with patch("src.agents.planner.create_action_plan") as planner:
        state = graph.invoke(new_run(UserEvent(kind="delete_document", document_id=passport.doc_id,
                                               confirmed=True), "test-user", scope=MILESTONE_SCOPE), config)
    planner.assert_not_called()
    assert state["procedure"].stage_status == "BLOCKED"
    assert "missing:passport_identity_evidence" in state["procedure"].stage_blockers
    assert not state["procedure"].stage_history


def test_plan_approval_alone_does_not_advance(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state.update(approval_decision=True, validation_errors=[], candidate_plan=[], plan_summary="Reviewed plan")
    update = state_update(state)
    assert update["run_status"] == "APPROVED" and "procedure" not in update
    assert state["procedure"].current_stage == "DOCUMENT_COLLECTION"


@pytest.mark.parametrize("procedure", [None, ProcedureState(current_stage="PAYMENT")])
def test_legacy_and_later_stages_are_untouched(ready_bundle, procedure):
    state = {**ready_bundle[2], "procedure": procedure}
    assert document_collection_update(state) == {}
    assert advance_procedure_if_ready(state, recorded_at=RECORDED_AT) == {}


def test_missing_procedure_is_not_implicitly_created():
    assert document_collection_update({}) == {}
    assert advance_procedure_if_ready({}, recorded_at=RECORDED_AT) == {}


def test_collection_confirmations_do_not_expand_legacy_scope():
    state = new_run(UserEvent(kind="confirm_information", document_id="__declarations__",
                             confirmed_fields={"biometric_photo": True}), "test-user")
    with pytest.raises(ValueError, match="Unsupported declaration"):
        verification_updates(state)


def test_transition_requires_trace_and_explicit_recording_time(ready_bundle):
    state = {**ready_bundle[2], "trace_id": ""}
    with pytest.raises(ValueError, match="trace_id"):
        advance_procedure_if_ready(state, recorded_at=RECORDED_AT)
    with pytest.raises(ValueError, match="recorded_at"):
        advance_procedure_if_ready(ready_bundle[2], recorded_at=None)


def test_sqlite_restart_keeps_transition_and_isolates_threads(ready_bundle, tmp_path):
    state = deepcopy(ready_bundle[2])
    db = tmp_path / "procedure-transition.db"
    config = {"configurable": {"thread_id": "transitioned"}}
    other = {"configurable": {"thread_id": "other-user"}}
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        graph.update_state(config, state)
        graph.update_state(other, {"procedure": ProcedureState()})
        graph.update_state(config, advance_procedure_if_ready(graph.get_state(config).values,
                                                             recorded_at=RECORDED_AT))
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        restored = graph.get_state(config).values
        procedure = restored["procedure"]
        assert isinstance(procedure, ProcedureState)
        assert procedure.current_stage == "EIKAMET_PREPARATION"
        assert procedure.completed_stages == ["DOCUMENT_COLLECTION"]
        assert procedure.stage_history[-1].recorded_at == RECORDED_AT
        for key in ("documents", "verification_records", "relevant_requirements", "extracted_facts", "residence_permit_context"):
            assert restored[key] == state[key]
        assert graph.get_state(other).values["procedure"] == ProcedureState()
        assert not restored.get("approval_decision") and not graph.get_state(config).interrupts
