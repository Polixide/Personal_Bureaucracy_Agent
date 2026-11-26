"""Offline first-stage V2 readiness; reuse real grounding and graph confirmations."""
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import json
import subprocess
import sys

from config import PROJECT_ROOT
from pydantic import ValidationError
from langgraph.types import Command
from src.agents.planner import PlannerUnavailable
from src.core.state import EIkametFieldState, EIkametPreparationState
from src.core.procedure import build_eikamet_preparation, eikamet_field_value, EIKAMET_FIELD_CATALOGUE
from src.memory.checkpointer import initialize_procedure_thread
from streamlit_app import procedure_view, start_eikamet_preparation

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
    from src.evaluation.scenarios import SYNTHETIC
    stamp = next(d for d in state["documents"] if d.doc_type == "turkiye_entry_stamp")
    confirm(graph, config, stamp.doc_id, {"latest_entry_date": "2026-09-15"})
    state = analyze(graph, config, tmp_path, "dorm_certificate",
                    text=SYNTHETIC["dorm_certificate"].replace("Evaluation Student", "Alex Example"))
    certificate = next(d for d in state["documents"] if d.doc_type == "dorm_certificate")
    confirm(graph, config, certificate.doc_id, {"dorm_certificate_format": True})
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


@pytest.mark.parametrize("proof,ready", [("dorm_certificate", True), ("rental_agreement", False), ("notarized_commitment", False)])
def test_reuses_accepted_address_evidence_only_for_its_method(ready_bundle, proof, ready):
    state = override(ready_bundle[2], "__declarations__", "address_proof_type", proof)
    assert assess(state).ready is ready


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


def test_insurance_start_not_required_under_active_v2_R015(ready_bundle):
    state = deepcopy(ready_bundle[2])
    doc = next(d for d in state["documents"] if d.doc_type == "health_insurance")
    state = override(state, doc.doc_id, "coverage_start_date", None)
    state = override(state, doc.doc_id, "insurance_valid_for_ikamet", None)
    state = override(state, "__declarations__", "insurance_exempt", False)
    # R015 requires end coverage only; normal private insurance needs no positive attestation.
    # Explicit invalidity is still a separate R016 blocker.
    general = {**state, "planning_scope": None}
    assert next(r for r in evaluate_requirements(general) if r.rule_id == "R015").passed is True
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


# Interaction and field-preparation phase: offline, no UI/browser automation.


def test_new_thread_initializes_once_and_refresh_does_not_write():
    graph = build_graph()
    config = {"configurable": {"thread_id": "brand-new"}}
    assert graph.get_state(config).created_at is None
    first = initialize_procedure_thread(graph, config)
    assert first.values["procedure"] == ProcedureState()
    assert not first.next and not first.interrupts
    second = initialize_procedure_thread(graph, config)
    assert second.config == first.config and second.values == first.values
    assert not second.values["procedure"].stage_history


@pytest.mark.parametrize("seed", [{"user_id": "legacy"}, {"procedure": None}, {"documents": []}])
def test_legacy_checkpoint_is_never_initialized_or_inferred(seed):
    graph = build_graph()
    config = {"configurable": {"thread_id": "legacy"}}
    graph.update_state(config, seed)
    original = graph.get_state(config)
    assert original.created_at is not None
    restored = initialize_procedure_thread(graph, config)
    assert restored.config == original.config and restored.values == original.values
    assert restored.values.get("procedure") is None
    assert procedure_view(restored.values)["legacy"]


def test_existing_v2_open_preserves_saved_procedure_exactly(ready_bundle):
    graph, config, _ = ready_bundle
    assert start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
    original = graph.get_state(config)
    for _ in range(3):
        snapshot = initialize_procedure_thread(graph, config)
        before = deepcopy(snapshot.values)
        view = procedure_view(snapshot.values)
        assert view["stage"] == "EIKAMET_PREPARATION"
        assert snapshot.values == before
        assert snapshot.config == original.config
        assert snapshot.values["procedure"] == original.values["procedure"]
        assert len(snapshot.values["procedure"].stage_history) == 1


def test_procedure_display_checklist_uses_deterministic_assessment(ready_bundle):
    state = ready_bundle[2]
    before = deepcopy(state)
    view = procedure_view(state)
    assert view["can_start"] and len(view["items"]) == 6
    assert {item["Status"] for item in view["items"]} == {"Ready / verified"}
    assert not view["blockers"] and state == before
    missing = procedure_view({"procedure": ProcedureState()})
    assert not missing["can_start"]
    assert {item["Status"] for item in missing["items"]} == {"Missing"}


def test_procedure_display_confirmation_and_conflict_statuses(ready_bundle):
    state = deepcopy(ready_bundle[2])
    stamp = next(d for d in state["documents"] if d.doc_type == "turkiye_entry_stamp")
    state["verification_records"][stamp.doc_id].pop("latest_entry_date")
    view = procedure_view(state)
    assert next(item for item in view["items"] if item["Prerequisite"].startswith("Latest"))["Status"] == "Needs confirmation"
    state["verification_conflicts"] = ["holder_name"]
    view = procedure_view(state)
    assert not view["can_start"]
    assert next(item for item in view["items"] if item["Prerequisite"].startswith("Passport"))["Status"] == "Conflict / blocked"


def test_explicit_ui_action_rechecks_latest_evidence_and_cannot_use_cached_ready(ready_bundle):
    graph, config, state = ready_bundle
    assert procedure_view(state)["can_start"]
    changed = override(state, "__declarations__", "biometric_photo", False)
    graph.update_state(config, changed)
    assert not start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
    procedure = graph.get_state(config).values["procedure"]
    assert procedure.current_stage == "DOCUMENT_COLLECTION" and not procedure.stage_history
    assert "missing:biometric_photo" in procedure.stage_blockers


def test_ui_transition_is_explicit_idempotent_and_preserves_provenance(ready_bundle):
    graph, config, _ = ready_bundle
    before = deepcopy(graph.get_state(config).values)
    assert start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
    after = graph.get_state(config).values
    assert {k: v for k, v in after.items() if k != "procedure"} == {k: v for k, v in before.items() if k != "procedure"}
    assert after["procedure"].current_stage == "EIKAMET_PREPARATION"
    assert after["procedure"].stage_status == "NOT_STARTED"
    assert after["procedure"].eikamet_preparation.prepared_field_count == 7
    assert not start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
    assert graph.get_state(config).values == after


def test_pending_approval_blocks_transition_and_approval_never_advances(ready_bundle):
    graph, config, _ = ready_bundle
    from src.core.state import PlanStep
    # Restore an already-validated legacy plan while collection is ready.
    # Generate no longer invents filler, but existing pending approvals still
    # must block an otherwise-ready transition and cannot advance it on approval.
    graph.update_state(config,{'candidate_plan':[PlanStep(step_id='saved-review',
        title='Review your existing student certificate',action_type='PROVIDE_GUIDANCE',
        reason='An existing plan awaits your decision',related_rule_id='R014',priority=1)],
        'validation_errors':[],'run_status':'VALIDATED'},as_node='gate_1')
    state=graph.invoke(None,config)
    assert state["__interrupt__"] and assess_document_collection(state).ready
    before = deepcopy(graph.get_state(config).values)
    assert not start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
    assert graph.get_state(config).values == before
    state = graph.invoke(Command(resume=True), config)
    assert state["run_status"] == "APPROVED"
    assert state["procedure"].current_stage == "DOCUMENT_COLLECTION"
    assert state["procedure"].eikamet_preparation is None


def test_new_thread_upload_does_not_advance_or_prepare_form(tmp_path):
    graph = build_graph()
    config = {"configurable": {"thread_id": "new-upload"}}
    initialize_procedure_thread(graph, config)
    state = analyze(graph, config, tmp_path, "passport_identity_page")
    assert state["procedure"].current_stage == "DOCUMENT_COLLECTION"
    assert not state["procedure"].stage_history
    assert state["procedure"].eikamet_preparation is None


@pytest.mark.parametrize("field_id,expected", [
    ("passport_number", "P1234567"), ("passport_expiry_date", "2030-08-30"),
    ("nationality", "Exampleland"),
])
def test_preparation_maps_grounded_passport_fields(ready_bundle, field_id, expected):
    state = ready_bundle[2]
    before = deepcopy(state)
    field = build_eikamet_preparation(state).fields[field_id]
    assert field.status == "VERIFIED" and field.source_type == "DOCUMENT"
    assert field.source_ref.startswith("/verification_records/")
    assert eikamet_field_value(state, field) == expected
    assert state == before


def test_preparation_reuses_the_verified_passport_name_parts_and_birth_date():
    from src.core.procedure import eikamet_preparation_inputs
    from src.core.state import Document
    from src.evaluation.scenarios import extract_synthetic
    from test_document_layouts_v2 import _checked_passport_mrz

    top, bottom = _checked_passport_mrz("ITA")
    doc = Document(doc_id="passport", doc_type="passport_copy", file_path="synthetic-passport.txt", status="analyzed",
                   info=extract_synthetic("passport_identity_page", text=top + "\n" + bottom))
    state = {**new_run(UserEvent(kind="reevaluate"), "synthetic", scope=MILESTONE_SCOPE),
             "procedure": ProcedureState(), "documents": [doc]}
    state.update(verification_updates(state))
    preparation, values = eikamet_preparation_inputs(state)
    assert {key: values[key] for key in ("first_name", "surname", "birth_date", "nationality", "passport_number")} == {
        "first_name": "TAYLOR", "surname": "EXAMPLE", "birth_date": "2000-01-01", "nationality": "ITA",
        "passport_number": "P1234567",
    }
    assert preparation.fields["first_name"].source_ref == "/verification_records/passport/first_name"
    assert preparation.fields["first_name"].status == "VERIFIED"
    assert "passport_issue_date" not in values


@pytest.mark.parametrize("signed", [True, False])
def test_address_prefill_requires_signed_source_or_explicit_confirmation(signed):
    from src.core.procedure import eikamet_preparation_inputs
    from src.core.state import Document
    from src.evaluation.scenarios import SYNTHETIC, extract_synthetic

    address = SYNTHETIC["address_form"] if signed else SYNTHETIC["address_form"].replace("yes", "no")
    documents = [
        Document(doc_id=kind, doc_type=kind, file_path="synthetic.txt", status="analyzed",
                 info=extract_synthetic(kind, text=address if kind == "address_form" else None))
        for kind in ("address_form", "utility_bill")
    ]
    state = {**new_run(UserEvent(kind="reevaluate"), "synthetic", scope=MILESTONE_SCOPE),
             "procedure": ProcedureState(current_stage="EIKAMET_PREPARATION"), "documents": documents}
    state.update(verification_updates(state))
    preparation, values = eikamet_preparation_inputs(state)
    assert values.get("turkiye_address") == ("Example Street 12" if signed else None)
    assert "address_province" not in values
    assert "address_district" not in values and "address_locality" not in values
    if signed:
        assert preparation.fields["turkiye_address"].source_ref == "/verification_records/address_form/declared_address"
        state = override(state, "__declarations__", "address_province", "Istanbul")
        state = override(state, "__declarations__", "residential_address", "Other Synthetic Street 10")
        preparation, values = eikamet_preparation_inputs(state)
        assert preparation.fields["turkiye_address"].status == preparation.fields["address_province"].status == "CONFLICT"
        assert "turkiye_address" not in values and "address_province" not in values
    else:
        state = override(state, "address_form", "declared_address", "Example Street 12")
        preparation, values = eikamet_preparation_inputs(state)
        assert values["turkiye_address"] == "Example Street 12"
        assert preparation.fields["turkiye_address"].status == "USER_CONFIRMED"


def test_preparation_maps_user_confirmation_without_copying_observations(ready_bundle):
    state = ready_bundle[2]
    preparation = build_eikamet_preparation(state)
    field = preparation.fields["latest_entry_date"]
    assert field.status == "USER_CONFIRMED" and field.source_type == "USER"
    assert eikamet_field_value(state, field) == "2026-09-15"
    assert "value" not in field.model_dump()
    assert "source_snippet" not in field.model_dump()
    assert "prepared_field_count" not in preparation.model_dump()
    assert "missing_field_ids" not in preparation.model_dump()
    assert "conflict_field_ids" not in preparation.model_dump()


@pytest.mark.parametrize("field_id", ["first_name", "surname", "birth_date", "gender", "passport_type",
    "passport_issue_date", "passport_issuing_country", "passport_issuing_authority", "email",
    "turkish_phone", "turkiye_address", "reason_of_stay", "requested_permit_start"])
def test_unsupported_or_absent_fields_remain_missing_and_uninvented(ready_bundle, field_id):
    state = deepcopy(ready_bundle[2])
    # Presence flags are not contact values; academic dates are not requested dates.
    state["residence_permit_context"].phone_present = True
    state["residence_permit_context"].email_present = True
    state["residence_permit_context"].turkey_address_present = True
    preparation = build_eikamet_preparation(state)
    field = preparation.fields[field_id]
    assert field.status == "MISSING" and field.source_ref is field.source_type is None
    assert eikamet_field_value(state, field) is None


def test_requested_start_projects_only_explicitly_accepted_requested_date(ready_bundle, tmp_path):
    graph, config, _ = ready_bundle
    state = analyze(graph, config, tmp_path, "ikamet_application_form")
    field = build_eikamet_preparation(state).fields["requested_permit_start"]
    assert field.status == "VERIFIED"
    assert eikamet_field_value(state, field) == "2026-09-15"


def test_preparation_detects_conflicting_accepted_passports_without_cached_conflict_list(ready_bundle):
    state = deepcopy(ready_bundle[2])
    passport = next(d for d in state["documents"] if d.doc_type == "passport_copy")
    other = passport.model_copy(deep=True, update={"doc_id": "other-passport"})
    state["documents"].append(other)
    state = override(state, other.doc_id, "passport_number", "P7654321")
    state["verification_conflicts"] = []
    preparation = build_eikamet_preparation(state)
    field = preparation.fields["passport_number"]
    assert field.status == "CONFLICT" and field.source_ref is None
    assert "passport_number" in preparation.conflict_field_ids
    assert eikamet_field_value(state, field) is None


def test_bare_context_and_unreviewed_source_cannot_populate_preparation(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state["verification_records"] = {}
    for doc in state["documents"]:
        doc.info.source_evidence = []
    preparation = build_eikamet_preparation(state)
    assert preparation.prepared_field_count == 0
    assert len(preparation.missing_field_ids) == len(EIKAMET_FIELD_CATALOGUE)


def test_old_preparation_reference_does_not_expose_removed_evidence(ready_bundle):
    state = ready_bundle[2]
    field = build_eikamet_preparation(state).fields["passport_number"]
    assert eikamet_field_value(drop_document(state, "passport_copy"), field) is None


def test_factual_events_refresh_preparation_without_later_transition(ready_bundle):
    graph, config, _ = ready_bundle
    assert start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
    before = deepcopy(graph.get_state(config).values["procedure"])
    passport = next(d for d in graph.get_state(config).values["documents"] if d.doc_type == "passport_copy")
    state = confirm(graph, config, passport.doc_id, {"expiry_date": "2031-01-01"})
    field = state["procedure"].eikamet_preparation.fields["passport_expiry_date"]
    assert field.status == "USER_CONFIRMED"
    assert eikamet_field_value(state, field) == "2031-01-01"
    assert state["procedure"].current_stage == before.current_stage == "EIKAMET_PREPARATION"
    assert state["procedure"].stage_status == before.stage_status
    assert state["procedure"].stage_history == before.stage_history
    state = graph.invoke(new_run(UserEvent(kind="delete_document", document_id=passport.doc_id, confirmed=True),
                                 "test-user", scope=MILESTONE_SCOPE), config)
    assert state["procedure"].eikamet_preparation.fields["passport_number"].status == "MISSING"
    assert state["procedure"].current_stage == "EIKAMET_PREPARATION"
    assert state["procedure"].stage_history == before.stage_history


@pytest.mark.parametrize("payload", [
    {"field_id": "passport_number", "status": "APPROVED"},
    {"field_id": "passport_number", "status": "VERIFIED"},
    {"field_id": "passport_number", "status": "MISSING", "source_ref": "/verification_records/x/y"},
    {"field_id": "passport_number", "value": "invented"},
    {"field_id": "passport_number", "source_snippet": "copied"},
])
def test_preparation_contract_rejects_unsupported_status_and_duplicated_fact_data(payload):
    with pytest.raises(ValidationError):
        EIkametFieldState.model_validate(payload)


def test_preparation_contract_uses_independent_defaults_and_derived_counts():
    first, second = EIkametPreparationState(), EIkametPreparationState()
    first.fields["email"] = EIkametFieldState(field_id="email")
    assert not second.fields and first.missing_field_ids == ["email"]
    with pytest.raises(ValidationError, match="field_id"):
        EIkametPreparationState(fields={"phone": EIkametFieldState(field_id="email")})


def test_sqlite_preparation_survives_restart_and_thread_open_without_rewriting(ready_bundle, tmp_path):
    state = deepcopy(ready_bundle[2])
    db = tmp_path / "eikamet-preparation.db"
    config = {"configurable": {"thread_id": "preparing"}}
    other = {"configurable": {"thread_id": "fresh-other"}}
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        graph.update_state(config, state, as_node="respond")
        initialize_procedure_thread(graph, other)
        assert start_eikamet_preparation(graph, config, recorded_at=RECORDED_AT)
        before = graph.get_state(config)
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        restored = initialize_procedure_thread(graph, config)
        assert restored.config == before.config and restored.values == before.values
        preparation = restored.values["procedure"].eikamet_preparation
        assert isinstance(preparation, EIkametPreparationState)
        assert isinstance(preparation.fields["passport_number"], EIkametFieldState)
        assert preparation.prepared_field_count == 7
        assert eikamet_field_value(restored.values, preparation.fields["passport_number"]) == "P1234567"
        assert initialize_procedure_thread(graph, other).values["procedure"] == ProcedureState()
        assert not restored.next and not restored.interrupts
    code = """
import json, sys
from src.core.graph import build_graph
from src.core.state import EIkametFieldState, EIkametPreparationState
from src.core.procedure import eikamet_field_value
from src.memory.checkpointer import sqlite_checkpointer, initialize_procedure_thread
with sqlite_checkpointer(sys.argv[1]) as saver:
    snapshot = initialize_procedure_thread(build_graph(saver), {'configurable': {'thread_id': 'preparing'}})
    p = snapshot.values['procedure'].eikamet_preparation
    print(json.dumps({'prepared': p.prepared_field_count, 'typed': isinstance(p, EIkametPreparationState),
        'field_typed': isinstance(p.fields['passport_number'], EIkametFieldState),
        'value': eikamet_field_value(snapshot.values, p.fields['passport_number']),
        'history_count': len(snapshot.values['procedure'].stage_history), 'pending': list(snapshot.next)}))
"""
    process = subprocess.run([sys.executable, "-c", code, str(db)], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(process.stdout) == {"prepared": 7, "typed": True, "field_typed": True,
                                        "value": "P1234567", "history_count": 1, "pending": []}


# V2 user experience: declarations are intent, never completed-stage evidence.
from streamlit_app import (needs_onboarding, save_entry_intent, user_case_view, technical_case_view,
    document_information_routes, user_text)
from src.core.state import ProcedureEntryIntent


@pytest.mark.parametrize("mode", ["START_FROM_SCRATCH","HAVE_DOCUMENTS","EIKAMET_COMPLETED","RESUME"])
def test_onboarding_records_intent_without_advancing_or_claiming_official_actions(mode):
    graph=build_graph();config={"configurable":{"thread_id":"onboarding"}}
    initial=initialize_procedure_thread(graph,config)
    assert needs_onboarding(initial.values)
    assert save_entry_intent(graph,config,mode,recorded_at=RECORDED_AT)
    state=graph.get_state(config).values
    assert isinstance(state["procedure_entry_intent"],ProcedureEntryIntent)
    assert state["procedure_entry_intent"].mode==mode
    assert not needs_onboarding(state) and state["procedure"]==ProcedureState()
    assert "online_application_completed" not in state
    assert not graph.get_state(config).next


def test_already_have_documents_still_requires_evidence(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"have-documents"}}
    initialize_procedure_thread(graph,config)
    save_entry_intent(graph,config,"HAVE_DOCUMENTS",recorded_at=RECORDED_AT)
    state=analyze(graph,config,tmp_path,"passport_identity_page")
    view=user_case_view(state)
    assert view["stage"]=="DOCUMENT_COLLECTION" and not view["can_start"]
    assert not state["procedure"].completed_stages


def test_claimed_completed_eikamet_and_uploaded_form_do_not_complete_later_stages(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"claimed-eikamet"}}
    initialize_procedure_thread(graph,config)
    save_entry_intent(graph,config,"EIKAMET_COMPLETED",recorded_at=RECORDED_AT)
    state=analyze(graph,config,tmp_path,"ikamet_application_form")
    assert state["residence_permit_context"].application_form_generated is True
    assert state["residence_permit_context"].online_application_completed is None
    assert state["procedure"].current_stage=="DOCUMENT_COLLECTION" and not state["procedure"].completed_stages
    for field in ("fee_receipt_present","fingerprint_step_completed","submitted_to_iro"):
        assert getattr(state["residence_permit_context"],field) is None


def test_user_mode_is_stage_specific_after_explicit_transition(ready_bundle):
    graph,config,state=ready_bundle
    before=user_case_view(state)
    assert before["title"]=="Collect and review your documents" and len(before["items"])==6
    assert start_eikamet_preparation(graph,config,recorded_at=RECORDED_AT)
    state=graph.get_state(config).values
    after=user_case_view(state)
    assert after["title"]=="Prepare your e-Ikamet application"
    assert after["current"]=="e-Ikamet preparation" and not after["items"]
    assert after["counts"]["Prepared fields"]==7 and after["field_groups"]["Need information from you"]
    assert next(item for item in after["progress"] if item["Stage"]=="Documents")["Status"]=="Complete"
    assert next(item for item in after["progress"] if item["Stage"]=="e-Ikamet preparation")["Status"]=="Current"


def test_user_mode_blockers_are_friendly_and_technical_mode_keeps_raw_data(ready_bundle):
    state=drop_document(ready_bundle[2],"passport_copy")
    before=deepcopy(state)
    normal=user_case_view(state)
    technical=technical_case_view(state)
    assert "Provide passport identity evidence." in normal["blockers"]
    assert "R008" not in json.dumps(normal) and "missing:" not in json.dumps(normal)
    assert "R008" in json.dumps(technical) and "source_evidence" in json.dumps(technical)
    technical["blockers"].append("display-only")
    assert state==before


def test_user_correction_route_is_derived_without_changing_extraction_metadata(ready_bundle):
    state=ready_bundle[2]
    passport=next(d for d in state["documents"] if d.doc_type=="passport_copy")
    raw=deepcopy(passport.info)
    changed=override(state,passport.doc_id,"holder_name","Alex Reviewed")
    assert document_information_routes(changed,passport)==["NATIVE_TEXT","USER_CORRECTION"]
    assert passport.info==raw


def test_onboarding_and_modes_do_not_migrate_legacy_cases():
    graph=build_graph();config={"configurable":{"thread_id":"legacy-ui"}}
    graph.update_state(config,{"user_id":"legacy-user"},as_node="respond")
    before=graph.get_state(config)
    snapshot=initialize_procedure_thread(graph,config)
    assert snapshot.config==before.config and snapshot.values==before.values
    assert not needs_onboarding(snapshot.values) and user_case_view(snapshot.values)["legacy"]


def test_entry_intent_survives_sqlite_restart_and_is_isolated(tmp_path):
    db=tmp_path/"onboarding.db";config={"configurable":{"thread_id":"first"}};other={"configurable":{"thread_id":"second"}}
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        initialize_procedure_thread(graph,config);initialize_procedure_thread(graph,other)
        save_entry_intent(graph,config,"EIKAMET_COMPLETED",recorded_at=RECORDED_AT)
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver);state=initialize_procedure_thread(graph,config).values
        assert isinstance(state["procedure_entry_intent"],ProcedureEntryIntent)
        assert state["procedure_entry_intent"].mode=="EIKAMET_COMPLETED"
        assert state["procedure"]==ProcedureState()
        assert graph.get_state(other).values.get("procedure_entry_intent") is None


def test_user_plan_prose_hides_rule_annotations_without_changing_plan():
    text="Upload your passport to address R006: missing expiry information"
    assert "R006" not in user_text(text)
    assert text=="Upload your passport to address R006: missing expiry information"



def test_guided_transition_ignores_stale_optional_plan_failure_but_preserves_facts(ready_bundle):
    from src.core.procedure import start_guided_preparation
    state=deepcopy(ready_bundle[2]);state['run_status']='HUMAN_REVIEW_REQUIRED'
    state['validation_errors']=['old optional plan error'];before=deepcopy(state)
    updates=start_guided_preparation(state,recorded_at=RECORDED_AT)
    assert updates['procedure'].current_stage=='EIKAMET_PREPARATION' and updates['validation_errors']==[]
    assert not ({'documents','verification_records','residence_permit_context','relevant_requirements'} & set(updates))
    assert state==before


def test_collection_confirmations_survive_stage_change(ready_bundle):
    from src.core.procedure import start_guided_preparation,_accepted_snapshot
    state=deepcopy(ready_bundle[2]);state.update(start_guided_preparation(state,recorded_at=RECORDED_AT))
    accepted=_accepted_snapshot(state)['residence_permit_context']
    assert accepted.address_proof_type=='dorm_certificate' and accepted.biometric_photo is True
    assert assess_document_collection(state).ready


def test_actual_form_path_requires_real_completion_confirmation_not_intent(ready_bundle,tmp_path):
    from src.core.procedure import application_form_review,continue_after_application_review,choose_eikamet_path
    graph,config,state=ready_bundle
    graph.update_state(config,choose_eikamet_path(state,'UPLOADED_FORM'))
    state=analyze(graph,config,tmp_path,'ikamet_application_form')
    review=application_form_review(state)
    assert not review['ready'] and 'online_application_completed' in review['confirmations']
    assert continue_after_application_review(state,recorded_at=RECORDED_AT)=={}
    doc=next(d for d in state['documents'] if d.doc_type=='ikamet_application_form')
    state=confirm(graph,config,doc.doc_id,{'online_application_completed':True,'application_form_signed':True})
    review=application_form_review(state)
    assert review['ready'] and 'passport_or_foreign_id' in review['matched']
    updates=continue_after_application_review(state,recorded_at=RECORDED_AT)
    assert updates['procedure'].current_stage=='PAYMENT' and updates['procedure'].reviewed_application_document_id==doc.doc_id
    assert 'sandbox_form' not in updates


def test_already_completed_reconstruction_does_not_falsely_complete_initial_collection(tmp_path):
    from src.core.procedure import application_form_review,continue_after_application_review,entry_intent_update
    graph=build_graph();config={'configurable':{'thread_id':'existing-form'}}
    graph.update_state(config,{'procedure':ProcedureState()})
    state=analyze(graph,config,tmp_path,'ikamet_application_form')
    graph.update_state(config,entry_intent_update(state,'EIKAMET_COMPLETED',user_id='existing-user',trace_id=state['trace_id'],recorded_at=RECORDED_AT))
    doc=next(d for d in state['documents'] if d.doc_type=='ikamet_application_form')
    state=confirm(graph,config,doc.doc_id,{'online_application_completed':True,'application_form_signed':True})
    assert application_form_review(state)['ready'] and not assess_document_collection(state).ready
    updates=continue_after_application_review(state,recorded_at=RECORDED_AT)
    assert updates['procedure'].current_stage=='PAYMENT'
    assert 'DOCUMENT_COLLECTION' not in updates['procedure'].completed_stages


@pytest.mark.parametrize('answers',[{'requested_start_date':'2028-01-01'}, {'birth_date':'2099-01-01'},
    {'email':'not an email'}, {'first_name':'Different','last_name':'Person'}])
def test_preparation_answers_reject_invalid_dates_contact_or_identity_before_state_write(ready_bundle,answers):
    from src.core.procedure import checked_form_answers
    state=ready_bundle[2];before=deepcopy(state)
    with pytest.raises(ValueError):checked_form_answers(state,answers)
    assert state==before


def test_stage_scoped_planning_keeps_known_earlier_failures_but_excludes_later_work(ready_bundle):
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import selected_needs
    state=deepcopy(ready_bundle[2])
    state['relevant_requirements']=evaluate_requirements(state)
    needs=actionable_needs(state)
    selected=selected_needs(needs)
    assert all(n['category']=='immediate' for n in selected)
    assert not any(n['rule_id'] in {'R007','R019','R021','R022','R023'} for n in selected)
    # A known expired legal stay remains actionable even when collecting later payment evidence.
    state['procedure']=state['procedure'].model_copy(update={'current_stage':'PAYMENT'})
    state=override(state,'__declarations__','legal_stay_expiry_date','2026-01-01')
    state=override(state,'__declarations__','application_date','2026-11-20')
    state['relevant_requirements']=evaluate_requirements(state)
    assert any(n['rule_id']=='R004' for n in selected_needs(actionable_needs(state)))



def test_composite_address_proof_explains_both_parts_and_independent_routes(tmp_path):
    from src.evaluation.scenarios import copilot_benchmark_cases,extract_synthetic,confirm
    from src.core.procedure import address_proof_status
    from src.core.state import Document
    state=copilot_benchmark_cases(tmp_path/'address')[2]['state']
    address=address_proof_status(state)
    assert not address['ready'] and [p['status'] for p in address['parts']]==['complete','missing']
    state['documents'].append(Document(doc_id='utility',file_path='synthetic.txt',doc_type='utility_bill',status='analyzed',info=extract_synthetic('utility_bill')))
    state.update(verification_updates(state));assert address_proof_status(state)['ready']
    state=confirm(state,'__declarations__',{'address_proof_type':'dorm_certificate'})
    state['documents']=[d for d in state['documents'] if d.doc_type not in {'utility_bill','address_form'}]
    assert address_proof_status(state)['ready'] and address_proof_status(state)['route']=='dorm_certificate'


def test_package_readiness_is_shared_with_email_and_planner(tmp_path):
    from src.evaluation.scenarios import copilot_benchmark_cases
    from src.core.procedure import iro_package_readiness,stage_document_checklist
    from src.agents.planner import canonical_candidates
    from src.tools.registry import execute_tool
    from src.tools.schemas import ToolCall
    cases=copilot_benchmark_cases(tmp_path/'package')
    state=cases[5]['state'];package=iro_package_readiness(state)
    assert package['ready_count']==7 and package['total']==9 and not package['ready']
    assert stage_document_checklist(state)==package['items']
    candidates=canonical_candidates(state)
    assert candidates and not any(c['target_document'] in {'passport_identity_page','biometric_photo'} for c in candidates)
    assert not any('passport' in c['fields'] or 'biometric_photo' in c['fields'] for c in candidates)
    result=execute_tool(state,ToolCall(name='prepare_iro_email',arguments={'purpose':'package_review'}),approved=True)
    assert result.status=='FAILED' and not result.state_updates
    complete=cases[9]['state'];package=iro_package_readiness(complete)
    assert package['ready'] and package['ready_count']==9
    preview=execute_tool(complete,ToolCall(name='prepare_iro_email',arguments={'purpose':'package_review','attachment_ids':package['attachment_ids']}))
    assert preview.status=='APPROVAL_REQUIRED'


def test_normal_payment_does_not_guess_extra_visa_fee_applicability(tmp_path):
    from src.evaluation.scenarios import copilot_benchmark_cases
    from src.core.procedure import payment_readiness
    state=copilot_benchmark_cases(tmp_path/'payment')[-1]['state']
    result=payment_readiness(state)
    assert result['ready'] and result['receipt_ready'] and not result['additional_fee_required']
    assert next(row for row in evaluate_requirements(state) if row.rule_id=='R011').status=='missing_information'
    assert state['residence_permit_context'].single_entry_visa_fee_required is None


def test_fingerprint_cannot_complete_without_reviewed_real_application(ready_bundle):
    from src.core.procedure import assess_current_stage,advance_current_stage
    state=override(ready_bundle[2],'__declarations__','fingerprint_step_completed',True)
    state=override(state,'__declarations__','gi_number_present',True)
    state['procedure']=state['procedure'].model_copy(update={'current_stage':'FINGERPRINT'})
    assert 'evidence:real_application_form' in assess_current_stage(state)['blockers']
    assert advance_current_stage(state,recorded_at=RECORDED_AT)=={}



@pytest.mark.parametrize('reason',['Tourism','Business trip'])
def test_educational_form_cannot_change_supported_exchange_reason(ready_bundle,reason):
    from src.core.procedure import checked_form_answers
    with pytest.raises(ValueError,match='student-exchange'):checked_form_answers(ready_bundle[2],{'reason_of_stay':reason})
