"""Synthetic end-to-end document evidence and cross-document consistency checks."""
from copy import deepcopy

import pytest

from src.agents.planner import compact_planner_context
from src.core.procedure import (
    address_proof_status, assess_document_collection, build_eikamet_preparation,
    iro_package_readiness, stage_document_checklist,
)
from src.core.state import Document, UserEvent, new_run
from src.evaluation.scenarios import copilot_benchmark_cases, extract_synthetic
from src.tools.registry import execute_tool
from src.tools.schemas import ToolCall
from src.workers.verification import (
    MILESTONE_SCOPE, accepted_sources, conflict_details, document_review, verification_updates,
)
from test_procedure import override, ready_bundle

PASSPORT = "P1234567"
OTHER_PASSPORT = "P7654321"
PASSPORT_FIELD = "passport_or_foreign_id"
COMBINED_ROUTE = "utility_bill_and_address_form"


def add_evidence(state, kind, doc_type=None, *, doc_id=None, text=None, status="analyzed"):
    document = Document(doc_id=doc_id or kind, file_path=f"synthetic-{doc_id or kind}.txt",
                        doc_type=doc_type or kind, status=status, info=extract_synthetic(kind, text=text))
    state["documents"] = [*state.get("documents", []), document]
    return document


def address_example(state, number=OTHER_PASSPORT):
    return add_evidence(
        state, "address_form",
        text=(
            f"Address Form\nFirst name: Alex\nSurname: Example\nPassport number: {number}\n"
            "Nationality: Exampleland\nDeclared address: Synthetic Street 10\n"
            "Phone: 05551234567\nForm date: 2026-11-20\nSignature present: yes"
        ),
    )


def recheck(state):
    state.update(new_run(UserEvent(kind="reevaluate"), "test-user", scope=MILESTONE_SCOPE))
    state.update(verification_updates(state))
    return state


def test_conflicting_passport_blocks_address_proof_until_explicit_correction(ready_bundle):
    state = deepcopy(ready_bundle[2])
    address = address_example(state)
    add_evidence(state, "utility_bill")
    state = override(state, "__declarations__", "address_proof_type", COMBINED_ROUTE)
    details = conflict_details(state)[PASSPORT_FIELD]
    original_passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    assert not details["resolved"]
    assert {(row["document_id"], row["value"], row["origin"]) for row in details["observations"]} == {
        (original_passport.doc_id, PASSPORT, "source_grounded"),
        (address.doc_id, OTHER_PASSPORT, "source_grounded"),
    }
    assert state["residence_permit_context"].passport_or_foreign_id is None
    assert "conflict:" + PASSPORT_FIELD in assess_document_collection(state).blockers
    assert [part["status"] for part in address_proof_status(state)["parts"]] == ["conflict", "complete"]
    prepared = build_eikamet_preparation(state).fields["passport_number"]
    assert prepared.status == "CONFLICT" and prepared.source_ref is None

    punctuation_differs = override(state, address.doc_id, "passport_number", "P123-4567")
    assert PASSPORT_FIELD in punctuation_differs["verification_conflicts"]

    corrected = override(state, address.doc_id, "passport_number", "p123 4567")
    assert PASSPORT_FIELD not in corrected["verification_conflicts"]
    assert assess_document_collection(corrected).ready
    assert address_proof_status(corrected)["ready"]
    row = next(row for row in document_review(corrected, address, full=True)["fields"]
               if row["name"] == "passport_number")
    assert row["value"] == "p123 4567" and row["origin"] == "user_confirmed"
    assert row["observation"] == OTHER_PASSPORT
    assert address.info.passport_number == OTHER_PASSPORT
    assert {row["value"] for row in accepted_sources(corrected)[PASSPORT_FIELD]} == {PASSPORT, "p123 4567"}


def test_combined_address_checklist_counts_one_requirement_and_only_current_parts(ready_bundle):
    state = deepcopy(ready_bundle[2])
    address = address_example(state, PASSPORT)
    state = override(state, "__declarations__", "address_proof_type", COMBINED_ROUTE)
    rows = stage_document_checklist(state)
    row = next(row for row in rows if row["key"] == "address_proof")
    assert len(rows) == 6 and row["status"] != "complete"
    assert [(part["document_type"], part["status"]) for part in row["parts"]] == [
        ("address_form", "complete"), ("utility_bill", "missing"),
    ]
    assert not assess_document_collection(state).ready

    add_evidence(state, "utility_bill")
    recheck(state)
    assert next(row for row in stage_document_checklist(state) if row["key"] == "address_proof")["status"] == "complete"
    assert assess_document_collection(state).ready

    state = override(state, "__declarations__", "address_proof_type", "dorm_certificate")
    row = next(row for row in stage_document_checklist(state) if row["key"] == "address_proof")
    assert [part["document_type"] for part in row["parts"]] == ["dorm_certificate"]
    assert document_review(state, address, full=True)["status"] == "complete"


@pytest.mark.parametrize(("label", "value", "target", "mismatch"), [
    ("Foreigner ID", "99000000001", "foreigner_id", False),
    ("Passport number", OTHER_PASSPORT, PASSPORT_FIELD, True),
    ("Passport / YKN", "99000000001", None, False),
])
def test_combined_identifier_requires_a_distinct_semantic_source_label(ready_bundle, label, value, target, mismatch):
    state = deepcopy(ready_bundle[2])
    form = add_evidence(
        state, "ikamet_application_form",
        text=(
            "Declaration for Short Term Residence Permit Registration Form\n"
            f"Applicant: Alex Example\n{label}: {value}\n"
            "Requested start date: 2026-09-15\nRequested end date: 2027-06-25"
        ),
    )
    state = override(state, form.doc_id, "passport_or_foreign_id", value)
    rows = [row for rows in accepted_sources(state).values() for row in rows
            if row["document_id"] == form.doc_id and row["field_name"] == "passport_or_foreign_id"]
    if target is None:
        assert rows == []
    else:
        assert [row["value"] for row in accepted_sources(state)[target] if row in rows] == [value]
    assert (PASSPORT_FIELD in state["verification_conflicts"]) is mismatch
    if not mismatch:
        assert state["residence_permit_context"].passport_or_foreign_id == PASSPORT


@pytest.mark.parametrize(("kind", "doc_type"), [
    ("passport_identity_page", "passport_copy"),
    ("turkiye_entry_stamp", "turkiye_entry_stamp"),
    ("student_enrollment_certificate", "student_certificate"),
    ("health_insurance_certificate", "health_insurance"),
    ("ikamet_application_form", "ikamet_application_form"),
    ("address_form", "address_form"),
    ("utility_bill", "utility_bill"),
    ("declaration_control_document", "declaration_control_document"),
    ("card_fee_receipt", "card_fee_receipt"),
    ("unknown", "unknown"),
])
def test_full_view_exposes_every_structured_field_of_each_saved_schema(kind, doc_type):
    state = new_run(UserEvent(kind="reevaluate"), "test-user", scope=MILESTONE_SCOPE)
    from src.evaluation.scenarios import SYNTHETIC
    source = SYNTHETIC.get(kind, {
        "declaration_control_document": "Declaration Control Document\nSigned: yes",
        "card_fee_receipt": "Residence Permit Card Fee Receipt\nPayment date: 2026-11-20\nAmount: 1234 TRY",
        "unknown": "A synthetic file without enough information to classify.",
    }.get(kind))
    document = add_evidence(state, kind, doc_type, text=source)
    recheck(state)
    rows = {row["name"]: row for row in document_review(state, document, full=True)["fields"]}
    saved = document.info.model_dump(mode="json")
    for name in document.info.observation_fields():
        assert rows[name]["observation"] == saved[name]
        if saved[name] is None and rows[name]["origin"] is None:
            assert rows[name]["value"] is None
    if doc_type != "unknown":
        assert document_review(state, document)["status"] == document_review(state, document, full=True)["status"]


def test_policy_passport_mismatch_can_be_corrected_from_the_policy_review(ready_bundle):
    state = deepcopy(ready_bundle[2])
    policy = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
    policy.info = extract_synthetic(
        "health_insurance_certificate",
        text="Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\nPolicy number: TEST-123\n"
             "Passport number: P7654321\nCoverage start: 2026-09-01\nCoverage end: 2027-07-01",
    )
    recheck(state)
    assert PASSPORT_FIELD in state["verification_conflicts"]
    assert PASSPORT_FIELD in {row["name"] for row in document_review(state, policy)["fields"]}
    corrected = override(state, policy.doc_id, PASSPORT_FIELD, PASSPORT)
    assert not corrected["verification_conflicts"]
    assert policy.info.passport_or_foreign_id == OTHER_PASSPORT


@pytest.mark.parametrize("legacy_confirmation", [False, True])
def test_reprocessing_keeps_the_confirmed_identifier_role(ready_bundle, monkeypatch, legacy_confirmation):
    graph, config, saved = ready_bundle
    state = deepcopy(saved)
    policy = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
    source = (
        "Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\n"
        "Policy number: TEST-123\n{}\nCoverage start: 2026-09-01\nCoverage end: 2027-07-01"
    )
    policy.info = extract_synthetic("health_insurance_certificate",
                                    text=source.format(f"Passport number: {OTHER_PASSPORT}"))
    state = override(state, policy.doc_id, PASSPORT_FIELD, PASSPORT)
    if legacy_confirmation:
        record = state["verification_records"][policy.doc_id][PASSPORT_FIELD]
        state["verification_records"][policy.doc_id][PASSPORT_FIELD] = record.model_copy(update={"identifier_role": None})
    graph.update_state(config, state, as_node="respond")

    reread = extract_synthetic("health_insurance_certificate", text=source.format("Foreigner ID: 99000000001"))
    monkeypatch.setattr("src.workers.document_analyzer.process_document", lambda *args, **kwargs: reread)
    event = UserEvent(kind="reprocess_document", document_id=policy.doc_id, document_type_hint="health_insurance_certificate")
    state = graph.invoke(new_run(event, "test-user", scope=MILESTONE_SCOPE), config=config)
    assert not state["verification_conflicts"]
    saved = state["verification_records"][policy.doc_id][PASSPORT_FIELD]
    assert saved.value == PASSPORT and saved.identifier_role == "passport_number"
    assert "foreigner_id" not in accepted_sources(state)
    active = next(doc for doc in state["documents"] if doc.doc_id == policy.doc_id)
    current = next(row for row in document_review(state, active, full=True)["fields"] if row["name"] == PASSPORT_FIELD)
    assert current["display_field"] == "passport_number"
    assert current["observation_field"] == "foreigner_id" and current["observation"] == "99000000001"


def test_foreign_id_review_does_not_recommend_a_passport_value(ready_bundle):
    from src.ui.documents import _passport_reference

    state = deepcopy(ready_bundle[2])
    form = add_evidence(
        state, "ikamet_application_form",
        text=(
            "Declaration for Short Term Residence Permit Registration Form\nApplicant: Alex Example\n"
            "Foreigner ID: 99000000001\nRequested start date: 2026-09-15\nRequested end date: 2027-06-25"
        ),
    )
    state = override(state, form.doc_id, PASSPORT_FIELD, None)
    assert _passport_reference(state, form, PASSPORT_FIELD) is None
    address = address_example(state)
    address.info.ambiguous_fields.append("passport_number")
    assert _passport_reference(state, address, "passport_number") == PASSPORT


def test_old_checkpoint_with_typed_foreign_id_does_not_show_a_stale_passport_conflict(ready_bundle):
    from src.ui.components import conflict_messages

    state = deepcopy(ready_bundle[2])
    add_evidence(
        state, "ikamet_application_form",
        text=(
            "Declaration for Short Term Residence Permit Registration Form\nApplicant: Alex Example\n"
            "Foreigner ID: 99000000001\nRequested start date: 2026-09-15\nRequested end date: 2027-06-25"
        ),
    )
    recheck(state)
    state["verification_conflicts"] = [PASSPORT_FIELD]
    assert not conflict_messages(state)
    assert assess_document_collection(state).ready
    assert build_eikamet_preparation(state).fields["passport_number"].status == "VERIFIED"


def test_shared_identity_fields_and_source_revocation(ready_bundle):
    state = deepcopy(ready_bundle[2])
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    address = address_example(state, PASSPORT)
    state = override(state, passport.doc_id, "birth_date", "2003-01-12")
    state = override(state, address.doc_id, "birth_date", "2003-01-13")
    assert assess_document_collection(state).prerequisites["passport_identity"] == "CONFLICT"
    state = override(state, passport.doc_id, "nationality", "ITA")
    state = override(state, address.doc_id, "nationality", "Italian")
    state = override(state, address.doc_id, "last_name", "Examples")
    assert {"birth_date", "holder_name"} <= set(state["verification_conflicts"])
    assert "nationality" not in state["verification_conflicts"]
    state = override(state, address.doc_id, "birth_date", "2003-01-12")
    state = override(state, address.doc_id, "last_name", "EXAMPLE")
    assert not state["verification_conflicts"]

    state = override(state, address.doc_id, "passport_number", OTHER_PASSPORT)
    assert PASSPORT_FIELD in state["verification_conflicts"]
    state.update(new_run(UserEvent(kind="remove_correction", document_id=address.doc_id,
                                    confirmed_fields={"passport_number": None}), "test-user", scope=MILESTONE_SCOPE))
    state.update(verification_updates(state))
    assert not state["verification_conflicts"]
    assert document_review(state, address, full=True)["status"] == "complete"


def test_graph_refreshes_legacy_conflicts_before_planning_and_after_passport_replacement(ready_bundle, monkeypatch):
    graph, config, saved = ready_bundle
    state = deepcopy(saved)
    address = address_example(state)
    state["verification_conflicts"] = []
    graph.update_state(config, state, as_node="respond")

    def inspect_plan_input(current):
        payload = compact_planner_context(current)
        assert PASSPORT_FIELD in payload["unresolved_conflicts"]
        assert any(PASSPORT_FIELD in item["fields"] and item["action_type"] == "CORRECT_INFORMATION"
                   for item in payload["canonical_candidates"])
        from src.core.state import ActionPlan
        return ActionPlan(summary="Synthetic inspection", actions=[])

    # Capture actual input after Gate 0 and rules refresh, without invoking Ollama.
    monkeypatch.setattr("src.agents.planner.create_action_plan", inspect_plan_input)
    graph.invoke(new_run(UserEvent(kind="generate_plan"), "test-user", scope=MILESTONE_SCOPE), config=config)
    assert PASSPORT_FIELD in graph.get_state(config).values["verification_conflicts"]

    doc = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    replacement = add_evidence(
        state, "passport_identity_page", "passport_copy", doc_id="replacement-passport", status="replacement_pending",
        text="PASSPORT\nHolder name: Alex Example\nNationality: Exampleland\n"
             f"Passport number: {OTHER_PASSPORT}\nExpiry date: 2030-08-30",
    )
    graph.update_state(config, {"documents": state["documents"]}, as_node="respond")
    resolved = graph.invoke(new_run(UserEvent(kind="replace_document", document_id=replacement.doc_id, confirmed=True),
                                    "test-user", scope=MILESTONE_SCOPE), config=config)
    assert all(item.doc_id != doc.doc_id or item.status == "deleted" for item in resolved["documents"])
    assert doc.doc_id not in resolved["verification_records"]
    assert not resolved["verification_conflicts"]
    assert {row["document_id"] for row in accepted_sources(resolved)[PASSPORT_FIELD]} == {address.doc_id, replacement.doc_id}
    assert assess_document_collection(resolved).ready


def test_sequential_confirmation_rejects_reversed_period(ready_bundle):
    state = ready_bundle[2]
    policy = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
    state = override(state, policy.doc_id, "coverage_start_date", "2027-01-01")
    with pytest.raises(ValueError, match="End date must not precede start date"):
        override(state, policy.doc_id, "coverage_end_date", "2026-12-20")


def test_clearing_conflicting_declaration_does_not_override_current_passport(ready_bundle):
    state = override(ready_bundle[2], "__declarations__", "passport_expiry_date", "2031-02-01")
    assert "passport_expiry_date" in state["verification_conflicts"]
    state = override(state, "__declarations__", "passport_expiry_date", None)
    assert "passport_expiry_date" not in state["verification_conflicts"]
    assert str(state["residence_permit_context"].passport_expiry_date) == "2030-08-30"


def test_period_confirmation_ignores_an_uncertain_other_endpoint(ready_bundle):
    state = deepcopy(ready_bundle[2])
    policy = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
    policy.info.ambiguous_fields.append("coverage_start_date")
    recheck(state)
    assert "coverage_start_date" not in state["verification_records"][policy.doc_id]

    corrected = override(state, policy.doc_id, "coverage_end_date", "2026-08-01")
    assert corrected["verification_records"][policy.doc_id]["coverage_end_date"].value == "2026-08-01"
    assert "coverage_start_date" not in corrected["verification_records"][policy.doc_id]


def test_offline_planner_targets_the_reported_identity_conflict(ready_bundle):
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import fallback_plan
    from src.guardrails.gate_1 import validate_plan
    from src.workers.rules_engine import evaluate_requirements

    state = deepcopy(ready_bundle[2])
    address_example(state)
    recheck(state)
    state["relevant_requirements"] = evaluate_requirements(state)
    assert state["verification_conflicts"] == [PASSPORT_FIELD]
    plan = fallback_plan(state, actionable_needs(state))
    assert any(step.action_type == "CORRECT_INFORMATION" and PASSPORT_FIELD in step.requested_fields
               for step in plan.actions)
    assert not any("holder_name" in step.requested_fields for step in plan.actions)
    assert not validate_plan(state, plan)


def test_deleted_passport_is_missing_and_its_facts_are_retired(ready_bundle):
    state = deepcopy(ready_bundle[2])
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    passport.status = "deleted"
    recheck(state)
    assert state["residence_permit_context"].passport_or_foreign_id is None
    assert passport.doc_id not in state["verification_records"]
    assert assess_document_collection(state).prerequisites["passport_identity"] == "MISSING"
    assert state["run_status"] == "DOCUMENT_ANALYZED"


def test_legacy_active_duplicate_cannot_establish_package_readiness_or_be_attached(tmp_path):
    state = copilot_benchmark_cases(tmp_path / "iro")[9]["state"]
    assert iro_package_readiness(state)["ready"]
    original = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    duplicate = add_evidence(state, "passport_identity_page", "passport_copy", doc_id="incomplete-duplicate",
                             text="Passport\nHolder name: Evaluation Student")
    package = iro_package_readiness(state)
    assert not package["ready"]
    row = next(row for row in package["items"] if row["document_type"] == "passport_copy")
    assert row["status"] == "ambiguous" and set(row["document_ids"]) == {original.doc_id, duplicate.doc_id}
    assert {original.doc_id, duplicate.doc_id}.isdisjoint(package["attachment_ids"])
    assert "package:passport_copy" in package["blockers"]
    result = execute_tool(state, ToolCall(name="prepare_iro_email", arguments={"purpose": "package_review"}),
                          approved=True)
    assert result.status == "FAILED" and not result.state_updates
    duplicate.status = "deleted"
    package = iro_package_readiness(state)
    assert package["ready"] and original.doc_id in package["attachment_ids"]
