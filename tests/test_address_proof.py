"""Synthetic address-evidence lifecycle checks for the supported first application."""
import pytest

from src.core.procedure import address_proof_status, assess_document_collection
from src.core.state import ProcedureState, ResidencePermitContext, UserEvent, new_run
from src.evaluation.scenarios import confirm
from src.workers.rules_engine import evaluate_requirements


def case():
    return {**new_run(UserEvent(kind="reevaluate"), "synthetic-address"),
            "procedure": ProcedureState()}


def address_rule(state):
    return next(rule for rule in evaluate_requirements(state) if rule.rule_id == "R017")


@pytest.mark.parametrize("method", ["dorm_certificate", "rental_agreement", "notarized_commitment"])
def test_method_selection_without_the_document_is_incomplete(method):
    state = confirm(case(), "__declarations__", {"address_proof_type": method})
    assert not address_proof_status(state)["ready"]
    assert "missing:address_proof" in assess_document_collection(state).blockers
    assert address_rule(state).passed is None


def test_cached_pair_flags_do_not_establish_readiness_or_override_a_different_selection():
    from src.guardrails.gate_0 import gate_0

    state = case()
    state["residence_permit_context"] = ResidencePermitContext(
        address_proof_type="rental_agreement", utility_bill_accepted=True, address_form_accepted=True,
    )
    row = address_rule(state)
    assert row.passed is None
    assert not address_proof_status(state)["ready"]
    state.update(event=UserEvent(kind="generate_plan"),
                 relevant_requirements=[row.model_copy(update={"passed": True, "status": "passed"})])
    assert gate_0(state)["relevant_requirements"] is None


SAMPLES = {
    "dorm_certificate": "Dormitory Certificate\nStudent name: Alex Example\nDormitory address: Example Campus 10",
    "rental_agreement": "Noter Onaylı Kira Sözleşmesi\nKiracı: Alex Example\nKiraya veren: Example Landlord\nKiralananın adresi: Synthetic Street 12",
    "notarized_commitment": "Taahhütname\nMisafir: Alex Example\nTaahhüt eden: Example Host\nİkamet adresi: Synthetic Street 12",
    "host_support": "Host supporting documents\nHost: Example Host\nAddress: Synthetic Street 12",
    "utility_bill": "Utility Bill\nHolder name: Example Host\nService address: Synthetic Street 12\nBilling date: 2026-11-12\nProvider: Example Energy",
    "utility_subscription": "Utility Subscription Agreement\nSubscriber: Example Host\nService address: Synthetic Street 12\nProvider: Example Energy",
}


def add_document(state, kind, *, doc_id=None, text=None, path=None, status="analyzed"):
    from src.core.state import Document
    from src.evaluation.scenarios import extract_synthetic

    source = text or SAMPLES[kind]
    document = Document(doc_id=doc_id or kind, file_path=str(path or f"synthetic-{kind}.txt"),
                        doc_type=kind, status=status, info=extract_synthetic(kind, text=source))
    if path:
        path.write_text(source, encoding="utf-8")
    state["documents"] = [*state.get("documents", []), document]
    return document


@pytest.mark.parametrize(("method", "attestation"), [
    ("dorm_certificate", "dorm_certificate_format"),
    ("rental_agreement", "notary_certification"),
])
def test_address_evidence_requires_document_specific_review(method, attestation):
    state = confirm(case(), "__declarations__", {"address_proof_type": method})
    uploaded = add_document(state, method)
    assert uploaded.info.holder_name == "Alex Example"
    assert uploaded.info.source_pages["holder_name"] == 1
    assert list(address_proof_status(state)["parts"])[0]["status"] == "needs confirmation"

    state = confirm(state, uploaded.doc_id, {attestation: False})
    assert address_rule(state).passed is None
    state = confirm(state, uploaded.doc_id, {attestation: True})
    assert address_rule(state).passed is True
    assert assess_document_collection(state).prerequisites["address_proof"] == "READY"

    state["documents"] = [uploaded.model_copy(update={"status": "deleted"})]
    assert address_rule(state).passed is None


def test_missing_and_conflicting_rental_fields_need_review():
    state = confirm(case(), "__declarations__", {"address_proof_type": "rental_agreement"})
    text = SAMPLES["rental_agreement"].replace("Kiraya veren: Example Landlord\n", "")
    doc = add_document(state, "rental_agreement", text=text + "\nRental address: A different synthetic address")
    assert doc.info.ambiguous_fields == ["declared_address"]
    assert "landlord_name" in doc.info.missing_fields
    state = confirm(state, doc.doc_id, {"notary_certification": True, "landlord_name": "Example Landlord"})
    assert address_proof_status(state)["parts"][0]["status"] == "needs confirmation"
    state = confirm(state, doc.doc_id, {"declared_address": "Synthetic Street 12"})
    assert address_rule(state).passed is True
    assert doc.info.declared_address is None


@pytest.mark.parametrize(("married", "turkish", "utility"), [
    (False, True, "utility_bill"),
    (True, False, "utility_subscription"),
])
def test_host_route_requires_the_applicable_support_and_current_named_utility(married, turkish, utility):
    state = confirm(case(), "__declarations__", {"address_proof_type": "notarized_commitment"})
    host = add_document(state, "notarized_commitment")
    state = confirm(state, host.doc_id, {"notary_certification": True, "host_identity_copy": True,
                                        "host_married": married, "host_is_turkish": turkish})
    support = add_document(state, "host_support")
    variant = "host_population_record" if turkish else "host_permit_copy"
    other = "host_permit_copy" if turkish else "host_population_record"
    state = confirm(state, support.doc_id, {"host_address_certificate": True, other: True})
    parts = {part["key"]: part for part in address_proof_status(state)["parts"]}
    assert variant in parts and other not in parts
    assert ("spouse_undertaking" in parts) is married
    assert parts[variant]["status"] == "needs confirmation"
    assert parts["host_identity_copy"]["document_ids"] == [host.doc_id]
    assert parts["host_address_certificate"]["document_ids"] == [support.doc_id]
    state = confirm(state, support.doc_id, {variant: True})

    bill = add_document(state, utility, text=SAMPLES[utility].replace("Example Host", "Other Person"))
    state = confirm(state, bill.doc_id, {"host_utility_current": True})
    assert not address_proof_status(state)["ready"]
    state = confirm(state, bill.doc_id, {"holder_name": "example host"})
    assert address_proof_status(state)["ready"] is (not married)
    if married:
        spouse = add_document(state, "host_support", doc_id="spouse-file")
        state = confirm(state, spouse.doc_id, {"spouse_undertaking": True})
        parts = {part["key"]: part for part in address_proof_status(state)["parts"]}
        assert parts["spouse_undertaking"]["document_ids"] == [spouse.doc_id]
    assert address_rule(state).passed is True


def test_local_iro_pair_requires_both_files_and_an_explicit_selection():
    from src.evaluation.scenarios import SYNTHETIC

    state = case()
    add_document(state, "address_form", text=SYNTHETIC["address_form"])
    state = confirm(state, "__declarations__", {"address_proof_type": "utility_bill_and_address_form"})
    assert [part["status"] for part in address_proof_status(state)["parts"]] == ["complete", "missing"]
    add_document(state, "utility_bill")
    assert address_rule(state).passed is True
    state = confirm(state, "__declarations__", {"address_proof_type": None})
    assert address_rule(state).passed is None
    state = confirm(state, "__declarations__", {"address_proof_type": "airbnb"})
    assert address_rule(state).passed is False


def test_an_unknown_upload_or_an_extension_cannot_establish_first_application_proof():
    from src.core.state import Document
    from src.document_processing.extractor import TextPage
    from src.document_processing.processor import process_document
    from src.evaluation.scenarios import ScriptedDocumentProvider

    state = confirm(case(), "__declarations__", {"address_proof_type": "dorm_certificate"})
    same_file = "synthetic-unrelated.txt"
    sample = [TextPage(1, "An unrelated synthetic page.")]
    info = process_document(same_file, pages=sample, provider=ScriptedDocumentProvider("dorm_certificate"))
    assert info.document_type == "unknown"
    state["documents"] = [Document(doc_id="unknown", doc_type="unknown", status="analyzed", file_path=same_file, info=info)]
    assert address_proof_status(state)["parts"][0]["status"] == "missing"
    with pytest.raises(ValueError, match="Reprocess"):
        confirm(state, "unknown", {"dorm_certificate_format": True})
    manual = process_document(same_file, pages=sample, document_type_hint="dorm_certificate",
                              provider=ScriptedDocumentProvider("dorm_certificate"))
    assert manual.document_type == "dorm_certificate" and manual.holder_name is None
    assert not manual.source_evidence
    doc = add_document(state, "dorm_certificate")
    state = confirm(state, doc.doc_id, {"dorm_certificate_format": True})
    context = state["residence_permit_context"]
    state["residence_permit_context"] = context.model_copy(update={"application_kind": "extension"})
    assert address_rule(state).passed is False
    assert not address_proof_status(state)["ready"]


def test_switching_method_reuses_saved_evidence_and_rejects_the_obsolete_plan():
    from src.agents.planner import actionable_needs, canonical_candidates
    from src.agents.planning_policy import fallback_plan
    from src.guardrails.gate_1 import validate_plan
    from src.core.procedure import eikamet_preparation_inputs, stage_document_checklist

    state = confirm(case(), "__declarations__", {"address_proof_type": "dorm_certificate"})
    dormant = next(item for item in canonical_candidates(state) if item["rule_ids"] == ["R017"])
    assert dormant["documents"] == ["dorm_certificate"]
    original = fallback_plan(state, actionable_needs(state))
    assert not validate_plan(state, original)

    dorm = add_document(state, "dorm_certificate")
    state = confirm(state, dorm.doc_id, {"dorm_certificate_format": True})
    assert address_rule(state).passed is True
    state = confirm(state, "__declarations__", {"address_proof_type": "rental_agreement"})
    assert address_rule(state).passed is None
    invalid = validate_plan(state, original)
    assert any(issue.code == "UNSUPPORTED_GROUNDING" for issue in invalid)
    rental = next(item for item in canonical_candidates(state) if item["rule_ids"] == ["R017"])
    assert rental["documents"] == ["rental_agreement"]
    assert dorm.doc_id in {doc.doc_id for doc in state["documents"]}

    contract = add_document(state, "rental_agreement")
    state = confirm(state, contract.doc_id, {"notary_certification": True})
    assert address_rule(state).passed is True
    assert not state["verification_conflicts"]
    _, values = eikamet_preparation_inputs(state)
    assert values["turkiye_address"] == "Synthetic Street 12"

    state = confirm(state, "__declarations__", {"address_proof_type": "dorm_certificate"})
    row = next(row for row in stage_document_checklist(state) if row.get("key") == "address_proof")
    assert row["status"] == "complete"
    assert [part["document_ids"] for part in row["parts"]] == [[dorm.doc_id]]
    _, values = eikamet_preparation_inputs(state)
    assert values["turkiye_address"] == "Example Campus 10"
    assert state["verification_records"][contract.doc_id]["notary_certification"].value is True


def test_replacement_and_removal_recompute_with_isolated_checkpoints(tmp_path):
    from src.core.graph import build_graph
    from src.memory.checkpointer import sqlite_checkpointer
    from src.workers.verification import MILESTONE_SCOPE

    state = confirm(case(), "__declarations__", {"address_proof_type": "dorm_certificate"})
    original = add_document(state, "dorm_certificate", path=tmp_path / "old.txt")
    state = confirm(state, original.doc_id, {"dorm_certificate_format": True})
    replacement = add_document(state, "dorm_certificate", doc_id="new-dorm", status="replacement_pending",
                               path=tmp_path / "new.txt")
    assert address_rule(state).passed is True
    cfg = {"configurable": {"thread_id": "synthetic-address"}, "recursion_limit": 60}
    db = tmp_path / "address.db"

    def dispatch(graph, kind, doc_id, **kwargs):
        return graph.invoke(new_run(UserEvent(kind=kind, document_id=doc_id, **kwargs), "synthetic-address",
                                    scope=MILESTONE_SCOPE), cfg)

    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        graph.update_state(cfg, state, as_node="respond")
        state = dispatch(graph, "replace_document", replacement.doc_id, confirmed=True)
        assert address_rule(state).passed is None
        assert original.doc_id not in state["verification_records"]
        state = dispatch(graph, "confirm_information", replacement.doc_id, confirmed_fields={"dorm_certificate_format": True})
        assert address_rule(state).passed is True
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        assert address_proof_status(graph.get_state(cfg).values)["parts"][0]["document_ids"] == [replacement.doc_id]
        state = dispatch(graph, "delete_document", replacement.doc_id, confirmed=True)
        assert address_rule(state).passed is None
        assert address_proof_status(state)["parts"][0]["status"] == "missing"
    assert (tmp_path / "old.txt").is_file() and (tmp_path / "new.txt").is_file()
