"""src/tools/extract_document.py: grounded observations become pack facts; nothing else does."""
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fakes import ScriptedBackend
from src.document_processing.processor import DocumentProcessingError
from src.document_processing.schemas import EntryStamp, HealthInsurance, SourceEvidence, UnknownDocument
from src.pack import load_pack
from src.tools.extract_document import extract_file

PACK = load_pack("ikamet_first")


def evidence(field, raw, value, snippet, page=1):
    return SourceEvidence(field=field, original_value=raw, normalized_value=value, snippet=snippet,
                          line_number=1, page_number=page)


def insurance(**extra):
    """A typed insurance observation whose end date is backed by evidence; the insurer is not."""
    return HealthInsurance(
        holder_name="Alex Example", insurer="Example Insurance", coverage_end_date="2027-06-30",
        source_pages={"holder_name": 1, "coverage_end_date": 1},
        source_snippets={"holder_name": "Insured person: Alex Example", "coverage_end_date": "Coverage end: 30.06.2027"},
        source_evidence=[evidence("holder_name", "Alex Example", "Alex Example", "Insured person: Alex Example"),
                         evidence("coverage_end_date", "30.06.2027", "2027-06-30", "Coverage end: 30.06.2027")],
        **extra)


def run(info, tmp_path, name="doc.txt"):
    path = tmp_path / name
    path.write_text("SYNTHETIC", encoding="utf-8")
    return extract_file(path, PACK, processor=lambda *args, **kwargs: info, bundle=lambda p, i: [i])


def test_grounded_document_fields_become_pack_facts(tmp_path):
    result = run(insurance(), tmp_path)
    facts = {f.field: f for f in result.observations[0].facts}
    assert facts["insurance_end_date"].value == date(2027, 6, 30)
    assert facts["insurance_end_date"].from_field == "coverage_end_date"
    assert facts["insurance_end_date"].page == 1 and "30.06.2027" in facts["insurance_end_date"].snippet
    assert facts["holder_name"].value == "Alex Example"


def test_values_without_source_evidence_are_not_proposed(tmp_path):
    facts = {f.field for f in run(insurance(), tmp_path).observations[0].facts}
    assert "insurance_insurer" not in facts


def test_ambiguous_fields_are_not_proposed(tmp_path):
    result = run(insurance(ambiguous_fields=["coverage_end_date"]), tmp_path)
    assert "insurance_end_date" not in {f.field for f in result.observations[0].facts}
    assert "coverage_end_date" in result.observations[0].ambiguous_fields


def test_entry_date_waits_for_confirmation_but_page_presence_does_not(tmp_path):
    stamp = EntryStamp(entry_stamp_present=True, latest_entry_date="2026-09-15",
                       source_pages={"entry_stamp_present": 1, "latest_entry_date": 1},
                       source_snippets={"entry_stamp_present": "TURKIYE ENTRY STAMP", "latest_entry_date": "Entry date: 15.09.2026"},
                       source_evidence=[evidence("entry_stamp_present", "TURKIYE ENTRY STAMP", "true", "TURKIYE ENTRY STAMP"),
                                        evidence("latest_entry_date", "15.09.2026", "2026-09-15", "Entry date: 15.09.2026")])
    facts = {f.field: f for f in run(stamp, tmp_path).observations[0].facts}
    assert facts["latest_entry_date"].requires_confirmation is True
    assert facts["turkey_entry_page_present"].value is True and not facts["turkey_entry_page_present"].requires_confirmation
    assert facts["entry_stamp_present"].value is True


def test_unknown_documents_give_no_facts(tmp_path):
    observation = run(UnknownDocument(), tmp_path).observations[0]
    assert observation.doc_type == "unknown" and observation.facts == []


def test_processing_errors_are_returned_not_raised(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("x", encoding="utf-8")

    def broken(*args, **kwargs):
        raise DocumentProcessingError("Local Ollama timed out during extraction")
    result = extract_file(path, PACK, processor=broken)
    assert result.observations == [] and "timed out" in result.error


def test_missing_file_is_an_error_result(tmp_path):
    assert "not found" in extract_file(tmp_path / "missing.txt", PACK).error


def test_real_processor_with_mocked_model_on_a_synthetic_certificate(tmp_path):
    path = tmp_path / "certificate.txt"
    path.write_text("SYNTHETIC EXAMPLE - NOT AN OFFICIAL DOCUMENT\nStudent Enrollment Certificate\n"
                    "Holder name: Alex Example\nStudent ID: SYN-001\nSabancı University\nIssue date: 2026-11-01",
                    encoding="utf-8")
    replies = [{"document_type": "student_enrollment_certificate"},
               {"holder_name": "Alex Example", "student_id": "SYN-001", "institution": "Sabancı University",
                "issue_date": "2026-11-01"}]
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [SimpleNamespace(message=SimpleNamespace(content=json.dumps(r))) for r in replies]
        result = extract_file(path, PACK)
    observation = result.observations[0]
    facts = {f.field: f.value for f in observation.facts}
    assert observation.doc_type == "student_enrollment_certificate"
    assert facts["student_id"] == "SYN-001" and facts["student_certificate_issue_date"] == date(2026, 11, 1)
    assert all(f.page == 1 and f.snippet for f in observation.facts)


# -------------------------------------------------------- generic path --
@pytest.fixture
def card_pack(tmp_path):
    """A tiny synthetic pack with one document type that has no typed schema."""
    folder = tmp_path / "packs" / "card_demo"
    folder.mkdir(parents=True)
    procedure = {"id": "card_demo", "title": "Synthetic", "status": "unverified", "scope": "test",
                 "sections": [{"id": "s", "title": "S"}],
                 "fields": [{"name": "holder_name", "type": "str"}, {"name": "permit_expiry_date", "type": "date"}],
                 "documents": [{"doc_type": "residence_permit_card", "description": "Residence permit card",
                                "extracts": ["holder_name", {"field": "permit_expiry_date", "from_field": "valid_until"}]}]}
    (folder / "procedure.json").write_text(json.dumps(procedure), encoding="utf-8")
    (folder / "rules.json").write_text("[]", encoding="utf-8")
    (folder / "sources.json").write_text("[]", encoding="utf-8")
    document = tmp_path / "card.txt"
    document.write_text("SYNTHETIC RESIDENCE PERMIT CARD\nName: Alex Example\nValid until: 31.07.2027\n", encoding="utf-8")
    return load_pack("card_demo", tmp_path / "packs"), document


def test_generic_path_keeps_only_grounded_values(card_pack):
    pack, document = card_pack
    backend = ScriptedBackend([{"fields": [
        {"field": "holder_name", "value": "Alex Example", "page": 1, "quote": "Name: Alex Example"},
        {"field": "valid_until", "value": "31.07.2027", "page": 1, "quote": "Valid until: 31.07.2027"}]}])
    observation = extract_file(document, pack, doc_type="residence_permit_card", backend=backend).observations[0]
    facts = {f.field: f.value for f in observation.facts}
    assert facts == {"holder_name": "Alex Example", "permit_expiry_date": date(2027, 7, 31)}
    enum = backend.calls[0]["schema"]["$defs"]["ProposedField"]["properties"]["field"]["enum"]
    assert enum == ["holder_name", "valid_until"]


@pytest.mark.parametrize("proposal,reason", [
    ({"field": "holder_name", "value": "Maria Other", "page": 1, "quote": "Name: Maria Other"}, "quote not found"),
    ({"field": "holder_name", "value": "Maria Other", "page": 1, "quote": "Name: Alex Example"}, "not present"),
    ({"field": "valid_until", "value": "31.07.2027", "page": 4, "quote": "Valid until: 31.07.2027"}, "does not exist"),
    ({"field": "nationality", "value": "X", "page": 1, "quote": "Name: Alex Example"}, "not a field"),
])
def test_generic_path_drops_invented_values(card_pack, proposal, reason):
    pack, document = card_pack
    # a misquote gets one second reading (reflection): here the model finds nothing more
    backend = ScriptedBackend([{"fields": [proposal]}, {"fields": []}])
    observation = extract_file(document, pack, doc_type="residence_permit_card", backend=backend).observations[0]
    assert observation.facts == []
    assert any(reason in issue for issue in observation.issues)


def test_generic_path_model_failure_means_no_facts(card_pack):
    pack, document = card_pack
    observation = extract_file(document, pack, doc_type="residence_permit_card",
                               backend=ScriptedBackend(["nope", "still nope"])).observations[0]
    assert observation.facts == [] and "unusable" in observation.issues[0]
