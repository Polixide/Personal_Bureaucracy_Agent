"""Offline classification/extraction regression tests; model transport is mocked."""
from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pymupdf
import pytest
from pydantic import ValidationError

from config import OLLAMA_MODEL, OLLAMA_HOST
from src.document_processing.providers import _create_openai_client as create_openai_for_test
from src.document_processing.extractor import TextPage
from src.document_processing.processor import (
    CHUNK_CHARS, MAX_INPUT_CHARS, MAX_CHUNKS, DocumentProcessingError,
    process_document, select_chunks, render_chunks,
)
from src.document_processing.schemas import (
    DocumentClassification, StudentCertificate, HealthInsurance,
    IkametApplicationForm, UnknownDocument,
)

STUDENT = dict(holder_name="Alex Example", student_id="TEST-0001",
               institution="Example University", issue_date="2026-10-01")
INSURANCE = dict(holder_name="Alex Example", insurer="Example Insurance",
                 policy_number="TEST-POL-0001", passport_or_foreign_id="P123456",
                 coverage_start_date="2026-10-01", coverage_end_date="2027-09-30")
APPLICATION = dict(applicant_name="Alex Example", application_number="APP-123",
                   passport_or_foreign_id="P123456", requested_start_date="2026-11-01",
                   requested_end_date="2027-06-01")


def response(payload):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))


def process_text(tmp_path, text, kind, payload=None):
    path = tmp_path / "document.txt"
    path.write_text(text, encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        replies = [response({"document_type": kind})]
        if payload is not None:
            replies.append(response(payload))
        client.return_value.chat.side_effect = replies
        result = process_document(path)
        return result, client.return_value.chat.call_args_list


def test_plain_text_is_unknown_and_skips_extraction(tmp_path):
    info, calls = process_text(tmp_path, "ciao sono marco", "unknown")
    assert isinstance(info, UnknownDocument)
    assert len(calls) == 1
    assert info.holder_name is None
    assert "student_id" not in info.model_dump()
    assert calls[0].kwargs["format"] == DocumentClassification.model_json_schema()


def test_ikamet_form_cannot_be_student_certificate(tmp_path):
    text = "RESIDENCE PERMIT APPLICATION FORM\nUniversity: Sabanci\nApplication number: APP-123\nApplicant: Alex Example\nPassport: P123456\nRequested period: 2026-11-01 to 2027-06-01"
    # Reproduce the live misclassification: deterministic heading guard corrects it.
    info, calls = process_text(tmp_path, text, "student_enrollment_certificate", APPLICATION)
    assert isinstance(info, IkametApplicationForm)
    assert info.application_number == "APP-123"
    assert info.applicant_name == "Alex Example"
    assert info.requested_end_date == date(2027,6,1)
    assert "student_id" not in calls[1].kwargs["format"]["properties"]
    assert info.source_pages["application_number"] == 1


def test_turkish_form_title(tmp_path):
    text = "\u0130KAMET \u0130ZN\u0130 BA\u015eVURU FORMU\nAlex Example"
    info, _ = process_text(tmp_path, text, "student_enrollment_certificate", APPLICATION)
    assert isinstance(info, IkametApplicationForm)


def test_student_type_fields_and_source_page(tmp_path):
    text = "Student Enrollment Certificate\nHolder name: Alex Example\nStudent ID: TEST-0001\nExample University\nIssue date: 2026-10-01"
    info, calls = process_text(tmp_path, text, "student_enrollment_certificate", STUDENT)
    assert isinstance(info, StudentCertificate)
    assert info.student_id == "TEST-0001"
    assert info.issue_date == date(2026,10,1)
    assert not info.missing_fields
    assert not info.validation_issues
    assert info.source_pages["student_id"] == 1
    assert set(calls[1].kwargs["format"]["properties"]) == set(StudentCertificate.model_fields) - {"missing_fields", "validation_issues", "source_evidence", "correction_notes", "selected_pages", "extraction_diagnostics"}
    assert "SourceEvidence" not in calls[1].kwargs["format"].get("$defs", {})
    assert all(call.kwargs["model"] == OLLAMA_MODEL for call in calls)


def test_insurance_uses_specific_schema_and_identity_field(tmp_path):
    info, calls = process_text(tmp_path, "Health Insurance Certificate\nPolicy: TEST-POL-0001\nPassport: P123456", "health_insurance_certificate", INSURANCE)
    assert isinstance(info, HealthInsurance)
    assert info.policy_number == "TEST-POL-0001"
    assert info.passport_or_foreign_id == "P123456"
    assert "student_id" not in info.model_dump()
    assert "student_id" not in calls[1].kwargs["format"]["properties"]
    assert "institution" not in calls[1].kwargs["format"]["properties"]


def test_insurance_student_id_is_discarded_and_flagged(tmp_path):
    info, _ = process_text(tmp_path, "Health Insurance Certificate", "health_insurance_certificate", {**INSURANCE,"student_id":"P123456"})
    assert "student_id" not in info.model_dump()
    assert any("student_id" in issue for issue in info.validation_issues)


@pytest.mark.parametrize("value", ["1310-10-20", "2200-01-01", "2026-02-30", "10/11/2026"])
def test_invalid_or_implausible_dates_become_null(tmp_path, value):
    info, _ = process_text(tmp_path, "Health Insurance Certificate", "health_insurance_certificate", {**INSURANCE,"coverage_start_date":value})
    assert info.coverage_start_date is None
    assert "coverage_start_date" in info.missing_fields
    assert any("date" in issue for issue in info.validation_issues)


@pytest.mark.parametrize("schema,fields,start,end", [
    (HealthInsurance, INSURANCE,"coverage_start_date","coverage_end_date"),
    (IkametApplicationForm, APPLICATION,"requested_start_date","requested_end_date"),
])
def test_end_before_start_is_discarded_and_flagged(schema,fields,start,end):
    info = schema(**{**fields,start:"2026-11-01",end:"2026-10-01"})
    assert getattr(info,end) is None
    assert any("precedes" in issue for issue in info.validation_issues)


def test_unknown_documents_cannot_invent_administrative_facts(tmp_path):
    info, calls = process_text(tmp_path, "hello world", "unknown", {"policy_number":"invented"})
    assert len(calls) == 1
    assert not ({"policy_number","student_id","application_number","requested_end_date"} & set(info.model_dump()))
    assert UnknownDocument(holder_name="fabricated").holder_name is None
    with pytest.raises(ValidationError):
        UnknownDocument(policy_number="invented")


def test_missing_fields_are_recomputed(tmp_path):
    info, _ = process_text(tmp_path, "Student Enrollment Certificate\nHolder name: Alex Example\nInstitution: Example University", "student_enrollment_certificate", {**STUDENT,"student_id":None,"issue_date":None,"missing_fields":["invented"],"validation_issues":["model claim"]})
    assert info.missing_fields == ["student_id","issue_date"]
    assert not info.validation_issues


def test_ambiguous_values_are_removed(tmp_path):
    info, _ = process_text(tmp_path, "Health Insurance Certificate", "health_insurance_certificate", {**INSURANCE,"ambiguous_fields":["policy_number","coverage_start_date"]})
    assert info.policy_number is None and info.coverage_start_date is None
    assert "policy_number" in info.missing_fields
    assert any("ambiguous" in issue for issue in info.validation_issues)


def test_inappropriate_ambiguous_field_is_rejected():
    with pytest.raises(ValidationError):
        HealthInsurance(ambiguous_fields=["student_id"])


def test_invalid_provenance_cannot_claim_an_unselected_page(tmp_path):
    info, _ = process_text(tmp_path, "Health Insurance Certificate\nPolicy: TEST-POL-0001", "health_insurance_certificate", {**INSURANCE,"source_pages":{"policy_number":99}})
    assert "policy_number" not in info.source_pages
    assert any("source page" in issue for issue in info.validation_issues)


def test_long_pdf_uses_relevant_pages_not_full_document(tmp_path):
    path = tmp_path / "long.pdf"
    with pymupdf.open() as pdf:
        for index in range(15):
            page = pdf.new_page()
            text = f"BOILERPLATE_PAGE_{index+1}\n" + "General terms and conditions.\n" * 15
            if index == 11:
                text = "Health Insurance Certificate\nPolicy: TEST-POL-0001\nInsurer: Example Insurance\nPassport: P123456\nCoverage: 2026-10-01 to 2027-09-30"
            page.insert_text((72,72),text)
        pdf.save(path)
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [response({"document_type":"health_insurance_certificate"}),response({**INSURANCE,"source_pages":{"policy_number":12}})]
        info = process_document(path)
        calls = client.return_value.chat.call_args_list
    assert 12 in info.selected_pages
    assert len(info.selected_pages) <= MAX_CHUNKS
    assert info.source_pages["policy_number"] == 12
    for call in calls:
        sent = call.kwargs["messages"][1]["content"]
        assert len(sent) <= MAX_INPUT_CHARS
        assert "[PAGE 12]" in sent
        assert "BOILERPLATE_PAGE_15" not in sent
        assert sent.count("[PAGE ") <= MAX_CHUNKS


def test_oversized_single_page_and_txt_are_chunked():
    page = TextPage(1,"intro " * 3000 + "Health Insurance policy coverage Passport TEST-P1" + " ending" * 3000)
    chunks = select_chunks([page],"health_insurance_certificate")
    assert len(chunks) <= MAX_CHUNKS
    assert all(len(c.text) <= CHUNK_CHARS for c in chunks)
    assert len(render_chunks(chunks)) <= MAX_INPUT_CHARS
    assert "TEST-P1" in render_chunks(chunks)


def test_unrelated_text_cannot_be_forced_to_student_by_model(tmp_path):
    info, calls = process_text(tmp_path,"ciao sono marco","student_enrollment_certificate",STUDENT)
    assert isinstance(info, UnknownDocument) and len(calls) == 1


def test_model_cannot_reclassify_during_extraction(tmp_path):
    info, _ = process_text(tmp_path,"Health Insurance Certificate","health_insurance_certificate",{**INSURANCE,"document_type":"student_enrollment_certificate"})
    assert isinstance(info, HealthInsurance)
    assert any("change" in issue for issue in info.validation_issues)


def test_model_and_host_configuration(tmp_path):
    path = tmp_path / "student.txt"; path.write_text("Student Enrollment Certificate")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [response({"document_type":"student_enrollment_certificate"}),response(STUDENT)]
        process_document(path,timeout=10)
        client.assert_called_once_with(host=OLLAMA_HOST,timeout=10)
        assert all(c.kwargs["model"] == OLLAMA_MODEL for c in client.return_value.chat.call_args_list)


@pytest.mark.parametrize("content",["not JSON","[]",'{"document_type":"invented"}'])
def test_bad_classification_output(tmp_path,content):
    path = tmp_path / "student.txt"; path.write_text("Student Certificate")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=content))
        with pytest.raises(DocumentProcessingError): process_document(path)


def test_connection_error(tmp_path):
    path = tmp_path / "student.txt"; path.write_text("Student Certificate")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=httpx.ConnectError("offline")
        with pytest.raises(DocumentProcessingError): process_document(path)


def test_empty_text_skips_model():
    with patch("src.document_processing.processor.extract_pages",return_value=[TextPage(1," ")]), patch("src.document_processing.processor.Client") as client:
        with pytest.raises(ValueError,match="no text"): process_document("empty.txt")
        client.assert_not_called()


def test_schema_rejects_irrelevant_and_wrong_types():
    for payload in ({"student_id":123},{"policy_number":"irrelevant"},{"ambiguous_fields":["invented"]}):
        with pytest.raises(ValidationError): StudentCertificate(**payload)


def test_legacy_catch_all_schema_is_flagged_for_reprocessing():
    from src.document_processing.schemas import DocumentInfo
    legacy = DocumentInfo.model_validate({"document_type":"health_insurance_certificate", "student_id":"P123", "institution":"Example", "issue_date":"2026-10-01"})
    assert legacy.document_type == "unknown"
    assert "student_id" not in legacy.model_dump()
    assert any("reprocess" in issue for issue in legacy.validation_issues)


def test_bad_json_during_extraction(tmp_path):
    path=tmp_path / "student.txt"; path.write_text("Student Enrollment Certificate")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"student_enrollment_certificate"}),SimpleNamespace(message=SimpleNamespace(content="bad JSON"))]
        with pytest.raises(DocumentProcessingError): process_document(path)


def test_long_ikamet_selects_applicant_and_requested_period_pages():
    pages = [TextPage(i, f"General administrative terms page {i}. " * 100) for i in range(1, 17)]
    pages[0] = TextPage(1, "RESIDENCE PERMIT APPLICATION FORM")
    pages[6] = TextPage(7, "Application applicant foreigner passport validity details")
    pages[13] = TextPage(14, "Residence permit application number APP-123\nApplicant: Alex Example\nForeigner passport validity requested period: 2026-11-01 to 2027-06-01")
    with patch("src.document_processing.processor.extract_pages", return_value=pages), \
         patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [
            response({"document_type": "ikamet_application_form"}),
            response({**APPLICATION, "source_pages": {"application_number": 14}}),
        ]
        info = process_document("long.pdf")
    assert info.selected_pages == [7, 14]
    assert info.source_pages["application_number"] == 14
    for call in client.return_value.chat.call_args_list:
        sent = call.kwargs["messages"][1]["content"]
        assert len(sent) <= MAX_INPUT_CHARS
        assert "General administrative terms" not in sent
        assert call.kwargs["think"] is False


def test_configurable_context_cap_keeps_provenance_inside_sent_text():
    text = "Student Enrollment Certificate\nStudent ID: TEST-0001\n" + "filler " * 500 + "UNSENT-INSTITUTION"
    with patch("src.document_processing.processor.extract_pages", return_value=[TextPage(1, text)]), \
         patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [
            response({"document_type": "student_enrollment_certificate"}),
            response({**STUDENT, "institution": "UNSENT-INSTITUTION", "source_pages": {"institution": 1}}),
        ]
        info = process_document("student.pdf", max_context_chars=120)
    for call in client.return_value.chat.call_args_list:
        assert len(call.kwargs["messages"][1]["content"]) <= 120
    assert "institution" not in info.source_pages
    assert any("source page" in issue for issue in info.validation_issues)
    assert info.source_pages["student_id"] == 1


@pytest.mark.parametrize("debug", [False, True])
def test_cli_debug_metadata_is_opt_in_and_leaves_json_clean(debug, capsys):
    import sys
    from src.document_processing.processor import main
    pages = [TextPage(1, "Student Enrollment Certificate\nAlex Example"), TextPage(2, "Extra page")]
    argv = ["processor", "student.pdf"] + (["--debug"] if debug else [])
    with patch.object(sys, "argv", argv), \
         patch("src.document_processing.processor.extract_pages", return_value=pages), \
         patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [
            response({"document_type": "student_enrollment_certificate"}), response(STUDENT),
        ]
        main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["document_type"] == "student_enrollment_certificate"
    if debug:
        assert "total_pdf_pages=2" in captured.err
        assert f"extracted_characters={sum(len(p.text) for p in pages)}" in captured.err
        for stage in ("classification", "extraction"):
            assert f"{stage}_duration_seconds=" in captured.err
            assert f"{stage}_selected_pages=[1, 2]" in captured.err
        sent = client.return_value.chat.call_args_list[1].kwargs["messages"][1]["content"]
        assert f"extraction_input_characters={len(sent)}" in captured.err
    else:
        assert "[document-debug]" not in captured.err
        for component in ("document_classification","document_extraction"):
            assert f"[LLM] {component} model=" in captured.err
        assert captured.err.count(" started")==2
        assert captured.err.count(" success elapsed=")==2


@pytest.mark.parametrize("stage", ["classification", "extraction"])
@pytest.mark.parametrize("timeout_error", [httpx.ReadTimeout, httpx.ConnectTimeout])
def test_timeout_identifies_stage_and_retains_debug_timing(stage, timeout_error, capsys):
    effects = [] if stage == "classification" else [response({"document_type": "student_enrollment_certificate"})]
    effects.append(timeout_error("synthetic timeout"))
    with patch("src.document_processing.processor.extract_pages", return_value=[TextPage(1, "Student Enrollment Certificate")]), \
         patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = effects
        with pytest.raises(DocumentProcessingError, match=f"timed out during {stage}") as caught:
            process_document("student.pdf", debug=True)
    assert caught.value.stage == stage
    assert caught.value.timed_out is True
    diagnostics = capsys.readouterr().err
    assert f"timeout_stage={stage}" in diagnostics
    assert f"{stage}_duration_seconds=" in diagnostics
    assert client.return_value.chat.call_count == (1 if stage == "classification" else 2)


@pytest.mark.parametrize("source,expected", [
    ("20.10.2026", "2026-10-20"), ("20/10/2026", "2026-10-20"),
    ("2026-10-20", "2026-10-20"), ("03/04/2026", "2026-04-03"),
    ("31.02.2026", None), ("29/02/2025", None), ("1310-10-20", None),
])
def test_source_date_candidates_use_deterministic_european_formats(source, expected):
    from src.document_processing.processor import source_candidates
    candidates, issues = source_candidates([TextPage(4, f"Coverage start: {source}")])
    dates = [candidate for candidate in candidates if candidate["kind"] == "date"]
    if expected is None:
        assert dates == []
        assert any("invalid source date" in issue for issue in issues)
    else:
        assert dates[0]["value"] == expected
        assert dates[0]["page"] == 4
        assert "Coverage start" in dates[0]["nearby"]
        assert not issues


def test_insurance_assigns_date_roles_only_from_source_candidates(tmp_path):
    text = ("Health Insurance Certificate\nInsured: Alex Example\n"
            "Policy number: TEST-POL-0001\nPassport: P123456\nInsurer: Example Insurance\n"
            "Coverage start: 01.10.2026\nCoverage end: 30/09/2027")
    info, calls = process_text(tmp_path, text, "health_insurance_certificate", INSURANCE)
    assert info.coverage_start_date == date(2026,10,1)
    assert info.coverage_end_date == date(2027,9,30)
    assert info.holder_name == "Alex Example"
    assert info.policy_number == "TEST-POL-0001"
    assert info.passport_or_foreign_id == "P123456"
    assert info.source_pages["coverage_start_date"] == 1
    assert info.source_pages["coverage_end_date"] == 1
    assert not info.validation_issues
    sent = calls[1].kwargs["messages"][1]["content"]
    assert "[SOURCE CANDIDATES]" in sent
    assert "date=2026-10-01" in sent and "date=2027-09-30" in sent
    assert "Coverage start" in sent and "Coverage end" in sent
    assert len(sent) <= MAX_INPUT_CHARS
    assert all(call.kwargs["think"] is False for call in calls)


def test_requested_dates_and_application_identity_are_grounded(tmp_path):
    text = ("RESIDENCE PERMIT APPLICATION FORM\nAd / Soyad: Alex Example\n"
            "Application number:APP-123\nForeign ID: P123456\n"
            "Requested start: 01/11/2026\nRequested end: 01.06.2027")
    payload = {**APPLICATION, "requested_start_date":"01/11/2026"}
    info, _ = process_text(tmp_path, text, "ikamet_application_form", payload)
    assert info.applicant_name == "Alex Example"
    assert info.application_number == "APP-123"
    assert info.passport_or_foreign_id == "P123456"
    assert info.requested_start_date == date(2026,11,1)
    assert info.requested_end_date == date(2027,6,1)
    assert not info.validation_issues


def test_absent_requested_end_date_cannot_be_invented(tmp_path):
    text = "RESIDENCE PERMIT APPLICATION FORM\nApplicant: Alex Example\nRequested start: 01.11.2026"
    info, _ = process_text(tmp_path, text, "ikamet_application_form", APPLICATION)
    assert info.requested_start_date == date(2026,11,1)
    assert info.requested_end_date is None
    assert any("requested_end_date" in issue for issue in info.validation_issues)


@pytest.mark.parametrize("kind,payload,field", [
    ("health_insurance_certificate", INSURANCE, "policy_number"),
    ("health_insurance_certificate", INSURANCE, "passport_or_foreign_id"),
    ("ikamet_application_form", APPLICATION, "application_number"),
])
def test_identifiers_absent_from_source_are_null(kind,payload,field,tmp_path):
    title = "Health Insurance Certificate" if kind.startswith("health") else "RESIDENCE PERMIT APPLICATION FORM"
    info, _ = process_text(tmp_path,title,kind,payload)
    assert getattr(info,field) is None
    assert any(field in issue and "source candidates" in issue for issue in info.validation_issues)


def test_impossible_source_date_is_flagged_even_if_model_returns_null(tmp_path):
    text = "Health Insurance Certificate\nCoverage start: 31.02.2026"
    info, _ = process_text(tmp_path,text,"health_insurance_certificate",{**INSURANCE,"coverage_start_date":None})
    assert info.coverage_start_date is None
    assert any("invalid source date" in issue for issue in info.validation_issues)


def test_source_backed_dates_still_enforce_start_end_order(tmp_path):
    text = "Health Insurance Certificate\nCoverage start: 01.11.2026\nCoverage end: 01.10.2026"
    info, _ = process_text(tmp_path,text,"health_insurance_certificate",{
        **INSURANCE,"coverage_start_date":"2026-11-01","coverage_end_date":"2026-10-01"})
    assert info.coverage_start_date == date(2026,11,1)
    assert info.coverage_end_date is None
    assert any("precedes" in issue for issue in info.validation_issues)


def test_unlabelled_names_are_not_trusted(tmp_path):
    info, _ = process_text(tmp_path,"Health Insurance Certificate\nAlex Example","health_insurance_certificate",INSURANCE)
    assert info.holder_name is None
    assert any("holder_name" in issue for issue in info.validation_issues)


def test_pdf_style_adjacent_labels_and_split_name_are_grounded(tmp_path):
    text = ("RESIDENCE PERMIT APPLICATION FORM\nName:\nAlex\nSurname:\nExample\n"
            "Application number:\nAPP-123\nPassport:\nP123456")
    info, _ = process_text(tmp_path,text,"ikamet_application_form",APPLICATION)
    assert info.applicant_name == "Alex Example"
    assert info.application_number == "APP-123"
    assert info.passport_or_foreign_id == "P123456"
    assert info.source_pages["applicant_name"] == 1


def test_insurance_never_uses_student_id_even_if_source_mentions_one(tmp_path):
    text = "Health Insurance Certificate\nStudent ID: STUDENT-123\nPassport: P123456"
    info, _ = process_text(tmp_path,text,"health_insurance_certificate",{**INSURANCE,"student_id":"STUDENT-123"})
    assert "student_id" not in info.model_dump()
    assert info.passport_or_foreign_id == "P123456"
    assert any("student_id" in issue for issue in info.validation_issues)


def test_unknown_with_dates_and_identifiers_still_skips_field_extraction(tmp_path):
    with patch("src.document_processing.processor.source_candidates") as extract:
        info, calls = process_text(tmp_path,"ciao sono marco\nDate: 01.11.2026\nPassport: P123456","unknown")
    extract.assert_not_called()
    assert len(calls) == 1
    assert info.document_type == "unknown" and info.holder_name is None
    assert not {"passport_or_foreign_id","coverage_start_date","application_number"} & info.model_dump().keys()


# Selectively adapted from origin/test/integration-abc evidence/reconciliation
# regressions. Canonical classification, bounded requests and typed schemas run.
def hybrid_insurance(tmp_path, text=None, payload=None):
    if text is None:
        text = ("Health Insurance Certificate\nInsured: Taylor Example\nInsurer: Example Insurance\n"
                "Policy number: DEMO-730\nPolicy approval date: 18/09/2028\n"
                "Document issuance date: 19/09/2028\n"
                "Coverage start date: 20/09/2028\nCoverage end date: 30/06/2029")
    return process_text(tmp_path, text, "health_insurance_certificate", {} if payload is None else payload)


def test_unique_labelled_insurance_evidence_fills_nulls_with_correction_notes(tmp_path):
    info, calls = hybrid_insurance(tmp_path, payload={"ambiguous_fields":["policy_number","issue_date","coverage_start_date","coverage_end_date"]})
    assert info.holder_name == "Taylor Example"
    assert info.insurer == "Example Insurance"
    assert info.policy_number == "DEMO-730"
    assert info.issue_date == date(2028,9,18)
    assert info.coverage_start_date == date(2028,9,20)
    assert info.coverage_end_date == date(2029,6,30)
    assert info.expiry_date is None  # Explicit expiry is not silently inferred.
    assert not info.ambiguous_fields and not info.missing_fields and not info.validation_issues
    assert any(note.startswith("issue_date:") for note in info.correction_notes)
    issue = next(e for e in info.source_evidence if e.field == "issue_date")
    assert issue.original_value == "18/09/2028" and issue.normalized_value == "2028-09-18"
    assert issue.snippet == "Policy approval date: 18/09/2028"
    assert issue.page_number == info.source_pages["issue_date"] == 1
    assert any(e.field == "document_issue_date" for e in info.source_evidence)
    assert all(len(c.kwargs["messages"][1]["content"]) <= MAX_INPUT_CHARS for c in calls)


def test_wrong_or_implausible_model_dates_corrected_only_by_labelled_role(tmp_path):
    info, _ = hybrid_insurance(tmp_path, payload={"issue_date":"2028-02-30", "coverage_start_date":"2028-09-18", "coverage_end_date":"3006-06-29", "policy_number":"FABRICATED-99"})
    assert info.issue_date == date(2028,9,18)
    assert info.coverage_start_date == date(2028,9,20)
    assert info.coverage_end_date == date(2029,6,30)
    assert info.policy_number == "DEMO-730"
    assert not info.validation_issues
    assert all(any(note.startswith(field+":") for note in info.correction_notes) for field in ("issue_date","policy_number","coverage_start_date","coverage_end_date"))


@pytest.mark.parametrize("extra,field", [
    ("Policy approval date: 19/09/2028","issue_date"),
    ("Coverage start date: 21/09/2028","coverage_start_date"),
    ("Policy number: OTHER-999","policy_number"),
    ("Insurer: Other Insurance","insurer"),
    ("Insured: Another Person","holder_name"),
])
def test_conflicting_source_values_are_not_resolved_by_model_choice(tmp_path, extra, field):
    text = ("Health Insurance Certificate\nInsured: Taylor Example\nInsurer: Example Insurance\n"
            "Policy number: DEMO-730\nPolicy approval date: 18/09/2028\n"
            "Coverage start date: 20/09/2028\nCoverage end date: 30/06/2029\n" + extra)
    values={"issue_date":"2028-09-18","coverage_start_date":"2028-09-20","policy_number":"DEMO-730","insurer":"Example Insurance","holder_name":"Taylor Example"}
    info, _ = hybrid_insurance(tmp_path,text,{field:values[field]})
    assert getattr(info,field) is None
    assert field in info.ambiguous_fields
    assert field not in info.source_snippets
    assert sum(issue.startswith(field+":") for issue in info.validation_issues) == 1


@pytest.mark.parametrize("raw", ["31/02/2028", "03/04/2028", "1310-10-20"])
def test_invalid_or_ambiguous_labelled_policy_date_stays_null(tmp_path,raw):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate\nPolicy approval date: "+raw,{"issue_date":"2028-04-03"})
    assert info.issue_date is None and "issue_date" in info.ambiguous_fields
    assert info.validation_issues
    assert any(e.original_value == raw and e.normalized_value is None for e in info.source_evidence if e.field == "issue_date")


def test_explicit_date_format_resolves_slash_ambiguity(tmp_path):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate\nFormat: DD/MM/YYYY\nPolicy approval date: 03/04/2028")
    assert info.issue_date == date(2028,4,3)
    assert not any(issue.startswith("issue_date:") for issue in info.validation_issues)


@pytest.mark.parametrize("other", ["Law reference: 18/09/1994", "Birth date: 18/09/1994", "Page 1: 18/09/2028"])
def test_legal_birth_or_footer_dates_do_not_disambiguate_policy_date(tmp_path,other):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate\n"+other+"\nPolicy approval date: 03/04/2028")
    assert info.issue_date is None
    assert "issue_date" in info.ambiguous_fields


def test_exact_snippets_replace_fabricated_model_quotes_as_correction_notes(tmp_path):
    info, _ = hybrid_insurance(tmp_path,payload={"issue_date":"2028-09-18","source_snippets":{"issue_date":"Invented wording 18/09/2028","coverage_end_date":"Invented ending 30/06/2029"}})
    assert info.issue_date == date(2028,9,18)
    assert info.source_snippets["issue_date"] == "Policy approval date: 18/09/2028"
    assert all("Invented" not in quote for quote in info.source_snippets.values())
    assert any("independently verified" in note for note in info.correction_notes)
    assert not info.validation_issues


def test_fabricated_value_and_quote_without_independent_evidence_require_review(tmp_path):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate\nNo policy dates supplied",{"issue_date":"2028-09-18","source_snippets":{"issue_date":"Invented 18/09/2028"}})
    assert info.issue_date is None
    assert "issue_date" not in info.source_snippets
    assert any("source snippet" in issue for issue in info.validation_issues)
    assert not any("independently verified" in note for note in info.correction_notes)


def test_source_evidence_and_correction_notes_cannot_be_fabricated_by_model(tmp_path):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate",{"source_evidence":[{"field":"invented"}],"correction_notes":["Trust this model"]})
    assert not info.source_evidence
    assert "Trust this model" not in info.correction_notes


@pytest.mark.parametrize("period,expected", [
    ("18/09/2028-00:00 - 20/09/2028-23:59", (date(2028,9,18),date(2028,9,20))),
    ("18/09/2028", (None,None)),
    ("18/09/2028\n20/09/2028\n21/09/2028", (None,None)),
    ("18/09/2028\nBirth date: 20/09/2028", (None,None)),
    ("18/09/2028\nLaw: 20/09/2028", (None,None)),
])
def test_split_period_requires_exactly_two_related_dates(tmp_path,period,expected):
    text="Health Insurance Certificate\nSEYAHAT BA\u015eLANGI\u00c7-\nB\u0130T\u0130\u015e TAR\u0130H\u0130\n"+period
    info, _ = hybrid_insurance(tmp_path,text)
    assert (info.coverage_start_date,info.coverage_end_date) == expected


def test_policy_prefix_and_decomposed_turkish_label_preserve_original_text(tmp_path):
    import unicodedata
    text=unicodedata.normalize("NFD","Health Insurance Certificate\nPOL\u0130\u00c7E NO\n\uff1a DEMO-42")
    info, _ = hybrid_insurance(tmp_path,text)
    assert info.policy_number == "DEMO-42"
    evidence=next(e for e in info.source_evidence if e.field == "policy_number")
    assert evidence.original_value.startswith("\uff1a")
    assert evidence.normalized_value == "DEMO-42"
    assert evidence.snippet in text


def test_policy_label_cannot_consume_unrelated_identifier(tmp_path):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate\nPOLICE NO\nFOREIGN ID: 1234567890",{"policy_number":"1234567890"})
    assert info.policy_number is None
    assert info.validation_issues


def test_duplicate_and_empty_labels_corroborate_instead_of_conflicting(tmp_path):
    text="Health Insurance Certificate\nPolicy approval date:\nPolicy approval date: 18/09/2028\nPolicy approval date: 2028-09-18"
    info, _ = hybrid_insurance(tmp_path,text)
    assert info.issue_date == date(2028,9,18)
    assert "issue_date" not in info.ambiguous_fields


def test_explicit_expiry_is_preserved_when_different_from_coverage_end(tmp_path):
    text="Health Insurance Certificate\nCoverage start date: 20/09/2028\nCoverage end date: 30/06/2029\nExpiry date: 01/07/2029"
    info, _ = hybrid_insurance(tmp_path,text)
    assert info.expiry_date == date(2029,7,1)
    assert info.coverage_end_date == date(2029,6,30)
    assert "expiry_date conflicts with coverage_end_date" in info.validation_issues


def test_policy_dates_do_not_cross_page_boundaries(tmp_path):
    pages=[TextPage(1,"Health Insurance Certificate\nPolicy approval date:"),TextPage(2,"18/09/2028")]
    with patch("src.document_processing.processor.extract_pages",return_value=pages), patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"health_insurance_certificate"}),response({"issue_date":"2028-09-18"})]
        info=process_document("synthetic.pdf")
    assert info.issue_date is None
    assert any(e.field == "issue_date" and e.page_number == 1 and e.normalized_value is None for e in info.source_evidence)


def test_real_pdf_evidence_keeps_original_page_and_snippet(tmp_path):
    path=tmp_path/"insurance.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72,72),"Health Insurance Certificate")
        pdf.new_page().insert_text((72,72),"Policy approval date: 18/09/2028\nPolicy number: DEMO-42")
        pdf.save(path)
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"health_insurance_certificate"}),response({})]
        info=process_document(path)
    assert info.issue_date == date(2028,9,18)
    assert info.source_pages["issue_date"] == 2
    evidence=next(e for e in info.source_evidence if e.field=="issue_date")
    assert evidence.page_number == 2 and evidence.snippet == "Policy approval date: 18/09/2028"


def test_insurer_next_line_is_reliable_labelled_evidence(tmp_path):
    info, _ = hybrid_insurance(tmp_path,"Health Insurance Certificate\nInsurer:\nExample Insurance")
    assert info.insurer == "Example Insurance"
    assert info.source_snippets["insurer"] == "Insurer:\nExample Insurance"


def test_enrollment_keeps_model_uncertainty_despite_unique_source(tmp_path):
    text="Student Enrollment Certificate\nHolder name: Alex Example\nStudent ID: TEST-0001\nInstitution: Example University\nIssue date: 2026-10-01"
    info, _ = process_text(tmp_path,text,"student_enrollment_certificate",{**STUDENT,"student_id":None,"ambiguous_fields":["issue_date"]})
    assert isinstance(info,StudentCertificate)
    assert info.student_id is None and info.issue_date is None
    assert not info.correction_notes
    assert "issue_date" in info.ambiguous_fields


def test_ikamet_roles_and_provenance_remain_supported(tmp_path):
    text="RESIDENCE PERMIT APPLICATION FORM\nApplicant: Alex Example\nApplication number: APP-123\nPassport: P123456\nRequested start: 2026-11-01\nRequested end: 2027-06-01"
    info, _ = process_text(tmp_path,text,"ikamet_application_form",APPLICATION)
    assert isinstance(info,IkametApplicationForm)
    assert info.requested_start_date == date(2026,11,1)
    assert info.requested_end_date == date(2027,6,1)
    assert info.source_snippets["requested_start_date"] == "Requested start: 2026-11-01"
    assert info.source_pages["application_number"] == 1
    assert not info.validation_issues


def test_hybrid_observations_and_correction_notes_survive_sqlite(tmp_path):
    from src.core.graph import build_graph
    from src.core.state import Document
    from src.memory.checkpointer import sqlite_checkpointer
    from src.document_processing.schemas import SourceEvidence
    info, _ = hybrid_insurance(tmp_path)
    document=Document(doc_id="policy",file_path="synthetic.txt",doc_type="health_insurance",status="analyzed",info=info)
    config={"configurable":{"thread_id":"hybrid-policy"}}
    db=tmp_path/"hybrid.db"
    with sqlite_checkpointer(db) as saver:
        build_graph(saver).update_state(config,{"documents":[document],"extracted_facts":{"policy":info.model_dump(mode="json")}})
    with sqlite_checkpointer(db) as saver:
        state=build_graph(saver).get_state(config).values
    restored=state["documents"][0].info
    assert restored == info
    assert isinstance(restored.source_evidence[0],SourceEvidence)
    assert restored.extraction_diagnostics==info.extraction_diagnostics
    assert state["extracted_facts"]["policy"]["correction_notes"] == info.correction_notes


def test_repeated_date_in_two_chunks_of_same_page_retains_real_snippet():
    from src.document_processing.evidence import source_candidates
    chunks=[TextPage(2,"Coverage start date: 20/09/2028"),TextPage(2,"Heading\nOther line\nCoverage start date: 20/09/2028")]
    candidates, _=source_candidates(chunks)
    assert any(c["snippet"] == "Coverage start date: 20/09/2028" and c["page_number"] == 2 for c in candidates)


def test_correction_notes_alone_do_not_block_gate_1(tmp_path):
    from src.core.state import ActionPlan, Document, PlanStep
    from src.guardrails.gate_1 import validate_plan
    info, _=hybrid_insurance(tmp_path)
    assert info.correction_notes and not info.validation_issues
    document=Document(doc_id="policy",file_path="synthetic.txt",status="analyzed",doc_type="health_insurance",info=info,expiry_date=info.coverage_end_date)
    plan=ActionPlan(actions=[PlanStep(step_id="review",priority=1,action="Review policy details",reason="Check the grounded observations",related_rule_id="R015",required_documents=["health_insurance"])],summary="Review proposed observations")
    from src.workers.rules_engine import evaluate_requirements
    assert not validate_plan({"documents":[document],"relevant_requirements":evaluate_requirements({})},plan)


@pytest.mark.parametrize("holder", ["NORA FICTIVA", None])
def test_insurance_person_dob_row_is_strong_source_evidence(tmp_path, holder):
    row="NORA FICTIVA 15/01/1999"
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\n"+row,{"holder_name":holder})
    assert info.holder_name == "NORA FICTIVA"
    assert info.source_snippets["holder_name"] == row
    assert info.source_pages["holder_name"] == 1
    evidence=next(e for e in info.source_evidence if e.field=="holder_name")
    assert evidence.original_value == evidence.normalized_value == "NORA FICTIVA"
    assert evidence.page_number == 1 and evidence.snippet == row


@pytest.mark.parametrize("row", ["THIS IS AN ARBITRARY SENTENCE", "PLEASE READ CAREFULLY 15/01/1999",
                               "UNITED STATES 15/01/1999", "HEALTH INSURANCE CERTIFICATE 15/01/1999",
                               "EXAMPLE INSURANCE COMPANY 15/01/1999", "NORA FICTIVA 31/02/1999"])
def test_uppercase_or_nonperson_insurance_rows_are_not_names(tmp_path,row):
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\n"+row,{"holder_name":row.split(" 26/")[0]})
    assert info.holder_name is None


def test_unlabelled_name_after_insured_information_heading(tmp_path):
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\nInsured information\nNORA FICTIVA",{"holder_name":"NORA FICTIVA"})
    assert info.holder_name == "NORA FICTIVA"
    assert info.source_snippets["holder_name"] == "NORA FICTIVA"


def test_unlabelled_insurance_company_header_is_grounded(tmp_path):
    company="GIG S\u0130GORTA A.\u015e."
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\n"+company,{"insurer":company})
    assert info.insurer == company
    assert info.source_snippets["insurer"] == company
    assert info.source_pages["insurer"] == 1
    evidence=next(e for e in info.source_evidence if e.field=="insurer")
    assert evidence.original_value == company and evidence.page_number == 1


@pytest.mark.parametrize("line", ["According to sigorta law, the insurer must comply",
                                "GIG SIGORTA A.S. tarafindan police duzenlenmistir",
                                "Legal reference to GIG SIGORTA A.S.",
                                "TRAVEL INSURANCE", "INSURANCE FOR FOREIGNERS", "SIGORTA GENEL SARTLARI"])
def test_legal_company_mentions_cannot_be_insurer(tmp_path,line):
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\n"+line,{"insurer":line})
    assert info.insurer is None
    assert any("insurer" in issue for issue in info.validation_issues)


def test_fabricated_holder_absent_from_source_is_rejected(tmp_path):
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\nNo person details supplied",{"holder_name":"FABRICATED PERSON"})
    assert info.holder_name is None
    assert any("holder_name" in issue for issue in info.validation_issues)


def test_strong_unlabelled_insurance_people_conflict_instead_of_guessing(tmp_path):
    info, _=hybrid_insurance(tmp_path,"Health Insurance Certificate\nNORA FICTIVA 15/01/1999\nANOTHER PERSON 20/05/2001",{"holder_name":"NORA FICTIVA"})
    assert info.holder_name is None and "holder_name" in info.ambiguous_fields


def test_insurance_only_person_row_does_not_relax_student_certificate(tmp_path):
    info, _=process_text(tmp_path,"Student Enrollment Certificate\nNORA FICTIVA 15/01/1999","student_enrollment_certificate",{"holder_name":"NORA FICTIVA"})
    assert info.holder_name is None



def test_complete_student_evidence_avoids_redundant_semantic_call(tmp_path):
    from src.document_processing.processor import process_document
    path=tmp_path/"complete-student.txt"
    path.write_text("STUDENT CERTIFICATE\nHolder name: Alex Example\nStudent ID: S123\nInstitution: Sabanci University\nIssue date: 2026-10-14\nThe exchange student will study between September 15, 2026 - June 25, 2027 at Sabanci University",encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value=response({"document_type":"student_enrollment_certificate"})
        info=process_document(path)
    assert client.return_value.chat.call_count==1
    assert info.exchange_student is True and info.program_start_date.isoformat()=="2026-09-15"
    assert not info.missing_fields and info.source_pages["student_id"]==1


@pytest.mark.parametrize("body,expected_start,expected_end", [
    ("Exchange student\nProgram start date: 2026-09-15\nProgram end date: 2027-06-25", "2026-09-15","2027-06-25"),
    ("Exchange student\nProgramme end date: 25.06.2027\nProgramme start date: 15.09.2026", "2026-09-15","2027-06-25"),
    ("Degisim ogrencisi\nProgram baslangic tarihi: 15/09/2026\nProgram bitis tarihi: 25/06/2027", "2026-09-15","2027-06-25"),
    ("Exchange student\nProgram start date: 03/04/2026\nProgram end date: 05/06/2027", None,None),
    ("Academic dates: 2026-09-15 and 2027-06-25", None,None),
])
def test_alternate_academic_layouts_keep_evidence_rules(body,expected_start,expected_end,tmp_path):
    path=tmp_path/"alternate-student.txt"
    path.write_text("STUDENT CERTIFICATE\n"+body,encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"student_enrollment_certificate"}),response({})]
        info=process_document(path)
    assert (info.program_start_date.isoformat() if info.program_start_date else None)==expected_start
    assert (info.program_end_date.isoformat() if info.program_end_date else None)==expected_end
    if expected_start:
        assert info.source_pages["program_start_date"]==1
        assert info.source_snippets["program_start_date"] in "STUDENT CERTIFICATE\n"+body


def test_turkish_only_short_term_heading_is_recognized(tmp_path):
    path=tmp_path/"turkish-form.txt"
    path.write_text("IKAMET IZNI KISA DONEM KAYIT FORMU",encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"ikamet_application_form"}),response({})]
        info=process_document(path)
    assert info.document_type=="ikamet_application_form" and info.permit_type=="short_term"
    assert info.source_pages["permit_type"]==1



def test_document_llm_diagnostics_are_present_without_verbose_content(tmp_path,capsys):
    path=tmp_path/"document-diagnostics.txt"
    path.write_text("Student Enrollment Certificate\nPRIVATE DOCUMENT CONTENT",encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"student_enrollment_certificate"}),response({})]
        info=process_document(path,debug=False)
    assert info.document_type=="student_enrollment_certificate"
    captured=capsys.readouterr()
    for component in ("document_classification","document_extraction"):
        assert f"[LLM] {component} model=" in captured.err
    assert captured.err.count(" started")==2 and captured.err.count(" success elapsed=")==2
    assert "PRIVATE" not in captured.err and "[document-debug]" not in captured.err
    assert captured.out==""



class FakeDocumentProvider:
    def __init__(self,payloads, *, name="ollama",images=False):
        self.name=name;self.model="test-document-model";self.supports_images=images
        self.payloads=payloads;self.calls=[]
    def generate_json(self,**request):
        from copy import deepcopy
        self.calls.append(request)
        payload=self.payloads[request["stage"]]
        if isinstance(payload,Exception):raise payload
        return deepcopy(payload)


QUALITY_INSURANCE=("Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\n"
                   "Policy number: TEST-POL-0001\nCoverage start: 01.10.2026\nCoverage end: 30.09.2027")


def poor_scan_pages(number=3):
    return [TextPage(number,"???","rapidocr",0,True,0.2,0.1,90,("insufficient_text","malformed_text","low_ocr_confidence"))]


def test_ollama_only_requires_neither_openai_credentials_nor_sdk(tmp_path,monkeypatch):
    from src.document_processing.providers import get_document_provider
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    provider=get_document_provider("ollama")
    assert provider.name=="ollama" and provider.model==OLLAMA_MODEL and not provider.supports_images
    info,_=process_text(tmp_path,"ciao sono marco","unknown")
    assert info.extraction_diagnostics["provider"]=="ollama"
    assert not info.extraction_diagnostics["fallback_attempted"]


def test_document_provider_selection_and_invalid_configuration():
    from src.document_processing.providers import get_document_provider
    assert get_document_provider("openai").supports_images
    with pytest.raises(DocumentProcessingError,match="Unsupported document provider"):
        get_document_provider("unsupported")


def test_optional_openai_credentials_and_sdk_fail_only_when_requested(monkeypatch):
    import builtins
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    with pytest.raises(DocumentProcessingError,match="OPENAI_API_KEY"):create_openai_for_test(10)
    monkeypatch.setenv("OPENAI_API_KEY","test-key-not-a-secret")
    real_import=builtins.__import__
    def without_sdk(name,*args,**kwargs):
        if name=="openai":raise ImportError("Optional package absent")
        return real_import(name,*args,**kwargs)
    with patch("builtins.__import__",side_effect=without_sdk):
        with pytest.raises(DocumentProcessingError,match="requirements-openai.txt"):create_openai_for_test(10)


def test_openai_sdk_factory_is_lazy_and_disables_automatic_retries(monkeypatch):
    import sys
    from unittest.mock import Mock
    monkeypatch.setenv("OPENAI_API_KEY","test-key-not-a-secret")
    factory=Mock()
    with patch.dict(sys.modules,{"openai":SimpleNamespace(OpenAI=factory)}):
        assert create_openai_for_test(23.5) is factory.return_value
    factory.assert_called_once_with(timeout=23.5,max_retries=0)


def test_openai_provider_accepts_numbered_images_and_strict_structured_output(monkeypatch,capsys):
    from unittest.mock import Mock
    from src.document_processing.providers import OpenAIProvider
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    client=Mock();client.responses.create.return_value=SimpleNamespace(status="completed",output_text='{"document_type":"unknown"}')
    provider=OpenAIProvider(model="configured-vision-model",client=client)
    assert provider.generate_json(stage="classification",prompt="Classify untrusted data",schema=DocumentClassification.model_json_schema(),text="PRIVATE CONTENT",images={7:b"synthetic PNG"})=={"document_type":"unknown"}
    request=client.responses.create.call_args.kwargs
    assert request["store"] is False and request["model"]=="configured-vision-model"
    content=request["input"][1]["content"]
    assert content[1]["text"]=="[PAGE IMAGE 7]" and content[2]["image_url"].startswith("data:image/png;base64,")
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"]["additionalProperties"] is False
    assert "PRIVATE" not in capsys.readouterr().err


@pytest.mark.parametrize("kind",["student_enrollment_certificate","health_insurance_certificate","ikamet_application_form","passport_identity_page","turkiye_entry_stamp","declaration_control_document","card_fee_receipt"])
def test_openai_wire_schemas_keep_all_profiles_without_dynamic_citation_maps(kind):
    from src.document_processing.providers import _openai_schema
    from src.document_processing.schemas import DOCUMENT_SCHEMAS
    original=DOCUMENT_SCHEMAS[kind].model_json_schema()
    # Match the processor's extraction contract (Python owns audit metadata).
    computed={"missing_fields","validation_issues","source_evidence","correction_notes","selected_pages","extraction_diagnostics"}
    schema={**original,"properties":{k:v for k,v in original["properties"].items() if k not in computed}}
    wire=_openai_schema(schema)
    assert not {"source_pages","source_snippets","extraction_diagnostics"}&set(wire["properties"])
    assert set(DOCUMENT_SCHEMAS[kind].observation_fields())<=set(wire["properties"])
    assert set(wire["required"])==set(wire["properties"])


def test_native_quality_metadata_is_python_owned_and_excluded_from_observations(tmp_path):
    info,_=process_text(tmp_path,QUALITY_INSURANCE,"health_insurance_certificate",{"extraction_diagnostics":{"provider":"invented","fallback_used":True}})
    d=info.extraction_diagnostics
    assert d["provider"]=="ollama" and d["model"]==OLLAMA_MODEL
    assert d["extraction_method"]=="native_text" and not d["ocr_used"] and not d["fallback_used"]
    assert d["native_text_character_count"]==len(QUALITY_INSURANCE)
    assert d["extraction_duration_seconds"]>=0 and "classification_duration_seconds" in d
    assert "extraction_diagnostics" not in info.observation_fields()
    assert not info.validation_issues


def test_poor_ocr_is_detected_without_enabling_cloud_fallback():
    provider=FakeDocumentProvider({"classification":{"document_type":"unknown"}})
    info=process_document("scan.pdf",pages=poor_scan_pages(),provider=provider)
    assert info.document_type=="unknown" and info.extraction_diagnostics["extraction_method"]=="rapidocr"
    assert info.extraction_diagnostics["fallback_eligible_reasons"]==["poor_ocr"]
    assert not info.extraction_diagnostics["fallback_used"] and not info.extraction_diagnostics["fallback_attempted"]


def test_multimodal_recovery_uses_same_grounding_and_original_page_provenance():
    from src.workers.verification import grounded_values
    from src.core.state import Document
    local=FakeDocumentProvider({})
    cloud=FakeDocumentProvider({"transcription":{"pages":[{"page_number":3,"text":QUALITY_INSURANCE}]},"classification":{"document_type":"health_insurance_certificate"},"extraction":{"policy_number":"FABRICATED-99","passport_or_foreign_id":"FABRICATED-ID","coverage_start_date":"1310-10-20"}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={3:b"image"}) as render:
        info=process_document("scan.pdf",pages=poor_scan_pages(),provider=local,fallback_provider=cloud)
    assert not local.calls
    assert [c["stage"] for c in cloud.calls]==["transcription","classification","extraction"]
    assert info.policy_number=="TEST-POL-0001" and info.coverage_start_date==date(2026,10,1)
    assert info.passport_or_foreign_id is None
    assert info.source_pages["policy_number"]==3 and info.selected_pages==[3]
    assert any(e.field=="policy_number" and e.page_number==3 for e in info.source_evidence)
    assert info.extraction_diagnostics["fallback_used"] and info.extraction_diagnostics["extraction_method"]=="multimodal_fallback"
    assert info.extraction_diagnostics["extracted_character_count"]==len(QUALITY_INSURANCE)
    assert info.extraction_diagnostics["fallback_duration_seconds"]>=0
    assert all(c["provider"]=="openai" for c in info.extraction_diagnostics["model_calls"])
    render.assert_called_once_with("scan.pdf",[3])
    document=Document(doc_id="scan",file_path="scan.pdf",doc_type="health_insurance",status="analyzed",info=info)
    accepted=grounded_values(document)
    assert "insurance_valid_for_ikamet" not in accepted and "extraction_diagnostics" not in accepted


@pytest.mark.parametrize("rows",[[],[{"page_number":99,"text":"invented"}],[{"page_number":3,"text":""}],[{"page_number":True,"text":"invented"}],[{"page_number":3,"text":"one"},{"page_number":3,"text":"two"}]])
def test_multimodal_transcription_rejects_invented_or_missing_page_provenance(rows):
    cloud=FakeDocumentProvider({"transcription":{"pages":rows}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={3:b"image"}):
        with pytest.raises(DocumentProcessingError,match="fallback failed") as error:
            process_document("scan.pdf",pages=poor_scan_pages(),provider=FakeDocumentProvider({}),fallback_provider=cloud)
    assert error.value.stage=="transcription" and error.value.diagnostics["fallback_outcome"]=="failed"


def test_fallback_never_sends_adequate_complete_native_pdf_to_cloud():
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages") as render:
        info=process_document("native.pdf",pages=[TextPage(1,QUALITY_INSURANCE)],provider=local,fallback_provider=cloud)
    assert not cloud.calls and not info.extraction_diagnostics["fallback_attempted"]
    render.assert_not_called()


def test_fallback_is_not_used_for_txt_or_unrelated_native_pdf():
    for path in ("unrelated.pdf","unrelated.txt"):
        local=FakeDocumentProvider({"classification":{"document_type":"unknown"}})
        cloud=FakeDocumentProvider({},name="openai",images=True)
        with patch("src.document_processing.processor.render_pdf_pages") as render:
            info=process_document(path,pages=[TextPage(1,"Unrelated private text is not an administrative document")],provider=local,fallback_provider=cloud)
        assert not cloud.calls and not info.extraction_diagnostics["fallback_attempted"]
        render.assert_not_called()


def test_missing_field_fallback_cannot_invent_evidence_and_keeps_better_local_result():
    pages=[TextPage(1,"Health Insurance Certificate\nPolicy number: TEST-POL-0001")]
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{"holder_name":"FABRICATED PERSON","insurer":"FABRICATED INSURANCE","coverage_start_date":"2026-10-01"}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"image"}):
        info=process_document("native.pdf",pages=pages,provider=local,fallback_provider=cloud)
    assert info.policy_number=="TEST-POL-0001" and info.holder_name is None and info.coverage_start_date is None
    assert info.extraction_diagnostics["fallback_attempted"] and not info.extraction_diagnostics["fallback_used"]
    assert info.extraction_diagnostics["fallback_outcome"]=="no_improvement"
    assert not any(call["stage"]=="transcription" for call in cloud.calls)


def test_optional_fallback_failure_keeps_partial_local_observations():
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({"classification":DocumentProcessingError("PRIVATE failure",stage="classification")},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"image"}):
        info=process_document("native.pdf",pages=[TextPage(1,"Health Insurance Certificate")],provider=local,fallback_provider=cloud)
    assert info.extraction_diagnostics["fallback_outcome"]=="failed" and not info.extraction_diagnostics["fallback_used"]
    assert info.missing_fields


def test_fallback_page_selection_is_bounded(monkeypatch):
    monkeypatch.setattr("src.document_processing.processor.DOCUMENT_FALLBACK_MAX_PAGES",2)
    pages=sum((poor_scan_pages(n) for n in range(1,7)),[])
    cloud=FakeDocumentProvider({"transcription":{"pages":[{"page_number":n,"text":"Unrelated readable document text"} for n in (1,2)]},"classification":{"document_type":"unknown"}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"one",2:b"two"}) as render:
        info=process_document("long-scan.pdf",pages=pages,provider=FakeDocumentProvider({}),fallback_provider=cloud)
    render.assert_called_once_with("long-scan.pdf",[1,2])
    assert info.extraction_diagnostics["fallback_pages"]==[1,2]


def test_invalid_source_dates_remain_null_and_have_quality_diagnostics(tmp_path):
    info,_=process_text(tmp_path,"Health Insurance Certificate\nCoverage end: 1310-10-20","health_insurance_certificate",{"coverage_end_date":"1310-10-20"})
    assert info.coverage_end_date is None
    assert info.extraction_diagnostics["invalid_field_count"]>0
    assert "coverage_end_date" in info.extraction_diagnostics["missing_expected_fields"]


def test_document_evaluator_reports_matches_counts_and_latency_without_private_values():
    from src.document_processing.processor import evaluate_documents
    info=StudentCertificate(**STUDENT)
    info.extraction_diagnostics={"extraction_method":"native_text","missing_expected_fields":[],"invalid_field_count":0,"fallback_used":False}
    cases=[{"document_path":"PRIVATE-FILE.pdf","expected_document_type":"student_enrollment_certificate","expected_critical_fields":{"student_id":"TEST-0001","issue_date":"2026-10-01"}}]
    reports=evaluate_documents(cases,processor=lambda path:info)
    report=reports[0]
    assert report["classification_correct"] and report["critical_fields_matched"]==2
    assert report["critical_field_exact_match"]=={"student_id":True,"issue_date":True}
    assert report["missing_field_count"]==report["invalid_field_count"]==0
    assert report["end_to_end_latency_seconds"]>=0 and report["extraction_method"]=="native_text"
    assert "TEST-0001" not in json.dumps(reports) and "PRIVATE-FILE" not in json.dumps(reports)


def test_document_evaluator_isolates_failures_and_reports_classification_mismatches():
    from src.document_processing.processor import evaluate_documents
    cases=[{"document_path":"one.pdf","expected_document_type":"student_enrollment_certificate","expected_critical_fields":{"student_id":"S123"}},
           {"document_path":"two.pdf","expected_document_type":"health_insurance_certificate","expected_critical_fields":{}}]
    def run(path):
        if path.name=="one.pdf":raise DocumentProcessingError("PRIVATE API failure")
        return UnknownDocument()
    reports=evaluate_documents(cases,processor=run)
    assert reports[0]["error_type"]=="DocumentProcessingError" and not reports[0]["classification_correct"]
    assert not reports[1]["classification_correct"] and "error_type" not in reports[1]
    assert "PRIVATE" not in json.dumps(reports)


def test_document_evaluation_cli_uses_local_manifest_and_keeps_output_json(tmp_path,capsys):
    import sys
    from src.document_processing.processor import main
    manifest=tmp_path/"cases.json"
    manifest.write_text(json.dumps([{"document_path":"one.txt","expected_document_type":"unknown","expected_critical_fields":{}}]),encoding="utf-8")
    with patch.object(sys,"argv",["processor","--evaluate",str(manifest)]),patch("src.document_processing.processor.process_document",return_value=UnknownDocument()) as processor:
        main()
    assert processor.call_args.args[0]==tmp_path/"one.txt"
    assert json.loads(capsys.readouterr().out)[0]["classification_correct"]


def test_provider_environment_configuration_is_optional_and_document_specific():
    import subprocess,sys,os
    environment=dict(os.environ)
    for key in ("LLM_PROVIDER","DOCUMENT_FALLBACK_PROVIDER","OPENAI_MODEL","OPENAI_API_KEY"):environment.pop(key,None)
    script="import config;print(config.LLM_PROVIDER,config.DOCUMENT_FALLBACK_PROVIDER,config.OLLAMA_MODEL)"
    environment["OLLAMA_MODEL"]="configured-local-model"
    result=subprocess.run([sys.executable,"-c",script],env=environment,capture_output=True,text=True,check=True,timeout=30)
    assert result.stdout.strip()=="ollama none configured-local-model"
    environment.update(LLM_PROVIDER="openai",DOCUMENT_FALLBACK_PROVIDER="openai",OPENAI_MODEL="configured-vision-model")
    result=subprocess.run([sys.executable,"-c",script+";print(config.OPENAI_MODEL)"],env=environment,capture_output=True,text=True,check=True,timeout=30)
    assert result.stdout.splitlines()==["openai openai configured-local-model","configured-vision-model"]


# V2 generalization: no private real document or live transport is used.
ALT_INSURANCE = ("YABANCILAR SAGLIK SIGORTASI\nSigortali Adi Soyadi: Alex Example\n"
    "Sigorta Sirketi: Example Insurance\nPolice Numarasi: SYN-2026-002\n"
    "Baslangic Tarihi: 01.10.2026\nBitis Tarihi: 30.09.2027\n"
    "Police Turu: Ozel Saglik Sigortasi\nIkamet izni icin duzenlenmistir.")


@pytest.mark.parametrize("text", [QUALITY_INSURANCE, ALT_INSURANCE,
    ALT_INSURANCE.replace("Baslangic Tarihi: 01.10.2026\nBitis Tarihi: 30.09.2027", "Seyahat Baslangic-Bitis Tarihi: 01.10.2026 - 30.09.2027"),
    ALT_INSURANCE.replace("Baslangic Tarihi: 01.10.2026\nBitis Tarihi: 30.09.2027", "Baslangic Tarihi\nBitis Tarihi\n01.10.2026 30.09.2027")])
def test_v2_insurance_semantics_generalize_across_labels_and_columns(text):
    provider = FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"}, "extraction":{}})
    info = process_document("synthetic.txt", pages=[TextPage(1,text)], provider=provider, mode="local-only")
    assert info.holder_name == "Alex Example" and info.insurer == "Example Insurance"
    assert info.coverage_start_date == date(2026,10,1) and info.coverage_end_date == date(2027,9,30)
    assert info.policy_number in {"TEST-POL-0001", "SYN-2026-002"}
    assert info.extraction_diagnostics["extraction_route"] == "NATIVE_TEXT"
    assert info.source_pages["coverage_start_date"] == 1


def test_unicode_turkish_aliases_preserve_exact_source_snippets():
    text = ALT_INSURANCE.replace("Sigortali Adi Soyadi", "Sigortal\u0131 Ad\u0131 Soyad\u0131").replace("Baslangic Tarihi", "Ba\u015flang\u0131\u00e7 Tarihi").replace("Bitis Tarihi", "Biti\u015f Tarihi")
    info = process_document("synthetic.txt", pages=[TextPage(4,text)], provider=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"}, "extraction":{}}), mode="local-only")
    assert info.holder_name == "Alex Example" and info.coverage_end_date == date(2027,9,30)
    assert "Ba\u015flang\u0131\u00e7" in info.source_snippets["coverage_start_date"]
    assert info.source_pages["coverage_end_date"] == 4


def test_insurance_purpose_and_category_are_source_claims_not_legal_acceptance():
    from src.workers.verification import grounded_values
    from src.core.state import Document
    info = process_document("synthetic.txt", pages=[TextPage(1,ALT_INSURANCE)],
        provider=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"}, "extraction":{}}),mode="local-only")
    assert info.insurance_evidence_type == "private_health_insurance"
    assert info.residence_permit_purpose is True
    accepted = grounded_values(Document(doc_id="insurance",file_path="synthetic.txt",doc_type="health_insurance",status="analyzed",info=info))
    assert accepted["insurance_evidence_type"] == "private_health_insurance"
    assert "insurance_valid_for_ikamet" not in accepted


def test_llm_cannot_invent_policy_purpose_or_insurance_category():
    info = process_document("synthetic.txt",pages=[TextPage(1,QUALITY_INSURANCE)],provider=FakeDocumentProvider({
        "classification":{"document_type":"health_insurance_certificate"},
        "extraction":{"insurance_evidence_type":"private_health_insurance","residence_permit_purpose":True}}),mode="local-only")
    assert info.insurance_evidence_type is info.residence_permit_purpose is None
    assert "GROUNDING_FAILURE" in info.extraction_diagnostics["failure_categories"]


def test_cover_page_does_not_hide_form_heading_on_later_page():
    provider=FakeDocumentProvider({"classification":{"document_type":"unknown"},"extraction":{"applicant_name":"Alex Example"}})
    info=process_document("synthetic.pdf",pages=[TextPage(1,"Administrative cover sheet"),TextPage(2,"DECLARATION FOR SHORT TERM RESIDENCE PERMIT REGISTRATION FORM\nApplicant: Alex Example")],provider=provider,mode="local-only")
    assert info.document_type=="ikamet_application_form" and info.applicant_name=="Alex Example"
    assert info.source_pages["permit_type"]==2


def test_poor_scan_auto_fallback_records_route_and_preserves_page():
    cloud=FakeDocumentProvider({"transcription":{"pages":[{"page_number":3,"text":ALT_INSURANCE}]},
        "classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={3:b"PNG"}):
        info=process_document("synthetic.pdf",pages=poor_scan_pages(),provider=FakeDocumentProvider({}),fallback_provider=cloud,mode="auto")
    assert info.extraction_diagnostics["extraction_route"]=="OPENAI_MULTIMODAL"
    assert info.source_pages["policy_number"]==3 and info.insurance_evidence_type=="private_health_insurance"


def test_cloud_diagnostic_mode_actually_sends_images_even_for_native_pdf():
    cloud=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"PNG"}):
        info=process_document("synthetic.pdf",pages=[TextPage(1,QUALITY_INSURANCE)],provider=cloud,mode="cloud")
    assert info.extraction_diagnostics["extraction_route"]=="OPENAI_MULTIMODAL"
    assert all(request["images"]=={1:b"PNG"} for request in cloud.calls)


def test_cloud_grounding_uses_whole_selected_image_page_without_enlarging_prompt():
    prefix="Health Insurance Certificate\n" + "Nonessential details. "*350
    text=prefix+"\n"+QUALITY_INSURANCE
    cloud=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"PNG"}):
        info=process_document("synthetic.pdf",pages=[TextPage(1,text)],provider=cloud,mode="cloud",max_context_chars=1000)
    assert info.policy_number=="TEST-POL-0001"
    assert all(len(request["text"])<=1000 for request in cloud.calls)
    assert info.source_snippets["policy_number"] in text


def test_local_schema_metadata_failure_can_trigger_safe_fallback():
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{"ambiguous_fields":123}})
    cloud=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"PNG"}):
        info=process_document("synthetic.pdf",pages=[TextPage(1,QUALITY_INSURANCE)],provider=local,fallback_provider=cloud,mode="auto")
    assert info.policy_number=="TEST-POL-0001"
    assert "SCHEMA_VALIDATION_FAILURE" in info.extraction_diagnostics["failure_categories"]
    assert info.extraction_diagnostics["fallback_used"]


def test_local_classification_failure_can_trigger_fallback():
    local=FakeDocumentProvider({"classification":DocumentProcessingError("Synthetic invalid JSON",stage="classification")})
    cloud=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"PNG"}):
        info=process_document("synthetic.pdf",pages=[TextPage(1,QUALITY_INSURANCE)],provider=local,fallback_provider=cloud,mode="auto")
    assert info.holder_name=="Alex Example" and "CLASSIFICATION_FAILURE" in info.extraction_diagnostics["failure_categories"]


def test_unresolved_administrative_profile_can_trigger_fallback_but_plain_unknown_does_not():
    local=FakeDocumentProvider({"classification":{"document_type":"unknown"}})
    cloud=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={1:b"PNG"}):
        info=process_document("synthetic.pdf",pages=[TextPage(1,QUALITY_INSURANCE)],provider=local,fallback_provider=cloud,mode="auto")
    assert info.document_type=="health_insurance_certificate"
    assert "unresolved_document_profile" in info.extraction_diagnostics["fallback_trigger_reasons"]


def test_local_only_overrides_fallback_configuration_and_never_loads_cloud():
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({},name="openai",images=True)
    info=process_document("synthetic.pdf",pages=[TextPage(1,"Health Insurance Certificate")],provider=local,fallback_provider=cloud,mode="local-only")
    assert not cloud.calls and not info.extraction_diagnostics["fallback_attempted"]


def test_auto_without_configuration_keeps_safe_partial_local_result(monkeypatch):
    monkeypatch.setattr("src.document_processing.processor.cloud_document_configured",lambda:False)
    info=process_document("synthetic.pdf",pages=[TextPage(1,"Health Insurance Certificate")],
        provider=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}}),mode="auto")
    assert info.policy_number is None and not info.extraction_diagnostics["fallback_attempted"]


def test_benchmark_comparison_reports_normalized_matches_and_values_only_on_request():
    from src.document_processing.processor import evaluate_documents
    info=process_document("synthetic.txt",pages=[TextPage(1,ALT_INSURANCE)],provider=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}}),mode="local-only")
    cases=[{"document_path":"synthetic.txt","expected_document_type":"health_insurance_certificate","expected_critical_fields":{"holder_name":"ALEX   EXAMPLE","coverage_end_date":"30.09.2027"}}]
    redacted=evaluate_documents(cases,processor=lambda path:info)[0]
    assert redacted["critical_field_normalized_match"]=={"holder_name":True,"coverage_end_date":True}
    assert "Alex Example" not in json.dumps(redacted)
    detailed=evaluate_documents(cases,processor=lambda path:info,include_values=True)[0]
    assert detailed["extracted_fields"]["holder_name"]=="Alex Example"
    assert detailed["expected_fields"]["coverage_end_date"]=="30.09.2027"
    assert "source_evidence" not in json.dumps(detailed)


def test_benchmark_compare_cli_runs_all_three_modes(tmp_path,capsys):
    import sys
    from src.document_processing.processor import main
    manifest=tmp_path/"synthetic.json"
    manifest.write_text(json.dumps([{"document_path":"synthetic.txt","expected_document_type":"unknown","expected_critical_fields":{}}]),encoding="utf-8")
    with patch.object(sys,"argv",["processor","--evaluate",str(manifest),"--compare"]),patch("src.document_processing.processor.process_document",return_value=UnknownDocument()) as processor:
        main()
    assert [call.kwargs["mode"] for call in processor.call_args_list]==["local-only","auto","cloud"]
    assert len(json.loads(capsys.readouterr().out))==3


def test_complete_native_evidence_avoids_cloud_despite_irrelevant_bad_scan():
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({},name="openai",images=True)
    pages=[TextPage(1,QUALITY_INSURANCE),*poor_scan_pages(9)]
    info=process_document("mixed.pdf",pages=pages,provider=local,fallback_provider=cloud,mode="auto")
    assert info.policy_number=="TEST-POL-0001" and not cloud.calls
    assert not info.extraction_diagnostics["fallback_attempted"]


def test_partial_native_evidence_is_not_replaced_by_vision_transcription():
    native="Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\nPolicy number: SYN-2026-002"
    recovered="Coverage start: 01.10.2026\nCoverage end: 30.09.2027"
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({"transcription":{"pages":[{"page_number":3,"text":recovered}]},
        "classification":{"document_type":"health_insurance_certificate"},"extraction":{}},name="openai",images=True)
    with patch("src.document_processing.processor.render_pdf_pages",return_value={3:b"scan",1:b"native"}):
        info=process_document("mixed.pdf",pages=[TextPage(1,native),*poor_scan_pages(3)],provider=local,fallback_provider=cloud,mode="auto")
    assert cloud.calls[0]["images"]=={3:b"scan"}
    assert info.policy_number=="SYN-2026-002" and info.coverage_end_date==date(2027,9,30)
    assert info.source_pages["holder_name"]==1 and info.source_pages["coverage_end_date"]==3
    assert info.source_snippets["holder_name"] in native


def test_document_fallback_does_not_depend_on_later_legal_coverage_failure():
    expired=QUALITY_INSURANCE.replace("01.10.2026","01.10.2020").replace("30.09.2027","30.09.2021")
    local=FakeDocumentProvider({"classification":{"document_type":"health_insurance_certificate"},"extraction":{}})
    cloud=FakeDocumentProvider({},name="openai",images=True)
    info=process_document("expired.pdf",pages=[TextPage(1,expired)],provider=local,fallback_provider=cloud,mode="auto")
    assert info.coverage_end_date==date(2021,9,30) and not cloud.calls


def test_new_provider_environment_names_override_legacy_without_model_literals():
    import os,subprocess,sys
    env=dict(os.environ)
    env.update(DOCUMENT_AI_PROVIDER="auto",OPENAI_DOCUMENT_MODEL="configured-image-model",OPENAI_MODEL="legacy-image-model")
    run=subprocess.run([sys.executable,"-c","import config;print(config.DOCUMENT_AI_PROVIDER,config.OPENAI_DOCUMENT_MODEL)"],
        env=env,capture_output=True,text=True,check=True,timeout=30)
    assert run.stdout.strip()=="auto configured-image-model"


def test_auto_without_document_model_does_not_enable_cloud(monkeypatch):
    from src.document_processing import providers
    monkeypatch.setenv("OPENAI_API_KEY","synthetic-key")
    monkeypatch.setattr(providers,"OPENAI_DOCUMENT_MODEL","")
    assert not providers.cloud_document_configured()


def test_cloud_transport_failure_is_controlled_and_logs_no_private_error(capsys):
    from unittest.mock import Mock
    from src.document_processing.providers import OpenAIProvider
    client=Mock();client.responses.create.side_effect=ConnectionError("PRIVATE REQUEST DETAILS")
    provider=OpenAIProvider(model="configured-image-model",client=client)
    with pytest.raises(DocumentProcessingError,match="request failed"):
        provider.generate_json(stage="extraction",prompt="Untrusted",schema=DocumentClassification.model_json_schema(),text="PRIVATE SOURCE",images={1:b"png"})
    assert "PRIVATE" not in capsys.readouterr().err
