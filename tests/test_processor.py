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
    assert calls[1].kwargs["format"] == StudentCertificate.model_json_schema()
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
        assert captured.err == ""


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
    plan=ActionPlan(actions=[PlanStep(step_id="review",priority=1,action="Review policy details",reason="Check the grounded observations",required_documents=["health_insurance"])],summary="Review proposed observations")
    assert not validate_plan({"documents":[document],"relevant_requirements":[]},plan)
