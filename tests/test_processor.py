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
    info, _ = process_text(tmp_path, "Student Enrollment Certificate\nHolder name: Alex Example", "student_enrollment_certificate", {**STUDENT,"student_id":None,"issue_date":None,"missing_fields":["invented"],"validation_issues":["model claim"]})
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
