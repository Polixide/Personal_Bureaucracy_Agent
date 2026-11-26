"""Synthetic document routes, explicit tax-purpose checks and persisted evidence."""
from datetime import date
import json
from pathlib import Path

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from scripts.benchmark_address_vision import TERRA, mocked_openai_client
from src.core.graph import build_graph
from src.core.model_provider import AIOptions, use_ai_options
from src.core.procedure import stage_document_checklist
from src.core.state import UserEvent, new_run
from src.document_processing.analysis_diagnostics import document_analysis_findings
from src.document_processing.extractor import TextPage
from src.document_processing.processor import get_document_provider, process_document
from src.document_processing.providers import OpenAIProvider
from src.document_processing.schemas import CardFeeReceipt, TaxIDCertificate
from src.memory.checkpointer import sqlite_checkpointer
from src.memory.twin_store import TwinStore
from src.workers.verification import MILESTONE_SCOPE, conflict_details, document_review, milestone_checklist
from test_milestone_documents import SAMPLES, analyze, confirm
from test_ui_workspaces import choose, entered, ui_case


PASSPORT = "P1234567"
TAX_ID = "0000000123"
ADDRESS = "Address Form\nFirst Name: Alex\nSurname: Example\nDeclared address: Synthetic Street 5\nSigned: Yes"
FEE = (
    "VERGİ DAİRESİ ALINDISI",
    "Vergi türü: 9207 — İkametgâh Tezkeresi Defter Satış Bedeli",
    "Mükellef Adı Soyadı: Alex Example",
    f"VKN: {TAX_ID}",
    "Ödeme Tarihi: 12.11.2026",
    "Alındı No.: AL-2026-101",
    "Ödenen tutar: 1.234,50 TL",
)
ASSIGNMENT = (
    "GELİR İDARESİ BAŞKANLIĞI",
    "POTANSİYEL VERGİ KİMLİK NUMARASI",
    f"Vergi Kimlik Numarası: {TAX_ID}",
    "Adı Soyadı: Alex Example",
    f"Pasaport No.: {PASSPORT}",
    "Tahsis Tarihi: 11.11.2026",
    "Belge Numarası: VKN-10023",
)


def scan(path, pages=1):
    with pymupdf.open() as source, pymupdf.open() as scans:
        for i in range(pages):
            p = source.new_page()
            p.insert_text((70, 85), f"Offline synthetic PDF image {i+1}")
            scans.new_page().insert_image(p.rect, pixmap=p.get_pixmap(dpi=130))
        scans.save(path)


def install_cloud_provider(monkeypatch, client):
    monkeypatch.setattr("src.document_processing.processor.get_document_provider",
                        lambda name, **kwargs: OpenAIProvider(model=TERRA, client=client) if name == "openai"
                        else get_document_provider(name, **kwargs))


def cloud_upload(graph, config, path):
    event = UserEvent(kind="analyze_document", file_path=str(path), original_filename=path.name)
    with use_ai_options(AIOptions(cloud_document=(event.kind, event.file_path))):
        return graph.invoke(new_run(event, config["configurable"]["thread_id"], scope=MILESTONE_SCOPE), config)


@pytest.mark.parametrize("line", [f"Passport number: {PASSPORT}", f"{PASSPORT}\n(Passport Number)",
                                 f"Pasaport No. / Passport Number: {PASSPORT}"])
def test_address_passport_is_read_only_from_the_labelled_source(line):
    info = process_document("synthetic.txt", pages=[TextPage(1, ADDRESS), TextPage(2, line)])
    assert info.passport_number == PASSPORT
    assert info.source_pages["passport_number"] == 2
    evidence = next(item for item in info.source_evidence if item.field == "passport_number")
    assert evidence.original_value == PASSPORT and PASSPORT in evidence.snippet


def test_address_vision_recovers_detached_passport_caption_on_later_page(tmp_path):
    path = tmp_path / "synthetic-address.pdf"
    scan(path, pages=2)
    pages = (tuple(ADDRESS.splitlines()), ("Nationality: Exampleland", PASSPORT, "(Passport Number)"))
    wire = []
    with mocked_openai_client(wire, transcribed_pages=pages) as client:
        info = process_document(path, mode="cloud", prefer_native=True, provider=OpenAIProvider(model=TERRA, client=client))
    assert info.passport_number == PASSPORT and info.source_pages["passport_number"] == 2
    assert info.selected_pages == [1, 2]
    assert len(wire) == 1 and wire[0]["page_markers"] == ["[PAGE IMAGE 1]", "[PAGE IMAGE 2]"]


@pytest.mark.parametrize("line", ["", "Passport Number:", f"Yabancı Kimlik No:\n{TAX_ID}\n(Passport Number)"])
def test_absent_address_passport_stays_distinct_from_case_reference(tmp_path, line):
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "missing-address-passport"}}
    analyze(graph, cfg, tmp_path, "passport_identity_page")
    state = analyze(graph, cfg, tmp_path, "address_form", text=ADDRESS + "\n" + line)
    doc = next(item for item in state["documents"] if item.doc_type == "address_form")
    assert doc.info.passport_number is None and "passport_number" not in doc.info.source_pages
    assert "passport_number" not in state["verification_records"][doc.doc_id]
    assert state["residence_permit_context"].passport_or_foreign_id == PASSPORT
    from src.ui.documents import _passport_reference
    assert _passport_reference(state, doc, "passport_number") == PASSPORT
    assert "passport_number" in doc.analysis_diagnostics["fields_not_found_in_processed_source"]


def test_address_ocr_conflict_and_correction_preserve_both_observations(tmp_path, monkeypatch):
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "address-ocr"}}
    analyze(graph, cfg, tmp_path, "passport_identity_page")
    path = tmp_path / "synthetic-address.pdf"
    scan(path)
    text = ADDRESS + "\nP7654321\n(Passport Number)"
    monkeypatch.setattr("src.document_processing.extractor.ocr_page", lambda *a, **kw: text)
    state = graph.invoke(new_run(UserEvent(kind="analyze_document", file_path=str(path)), "m1-user", scope=MILESTONE_SCOPE), cfg)
    doc = next(item for item in state["documents"] if item.doc_type == "address_form")
    review = document_review(state, doc)
    assert review["status"] == "conflict"
    assert [row["name"] for row in review["fields"] if row["issue"]] == ["passport_number"]
    assert {row["document_type"] for row in conflict_details(state)["passport_or_foreign_id"]["observations"]} == {"address_form", "passport_copy"}
    corrected = confirm(graph, cfg, doc.doc_id, {"passport_number": PASSPORT})
    record = corrected["verification_records"][doc.doc_id]["passport_number"]
    assert not corrected["verification_conflicts"]
    assert next(item for item in corrected["documents"] if item.doc_id == doc.doc_id).info.passport_number == "P7654321"
    monkeypatch.setattr("src.document_processing.extractor.ocr_page", lambda *a, **kw: ADDRESS + f"\n{PASSPORT}\n(Passport Number)")
    event = UserEvent(kind="reprocess_document", document_id=doc.doc_id, document_type_hint="address_form")
    state = graph.invoke(new_run(event, "m1-user", scope=MILESTONE_SCOPE), cfg)
    updated = next(item for item in state["documents"] if item.doc_id == doc.doc_id)
    assert state["verification_records"][doc.doc_id]["passport_number"] == record
    assert updated.info.passport_number == PASSPORT
    assert updated.prior_observations["passport_number"] == "P7654321"


@pytest.mark.parametrize("heading", ["VERGİ DAİRESİ ALINDISI", "TAHSİLAT ALINDISI", "Vergi Dairesi E-Alındı"])
def test_9207_scan_pairs_clear_ocr_columns_and_leaves_weak_date_unknown(tmp_path, monkeypatch, heading):
    path = tmp_path / "receipt.pdf"
    scan(path)
    lines = (
        heading, "ABC2026100000101", "Alındı No.", "Ödeme Tarihi", "2026/11/1O",
        "Tutarı", "Gecikme zammı", "9207-İKAMETG.TEZK.DFT.SAT.BED.", "#934,50#", "#0,00#",
        f"VKN: {TAX_ID}", "Mükellef Adı Soyadı: Alex Example",
    )
    boxes = (
        (200, 10, 320, 24), (320, 30, 440, 48), (190, 34, 279, 50), (190, 55, 279, 70), (320, 55, 450, 70),
        (300, 90, 360, 106), (430, 90, 550, 106), (10, 117, 270, 132), (320, 117, 399, 132), (465, 117, 500, 132),
        (30, 135, 151, 155), (30, 162, 240, 180),
    )

    def local_ocr(_page, _number, *, metadata):
        metadata.update(ocr_confidence=0.9, ocr_bounds=boxes, low_confidence_lines=(lines[4],))
        return "\n".join(lines)

    monkeypatch.setattr("src.document_processing.extractor.ocr_page", local_ocr)
    info = process_document(path, mode="local-only")
    assert isinstance(info, CardFeeReceipt)
    assert info.payment_code == "9207" and info.receipt_present is True
    assert (info.reference_number, info.amount, info.taxpayer_id, info.holder_name) == ("ABC2026100000101", "934,50", TAX_ID, "Alex Example")
    assert info.payment_date is None and "payment_date" in info.ambiguous_fields
    assert all(info.source_pages[field] == 1 for field in ("reference_number", "amount", "taxpayer_id"))
    assert next(e.original_value for e in info.source_evidence if e.field == "amount") == "#934,50#"


def test_9207_vision_persists_payment_and_counts_for_r010(ui_case, tmp_path, monkeypatch):
    path = tmp_path / "synthetic-payment.pdf"
    scan(path, pages=2)
    wire = []
    ui_case.seed(entered(new_run(UserEvent(kind="reevaluate"), "demo-user", scope=MILESTONE_SCOPE)))
    with mocked_openai_client(wire, transcribed_pages=(FEE[:4], FEE[4:])) as client:
        install_cloud_provider(monkeypatch, client)
        with sqlite_checkpointer(tmp_path / "ui.db") as checkpointer:
            state = cloud_upload(build_graph(checkpointer), ui_case.config, path)
    doc = state["documents"][0]
    assert isinstance(doc.info, CardFeeReceipt)
    assert doc.info.payment_date == date(2026, 11, 12)
    assert doc.info.reference_number == "AL-2026-101" and doc.info.taxpayer_id == TAX_ID
    assert doc.info.amount == "1.234,50 TL" and doc.info.payment_code == "9207"
    assert doc.info.source_pages["payment_date"] == 2 and doc.info.source_pages["taxpayer_id"] == 1
    assert any(rule.rule_id == "R010" and rule.passed is True for rule in state["relevant_requirements"])
    assert state["residence_permit_context"].fee_receipt_present is True
    assert state["residence_permit_context"].foreigner_id is None
    assert len(wire) == 1
    assert doc.analysis_diagnostics["image_pages"] == [1, 2]
    assert doc.analysis_diagnostics["requests"][0]["structured_output_valid"] is True
    assert doc.analysis_diagnostics["total_tokens"] == 730
    with TwinStore(":memory:") as store:
        store.save_twin(state)
        assert store.load_twin("demo-user")["documents"][0] == doc
    assert ui_case.read()["documents"][0].info == doc.info

    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reopened.session_state["workspace_profile_id"] = "demo-user"
    reopened.session_state[f"view_{ui_case.key}"] = "documents"
    reopened.run()
    choose(reopened, "selectbox", "Saved document", doc.doc_id)
    reopened.run()
    cards = "\n".join(item.proto.body for item in reopened.get("html"))
    assert all(word in cards for word in ("9207", "AL-2026-101", TAX_ID, "Payer or taxpayer name"))
    assert len(wire) == 1
    reopened.session_state[f"view_{ui_case.key}"] = "technical"
    reopened.run()
    choose(reopened, "selectbox", "Advanced", "Document diagnostics")
    reopened.run()
    assert "Extraction completed" in [row.value for row in reopened.markdown]
    diagnostics = "\n".join(row.value for row in reopened.json)
    assert all(word not in diagnostics for word in ("AL-2026-101", TAX_ID, "Alex Example", path.name))


@pytest.mark.parametrize("lines", [
    ("VERGİ DAİRESİ ALINDISI", "Vergi Kodu: 9999", "Ödeme Tarihi: 12.11.2026", f"VKN: {TAX_ID}"),
    ("VERGİ DAİRESİ ALINDISI", "Alındı No.: 9207", "Tutar:", "9207", "Vergi Kodu: 9999"),
])
def test_other_tax_office_receipts_remain_unsupported(lines):
    info = process_document("other.txt", pages=[TextPage(1, "\n".join(lines))], document_type_hint="card_fee_receipt")
    assert info.document_type == "unknown"


def test_fee_label_and_taxpayer_id_do_not_establish_a_transaction():
    info = process_document("synthetic.txt", pages=[TextPage(1, "\n".join(FEE[:4]))])
    assert isinstance(info, CardFeeReceipt)
    assert info.receipt_present is None and info.reference_number is None
    assert info.taxpayer_id == TAX_ID


@pytest.mark.parametrize("missing", [False, True])
def test_genuine_tax_assignment_pdf_is_only_supporting_evidence(tmp_path, monkeypatch, missing):
    path = tmp_path / "tax-assignment.pdf"
    scan(path)
    wire = []
    pages = (ASSIGNMENT[:2] if missing else ASSIGNMENT,)
    with mocked_openai_client(wire, transcribed_pages=pages) as client:
        install_cloud_provider(monkeypatch, client)
        graph = build_graph()
        cfg = {"configurable": {"thread_id": "assignment"}}
        state = cloud_upload(graph, cfg, path)
    doc = state["documents"][0]
    assert isinstance(doc.info, TaxIDCertificate)
    assert doc.info.tax_identification_number == (None if missing else TAX_ID)
    fields = {row["name"] for row in document_review(state, doc)["fields"] if row["issue"]}
    assert fields == ({"holder_name", "tax_identification_number"} if missing else set())
    if not missing:
        assert doc.info.issue_date == date(2026, 11, 11) and doc.info.issuing_authority
        assert doc.info.document_reference == "VKN-10023"
        assert doc.info.source_pages["passport_number"] == 1
    assert state["residence_permit_context"].fee_receipt_present is None
    assert state["residence_permit_context"].passport_or_foreign_id is None
    assert next(r for r in state["relevant_requirements"] if r.rule_id == "R010").passed is not True
    assert "tax_id_certificate" not in {row["document_type"] for row in milestone_checklist(state)}
    assert "tax_id_certificate" not in {row["document_type"] for row in stage_document_checklist(state)}
    assert len(wire) == 1
    with TwinStore(":memory:") as store:
        store.save_twin(state)
        assert store.load_twin("assignment")["documents"][0].info == doc.info


def test_optional_assignment_passport_uses_existing_identity_conflict_check(tmp_path):
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "supporting-identity"}}
    analyze(graph, cfg, tmp_path, "passport_identity_page")
    different = tuple(line.replace(PASSPORT, "P7654321") for line in ASSIGNMENT)
    state = analyze(graph, cfg, tmp_path, "tax_id_certificate", text="\n".join(different))
    rows = conflict_details(state)["passport_or_foreign_id"]["observations"]
    assert {row["document_type"] for row in rows} == {"tax_id_certificate", "passport_copy"}
    assert state["residence_permit_context"].fee_receipt_present is None


def test_manual_tax_hint_cannot_turn_arbitrary_text_into_evidence():
    info = process_document("synthetic.txt", document_type_hint="tax_id_certificate", pages=[TextPage(1, f"Shopping account\nVKN: {TAX_ID}")])
    assert info.document_type == "unknown"


@pytest.mark.parametrize(("network_error", "finding"), [("connection", "API connection error"), ("timeout", "API timeout")])
def test_failed_vision_request_has_different_diagnostics_from_unsupported_document(tmp_path, monkeypatch, network_error, finding):
    import httpx2
    import openai

    path = tmp_path / "synthetic-scan.pdf"
    scan(path)
    cfg = {"configurable": {"thread_id": f"failure-{network_error}"}}

    def failed(request):
        error_type = httpx2.ConnectError if network_error == "connection" else httpx2.ReadTimeout
        raise error_type("SYNTHETIC SDK PRIVATE PAYLOAD", request=request)

    with openai.OpenAI(api_key="synthetic-offline", max_retries=0, http_client=httpx2.Client(
        transport=httpx2.MockTransport(failed)
    )) as client:
        install_cloud_provider(monkeypatch, client)
        doc = cloud_upload(build_graph(), cfg, path)["documents"][0]
    assert doc.status == "rejected"
    assert doc.analysis_diagnostics["failure_category"] == network_error
    assert doc.analysis_diagnostics["processing_mode"] == "cloud vision"
    assert doc.analysis_diagnostics["image_pages"] == [1]
    assert document_analysis_findings(doc.info, doc.analysis_diagnostics)[1] == [finding]
    assert "PRIVATE PAYLOAD" not in json.dumps(doc.analysis_diagnostics)

    unknown = ("VERGİ DAİRESİ ALINDISI", "Vergi Kodu: 9999", "Makbuz No.: ADMIN-101")
    with mocked_openai_client([], transcribed_pages=(unknown,)) as client:
        install_cloud_provider(monkeypatch, client)
        doc = cloud_upload(build_graph(), cfg, path)["documents"][0]
    assert doc.status == "analyzed" and doc.doc_type == "unknown"
    assert document_analysis_findings(doc.info, doc.analysis_diagnostics)[1] == ["Unsupported document type"]

def test_conflicting_transaction_references_do_not_by_themselves_accept_fee_receipt():
    readings = (*FEE[:4], "Alındı No.: AL-2026-101", "Alındı No.: AL-2026-102")
    info = process_document("synthetic.txt", pages=[TextPage(1, "\n".join(readings))])
    assert isinstance(info, CardFeeReceipt)
    assert info.reference_number is None and "reference_number" in info.ambiguous_fields
    assert info.receipt_present is None
