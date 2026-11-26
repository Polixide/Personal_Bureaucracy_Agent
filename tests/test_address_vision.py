"""Exercise the real document pipeline against a synthetic Address Form."""
import json

import pytest

import config
from scripts.benchmark_address_vision import EXPECTED, PAGES, TERRA, make_pdf, mocked_openai_client
from src.core.llm_diagnostics import collect_model_calls
from src.core.model_provider import AIOptions, use_ai_options
from src.core.state import Document, UserEvent
from src.document_processing.extractor import TextPage
from src.document_processing.processor import process_document
from src.document_processing.providers import OpenAIProvider
from src.workers.document_analyzer import document_analyzer
from src.workers.verification import document_review, grounded_values
from test_procedure import override
from test_ui_workspaces import choose, entered, ui_case


def test_local_ocr_retains_spelling_for_explicit_review(monkeypatch, tmp_path):
    path = tmp_path / "synthetic-address.pdf"
    make_pdf(path)
    ocr_lines = "\n".join(PAGES[0]).replace("BELLARIVA", "BELlARIVA")
    monkeypatch.setattr("src.document_processing.extractor.ocr_page", lambda *a, **kw: ocr_lines)
    with collect_model_calls() as calls:
        result = process_document(path, mode="local-only")
    assert not calls
    assert result.model_dump(mode="json", include=set(EXPECTED)) == {**EXPECTED, "last_name": "BELlARIVA"}
    assert result.source_pages["passport_number"] == 1 and result.source_pages["declared_address"] == 2
    assert result.extraction_diagnostics["field_read_methods"]["passport_number"] == "pdf_form"
    assert result.signature_present is None
    address = Document(doc_id="synthetic-form", doc_type="address_form", file_path=str(path), status="analyzed", info=result)
    assert "last_name" not in grounded_values(address)
    review = {row["name"]: row for row in document_review({}, address, full=True)["fields"]}
    assert review["last_name"]["observation"] == "BELlARIVA"
    assert review["last_name"]["status"] == "needs_confirmation"


@pytest.mark.parametrize(("explicit_type", "image_pages"), [(False, [1]), (True, [1, 2])])
def test_real_sdk_serializes_page_images_and_returns_all_grounded_observations(tmp_path, monkeypatch, explicit_type, image_pages):
    monkeypatch.setattr(config, "OPENAI_REASONING_EFFORT", "low")
    path = tmp_path / "synthetic-address.pdf"
    make_pdf(path)
    wire = []
    with mocked_openai_client(wire) as client, collect_model_calls() as calls:
        info = process_document(path, mode="cloud", prefer_native=True,
                                document_type_hint="address_form" if explicit_type else None,
                                provider=OpenAIProvider(model=TERRA, client=client))
    assert info.model_dump(mode="json", include=set(EXPECTED)) == EXPECTED
    assert info.signature_present is None and info.selected_pages == [1, 2]
    assert len(wire) == len(calls) == 1
    assert wire[0]["model_requested_on_wire"] == TERRA and wire[0]["reasoning"] == "low"
    assert wire[0]["page_markers"] == [f"[PAGE IMAGE {number}]" for number in image_pages]
    assert wire[0]["images"] == [{"detail": "high", "size": (1653, 2339)}] * len(image_pages)
    call = calls[0]
    assert call.response_model == TERRA and call.structured_output_valid is True
    assert [image["page"] for image in call.image_inputs] == image_pages
    assert (call.image_count, call.prompt_tokens, call.completion_tokens, call.total_tokens) == (len(image_pages), 540, 190, 730)
    assert info.extraction_diagnostics["pdf_page_count"] == 2


def test_conflicting_or_unreadable_values_remain_unresolved(tmp_path):
    path = tmp_path / "synthetic-address.pdf"
    make_pdf(path)
    altered = (
        tuple(line.replace("BELLARIVA", "[unreadable]") for line in PAGES[0]),
        tuple(line.replace("Example Hall 9", "Example Hall 19") for line in PAGES[1]),
    )
    with mocked_openai_client([], transcribed_pages=altered) as client:
        info = process_document(path, mode="cloud", document_type_hint="address_form",
                                provider=OpenAIProvider(model=TERRA, client=client))
    assert info.last_name is None and info.declared_address is None
    assert {"last_name", "declared_address"} <= set(info.ambiguous_fields)
    assert {x.original_value for x in info.source_evidence if x.field == "declared_address"} == {
        "Example Hall 9, Sample District", "Example Hall 19, Sample District",
    }
    assert info.signature_present is None


def test_type_hint_alone_does_not_upgrade_unrelated_evidence():
    info = process_document("synthetic.txt", document_type_hint="address_form",
                            pages=[TextPage(1, "Unrelated form\nName: Alex\nPassport: TEST123\nSigned: yes")])
    assert info.document_type == "address_form"
    assert info.first_name is None and info.signature_present is None


def test_local_cache_reuse_has_no_current_model_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "synthetic-address.txt"
    path.write_text("Address Form\nFirst Name: RIVA\nSurname: BELLARIVA", encoding="utf-8")
    event = UserEvent(kind="analyze_document", file_path=str(path))
    state = {"user_id": "synthetic", "event": event, "documents": [Document(doc_id="form", file_path=str(path))]}
    # A cloud choice for a different file cannot change the mode of this event.
    with use_ai_options(AIOptions(cloud_document=("analyze_document", str(tmp_path / "other.pdf")))):
        first = document_analyzer(state)["documents"][0]
        second = document_analyzer(state)["documents"][0]
    assert first.info == second.info.model_copy(update={"extraction_diagnostics": first.info.extraction_diagnostics})
    assert first.analysis_diagnostics["selected_provider"] == "local"
    assert second.analysis_diagnostics["cache_hit"] is True
    assert second.analysis_diagnostics["requests"] == []
    assert second.analysis_diagnostics["image_count"] == 0
    report = json.dumps(second.analysis_diagnostics)
    assert "BELLARIVA" not in report and "synthetic-address" not in report

def test_streamlit_proves_file_authorization_persistence_and_no_requests_on_rerun(
    ui_case, tmp_path, monkeypatch,
):
    from src.core.state import ProcedureState

    monkeypatch.setattr(config, "OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setattr("src.ui.model_settings.cloud_configuration_issue", lambda *args: None)
    app = ui_case.app
    app.session_state[f"selected_document_provider_{ui_case.key}"] = "cloud"
    path = tmp_path / "synthetic-address.pdf"
    make_pdf(path)
    original_pdf = path.read_bytes()
    old = process_document("synthetic.txt", pages=[TextPage(1, "Address Form\nSurname: OLDREADING")])
    document = Document(doc_id="address-fixture", doc_type="address_form", file_path=str(path), status="analyzed", info=old)
    state = {"procedure": ProcedureState(), "documents": [document], "user_id": "demo-user", "trace_id": "synthetic"}
    state = override(state, document.doc_id, "last_name", "CONFIRMEDNAME")
    ui_case.seed(entered(state))
    requests = []
    from src.document_processing.processor import get_document_provider

    with mocked_openai_client(requests) as client:
        def document_provider(name, **options):
            if name == "openai":
                return OpenAIProvider(model=TERRA, client=client)
            return get_document_provider(name, **options)

        monkeypatch.setattr("src.document_processing.processor.get_document_provider", document_provider)
        ui_case.nav("documents")
        choose(app, "selectbox", "Saved document", document.doc_id)
        app.run()
        assert not requests
        assert "OpenAI Vision is available for this PDF" in ui_case.text()
        choose(app, "toggle", "Document options", True)
        app.run()
        assert not requests
        ui_case.click("Analyze this document with OpenAI Vision")
        saved = ui_case.read()
        updated = saved["documents"][0]
        assert updated.status == "analyzed"
        assert updated.info.model_dump(mode="json", include=set(EXPECTED)) == EXPECTED
        assert updated.prior_observations == {"last_name": "OLDREADING"}
        from src.memory.twin_store import TwinStore
        with TwinStore(tmp_path / "ui.db") as store:
            assert store.load_twin("demo-user")["documents"] == saved["documents"]
        assert saved["verification_records"][document.doc_id]["last_name"].value == "CONFIRMEDNAME"
        assert updated.analysis_diagnostics["image_pages"] == [1, 2]
        assert updated.analysis_diagnostics["requests"][0]["total_tokens"] == 730
        assert len(requests) == 1
        assert "OpenAI Vision analyzed PDF page images" in ui_case.text()
        assert "Readings changed by reanalysis" in ui_case.text()
        html = "\n".join(element.proto.body for element in app.get("html"))
        assert "CONFIRMEDNAME" in html and "BELLARIVA" in html and "Original reading" in html
        assert EXPECTED["declared_address"] in html and EXPECTED["passport_number"] in html and EXPECTED["phone"] in html
        assert "Filled PDF field" in html
        ui_case.nav("technical")
        choose(app, "selectbox", "Advanced", "Document diagnostics")
        app.run()
        rendered = "\n".join(element.value for element in app.json)
        assert TERRA in rendered and "730" in rendered and "2339" in rendered
        assert "CONFIRMEDNAME" not in rendered and "BELLARIVA" not in rendered
        assert "synthetic-address.pdf" not in rendered and "TEST990001" not in rendered
        assert "N/A" in rendered and "data:image" not in rendered
        app.run()
        ui_case.healthy()
        metrics = app.session_state[f"ai_session_metrics_{ui_case.key}"]
        assert metrics["openai_requests"] == 1
        assert len(requests) == 1 and ui_case.read()["trace_id"] == saved["trace_id"]
    assert path.read_bytes() == original_pdf


def test_failed_vision_attempt_preserves_saved_readings_and_has_safe_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "synthetic-address.pdf"
    make_pdf(path)
    old = process_document("synthetic.txt", pages=[TextPage(1, "Address Form\nSurname: OLDREADING")])
    document = Document(doc_id="form", doc_type="address_form", file_path=str(path), status="pending", info=old)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        status="incomplete", model=TERRA, output_text="synthetic private error text", usage=None,
    )))
    monkeypatch.setattr("src.document_processing.processor.get_document_provider",
                        lambda *a, **kw: OpenAIProvider(model=TERRA, client=client))
    event = UserEvent(kind="reprocess_document", document_id=document.doc_id, document_type_hint="address_form")
    with use_ai_options(AIOptions(cloud_document=(event.kind, document.doc_id))), collect_model_calls() as calls:
        updated = document_analyzer({"documents": [document], "event": event})["documents"][0]
    assert updated.status == "rejected" and updated.info == old
    report = updated.analysis_diagnostics
    assert report["selected_provider"] == "openai" and report["outcome"] == "failed"
    assert report["image_pages"] == [1, 2] and len(calls) == 1
    request = report["requests"][0]
    assert request["outcome"] == "invalid_output" and request["structured_output_valid"] is False
    assert request["total_tokens"] is None
    assert "synthetic private error text" not in json.dumps(report)
    from src.memory.twin_store import TwinStore

    without_prior_result = updated.model_copy(update={"info": None})
    with TwinStore(":memory:") as store:
        store.save_twin({"user_id": "synthetic", "documents": [without_prior_result]})
        assert store.load_twin("synthetic")["documents"] == [without_prior_result]
