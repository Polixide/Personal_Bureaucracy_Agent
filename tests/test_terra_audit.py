"""Synthetic verification of the active launcher, real SDK payloads and public assets."""
import base64
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from src.core.llm_diagnostics import collect_model_calls
from src.core.model_provider import AIOptions, OpenAIRequestError, RequestBudget, openai_response, use_ai_options
from src.document_processing.processor import process_document
from src.document_processing.providers import OpenAIProvider
from src.rag.public_sources import load_public_sources
from src.rag.retriever import retrieval_status, search_official_sources
from test_procedure import ready_bundle
from test_ui_workspaces import choose, entered, ui_case


TERRA = "gpt-6.1-terra"
TEST_PAGES = [
    "Health Insurance Certificate\nInsured: Alex Example\nInsurer: Example Insurance\n"
    "Policy number: TEST-POL-0001\nCoverage start: 01.10.2026",
    "Health Insurance Certificate continued\nCoverage end: 30.09.2027\nSynthetic test page for this policy.",
]


def _response(payload):
    return {
        "id": "resp_synthetic", "object": "response", "created_at": 0, "model": TERRA, "status": "completed",
        "output": [{"id": "msg_synthetic", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}]}],
        "usage": {"input_tokens": 41, "output_tokens": 12, "total_tokens": 53,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 4}},
    }


def _sdk(handler):
    sdk = pytest.importorskip("openai")
    http = pytest.importorskip("httpx2")
    return sdk.OpenAI(api_key="test-synthetic-no-remote", max_retries=0,
                      http_client=http.Client(transport=http.MockTransport(handler)))


def test_installed_openai_sdk_serializes_terra_and_low_effort(monkeypatch):
    http = pytest.importorskip("httpx2")
    observed = []
    monkeypatch.setattr("config.OPENAI_REASONING_EFFORT", "low")

    def respond(request):
        assert request.url.path == "/v1/responses"
        observed.append(json.loads(request.content))
        return http.Response(200, json=_response({"answer": "synthetic"}))

    with _sdk(respond) as client, collect_model_calls() as calls, use_ai_options(AIOptions(
        provider="openai", budget=RequestBudget(1)
    )):
        output = openai_response(model=TERRA, task="synthetic_text", client=client, max_output_tokens=120,
            messages=[{"role": "user", "content": "Synthetic question"}],
            schema={"type": "object", "properties": {"answer": {"type": "string"}}})
    assert json.loads(output) == {"answer": "synthetic"}
    assert observed[0]["model"] == TERRA and observed[0]["reasoning"] == {"effort": "low"}
    assert observed[0]["store"] is False and observed[0]["text"]["format"]["strict"] is True
    assert len(calls) == 1 and calls[0].response_model == TERRA
    assert (calls[0].total_tokens, calls[0].reasoning_tokens, calls[0].image_count) == (53, 4, 0)


def test_model_access_failure_has_a_safe_dedicated_error(capsys):
    http = pytest.importorskip("httpx2")

    def missing(request):
        return http.Response(404, json={"error": {"message": "SYNTHETIC PRIVATE ERROR PAYLOAD",
                                                  "code": "model_not_found", "type": "invalid_request_error"}})

    with _sdk(missing) as client, collect_model_calls() as calls, pytest.raises(OpenAIRequestError) as failure:
        openai_response(model=TERRA, task="missing_model", messages=[{"role": "user", "content": "Synthetic"}],
                        schema={"type": "object", "properties": {}}, max_output_tokens=100, client=client)
    assert failure.value.code == "model_unavailable" and len(calls) == 1
    assert "PRIVATE ERROR PAYLOAD" not in str(failure.value) + capsys.readouterr().err


def _synthetic_pdf(path, scanned):
    with pymupdf.open() as native, pymupdf.open() as scans:
        for content in TEST_PAGES:
            page = native.new_page()
            page.insert_text((70, 75), content, fontsize=12)
            if scanned:
                scans.new_page().insert_image(page.rect, pixmap=page.get_pixmap(dpi=120))
        (scans if scanned else native).save(path)
    with pymupdf.open(path) as pdf:
        assert len(pdf) == 2 and all(bool(page.get_text().strip()) != scanned for page in pdf)


@pytest.mark.parametrize("scanned", [False, True])
def test_authorized_native_and_image_only_pdf_through_actual_sdk(tmp_path, monkeypatch, scanned):
    http = pytest.importorskip("httpx2")
    monkeypatch.setattr("config.OPENAI_REASONING_EFFORT", "low")
    path = tmp_path / "synthetic-insurance.pdf"
    _synthetic_pdf(path, scanned)
    requests = []

    def respond(request):
        wire = json.loads(request.content)
        requests.append(wire)
        assert wire["model"] == TERRA and wire["reasoning"] == {"effort": "low"}
        stage = wire["text"]["format"]["name"]
        content = wire["input"][1]["content"]
        images = [item for item in content if item["type"] == "input_image"]
        if stage == "document_transcription":
            assert len(images) == 2
            assert [part["text"] for part in content if part.get("text", "").startswith("[PAGE IMAGE")] == [
                "[PAGE IMAGE 1]", "[PAGE IMAGE 2]"]
            assert all(image["detail"] == "high" and base64.b64decode(
                image["image_url"].split(",", 1)[1]).startswith(b"\x89PNG") for image in images)
            answer = {"pages": [{"page_number": index, "text": text} for index, text in enumerate(TEST_PAGES, 1)]}
        elif stage == "document_classification":
            assert not images
            assert "Health Insurance Certificate" in content[0]["text"]
            answer = {"document_type": "health_insurance_certificate"}
        else:
            assert stage == "document_extraction" and not images
            answer = {"passport_or_foreign_id": "FABRICATED-SYNTHETIC-NOT-IN-PAGES"}
        return http.Response(200, json=_response(answer))

    with _sdk(respond) as client, collect_model_calls() as calls:
        info = process_document(path, mode="cloud", prefer_native=True, provider=OpenAIProvider(model=TERRA, client=client))
    assert info.holder_name == "Alex Example"
    assert (info.coverage_start_date, info.coverage_end_date) == (date(2026, 10, 1), date(2027, 9, 30))
    assert info.source_pages["coverage_end_date"] == 2 and info.passport_or_foreign_id is None
    vision = [call for call in calls if call.image_count]
    assert len(vision) == int(scanned)
    if scanned:
        assert vision[0].task == "document_transcription" and vision[0].image_count == 2
        assert vision[0].structured_output_valid and vision[0].response_model == TERRA
    route = info.extraction_diagnostics["extraction_route"]
    assert route == ("OPENAI_MULTIMODAL" if scanned else "NATIVE_TEXT")
    assert len(requests) == len(calls)


def test_exact_public_corpus_and_real_dormitory_retrieval():
    sources = load_public_sources()
    assert sources.loaded == sources.sources == 45 and not sources.issues
    assert {chunk.procedure for chunk in sources.chunks} == {
        "public_web", "ikamet_first", "ikamet_extension", "tc_kimlik_first", "tc_kimlik_renewal"}
    assert all(chunk.file_ref.startswith(("knowledge/corpus/", "procedures/")) for chunk in sources.chunks)
    details = {}
    rows = search_official_sources("What address proof do I need if I live in a dormitory?", embedding_model="", details=details)
    assert any("dorm" in row.content.casefold() and row.source_id.startswith("WEB_IRO") for row in rows)
    assert details["indexed_chunks"] == retrieval_status("case", embedding_model="")["indexed_chunks"]
    assert details["result_count"] == len(rows) and details["fingerprint"]
    assert not search_official_sources("What does official guidance say about quantum xenon flux?", embedding_model="")


def test_app_py_details_uses_server_provider_and_separates_retrieval_from_generation(ui_case, ready_bundle, monkeypatch):
    monkeypatch.setattr("config.LLM_PROVIDER", "openai")
    monkeypatch.setattr("config.DOCUMENT_PROVIDER", "cloud")
    monkeypatch.setattr("config.OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setattr("src.ui.model_settings.OPENAI_MODEL", TERRA)
    monkeypatch.setattr("src.ui.model_settings.OPENAI_DOCUMENT_MODEL", TERRA)
    monkeypatch.setattr("src.ui.model_settings.cloud_configuration_issue", lambda *a: None)
    monkeypatch.setattr("src.agents.assistant.OPENAI_MODEL", TERRA)
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(
        model=TERRA, status="completed", output_text='{"source_indices":[0]}',
        usage=SimpleNamespace(input_tokens=19, output_tokens=6, total_tokens=25)))), close=Mock())
    monkeypatch.setattr("src.core.model_provider.create_openai_client", lambda timeout: client)
    ui_case.seed(entered(ready_bundle[2]))
    original = ui_case.read()
    launcher = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=25)
    launcher.session_state["workspace_profile_id"] = "demo-user"
    launcher.run()
    next(button for button in launcher.button if button.label == "Your details").click()
    launcher.run()
    choose(launcher, "radio", "Details section", "knowledge")
    launcher.run()
    assert not launcher.exception and client.responses.create.call_count == 0
    choose(launcher, "text_input", "Test a question", "What address proof do I need if I live in a dormitory?")
    next(button for button in launcher.button if button.label == "Retrieve evidence").click()
    launcher.run()
    assert not launcher.exception and client.responses.create.call_count == 0
    assert any(item.value == "#### Retrieved sources" for item in launcher.markdown)
    next(button for button in launcher.button if button.label == "Generate answer").click()
    launcher.run()
    assert not launcher.exception and client.responses.create.call_count == 1
    wire = client.responses.create.call_args.kwargs
    assert wire["model"] == TERRA and wire["reasoning"] == {"effort": "low"}
    assert any("Answer model" in item.value and TERRA in item.value for item in launcher.caption)
    assert any(item.label == "Whole-session usage" for item in launcher.get("expander"))
    launcher.run()
    assert client.responses.create.call_count == 1 and ui_case.read() == original
    next(button for button in launcher.button if button.label == "Documents").click()
    launcher.run()
    assert any(button.label == "Upload and analyze with OpenAI" for button in launcher.button)

def test_cloud_passport_bundle_uses_the_transcribed_pages_and_stays_on_openai(monkeypatch):
    from src.core.state import Document, UserEvent
    from src.document_processing.extractor import TextPage
    from src.document_processing.schemas import DOCUMENT_SCHEMAS
    from src.workers import document_analyzer as worker

    original_pages = [TextPage(1, "Synthetic passport identity page"), TextPage(2, "Synthetic arrival marker")]
    primary = DOCUMENT_SCHEMAS["passport_identity_page"](selected_pages=[1])
    stamp = DOCUMENT_SCHEMAS["turkiye_entry_stamp"](selected_pages=[2])

    def analyze(path, **options):
        assert options["mode"] == "cloud"
        if options.get("page_sink") is not None:
            options["page_sink"].extend(original_pages)
        return primary

    def classify(proposed, pages):
        return "turkiye_entry_stamp" if pages[0].number == 2 else "unknown"

    def summarize(path, **kwargs):
        assert kwargs["mode"] == "cloud" and kwargs["prefer_native"]
        assert kwargs["pages"] == [original_pages[1]]
        return stamp

    monkeypatch.setattr(worker, "process_document", analyze)
    monkeypatch.setattr(worker, "_cached_analyze", analyze)
    local = Mock(side_effect=RuntimeError("Local OCR unavailable"))
    monkeypatch.setattr("src.document_processing.processor.extract_pages", local)
    monkeypatch.setattr("src.document_processing.processor.checked_classification", classify)
    monkeypatch.setattr("src.document_processing.processor.process_document", summarize)
    path = "synthetic-bundle.pdf"
    event = UserEvent(kind="analyze_document", file_path=path)
    state = {"event": event, "user_id": "synthetic", "documents": [Document(doc_id="fixture", file_path=path)]}
    with use_ai_options(AIOptions(provider="ollama", cloud_document=(event.kind, path))):
        result = worker.document_analyzer(state)
    assert [doc.doc_type for doc in result["documents"]] == ["passport_copy", "turkiye_entry_stamp"]
    assert [doc.info.selected_pages for doc in result["documents"]] == [[1], [2]]
    local.assert_not_called()


def test_mixed_dimension_vector_file_does_not_report_ready(tmp_path):
    from src.rag.semantic import _index_file, _write_index, index_status

    model = "test-vector-model"
    rows = [SimpleNamespace(chunk_id="a"), SimpleNamespace(chunk_id="b")]
    _write_index(_index_file(model, tmp_path), model, {"a": [1.0, 0.0], "b": [1.0, 0.0, 0.0]})
    status = index_status(rows, model, directory=tmp_path)
    assert status["vector_status"] == "partial"
    assert status["indexed_vectors"] == 1

def test_case_index_refresh_preserves_other_current_public_vectors(tmp_path, monkeypatch):
    from src.rag import __main__ as cli, semantic

    model = "test-vector-model"
    supported = SimpleNamespace(chunk_id="case", content="updated case source")
    other = SimpleNamespace(chunk_id="other", content="different public scope")
    monkeypatch.setattr(semantic, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(cli, "retrieval_chunks", lambda scope: [supported] if scope == "case" else [supported, other])
    mock = Mock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(semantic, "local_embedder", lambda name: mock)
    semantic._write_index(semantic._index_file(model, tmp_path), model, {"other": [0.0, 1.0]})
    assert cli.main(["index", "--scope", "case", "--model", model]) == 0
    assert semantic.index_status([supported, other], model)["vector_status"] == "ready"
    mock.assert_called_once_with(["updated case source"], "rag_embedding_index")
