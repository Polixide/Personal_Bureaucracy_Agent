"""Documents read by the cloud model (DOCUMENT_PROVIDER=cloud), with fake clients: no network, no Ollama, no OCR.

What is checked: page images reach GPT and Claude in the providers' formats; a scan is transcribed by the cloud
model instead of RapidOCR (and RapidOCR takes over if the model fails); a text PDF is never sent as an image; the
typed processor and the generic reader run on the cloud model with the whole document; and the checks by code after
the model are unchanged (an invented policy number is still replaced by the one on the page).
"""
import json
from datetime import date
from unittest.mock import Mock, patch

import pymupdf

from src.document_processing.extractor import TextPage, extract_pages
from src.document_processing.generic import extract_generic
from src.document_processing.processor import process_document
from src.document_processing.transcribe import CloudTranscriber, Transcription
from src.llm import AnthropicBackend, LLMError, OpenAIBackend
from src.pack import load_pack
from src.tools.extract_document import extract_file
from test_llm_cloud import FakeAnthropic, FakeEmbedder, FakeOpenAI

PACK = load_pack("ikamet_first")
PNG = b"\x89PNG fake image"
INSURANCE = ("Health Insurance Certificate\nInsured: Taylor Example\nInsurer: Example Insurance\n"
             "Policy number: DEMO-OCR-42\nPolicy approval date: 18/09/2028\n"
             "Coverage start date: 20/09/2028\nCoverage end date: 30/06/2029")


def scanned_pdf(path, embedded_text=None):
    """A synthetic raster page (an image covering the page), optionally with a text layer like a scanner app adds."""
    with pymupdf.open() as document:
        page = document.new_page()
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
        image.clear_with(255)
        page.insert_image(page.rect, stream=image.tobytes("png"))
        if embedded_text:
            page.insert_text((72, 72), embedded_text)
        document.save(path)


class FakeCloudReader:
    """A cloud backend double: `see` transcribes the page, `chat` answers each document prompt by component."""
    cloud = True

    def __init__(self, page_text="", replies=None):
        self.page_text = page_text
        self.replies = replies or {}
        self.seen: list[str] = []
        self.images: list[bytes] = []
        self.asked: list[tuple[str, list[dict]]] = []

    def see(self, prompt, images, schema, component):
        self.seen.append(component)
        self.images += images
        if component == "document_transcription":
            return json.dumps({"lines": self.page_text.splitlines()})
        return json.dumps({"doc_type": "this", "fields": []})

    def chat(self, messages, schema, component):
        self.asked.append((component, messages))
        return json.dumps(self.replies.get(component, {"fields": []}))

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


# ------------------------------------------------------------ backends --
def test_gpt_sees_page_images_as_high_detail_data_urls():
    client = FakeOpenAI('{"lines": ["Passport"]}')
    backend = OpenAIBackend(model="gpt-6.1-terra", client=client, embedder=FakeEmbedder())
    text = backend.see("Transcribe.", [PNG], Transcription.model_json_schema(), "document_transcription")
    request = client.requests[0]
    prompt, image = request["messages"][0]["content"]
    assert text == '{"lines": ["Passport"]}' and request["model"] == "gpt-6.1-terra"
    assert prompt == {"type": "text", "text": "Transcribe."}
    assert image["image_url"]["url"].startswith("data:image/png;base64,") and image["image_url"]["detail"] == "high"
    assert request["response_format"]["type"] == "json_schema"


def test_claude_sees_page_images_as_base64_blocks_before_the_question():
    client = FakeAnthropic('{"lines": ["Passport"]}')
    backend = AnthropicBackend(client=client, embedder=FakeEmbedder())
    backend.see("Transcribe.", [PNG], Transcription.model_json_schema(), "document_transcription")
    request = client.requests[0]
    image, prompt = request["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert prompt == {"type": "text", "text": "Transcribe."}
    assert "system" not in request                     # no empty system text is sent
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_a_jpeg_page_is_sent_as_image_jpeg_to_both_providers():
    jpeg = b"\xff\xd8\xff fake jpeg"
    schema = Transcription.model_json_schema()
    gpt = FakeOpenAI('{"lines": []}')
    OpenAIBackend(client=gpt, embedder=FakeEmbedder()).see("Read.", [jpeg], schema, "document_transcription")
    assert gpt.requests[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    claude = FakeAnthropic('{"lines": []}')
    AnthropicBackend(client=claude, embedder=FakeEmbedder()).see("Read.", [jpeg], schema, "document_transcription")
    assert claude.requests[0]["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


# ----------------------------------------------------- transcription --
def test_a_scan_is_transcribed_by_the_cloud_model_and_rapidocr_is_not_used(tmp_path):
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    pages = extract_pages(path, lambda page, number: "Ağrı Mahallesi\nNo: 12")    # RapidOCR would fail (conftest)
    assert pages[0].text == "Ağrı Mahallesi\nNo: 12" and pages[0].ocr


def test_a_failed_transcription_falls_back_to_rapidocr(tmp_path):
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    engine = Mock(return_value=Mock(txts=("Local OCR text",), scores=None))
    with patch("src.document_processing.extractor._create_ocr_engine", return_value=engine):
        pages = extract_pages(path, lambda page, number: "")
    assert pages[0].text == "Local OCR text"


def test_a_scan_with_a_scanner_text_layer_is_read_from_its_image_in_cloud_mode(tmp_path):
    path = tmp_path / "scan.pdf"
    scanned_pdf(path, embedded_text="Studnet Enro11ment Certifcate with a poor scanner text layer")
    assert "Studnet" in extract_pages(path)[0].text                                  # local: the text layer
    assert extract_pages(path, lambda page, number: "Student Enrollment Certificate")[0].text == \
        "Student Enrollment Certificate"


def test_a_text_pdf_is_never_sent_as_an_image(tmp_path):
    path = tmp_path / "text.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Rental agreement between the owner and the tenant, Tuzla")
        document.save(path)
    transcribe = Mock(return_value="should not be used")
    assert "Rental agreement" in extract_pages(path, transcribe)[0].text
    transcribe.assert_not_called()


def test_the_transcriber_keeps_the_lines_and_reports_failed_pages(tmp_path):
    path = tmp_path / "scan.pdf"
    scanned_pdf(path)
    backend = FakeCloudReader(" Line one \n\nLine two")
    reader = CloudTranscriber(backend)
    with pymupdf.open(path) as document:
        assert reader.read(document[0], 1) == "Line one\nLine two"
    assert backend.images[0].startswith(b"\xff\xd8")           # a JPEG: a photographed page stays about 1 MB
    broken = Mock(cloud=True, see=Mock(side_effect=LLMError("down")))
    failing = CloudTranscriber(broken)
    with pymupdf.open(path) as document:
        assert failing.read(document[0], 1) == ""
    assert failing.failed == [1] and "page(s) 1" in failing.note()


# --------------------------------------------------------- extraction --
def test_the_typed_processor_reads_the_whole_document_with_the_cloud_model():
    text = "Health Insurance Certificate\n" + "Clause text. " * 500 + "\nCoverage end date: 30/06/2029"
    backend = FakeCloudReader(replies={"document_extraction": {"document_type": "health_insurance_certificate"}})
    process_document("insurance.txt", pages=[TextPage(1, text)], document_type_hint="health_insurance_certificate",
                     backend=backend, max_context_chars=40000)                     # no Ollama client (conftest)
    (component, messages), = backend.asked
    assert component == "document_extraction" and "30/06/2029" in messages[1]["content"]


def test_the_generic_reader_sends_a_whole_document_to_the_cloud_model_in_one_request():
    text = "Electricity bill\n" + "Consumption details. " * 400 + "\nEND OF BILL"
    backend = FakeCloudReader()
    extract_generic([TextPage(1, text)], PACK.document("utility_bill"), PACK, backend)
    calls = [messages for component, messages in backend.asked if component == "document_generic_extraction"]
    assert len(calls) == 1 and "END OF BILL" in calls[0][1]["content"]


def test_a_scanned_insurance_read_in_the_cloud_is_still_checked_by_code(tmp_path):
    path = tmp_path / "insurance-scan.pdf"
    scanned_pdf(path)
    backend = FakeCloudReader(INSURANCE, {
        "document_classification": {"document_type": "health_insurance_certificate"},
        "document_extraction": {"policy_number": "INVENTED-99", "source_snippets": {"policy_number": "Invented"}}})
    result = extract_file(path, PACK, backend=backend)          # Ollama and RapidOCR would fail the test
    assert result.error is None and backend.seen[0] == "document_transcription"
    observation, = result.observations
    facts = {fact.field: fact.value for fact in observation.facts}
    assert observation.doc_type == "health_insurance_certificate"
    assert facts["insurance_policy_number"] == "DEMO-OCR-42"        # the page wins over the model's invention
    assert facts["insurance_end_date"] == date(2029, 6, 30)
