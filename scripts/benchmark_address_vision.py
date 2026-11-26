"""Benchmark a synthetic Address Form with real local OCR and an offline OpenAI SDK transport."""
import base64
import json
from pathlib import Path
from struct import unpack
from time import perf_counter

from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pymupdf

import config

from src.core.llm_diagnostics import collect_model_calls
from src.document_processing.processor import process_document
from src.document_processing.providers import OllamaProvider, OpenAIProvider
from src.document_processing.schemas import AddressForm

TERRA = "gpt-6.1-terra"
EXPECTED = {
    "first_name": "RIVA", "last_name": "BELLARIVA", "passport_number": "TEST990001",
    "nationality": "ITALIAN", "birth_date": "2000-03-19", "phone": "+00 555 010 0299",
    "declared_address": "Example Hall 9, Sample District", "form_date": "2026-10-21",
}
PAGES = (
    (
        "Address Form",
        "First Name: RIVA",
        "Last / Family Name: BELLARIVA",
        "Nationality: ITALIAN",
        "Date of Birth: 19.03.2000",
    ),
    (
        "Contact Section",
        "Mobile Number: +00 555 010 0299",
        "Address in Turkey:",
        "Example Hall 9, Sample District",
        "Date: 21.10.2026",
        "Signature:",
    ),
)


def make_pdf(path):
    with pymupdf.open() as source, pymupdf.open() as pdf:
        scanned = source.new_page()
        for i, line in enumerate(PAGES[0]):
            scanned.insert_text((70, 75 + i * 32), line, fontsize=16)
        first = pdf.new_page()
        first.insert_image(first.rect, pixmap=scanned.get_pixmap(dpi=160))
        widget = pymupdf.Widget()
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        widget.field_name = "passport_number"
        widget.field_label = "Passport number"
        widget.field_value = EXPECTED["passport_number"]
        widget.rect = pymupdf.Rect(70, 270, 300, 295)
        first.add_widget(widget)
        second = pdf.new_page()
        for i, line in enumerate(PAGES[1]):
            second.insert_text((70, 75 + i * 32), line, fontsize=16)
        second.draw_line((170, 235), (285, 235))
        second.draw_polyline([(202, 233), (214, 224), (208, 233), (227, 225), (222, 235)])
        pdf.save(path)


def mocked_openai_client(observed, *, transcribed_pages=PAGES):
    import httpx2
    import openai

    def respond(request):
        assert request.url.path == "/v1/responses"
        wire = json.loads(request.content)
        body = wire["input"][1]["content"]
        images = [item for item in body if item["type"] == "input_image"]
        stage = wire["text"]["format"]["name"]
        observed.append({
            "stage": stage, "model_requested_on_wire": wire["model"],
            "reasoning": wire.get("reasoning", {}).get("effort"),
            "page_markers": [item["text"] for item in body if item.get("text", "").startswith("[PAGE IMAGE")],
            "images": [{"detail": item["detail"], "size": unpack(">II", base64.b64decode(
                item["image_url"].split(",", 1)[1])[16:24])} for item in images],
        })
        if stage == "document_transcription":
            answer = {"pages": [{"page_number": n, "text": "\n".join(transcribed_pages[n - 1])}
                                for n in (int(marker.split()[-1][:-1]) for marker in observed[-1]["page_markers"])]}
        elif stage == "document_classification":
            answer = {"document_type": "address_form"}
        else:
            answer = {"document_type": "address_form", **EXPECTED, "signature_present": True}
        return httpx2.Response(200, json={
            "id": "resp_synthetic", "object": "response", "created_at": 0, "model": TERRA, "status": "completed",
            "output": [{"id": "msg_synthetic", "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps(answer), "annotations": []}]}],
            "usage": {"input_tokens": 540, "output_tokens": 190, "total_tokens": 730,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 50}},
        })

    return openai.OpenAI(api_key="synthetic-local-mock", max_retries=0,
                         http_client=httpx2.Client(transport=httpx2.MockTransport(respond)))


def evaluate(label, path, **options):
    started = perf_counter()
    with collect_model_calls() as calls:
        info = process_document(path, **options)
    observed = info.model_dump(mode="json", include=info.observation_fields(), exclude_none=True)
    exact = {name: observed.get(name) == expect for name, expect in EXPECTED.items()}
    correct = sum(exact.values())
    missing = EXPECTED.keys() - observed.keys()
    wrong = {name for name in EXPECTED.keys() & observed.keys() if not exact[name]}
    extra = observed.keys() - EXPECTED.keys()

    def total(attribute):
        values = [getattr(c, attribute) for c in calls]
        return sum(values) if values and all(v is not None for v in values) else None
    return {
        "case": label, "route": info.extraction_diagnostics.get("extraction_route"),
        "document_type": info.document_type, "matched_fields": sorted(k for k, value in exact.items() if value),
        "identity_exact": {name: exact[name] for name in ("first_name", "last_name", "passport_number")},
        "precision": correct / len(observed) if observed else None, "recall": correct / len(EXPECTED),
        "missing_fields": sorted(missing), "missing_rate": len(missing) / len(EXPECTED),
        "wrong_fields": sorted(wrong), "wrong_rate": len(wrong) / len(EXPECTED),
        "unexpected_fields": sorted(extra), "validation_issue_count": len(info.validation_issues),
        "typed_schema_valid": isinstance(info, AddressForm),
        "model_response_schema_valid": [c.structured_output_valid for c in calls],
        "elapsed_seconds": round(perf_counter() - started, 3),
        "input_tokens": total("prompt_tokens"), "output_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"), "estimated_api_cost_usd": None,
    }


class LocalStructuredStub:
    """Supply fixed model responses if the pipeline needs local model fallback."""

    def __init__(self, **_):
        pass

    def chat(self, *, format, **_):
        payload = {"document_type": "address_form"}
        payload.update({field: value for field, value in EXPECTED.items() if field in format["properties"]})
        if "signature_present" in format["properties"]:
            payload["signature_present"] = True
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))


def main():
    config.OPENAI_REASONING_EFFORT = "low"
    base = Path(".test-temp").resolve()
    if not base.is_relative_to(Path.cwd().resolve()):
        raise ValueError("The benchmark scratch directory must be inside the current checkout")
    base.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="vision-address-", dir=base) as directory:
        path = Path(directory) / "synthetic-address.pdf"
        make_pdf(path)
        local = evaluate("Real RapidOCR / stubbed local model if needed", path, mode="local-only",
                         provider=OllamaProvider(client_factory=LocalStructuredStub))
        wire = []
        with mocked_openai_client(wire) as client:
            cloud = evaluate("OpenAI SDK / offline HTTP mock", path, mode="cloud", prefer_native=True,
                             provider=OpenAIProvider(model=TERRA, client=client))
    print(json.dumps({
        "fixture": "Two-page synthetic Address Form, with scanned text, a filled PDF widget and native contact fields",
        "token_measurement": "Mock response metadata for the SDK route. Live OpenAI tokens and cost are N/A.",
        "results": [local, cloud],
        "mock_wire_requests": wire,
    }, indent=2))


if __name__ == "__main__":
    main()
