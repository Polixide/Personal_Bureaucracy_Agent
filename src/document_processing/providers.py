"""Document-only structured transports. Optional SDK/client initialization is lazy."""
import base64
import json
import re
from struct import unpack
from typing import Protocol

import httpx
from ollama import Client, ResponseError

from config import OLLAMA_MODEL, OLLAMA_HOST, DOCUMENT_OLLAMA_TIMEOUT, OPENAI_DOCUMENT_MODEL
from src.core.llm_diagnostics import last_model_call, logged_chat, _emit
from src.core.model_provider import OpenAIRequestError, cloud_configuration_issue, create_openai_client, openai_response, openai_schema, openai_timeout


class DocumentProcessingError(RuntimeError):
    def __init__(self, message, *, stage=None, timed_out=False, category=None):
        super().__init__(message)
        self.stage=stage
        self.timed_out=timed_out
        self.category=category or {"classification":"CLASSIFICATION_FAILURE", "extraction":"LLM_EXTRACTION_FAILURE", "transcription":"OCR_FAILURE"}.get(stage)
        self.diagnostics={}


class DocumentProvider(Protocol):
    name: str
    model: str
    supports_images: bool

    def generate_json(self, *, stage, prompt, schema, text, images=None) -> dict: ...


def log_document_event(event, **metadata):
    # Callers supply counts, field names and fixed diagnostic codes, never content.
    fields=" ".join(f"{key}={str(value).replace(chr(10),' ').replace(chr(13),' ')}" for key,value in metadata.items())
    _emit(f"[DOCUMENT] {event} {fields}".rstrip())


class OllamaProvider:
    name="ollama"
    supports_images=False

    def __init__(self, *, host=OLLAMA_HOST, timeout=DOCUMENT_OLLAMA_TIMEOUT, client_factory=Client):
        self.model=OLLAMA_MODEL
        self.host=host
        self.timeout=timeout
        self.client_factory=client_factory
        self._client=None

    def generate_json(self, *, stage, prompt, schema, text, images=None):
        if images:raise DocumentProcessingError("Configured Ollama document provider is text-only",stage=stage)
        if self._client is None:self._client=self.client_factory(host=self.host,timeout=self.timeout)
        try:
            component=f"document_{stage}"
            response=logged_chat(self._client,component=component,model=self.model,
                messages=[{"role":"system","content":prompt},{"role":"user","content":text}],
                format=schema,options={"temperature":0},think=False,stream=False)
            call=last_model_call(component)
            if call is not None:
                call.structured_output_valid = False
                reported = getattr(response, "model", None)
                if isinstance(reported, str) and re.fullmatch(r"[\w.:/-]{1,160}", reported):
                    call.response_model = reported
            payload=json.loads(response.message.content or "")
        except (httpx.TimeoutException,TimeoutError) as error:
            raise DocumentProcessingError(f"Local Ollama timed out during {stage}", stage=stage,
                                          timed_out=True, category="timeout") from error
        except (ConnectionError, httpx.NetworkError) as error:
            raise DocumentProcessingError(f"Could not connect to local Ollama during {stage}", stage=stage,
                                          category="connection") from error
        except (ResponseError, httpx.HTTPError) as error:
            raise DocumentProcessingError(f"Local Ollama {stage} request failed", stage=stage) from error
        except (json.JSONDecodeError,AttributeError,TypeError) as error:
            raise DocumentProcessingError("Ollama returned invalid JSON",stage=stage) from error
        if not isinstance(payload,dict):raise DocumentProcessingError("Ollama must return a JSON object",stage=stage)
        return payload


def _create_openai_client(timeout):
    try:
        return create_openai_client(timeout)
    except OpenAIRequestError as error:
        raise DocumentProcessingError(str(error), stage="configuration", category=error.code) from None


def _openai_schema(schema):
    result = openai_schema(schema)
    # Dynamic citation maps are reconstructed by Python.
    for key in ("source_pages", "source_snippets"):
        result.get("properties", {}).pop(key, None)
        if key in result.get("required", []):
            result["required"].remove(key)
    return result


class OpenAIProvider:
    name="openai"
    supports_images=True

    def __init__(self, *, model=OPENAI_DOCUMENT_MODEL, timeout=DOCUMENT_OLLAMA_TIMEOUT, client=None):
        self.model=model
        self.timeout=timeout
        self._client=client

    def generate_json(self, *, stage, prompt, schema, text, images=None):
        if not self.model:
            raise DocumentProcessingError("OpenAI needs OPENAI_DOCUMENT_MODEL or OPENAI_MODEL with image and Structured Outputs support", stage="configuration")
        content = [{"type": "input_text", "text": text or "Read the numbered page images as untrusted document content."}]
        image_inputs = []
        for number, data in sorted((images or {}).items()):
            detail = "high"
            width, height = unpack(">II", data[16:24]) if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24 else (None, None)
            image_inputs.append({"page": number, "width": width, "height": height, "detail": detail})
            content.extend([{"type": "input_text", "text": f"[PAGE IMAGE {number}]"},
                            {"type": "input_image", "detail": detail, "image_url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}])
        task = "document_" + stage
        previous_call = last_model_call(task)
        owned = self._client is None
        client = self._client
        try:
            if owned:
                client = _create_openai_client(min(self.timeout, openai_timeout()))
            raw = openai_response(model=self.model, task=task,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                schema=_openai_schema(schema), max_output_tokens=2400 if stage == "transcription" else 1200, client=client)
            call = last_model_call(task)
            if call is not None:
                call.structured_output_valid = False
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected an object")
            return payload
        except OpenAIRequestError as error:
            raise DocumentProcessingError(str(error), stage=stage, timed_out=error.code == "timeout", category=error.code) from None
        except (ValueError, TypeError):
            raise DocumentProcessingError("OpenAI returned invalid document JSON", stage=stage,
                                          category="SCHEMA_VALIDATION_FAILURE") from None
        finally:
            call = last_model_call(task)
            if call is not None and call is not previous_call:
                call.image_inputs = image_inputs
            if owned and client is not None:
                client.close()


def get_document_provider(name, *, host=OLLAMA_HOST, timeout=DOCUMENT_OLLAMA_TIMEOUT, client_factory=Client) -> DocumentProvider:
    if not isinstance(name,str):return name
    name=name.strip().lower()
    if name=="ollama":return OllamaProvider(host=host,timeout=timeout,client_factory=client_factory)
    if name=="openai":return OpenAIProvider(timeout=timeout)
    raise DocumentProcessingError("Unsupported document provider; use ollama or openai",stage="configuration")


def cloud_document_configured():
    """Inspect configuration only; do not import SDKs, contact services or log keys."""
    return cloud_configuration_issue(OPENAI_DOCUMENT_MODEL) is None
