"""Every LLM and embedding call goes through this module.

Contract for structured calls (`ask_structured`):
  1. the model gets a JSON schema (Ollama `format=`), temperature 0, thinking off;
  2. the reply is validated with Pydantic;
  3. on failure the model gets ONE retry with the validation error;
  4. after that the caller gets `ok=False` and uses its deterministic fallback.
So bad model output never crashes the app.

Document extraction (src/document_processing/processor.py) uses `logged_chat` directly. It has its
own schema validation and its fallback is "no facts, ask the user to confirm"; a retry there would
change its tested call counts.

Backends: `OllamaBackend` talks to the local Ollama server. Tests pass a scripted backend with the
same `chat` / `embed` methods (see tests/fakes.py). Logs never contain prompt or reply text.
"""
import json
import sys
from time import perf_counter
from typing import Any, Protocol

import httpx
from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

from config import EMBED_MODEL, LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, OLLAMA_HOST, OLLAMA_TIMEOUT

TRANSPORT_ERRORS = (ResponseError, ConnectionError, httpx.HTTPError, TimeoutError)


class LLMError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


class Backend(Protocol):
    """What the agent needs from a model: schema-constrained chat and embeddings."""

    def chat(self, messages: list[dict], schema: dict, component: str) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ------------------------------------------------------------ diagnostics --
def _emit(message: str) -> None:
    """Write one diagnostics line to stderr; a closed stream must never break the agent."""
    try:
        if sys.stderr is None:
            return
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError, AttributeError):
        pass


def logged_chat(client: Any, *, component: str, **kwargs: Any) -> Any:
    """Forward one `client.chat` call and log only model, component, outcome and duration."""
    model = str(kwargs.get("model", "")).replace("\r", " ").replace("\n", " ")
    prefix = f"[LLM] {component} model={model}"
    started = perf_counter()
    _emit(prefix + " started")
    try:
        response = client.chat(**kwargs)
    except Exception as error:
        outcome = "timeout" if isinstance(error, (TimeoutError, httpx.TimeoutException)) else "failure"
        _emit(f"{prefix} {outcome} elapsed={perf_counter() - started:.1f}s")
        raise
    _emit(f"{prefix} success elapsed={perf_counter() - started:.1f}s")
    return response


# ---------------------------------------------------------------- Ollama --
class OllamaBackend:
    """The local Ollama server (models and host from config.py)."""

    def __init__(self, model: str = LLM_MODEL, host: str = OLLAMA_HOST,
                 timeout: float = OLLAMA_TIMEOUT, embed_model: str = EMBED_MODEL):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.embed_model = embed_model

    def chat(self, messages: list[dict], schema: dict, component: str) -> str:
        """One schema-constrained chat call; returns the raw reply text."""
        client = Client(host=self.host, timeout=self.timeout)
        try:
            response = logged_chat(client, component=component, model=self.model, messages=messages,
                                   format=schema, options={"temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX},
                                   think=False, stream=False)
        except TRANSPORT_ERRORS as error:
            raise LLMError(f"Ollama unavailable: {error}") from error
        return response.message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors for `texts` (one per text)."""
        client = Client(host=self.host, timeout=self.timeout)
        try:
            response = client.embed(model=self.embed_model, input=texts)
        except TRANSPORT_ERRORS as error:
            raise LLMError(f"Ollama embeddings unavailable: {error}") from error
        return [list(vector) for vector in response.embeddings]


# --------------------------------------------------------- structured --
class StructuredResult(BaseModel):
    """Outcome of one structured call (including its retry)."""
    ok: bool
    value: Any = None
    error: str = ""
    attempts: int = 0
    raw: str = ""


def clean_reply(raw: str) -> str:
    """Remove a ``` fence some models add around JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def short_validation_error(error: ValidationError) -> str:
    """At most three `location: message` pairs, without echoing the input values."""
    parts = []
    for item in error.errors(include_url=False, include_input=False)[:3]:
        where = ".".join(str(part) for part in item["loc"]) or "(root)"
        parts.append(f"{where}: {item['msg']}")
    return "; ".join(parts)


def parse_reply(raw: str, output_model: type[BaseModel]) -> tuple[BaseModel | None, str]:
    """Validate one reply. Returns (value, "") or (None, error message)."""
    try:
        return output_model.model_validate_json(clean_reply(raw)), ""
    except ValidationError as error:
        return None, short_validation_error(error)


def retry_message(error: str) -> dict:
    """The single correction message sent before the retry."""
    return {"role": "user",
            "content": f"Your last reply was invalid: {error}. Reply again with ONLY one JSON object "
                       f"that matches the schema."}


def ask_structured(backend: Backend, messages: list[dict], output_model: type[BaseModel],
                   component: str, schema: dict | None = None) -> StructuredResult:
    """Schema-constrained call with Pydantic validation and one retry. Never raises."""
    schema = schema or output_model.model_json_schema()
    conversation = list(messages)
    error = ""
    raw = ""
    for attempt in (1, 2):
        try:
            raw = backend.chat(conversation, schema, component)
        except LLMError as transport:
            error = str(transport)
            continue
        value, error = parse_reply(raw, output_model)
        if value is not None:
            return StructuredResult(ok=True, value=value, attempts=attempt, raw=raw)
        conversation = conversation + [{"role": "assistant", "content": raw}, retry_message(error)]
    return StructuredResult(ok=False, error=error, attempts=2, raw=raw)


def to_json(data: Any) -> str:
    """Compact JSON for prompts (dates and models become text)."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
