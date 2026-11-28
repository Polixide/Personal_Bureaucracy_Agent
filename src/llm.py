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

Backends: `OllamaBackend` talks to the local Ollama server. `AnthropicBackend` (Claude) and `OpenAIBackend` (GPT)
use the providers' official SDKs when LLM_PROVIDER says so (`agent_backend`); they are used by the chat assistant,
the orchestrator and its workers. Reading documents uses `document_backend`: the local Ollama server, so document
pages never leave the computer, unless DOCUMENT_PROVIDER=cloud makes the cloud model read them too (it then
also sees page images through `see`). Embeddings always stay local. Tests pass a scripted backend with the
same `chat` / `embed` methods (see tests/fakes.py). Logs never contain prompt or reply text.
"""
import base64
import json
import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any, Protocol

import anthropic
import httpx
import openai
from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

from config import (ANTHROPIC_MODEL, CHAT_MODEL, CLOUD_EFFORT, CLOUD_MAX_TOKENS, CLOUD_PROVIDERS, CLOUD_TIMEOUT,
                    DOCUMENT_PROVIDER, EMBED_MODEL, LLM_MODEL, LLM_NUM_CTX, LLM_PROVIDER, LLM_TEMPERATURE,
                    OLLAMA_HOST, OLLAMA_TIMEOUT, OPENAI_MODEL, OPENAI_REASONING_EFFORT, VISION_MODEL, VISION_NUM_CTX,
                    VISION_TIMEOUT)

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


def timed_call(component: str, model: str, call: Callable[[], Any]) -> Any:
    """Run one model call and log only model, component, outcome and duration (never prompt or reply text)."""
    model = str(model).replace("\r", " ").replace("\n", " ")
    prefix = f"[LLM] {component} model={model}"
    started = perf_counter()
    _emit(prefix + " started")
    try:
        response = call()
    except Exception as error:
        outcome = "timeout" if isinstance(error, (TimeoutError, httpx.TimeoutException)) else "failure"
        _emit(f"{prefix} {outcome} elapsed={perf_counter() - started:.1f}s")
        raise
    _emit(f"{prefix} success elapsed={perf_counter() - started:.1f}s")
    return response


def logged_chat(client: Any, *, component: str, **kwargs: Any) -> Any:
    """Forward one `client.chat` call (Ollama) through timed_call."""
    return timed_call(component, kwargs.get("model", ""), lambda: client.chat(**kwargs))


# ---------------------------------------------------------------- Ollama --
class OllamaBackend:
    """The local Ollama server (models and host from config.py)."""
    cloud = False

    def __init__(self, model: str = LLM_MODEL, host: str = OLLAMA_HOST,
                 timeout: float = OLLAMA_TIMEOUT, embed_model: str = EMBED_MODEL, vision_model: str = VISION_MODEL):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.embed_model = embed_model
        self.vision_model = vision_model

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

    def see(self, prompt: str, images: list[bytes], schema: dict, component: str) -> str:
        """One schema-constrained call to the vision model with page images (PNG bytes)."""
        client = Client(host=self.host, timeout=VISION_TIMEOUT)
        try:
            response = logged_chat(client, component=component, model=self.vision_model,
                                   messages=[{"role": "user", "content": prompt, "images": images}], format=schema,
                                   options={"temperature": LLM_TEMPERATURE, "num_ctx": VISION_NUM_CTX},
                                   think=False, stream=False)
        except TRANSPORT_ERRORS as error:
            raise LLMError(f"Ollama vision model unavailable: {error}") from error
        return response.message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors for `texts` (one per text)."""
        client = Client(host=self.host, timeout=self.timeout)
        try:
            response = client.embed(model=self.embed_model, input=texts)
        except TRANSPORT_ERRORS as error:
            raise LLMError(f"Ollama embeddings unavailable: {error}") from error
        return [list(vector) for vector in response.embeddings]


# ----------------------------------------------------------------- cloud --
# Keywords Claude's structured outputs do not accept (Pydantic still checks them after the reply).
UNSUPPORTED_KEYWORDS = ("minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                        "multipleOf", "minItems", "maxItems", "pattern", "default")
SCHEMA_MAPS = ("properties", "$defs", "definitions")          # name -> schema mappings
SCHEMA_LISTS = ("items", "anyOf", "allOf", "oneOf", "prefixItems", "not")
JSON_INSTRUCTION = "Reply with ONLY one JSON object (no text around it) that matches this JSON schema: "


def strict_schema(schema: dict) -> dict | None:
    """The schema as Claude's structured outputs accept it, or None when it cannot be written that way.

    Every object needs `additionalProperties: false`; length and number limits are dropped (Pydantic checks them).
    A free-form object (any keys, like a tool's `args`) cannot be expressed: then the caller asks for JSON in words.
    """
    copy = json.loads(json.dumps(schema))
    return copy if _make_strict(copy) else None


def _make_strict(node: Any) -> bool:
    """Adapt one schema node in place; False when it (or a child) is a free-form object."""
    if isinstance(node, list):
        return all([_make_strict(item) for item in node])
    if not isinstance(node, dict):
        return True
    for keyword in UNSUPPORTED_KEYWORDS:
        node.pop(keyword, None)
    if node.get("type") == "object" or "properties" in node:
        if not isinstance(node.get("properties"), dict):
            return False
        node["additionalProperties"] = False
    results = [True]
    for key, value in node.items():
        if key in SCHEMA_MAPS and isinstance(value, dict):
            results += [_make_strict(child) for child in value.values()]
        elif key in SCHEMA_LISTS:
            results.append(_make_strict(value))
    return all(results)


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Ollama-style messages -> (system text, turns) as the Messages API wants them.

    The turns start with the user and end with the user (current Claude models refuse a prefilled assistant turn).
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "Start."})
    if turns[-1]["role"] != "user":
        turns.append({"role": "user", "content": "Continue."})
    return system, turns


def image_base64(image: bytes) -> str:
    """Image bytes as base64 text, the way both providers take an image inside a request."""
    return base64.b64encode(image).decode("ascii")


def image_media_type(image: bytes) -> str:
    """image/jpeg (pages sent to a cloud model) or image/png, from the file's first bytes."""
    return "image/jpeg" if image.startswith(b"\xff\xd8") else "image/png"


class AnthropicBackend:
    """Claude through the official `anthropic` SDK (LLM_PROVIDER=anthropic). Embeddings stay local (Ollama)."""
    cloud = True

    def __init__(self, model: str = ANTHROPIC_MODEL, client: Any = None, embedder: Any = None):
        self.model = model
        self.client = client
        self.embedder = embedder or OllamaBackend()

    def chat(self, messages: list[dict], schema: dict, component: str) -> str:
        """One JSON reply: schema-constrained when the schema allows it, otherwise asked for in the system text."""
        system, turns = split_system(messages)
        return self._structured(system, turns, schema, component)

    def see(self, prompt: str, images: list[bytes], schema: dict, component: str) -> str:
        """One JSON reply about page images (JPEG/PNG bytes, DOCUMENT_PROVIDER=cloud): the images, then the question."""
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": image_media_type(image),
                                               "data": image_base64(image)}} for image in images]
        turns = [{"role": "user", "content": blocks + [{"type": "text", "text": prompt}]}]
        return self._structured("", turns, schema, component)

    def _structured(self, system: str, turns: list[dict], schema: dict, component: str) -> str:
        """Strict JSON schema first; if Claude refuses that schema, the schema asked for in the system text."""
        strict = strict_schema(schema)
        attempts = [strict, None] if strict is not None else [None]    # schema not accepted: ask for JSON in words
        error: Exception | None = None
        for constraint in attempts:
            try:
                return self._send(system, turns, constraint, schema, component)
            except anthropic.BadRequestError as refused:
                error = refused
        raise LLMError(f"Claude refused the request: {error}")

    def _send(self, system: str, turns: list[dict], strict: dict | None, schema: dict, component: str) -> str:
        output_config: dict[str, Any] = {"effort": CLOUD_EFFORT} if CLOUD_EFFORT else {}
        if strict is not None:
            output_config["format"] = {"type": "json_schema", "schema": strict}
        else:
            system = f"{system}\n\n{JSON_INSTRUCTION}{to_json(schema)}".strip()
        request: dict[str, Any] = {"model": self.model, "max_tokens": CLOUD_MAX_TOKENS, "messages": turns,
                                   "cache_control": {"type": "ephemeral"}}
        if system:                     # an image question has no system text
            request["system"] = system
        if output_config:
            request["output_config"] = output_config
        try:
            response = timed_call(component, self.model, lambda: self._client().messages.create(**request))
        except anthropic.BadRequestError:
            raise
        except anthropic.AnthropicError as error:
            raise LLMError(f"Claude unavailable: {error}") from error
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this request")
        return "".join(block.text for block in response.content if block.type == "text")

    def _client(self) -> Any:
        if self.client is None:
            self.client = anthropic.Anthropic(timeout=CLOUD_TIMEOUT, max_retries=2)
        return self.client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings stay on the local Ollama server (the official sources are indexed there)."""
        return self.embedder.embed(texts)


class OpenAIBackend:
    """GPT through the official `openai` SDK (LLM_PROVIDER=openai). Embeddings stay local (Ollama)."""
    cloud = True

    def __init__(self, model: str = OPENAI_MODEL, client: Any = None, embedder: Any = None):
        self.model = model
        self.client = client
        self.embedder = embedder or OllamaBackend()

    def chat(self, messages: list[dict], schema: dict, component: str) -> str:
        """One JSON reply to text messages."""
        turns = [{"role": m["role"], "content": m["content"]} for m in messages]
        return self._structured(turns, schema, component)

    def see(self, prompt: str, images: list[bytes], schema: dict, component: str) -> str:
        """One JSON reply about page images (JPEG/PNG bytes, DOCUMENT_PROVIDER=cloud), sent in high detail."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": f"data:{image_media_type(image)};base64,"
                                                               f"{image_base64(image)}", "detail": "high"}}
                    for image in images]
        return self._structured([{"role": "user", "content": content}], schema, component)

    def _structured(self, turns: list[dict], schema: dict, component: str) -> str:
        """json_schema mode (not strict, so any schema works); JSON mode if the schema is refused."""
        guided = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": False}}
        hint = {"role": "system", "content": JSON_INSTRUCTION + to_json(schema)}
        error: Exception | None = None
        for attempt, response_format in ((turns, guided), ([hint] + turns, {"type": "json_object"})):
            try:
                return self._send(attempt, response_format, component)
            except openai.BadRequestError as refused:
                error = refused
        raise LLMError(f"OpenAI refused the request: {error}")

    def _send(self, turns: list[dict], response_format: dict, component: str) -> str:
        request: dict[str, Any] = {"model": self.model, "messages": turns, "response_format": response_format,
                                   "max_completion_tokens": CLOUD_MAX_TOKENS}
        if OPENAI_REASONING_EFFORT:
            request["reasoning_effort"] = OPENAI_REASONING_EFFORT
        try:
            response = timed_call(component, self.model, lambda: self._client().chat.completions.create(**request))
        except openai.BadRequestError:
            raise
        except openai.OpenAIError as error:
            raise LLMError(f"OpenAI unavailable: {error}") from error
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError("The model declined to answer this request")
        return message.content or ""

    def _client(self) -> Any:
        if self.client is None:
            self.client = openai.OpenAI(timeout=CLOUD_TIMEOUT, max_retries=2)
        return self.client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings stay on the local Ollama server."""
        return self.embedder.embed(texts)


def is_cloud() -> bool:
    """True when the agent runs on a cloud model (LLM_PROVIDER=anthropic or openai)."""
    return LLM_PROVIDER in CLOUD_PROVIDERS


def agent_backend(kind: str = "chat") -> Any:
    """The model that plans, talks and repairs: the cloud model when LLM_PROVIDER says so, otherwise Ollama.

    kind "chat" is the chat assistant (small fast local model), "team" the orchestrator and its workers.
    """
    if LLM_PROVIDER == "anthropic":
        return AnthropicBackend()
    if LLM_PROVIDER == "openai":
        return OpenAIBackend()
    return OllamaBackend(model=CHAT_MODEL if kind == "chat" else LLM_MODEL)


def local_backend() -> OllamaBackend:
    """The local Ollama server: embeddings always, documents unless DOCUMENT_PROVIDER=cloud."""
    return OllamaBackend()


def is_cloud_backend(backend: Any) -> bool:
    """True for a cloud backend (Claude or GPT); False for Ollama, test doubles and None."""
    return bool(getattr(backend, "cloud", False))


def documents_in_cloud() -> bool:
    """True when the cloud model reads the documents (DOCUMENT_PROVIDER=cloud with a cloud LLM_PROVIDER)."""
    return DOCUMENT_PROVIDER == "cloud" and is_cloud()


def document_backend() -> Any:
    """Who reads documents (OCR of scans, extraction, vision): the cloud model when documents_in_cloud(), else Ollama.

    Its `embed` stays local either way (the cloud backends hand embeddings to Ollama).
    """
    return agent_backend("team") if documents_in_cloud() else local_backend()


def document_model_label() -> str:
    """Which model reads the documents, in words, for the app."""
    if documents_in_cloud():
        return agent_model_label("team")
    return f"local models only (text, OCR, {VISION_MODEL})"


def agent_model_label(kind: str = "chat") -> str:
    """Which model answers, in words, for the app."""
    if LLM_PROVIDER == "anthropic":
        return f"{ANTHROPIC_MODEL} (Anthropic, cloud)"
    if LLM_PROVIDER == "openai":
        return f"{OPENAI_MODEL} (OpenAI, cloud)"
    return f"{CHAT_MODEL if kind == 'chat' else LLM_MODEL} (local)"


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


def ask_vision(backend: Any, prompt: str, images: list[bytes], output_model: type[BaseModel], component: str,
               schema: dict | None = None) -> StructuredResult:
    """Like ask_structured, for the vision model: validation, one retry, never raises.

    A backend without a `see` method (a test double, or vision switched off) gives ok=False.
    """
    see = getattr(backend, "see", None)
    if see is None:
        return StructuredResult(ok=False, error="no vision model")
    schema = schema or output_model.model_json_schema()
    error, raw = "", ""
    for attempt in (1, 2):
        text = prompt if attempt == 1 else f"{prompt}\n\nYour last reply was invalid: {error}. Reply with ONLY the JSON."
        try:
            raw = see(text, images, schema, component)
        except LLMError as transport:
            return StructuredResult(ok=False, error=str(transport), attempts=attempt)
        value, error = parse_reply(raw, output_model)
        if value is not None:
            return StructuredResult(ok=True, value=value, attempts=attempt, raw=raw)
    return StructuredResult(ok=False, error=error, attempts=2, raw=raw)


def to_json(data: Any) -> str:
    """Compact JSON for prompts (dates and models become text)."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
