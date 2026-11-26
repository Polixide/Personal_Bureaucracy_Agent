"""Request-scoped provider selection and the optional OpenAI Responses transport."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from importlib.util import find_spec
from math import isfinite
from threading import Lock
import os
import tomllib

import httpx

import config

from src.core.llm_diagnostics import last_model_call, logged_request
from config import OPENAI_MODEL


ERROR_MESSAGES = {
    "missing_key": "Set OPENAI_API_KEY on the server to use OpenAI.",
    "missing_model": "Set OPENAI_MODEL (or OPENAI_DOCUMENT_MODEL for documents) on the server.",
    "missing_sdk": "Install the optional SDK with pip install -r requirements-openai.txt.",
    "model_unavailable": "Your OpenAI project could not access the configured model. Check the model ID and project permissions.",
    "configuration": "Choose a supported OPENAI_REASONING_EFFORT for the configured model.",
    "authentication": "OpenAI could not authenticate. Check the server API key and its permissions.",
    "rate_limit": "OpenAI is rate limiting requests or the account quota is exhausted. Check your API usage before retrying.",
    "timeout": "OpenAI did not respond before the request timeout. Try again when the service is available.",
    "connection": "The OpenAI request failed because the service could not be reached. Check your connection and try again.",
    "budget": "The configured OpenAI request limit was reached.",
    "response": "OpenAI could not complete this request. Check model access and Structured Outputs support.",
    "invalid_output": "OpenAI returned incomplete or unusable output.",
}


class OpenAIRequestError(RuntimeError):
    """A safe error code and message, independent of SDK request/response data."""

    def __init__(self, code):
        self.code = code
        super().__init__(ERROR_MESSAGES[code])


class RequestBudget:
    """Count outbound attempts, including failures, across one session or benchmark."""

    def __init__(self, maximum=None):
        if maximum is not None and (type(maximum) is not int or maximum < 0):
            raise ValueError("OpenAI request limit must be a nonnegative integer")
        self.maximum = maximum
        self.used = 0
        self._lock = Lock()

    def reserve(self):
        with self._lock:
            if self.maximum is not None and self.used >= self.maximum:
                raise OpenAIRequestError("budget")
            self.used += 1


def configured_budget():
    value = os.getenv("OPENAI_MAX_REQUESTS", "").strip()
    try:
        maximum = int(value) if value else None
        return RequestBudget(maximum)
    except ValueError:
        return RequestBudget(0)


@dataclass(frozen=True)
class AIOptions:
    provider: str = "ollama"
    budget: RequestBudget | None = None
    cloud_document: tuple[str, str] | None = None
    rag_enabled: bool = True
    embedding_model: str | None = None

    def __post_init__(self):
        if self.provider not in {"ollama", "openai"}:
            raise ValueError("Choose Ollama or OpenAI")


_options = ContextVar("ai_request_options", default=None)


def ai_options():
    active = _options.get()
    return active if active is not None else AIOptions(provider=config.LLM_PROVIDER)


@contextmanager
def use_ai_options(options):
    token = _options.set(options)
    try:
        yield
    finally:
        _options.reset(token)


def _openai_key():
    """Read a server-side credential without putting it in session state."""
    return os.getenv("OPENAI_API_KEY", "").strip() or _read_secret()


def _read_secret():
    try:
        with (config.PROJECT_ROOT / ".streamlit" / "secrets.toml").open("rb") as stream:
            value = tomllib.load(stream).get("OPENAI_API_KEY", "")
    except (OSError, ValueError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def cloud_configuration_issue(model=OPENAI_MODEL):
    """Check availability without constructing a client or exposing credentials."""
    if not model:
        return "missing_model"
    if not _openai_key():
        return "missing_key"
    try:
        if find_spec("openai") is None:
            return "missing_sdk"
    except (ImportError, ValueError):
        return "missing_sdk"
    if config.OPENAI_REASONING_EFFORT not in {"", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        return "configuration"
    return None


def openai_timeout():
    try:
        seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0
    return seconds if isfinite(seconds) and 0 < seconds <= 120 else 30.0


def create_openai_client(timeout):
    api_key = _openai_key()
    if not api_key:
        raise OpenAIRequestError("missing_key")
    try:
        from openai import OpenAI
    except ImportError:
        raise OpenAIRequestError("missing_sdk") from None
    # Fixed official endpoint, with no SDK retries or inherited custom endpoint.
    return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1", timeout=timeout, max_retries=0)


def openai_schema(schema):
    """Adapt Pydantic's schema to the supported strict JSON-schema subset."""
    result = deepcopy(schema)

    def strict(node):
        if isinstance(node, dict):
            for key in ("default", "uniqueItems"):
                node.pop(key, None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(result)
    return result


def _error_code(error):
    name = type(error).__name__
    if name == "APITimeoutError" or isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    status = getattr(error, "status_code", None)
    if status == 404 or getattr(error, "code", None) in ("model_not_found", "model_not_available", "model_access_denied"):
        return "model_unavailable"
    if status in {401, 403} or name in {"AuthenticationError", "PermissionDeniedError"}:
        return "authentication"
    if status == 429 or name == "RateLimitError":
        return "rate_limit"
    if name == "APIConnectionError" or isinstance(error, (ConnectionError, httpx.NetworkError)):
        return "connection"
    return "response"


def openai_response(*, model, task, messages, schema, max_output_tokens, retry=0, client=None):
    """Submit exactly one request; leave validation and repairs to the caller."""
    if not model:
        raise OpenAIRequestError("missing_model")
    effort = config.OPENAI_REASONING_EFFORT
    if effort not in {"", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise OpenAIRequestError("configuration")
    request = {
        "model": model, "input": messages, "max_output_tokens": max_output_tokens, "store": False,
        "text": {"format": {"type": "json_schema", "name": task, "schema": openai_schema(schema), "strict": True}},
    }
    if effort:
        request["reasoning"] = {"effort": effort}
    owned = client is None
    budget = ai_options().budget
    try:
        if owned:
            client = create_openai_client(openai_timeout())
        if budget is not None:
            budget.reserve()
        response = logged_request(
            client.responses.create, provider="openai", model=model, task=task, retry=retry,
            model_kwargs=request,
        )
        call = last_model_call(task)
        if not getattr(response, "output_text", None) or getattr(response, "status", "completed") != "completed":
            if call is not None:
                call.structured_output_valid = False
                call.error_code = "invalid_output"
            raise OpenAIRequestError("invalid_output")
        return response.output_text
    except (OpenAIRequestError, AssertionError):
        raise
    except Exception as error:
        code = _error_code(error)
        call = last_model_call(task)
        if call is not None:
            call.error_code = code
            call.timed_out = code == "timeout"
        raise OpenAIRequestError(code) from None
    finally:
        if owned and client is not None:
            client.close()
