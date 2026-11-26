"""Content-free diagnostics and native usage metadata for model requests."""
import sys
import json
import os
import re
from datetime import date
from math import isfinite
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import httpx


@dataclass(slots=True)
class ModelCall:
    model: str
    task: str
    retry: int
    request_success: bool
    timed_out: bool
    wall_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    server_total_seconds: float | None = None
    load_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    generation_seconds: float | None = None
    generation_tokens_per_second: float | None = None
    ttft_seconds: float | None = None
    structured_output_valid: bool | None = None
    candidate_contract_valid: bool | None = None
    fallback_used: bool = False
    proposed_actions: int | None = None
    unsupported_selector_actions: int | None = None
    duplicate_candidate_references: int | None = None
    unknown_candidate_references: int | None = None
    supplied_rule_references: int | None = None
    proposed_rule_references: int | None = None
    unsupported_rule_wording_actions: int | None = None
    provider: str = "ollama"
    response_model: str | None = None
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None
    image_count: int | None = None
    image_inputs: list[dict] | None = None
    document_text_pages: list[int] | None = None
    document_partial_text_pages: list[int] | None = None
    error_code: str | None = None
    cached_prompt_tokens: int | None = None
    estimated_cost_usd: float | None = None
    pricing: dict | None = None


_calls = ContextVar("model_call_collection", default=None)
_last_call = ContextVar("last_model_call", default=None)


@contextmanager
def collect_model_calls(*, propagate=False):
    """Collect content-free metrics, optionally including them in an enclosing run."""
    calls = []
    parent = _calls.get()
    collection = _calls.set(calls)
    last = _last_call.set(None)
    try:
        yield calls
    finally:
        _last_call.reset(last)
        _calls.reset(collection)
        if propagate and parent is not None:
            parent.extend(calls)


def last_model_call(component):
    calls = _calls.get()
    call = _last_call.get()
    return call if calls is not None and call is not None and call.task == component else None


def ollama_usage(response):
    """Read native metadata without retaining the response or approximating tokens."""
    def native(key):
        value = response.get(key) if isinstance(response, Mapping) else getattr(response, key, None)
        return value if type(value) is int and 0 <= value <= 2**63-1 else None

    def seconds(key):
        value = native(key)
        return value / 1_000_000_000 if value is not None else None

    prompt, completion = native("prompt_eval_count"), native("eval_count")
    generation = seconds("eval_duration")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        "server_total_seconds": seconds("total_duration"),
        "load_seconds": seconds("load_duration"),
        "prompt_eval_seconds": seconds("prompt_eval_duration"),
        "generation_seconds": generation,
        "generation_tokens_per_second": completion / generation if completion is not None and generation else None,
    }


def _native_count(obj, key):
    value = obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _pricing_for(model):
    try:
        price = json.loads(os.getenv("OPENAI_PRICING_JSON", "{}"))[model]
        as_of = date.fromisoformat(price["as_of"]).isoformat()
        rates = {key: price[key] for key in ("input_per_million", "output_per_million")}
        if "cached_input_per_million" in price:
            rates["cached_input_per_million"] = price["cached_input_per_million"]
        if not all(type(value) in (int, float) and isfinite(value) and 0 <= value <= 1_000_000
                   for value in rates.values()):
            return None
    except (ValueError, TypeError, KeyError, OverflowError):
        return None
    return {"as_of": as_of, **rates}


def openai_usage(response, model):
    usage = getattr(response, "usage", None)
    prompt = _native_count(usage, "input_tokens")
    completion = _native_count(usage, "output_tokens")
    total = _native_count(usage, "total_tokens")
    reasoning = _native_count(getattr(usage, "output_tokens_details", None), "reasoning_tokens")
    reported = getattr(response, "model", None)
    response_model = reported if isinstance(reported, str) and re.fullmatch(r"[\w.:-]{1,160}", reported) else None
    cached = _native_count(getattr(usage, "input_tokens_details", None), "cached_tokens")
    pricing = _pricing_for(model)
    cost = None
    if pricing is not None and prompt is not None and completion is not None:
        regular_rate = pricing["input_per_million"]
        if "cached_input_per_million" not in pricing:
            cost = (prompt * regular_rate + completion * pricing["output_per_million"]) / 1_000_000
        elif cached is not None and cached <= prompt:
            cost = ((prompt - cached) * regular_rate + cached * pricing["cached_input_per_million"]
                    + completion * pricing["output_per_million"]) / 1_000_000
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
            "cached_prompt_tokens": cached, "estimated_cost_usd": cost, "pricing": pricing,
            "reasoning_tokens": reasoning, "response_model": response_model}


def _record_call(model, task, retry, outcome, elapsed, response=None, *, provider="ollama", request=None):
    calls = _calls.get()
    if calls is not None:
        usage = openai_usage(response, model) if provider == "openai" else ollama_usage(response)
        entry = ModelCall(model=model, task=task, retry=retry, request_success=outcome == "success",
                          timed_out=outcome == "timeout", wall_seconds=elapsed, provider=provider, **usage)
        if request is not None and provider == "openai":
            entry.reasoning_effort = request.get("reasoning", {}).get("effort")
            entry.image_count = sum(
                part.get("type") == "input_image"
                for message in request.get("input", [])
                if isinstance(message.get("content"), list)
                for part in message["content"]
            )
        calls.append(entry)
        _last_call.set(entry)


def _emit(message):
    try:
        if sys.stderr is None:return
        print(message,file=sys.stderr,flush=True)
    except (OSError,ValueError,AttributeError):
        # A closed diagnostics stream must not change agent decisions or errors.
        pass


def logged_chat(client, *, component, retry=0, **kwargs):
    """Forward the exact request/result/errors; never log request or response data."""
    model=str(kwargs.get("model","")).replace("\r"," ").replace("\n"," ")
    prefix=f"[LLM] {component} model={model}"
    _emit(prefix+" started")
    if component in {"planner","reflection_replan"}:
        _emit(f"[LLM] {component} input_chars={sum(len(m.get('content','')) for m in kwargs.get('messages',[]) if isinstance(m.get('content'),str))}")
    started=perf_counter()
    try:
        response=client.chat(**kwargs)
    except Exception as error:
        elapsed=perf_counter()-started
        outcome="timeout" if isinstance(error,(TimeoutError,httpx.TimeoutException)) else "failure"
        _record_call(model, component, retry, outcome, elapsed)
        _emit(f"{prefix} {outcome} elapsed={elapsed:.1f}s")
        raise
    elapsed=perf_counter()-started
    _record_call(model, component, retry, "success", elapsed, response)
    if component in {"planner","reflection_replan"}:
        message=getattr(response,"message",None)
        content=message.get("content","") if isinstance(message,dict) else getattr(message,"content","")
        _emit(f"[LLM] {component} output_chars={len(content) if isinstance(content,str) else 0}")
    _emit(f"{prefix} success elapsed={elapsed:.1f}s")
    return response


def logged_request(request, *, provider, model, task, retry=0, model_kwargs):
    """Measure one provider request without logging payloads or SDK error text."""
    model = model.replace("\r", " ").replace("\n", " ")
    prefix = f"[LLM] {task} provider={provider} model={model}"
    _emit(prefix + " started")
    started = perf_counter()
    try:
        response = request(**model_kwargs)
    except Exception as error:
        elapsed = perf_counter() - started
        timeout = isinstance(error, (TimeoutError, httpx.TimeoutException)) or type(error).__name__ == "APITimeoutError"
        outcome = "timeout" if timeout else "failure"
        _record_call(model, task, retry, outcome, elapsed, provider=provider, request=model_kwargs)
        _emit(f"{prefix} {outcome} elapsed={elapsed:.3f}s")
        raise
    elapsed = perf_counter() - started
    _record_call(model, task, retry, "success", elapsed, response, provider=provider, request=model_kwargs)
    _emit(f"{prefix} success elapsed={elapsed:.3f}s")
    return response


def log_planner_fallback():
    for component in ("planner", "reflection_replan"):
        call = last_model_call(component)
        if call is not None:
            call.fallback_used = True
    _emit("[FALLBACK] deterministic planner activated")


def plan_hash(plan):
    """Compare structured actions without writing their contents to diagnostics."""
    return sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), default=lambda x: x.model_dump(mode="json")).encode()).hexdigest()


def log_plan_rules(label, rule_ids):
    safe = sorted({rid if isinstance(rid, str) and re.fullmatch(r"R\d{3}", rid) else "<invalid_rule_id>" for rid in rule_ids})
    _emit(f"[PLANNER] {label}={json.dumps(safe)}")


def log_validation_rejections(errors):
    for error in errors:
        rid = error.rule_id if error.rule_id and re.fullmatch(r"R\d{3}", error.rule_id) else "none"
        code = error.code if re.fullmatch(r"[A-Z_]+", error.code) else "VALIDATION_ERROR"
        _emit(f"[GATE1] rejected_rule_id={rid} rejection_reason={code}")
        if code=="PLANNER_SCHEMA_ERROR":
            match=re.fullmatch(r"Planner schema issue: ([a-z_]+)",error.message)
            _emit("[GATE1] schema_issue="+(match.group(1) if match else "invalid_plan_structure"))


def log_reflection(attempt, errors=(), *, previous_plan_hash=None, new_plan_hash=None):
    _emit(f"[REFLECTION] attempt={int(attempt)}")
    if new_plan_hash is not None:
        _emit(f"[REFLECTION] new_plan_hash={new_plan_hash}")
    else:
        codes = [e.code if re.fullmatch(r"[A-Z_]+", e.code) else "VALIDATION_ERROR" for e in errors]
        _emit(f"[REFLECTION] received_validation_errors={json.dumps(codes)}")
        _emit(f"[REFLECTION] previous_plan_hash={previous_plan_hash}")
