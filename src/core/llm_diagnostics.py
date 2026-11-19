"""Content-free stderr diagnostics around existing Ollama transport calls."""
import sys
import json
import re
from hashlib import sha256
from time import perf_counter

import httpx


def _emit(message):
    try:
        if sys.stderr is None:return
        print(message,file=sys.stderr,flush=True)
    except (OSError,ValueError,AttributeError):
        # A closed diagnostics stream must not change agent decisions or errors.
        pass


def logged_chat(client, *, component, **kwargs):
    """Forward the exact request/result/errors; never log request or response data."""
    model=str(kwargs.get("model","")).replace("\r"," ").replace("\n"," ")
    prefix=f"[LLM] {component} model={model}"
    started=perf_counter()
    _emit(prefix+" started")
    if component in {"planner","reflection_replan"}:
        _emit(f"[LLM] {component} input_chars={sum(len(m.get('content','')) for m in kwargs.get('messages',[]) if isinstance(m.get('content'),str))}")
    try:
        response=client.chat(**kwargs)
    except Exception as error:
        outcome="timeout" if isinstance(error,(TimeoutError,httpx.TimeoutException)) else "failure"
        _emit(f"{prefix} {outcome} elapsed={perf_counter()-started:.1f}s")
        raise
    if component in {"planner","reflection_replan"}:
        message=getattr(response,"message",None)
        content=message.get("content","") if isinstance(message,dict) else getattr(message,"content","")
        _emit(f"[LLM] {component} output_chars={len(content) if isinstance(content,str) else 0}")
    _emit(f"{prefix} success elapsed={perf_counter()-started:.1f}s")
    return response


def log_planner_fallback():
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
