"""Deterministic outcome/trajectory metrics; no production self-grading."""
from collections import Counter
import json
from math import comb

LOOP_FAILURE_CATEGORIES = {"TOOL_LOOP_LIMIT", "RETRIEVAL_LOOP_LIMIT", "TOOL_BUDGET_EXHAUSTED",
    "TOOL_INVOCATION_LIMIT", "IDENTICAL_TOOL_FAILURE_LIMIT", "DUPLICATE_TOOL_CALL"}


def trajectory_metrics(events, expected_tools=None, expected_arguments=None):
    rows = [item.model_dump(mode="json") if hasattr(item,"model_dump") else item for item in events]
    rows = [row.get("observation", row) for row in rows if isinstance(row,dict)]
    tools = [row for row in rows if row.get("name") and row.get("status") in {"SUCCEEDED","FAILED"} and not row.get("replayed")]
    expected = set(expected_tools or [])
    fingerprints = [row.get("fingerprint") or json.dumps({"name":row.get("name"),"arguments":row.get("arguments",{})},sort_keys=True,default=str) for row in tools]
    schema_errors = sum(row.get("failure_category")=="TOOL_SCHEMA_ERROR" for row in tools)
    selected_correct = sum(row.get("name") in expected for row in tools)
    argument_rows = [row for row in tools if expected_arguments and row.get("name") in expected_arguments]
    argument_correct = sum(all(row.get("arguments", {}).get(key)==value for key,value in expected_arguments[row["name"]].items()) for row in argument_rows)
    return {
        "path_length":len(rows), "tool_calls":len(tools),
        "tool_selection_precision":selected_correct/len(tools) if tools and expected else None,
        "schema_adherence":1-schema_errors/len(tools) if tools else 1.0,
        "argument_accuracy":argument_correct/len(argument_rows) if argument_rows else None,
        "reflection_count":sum(row.get("kind")=="reflection" or row.get("event")=="reflection" for row in rows),
        "duplicate_tool_calls":sum(count-1 for count in Counter(fingerprints).values()),
        "loop_stopped":any(row.get("failure_category") in LOOP_FAILURE_CATEGORIES for row in rows),
        "tool_elapsed_seconds":sum(float(row.get("elapsed_seconds",0)) for row in tools),
        "approval_previews":sum(row.get("status")=="APPROVAL_REQUIRED" for row in rows),
        "human_rejections":sum(row.get("status")=="REJECTED" for row in rows),
        "llm_calls":sum(int(row.get("llm_calls",0)) for row in rows),
        "input_characters":sum(int(row.get("input_characters",0)) for row in rows),
        "output_characters":sum(int(row.get("output_characters",0)) for row in rows),
    }


def reliability_metrics(successes, k=3):
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive")
    if any(type(item) is not bool for item in successes):
        raise ValueError("Success outcomes must be deterministic booleans")
    n, correct = len(successes), sum(successes)
    groups = [successes[index:index+k] for index in range(0,n,k) if len(successes[index:index+k])==k]
    return {"runs":n,"successes":correct,"k":k,
        "single_run_success":correct/n if n else None,
        "pass_at_k":1-(comb(n-correct,k)/comb(n,k) if n-correct>=k else 0) if n>=k else None,
        "pass_power_k":sum(all(group) for group in groups)/len(groups) if groups else None,
        "complete_repeat_groups":len(groups)}


def state_metrics(state):
    """Read-only technical summary joining existing traces without prompt replay."""
    events = list(state.get("tool_outcomes", []))
    events += [{"kind":"reflection"} for _ in state.get("repair_history", [])]
    result = trajectory_metrics(events)
    diagnostics = state.get("assistant_diagnostics", {})
    result["loop_stopped"] = result["loop_stopped"] or bool(LOOP_FAILURE_CATEGORIES.intersection(diagnostics.get("failure_categories", [])))
    result.update(llm_calls=diagnostics.get("llm_calls", 0), input_characters=diagnostics.get("llm_input_chars") or diagnostics.get("input_chars", 0),
        output_characters=diagnostics.get("output_chars", 0), chat_elapsed_seconds=diagnostics.get("elapsed_seconds", 0),
        planner_seconds=state.get("planner_diagnostics", {}).get("duration_seconds", 0))
    document_diagnostics = []
    for document in state.get("documents", []):
        info = getattr(document, "info", None)
        if info is not None: document_diagnostics.append(getattr(info, "extraction_diagnostics", {}) or {})
    result["document_classification_seconds"] = sum(call.get("duration_seconds", 0) for data in document_diagnostics for call in data.get("model_calls", []) if call.get("stage")=="classification")
    result["document_extraction_seconds"] = sum(call.get("duration_seconds", 0) for data in document_diagnostics for call in data.get("model_calls", []) if call.get("stage")=="extraction")
    result["ocr_seconds"] = sum(data.get("ocr_duration_seconds", 0) for data in document_diagnostics)
    result["cloud_fallback_seconds"] = sum(data.get("fallback_duration_seconds", 0) for data in document_diagnostics)
    result["cloud_fallback_frequency"] = sum(bool(data.get("fallback_used")) for data in document_diagnostics)/len(document_diagnostics) if document_diagnostics else 0
    result["rag_seconds"] = sum(row.get("elapsed_seconds", 0) for row in events if row.get("name")=="search_official_sources")
    return result
