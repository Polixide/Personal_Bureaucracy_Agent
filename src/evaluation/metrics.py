"""Deterministic outcome/trajectory metrics; no production self-grading."""
from collections import Counter
import json
from math import ceil, comb
from statistics import fmean, median

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


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def exact_action_score(observed, alternatives):
    """Best F1 among labeled sets, breaking ties by fewer misses and extras."""
    if alternatives is None:
        return None
    observed = set(observed)
    scores = []
    for alternative in alternatives:
        required = set(alternative)
        if not required and not observed:
            scores.append((1.0, 1.0, 1.0, 0, 0))
            continue
        common = len(observed & required)
        precision = common / len(observed) if observed else 0.0
        recall = common / len(required) if required else 1.0
        f1 = ratio(2 * precision * recall, precision + recall) or 0.0
        scores.append((f1, precision, recall, len(required - observed), len(observed - required)))
    f1, precision, recall, misses, extras = max(scores, key=lambda score: (score[0], -score[3], -score[4]))
    return {
        "action_precision": precision, "action_recall": recall, "action_f1": f1,
        "missing_required_actions": misses, "extra_actions": extras,
    }


def outcome_assertions(spec, observed, action_score):
    """Evaluate labeled outcomes, even when the architecture's internal gate passes."""
    expected = spec["expected"]
    budget = spec["budget"]
    checks = {
        "stopping_condition": observed["stop"] == expected["stop"],
        "rule_outcomes": all(observed["rule_outcomes"].get(key) == value
                             for key, value in expected["rule_outcomes"].items()),
        "fallback_attribution": observed["fallback_used"] == expected["fallback"],
        "model_call_budget": observed["llm_calls"] <= budget["model_calls"],
        "action_budget": observed["validated_actions"] <= budget["actions"],
        "no_automatic_approval_or_execution": observed["safety_boundary_intact"],
    }
    if action_score is not None:
        checks["exact_action_set"] = action_score["action_f1"] == 1.0
        checks["unique_action_candidates"] = observed["validated_duplicate_references"] == 0
    if "action_types" in expected:
        checks["action_types"] = all(observed["action_types"].get(key) == expected["action_types"].get(key)
                                     for key in observed["action_types"])
    if "withhold_context" in expected:
        checks["withheld_conflicting_facts"] = set(expected["withhold_context"]) <= set(observed["withheld_context"])
    if "advisories" in expected:
        checks["preparation_advisory"] = set(expected["advisories"]) <= set(observed["advisories"])
        checks["rule_blocker_retained"] = set(expected["retained_blockers"]) <= set(observed["blocker_rule_ids"])
    for name in ("applied_rule_count", "preparation_complete", "reflection_attempts", "terminal_issue"):
        if name in expected:
            checks[name] = observed[name] == expected[name]
    if "extraction" in expected:
        checks["document_type"] = observed["classification_correct"]
        checks["exact_fields"] = observed["extraction_precision"] == observed["extraction_recall"] == 1.0
        checks["source_pages"] = observed["document_source_accuracy"] == 1.0
    if observed["source_attribution_accuracy"] is not None:
        checks["rule_source_attribution"] = observed["source_attribution_accuracy"] == 1.0
    if observed["scope_accuracy"] is not None:
        checks["procedure_scope"] = observed["scope_accuracy"] == 1.0
    return checks


def complete_total(items, key):
    """Zero calls have zero cost. One unknown value makes a measured total unknown."""
    values = [item.get(key) for item in items]
    return sum(values) if all(value is not None for value in values) else None


def summarize_calls(calls):
    received = [call for call in calls if call["request_success"]]
    evaluated = [call for call in received if call["structured_output_valid"] is not None]
    contractual = [call for call in received if call["candidate_contract_valid"] is not None]
    cloud = [call for call in calls if call.get("provider") == "openai"]
    embeddings = [call for call in calls if call["task"].startswith("rag_embedding_")]
    text_calls = [call for call in calls if not call["task"].startswith("rag_embedding_")]
    return {
        "llm_calls": len(calls), "transport_failures": len(calls) - len(received),
        "timeouts": sum(call["timed_out"] for call in calls),
        "model_outputs_received": len(received),
        "schema_outputs_evaluated": len(evaluated),
        "schema_valid_outputs": sum(call["structured_output_valid"] for call in evaluated),
        "schema_validity_rate": _mean(call["structured_output_valid"] for call in evaluated),
        "candidate_contract_outputs": len(contractual),
        "candidate_contract_valid_outputs": sum(call["candidate_contract_valid"] for call in contractual),
        "prompt_tokens": complete_total(text_calls, "prompt_tokens"),
        "completion_tokens": complete_total(text_calls, "completion_tokens"),
        "total_tokens": complete_total(text_calls, "total_tokens"),
        "provider_calls": dict(sorted(Counter(call.get("provider", "ollama") for call in calls).items())),
        "request_successes": len(received),
        "transport_retries": 0,  # Both transports dispatch once; Reflection is measured separately.
        "fallback_calls": sum(bool(call.get("fallback_used")) for call in calls),
        "embedding_calls": len(embeddings),
        "embedding_input_tokens": complete_total(embeddings, "prompt_tokens") if embeddings else None,
        "openai_requests": len(cloud),
        "openai_input_tokens": complete_total(cloud, "prompt_tokens") if cloud else None,
        "openai_output_tokens": complete_total(cloud, "completion_tokens") if cloud else None,
        "openai_reasoning_tokens": complete_total(cloud, "reasoning_tokens") if cloud else None,
        "openai_vision_requests": sum((row.get("image_count") or 0) > 0 for row in cloud),
        "openai_image_inputs": complete_total(cloud, "image_count") if cloud else None,
        "estimated_cost_usd": complete_total(cloud, "estimated_cost_usd") if cloud else None,
        "generation_tokens_per_second": _mean(call.get("generation_tokens_per_second") for call in text_calls),
        "llm_latency_seconds": complete_total(calls, "wall_seconds"),
    }


def retrieval_metrics(enabled, queries=(), *, k=None):
    """Compute retrieval metrics from labeled ordered results where available."""
    queries = list(queries) if enabled else []
    ranked = [query for query in queries if query.get("relevant") and query.get("retrieved") is not None] if k else []

    def scores(query):
        labels = set(query["relevant"])
        selected = list(dict.fromkeys(query["retrieved"][:k]))
        hits = len(labels.intersection(selected))
        first = next((index for index, item in enumerate(query["retrieved"][:k], 1) if item in labels), None)
        return hits / k, hits / len(labels), bool(hits), 1 / first if first else 0.0

    samples = [scores(query) for query in ranked]
    values = [fmean(column) for column in zip(*samples)] if samples else [None] * 4
    return {
        "rag_enabled": enabled, "rag_used": bool(queries), "retrieval_k": k if enabled else None,
        **dict(zip(("retrieval_precision_at_k", "retrieval_recall_at_k", "retrieval_hit_rate_at_k",
                    "retrieval_mrr_at_k"), values)),
        "retrieval_latency_seconds": _mean(query.get("latency_seconds") for query in queries),
        "retrieved_chunk_count": _mean(len(query["retrieved"]) for query in queries if query.get("retrieved") is not None),
        "retrieval_source_coverage": _mean(query.get("source_coverage") for query in queries),
    }


def _mean(values):
    known = [value for value in values if value is not None]
    return fmean(known) if known else None


def aggregate(rows):
    """Micro-average event ratios and macro-average labeled per-case measures."""
    def summed(key):
        return complete_total(rows, key)

    def rate(numerator, denominator):
        n, d = summed(numerator), summed(denominator)
        return ratio(n, d) if n is not None and d is not None else None

    latencies = sorted(row["end_to_end_latency_seconds"] for row in rows
                       if row.get("end_to_end_latency_seconds") is not None)
    planning = [row for row in rows if row.get("planning_requested")]
    repair = [row for row in planning if (row.get("reflection_attempts") or 0) > 0]
    fallback = [row["fallback_used"] for row in planning if row.get("fallback_used") is not None]
    failures = Counter(row["terminal_issue"] for row in rows if row.get("terminal_issue"))
    repair_failures = Counter(row["terminal_issue"] for row in repair if row.get("terminal_issue"))
    tokens = summed("total_tokens")
    return {
        "runs": len(rows),
        "task_success_rate": _mean(row.get("task_success") for row in rows),
        "latency_samples": len(latencies),
        "mean_end_to_end_latency_seconds": _mean(latencies),
        "p50_end_to_end_latency_seconds": median(latencies) if latencies else None,
        "p95_end_to_end_latency_seconds": latencies[ceil(0.95 * len(latencies)) - 1] if len(latencies) >= 20 else None,
        "mean_llm_calls": _mean(row.get("llm_calls") for row in rows),
        "openai_requests": summed("openai_requests"),
        "estimated_cost_usd": complete_total([row for row in rows if row.get("openai_requests")], "estimated_cost_usd")
                              if any(row.get("openai_requests") for row in rows) else None,
        "rag_queries": summed("rag_queries"),
        **{key: _mean(row.get(key) for row in rows) for key in (
            "retrieval_precision_at_k", "retrieval_recall_at_k", "retrieval_hit_rate_at_k",
            "retrieval_mrr_at_k", "retrieval_latency_seconds", "retrieved_chunk_count",
            "retrieval_source_coverage", "answer_correct", "citation_accuracy",
            "grounded_claim_rate", "unsupported_claim_rate", "faithfulness")},
        "mean_total_tokens": ratio(tokens, len(rows)) if tokens is not None else None,
        "mean_known_case_total_tokens": _mean(row.get("total_tokens") for row in rows),
        "token_total_samples": sum(row.get("total_tokens") is not None for row in rows),
        "schema_validity_rate": rate("schema_valid_outputs", "schema_outputs_evaluated"),
        "candidate_contract_validity_rate": rate("candidate_contract_valid_outputs", "candidate_contract_outputs"),
        "plan_validation_rate": rate("gate_acceptances", "materialized_gate_submissions"),
        "gate_attempt_acceptance_rate": rate("gate_acceptances", "gate_submissions"),
        "hallucinated_action_rate": rate("unsupported_selector_actions", "raw_proposed_actions"),
        "rule_reference_accuracy": rate("supplied_rule_references", "proposed_rule_references"),
        "fallback_rate": _mean(fallback) if len(fallback) == len(planning) else None,
        "reflection_recovery_rate": _mean(row.get("recovered_by_llm_repair") for row in repair),
        "mean_repair_attempts": _mean(row.get("reflection_attempts") for row in repair),
        "mean_reflection_attempts_per_case": _mean(row.get("reflection_attempts") for row in planning),
        "cases_entering_reflection": len(repair),
        "initial_rejections": summed("initial_rejection"),
        "llm_recoveries": summed("recovered_by_llm_repair"),
        "fallback_recoveries": summed("recovered_by_fallback"),
        "unchanged_invalid_replans": summed("unchanged_invalid_replans"),
        "changed_invalid_replans": summed("changed_invalid_replans"),
        "retry_exhaustions": summed("retry_exhausted"),
        "reflection_calls": summed("reflection_calls"),
        "reflection_prompt_tokens": summed("reflection_prompt_tokens"),
        "reflection_completion_tokens": summed("reflection_completion_tokens"),
        "reflection_total_tokens": summed("reflection_total_tokens"),
        "reflection_latency_seconds": summed("reflection_latency_seconds"),
        "human_review_cases": summed("human_review_required"),
        "terminal_failure_distribution": dict(sorted(failures.items())),
        "repair_failure_distribution": dict(sorted(repair_failures.items())),
        "validated_plan_origins": dict(sorted(Counter(row["validated_plan_origin"] for row in rows
                                                     if row.get("validated_plan_origin")).items())),
        **{f"mean_{key}": _mean(row.get(key) for row in rows) for key in ("worker_calls", "tool_calls", "graph_steps")},
        **{key: summed(key) for key in ("dependency_violations", "premature_submission_rejections",
                                       "premature_completion_claims", "unsupported_legal_claim_rejections",
                                       "unknown_as_failure_rejections", "unsafe_actions",
                                       "duplicate_candidate_references", "unknown_candidate_references",
                                       "missing_required_actions")},
        "mean_actions_per_validated_plan": _mean(row.get("validated_actions") for row in rows if row.get("plan_valid")),
        "plans_with_unresolved_blockers": sum(bool(row.get("remaining_blockers")) for row in rows if row.get("plan_valid")),
        "blocking_plan_samples": sum(row.get("remaining_blockers") is not None for row in rows if row.get("plan_valid")),
        "mean_validated_action_coverage": _mean(row.get("validated_action_coverage") for row in rows),
        **{key: _mean(row.get(key) for row in rows) for key in (
            "action_precision", "action_recall", "action_f1", "source_attribution_accuracy",
            "scope_accuracy", "extraction_precision", "extraction_recall", "document_source_accuracy",
            "contradiction_rate", "unknown_preservation_rate")},
    }
