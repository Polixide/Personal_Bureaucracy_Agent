"""Run or compare local synthetic preparation benchmarks and export CSV/JSON."""
import argparse
import csv
import json
import os
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from config import (
    DOCUMENT_OLLAMA_TIMEOUT, MAX_REFLECTION_ATTEMPTS, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT, PLANNER_MODEL,
    PLANNER_MAX_ACTIONS, PLANNER_MAX_OUTPUT_TOKENS, PROJECT_ROOT, ASSISTANT_MODEL, OPENAI_MODEL, OPENAI_DOCUMENT_MODEL,
)
from src.evaluation.metrics import aggregate, summarize_calls
from src.core.model_provider import AIOptions, ERROR_MESSAGES, RequestBudget, cloud_configuration_issue, openai_timeout

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
SPEC_PATH = EVALUATION_DIR / "benchmark.v1.json"
ARCHITECTURE = "deterministic_router"
MODES = ("offline", "live", "document-offline", "document-live", "rag-offline", "rag-live")
LIVE_MODES = {"live", "document-live", "rag-live"}
CSV_FIELDS = (
    "run_id", "benchmark_version", "scenario_id", "architecture", "source_version", "model",
    "execution_mode", "timestamp", "repeat_index", "document_format", "task_success", "schema_valid",
    "plan_valid", "action_precision", "action_recall", "action_f1", "llm_calls", "prompt_tokens",
    "completion_tokens", "total_tokens", "llm_latency_seconds", "end_to_end_latency_seconds",
    "worker_calls", "tool_calls", "graph_steps", "reflection_attempts", "fallback_used",
    "human_review_required", "human_approval_reached", "unsafe_actions", "remaining_blockers",
    "rag_enabled", "rag_queries", "retrieval_precision_at_k", "retrieval_recall_at_k", "retrieval_latency_seconds",
    "gate_submissions", "materialized_gate_submissions", "gate_acceptances", "gate_rejections",
    "validated_plan_origin", "transport_failures", "timeouts",
    "schema_outputs_evaluated", "schema_valid_outputs", "raw_proposed_actions",
    "unsupported_selector_actions", "duplicate_candidate_references", "unknown_candidate_references",
    "missing_required_actions", "validated_action_coverage", "failed_tool_calls", "terminal_issue",
    "provider", "request_successes", "transport_retries", "fallback_calls", "embedding_calls",
    "embedding_input_tokens", "openai_requests", "openai_input_tokens", "openai_output_tokens", "estimated_cost_usd",
    "generation_tokens_per_second", "reflection_calls", "rag_used", "retrieval_k", "retrieval_mrr_at_k",
    "retrieval_hit_rate_at_k", "retrieved_chunk_count", "retrieval_source_coverage", "retrieval_mode",
    "embedding_model", "answer_correct", "citation_accuracy", "grounded_claim_rate", "unsupported_claim_rate", "faithfulness",
    "openai_reasoning_tokens", "openai_vision_requests", "openai_image_inputs", "index_status", "indexed_chunks",
)


def load_suite(path=SPEC_PATH):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def export_results(directory, label, payload):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label):
        raise ValueError("Use a short alphanumeric export label with optional dashes or underscores")
    json_path, csv_path = (directory / f"{label}.{kind}" for kind in ("json", "csv"))
    if json_path.exists() or csv_path.exists():
        raise ValueError("An export with this label already exists")
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["runs"])
    return json_path, csv_path


def _packages():
    result = {}
    for name in ("langgraph", "ollama", "openai", "pydantic", "httpx"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _selected_model(args):
    if args.provider == "openai":
        return OPENAI_DOCUMENT_MODEL if args.mode == "document-live" else OPENAI_MODEL
    return {"document-live": OLLAMA_MODEL, "rag-live": ASSISTANT_MODEL}.get(args.mode, PLANNER_MODEL)


def _metadata(args):
    live = args.mode in LIVE_MODES
    return {
        "architecture": ARCHITECTURE, "source_version": args.source_version,
        "execution_mode": args.mode,
        "provider": args.provider if live else "scripted",
        "model": _selected_model(args) if live else "scripted",
        "max_openai_requests": args.max_openai_requests if args.provider == "openai" else None,
        "retrieval_configuration": {"embedding_model": os.getenv("OLLAMA_EMBED_MODEL") or None if args.mode == "rag-live" else None,
                                    "configured": args.mode.startswith("rag-")},
        "cost_method": "Operator-supplied USD prices per million tokens; missing rates or usage leave cost unknown."
                       if args.provider == "openai" else None,
        "model_revision": args.model_revision if live else None,
        "python": platform.python_version(), "packages": _packages(),
        "ollama_server_version": args.ollama_server_version if live else None,
        "hardware": args.hardware, "warm_cold": args.model_state if live else None,
        "repeats": args.repeat, "seed": None,
        "document_format": args.document_format if args.mode.startswith("document-") else None,
        "adapter_scope": "marco_v2_milestone_document_preparation",
        "document_provider": ("openai_opt_in" if args.provider == "openai" else "ollama_local_only")
                             if args.mode.startswith("document-") else None,
        "timing_scope": "synthetic_input_to_terminal_graph_snapshot",
        "worker_count_scope": "router_specialized_nodes",
        "graph_step_scope": "router_named_update_events",
        "model_configuration": {
            "temperature": 0, "think": False, "stream": False, "planner_num_predict": PLANNER_MAX_OUTPUT_TOKENS,
            "planner_timeout_seconds": openai_timeout() if args.provider == "openai" else OLLAMA_TIMEOUT,
            "document_timeout_seconds": openai_timeout() if args.provider == "openai" else DOCUMENT_OLLAMA_TIMEOUT,
            "planner_selected_candidate_budget": PLANNER_MAX_ACTIONS, "max_reflection_attempts": MAX_REFLECTION_ATTEMPTS,
        },
    }


def _pack_runs(rows, metadata, benchmark_version):
    groups = sorted({row["scenario_id"] for row in rows})
    tasks = sorted({call["task"] for row in rows for call in row["model_calls"]})
    return {
        "benchmark_version": benchmark_version, "metadata": metadata, "runs": rows,
        "summary": aggregate(rows),
        "by_scenario": {key: aggregate([row for row in rows if row["scenario_id"] == key]) for key in groups},
        "by_model_task": {key: summarize_calls([call for row in rows for call in row["model_calls"] if call["task"] == key])
                          for key in tasks},
    }


def compare_exports(first, second):
    """Require the same benchmark, mode and case counts; flag environmental mismatches."""
    if first["benchmark_version"] != second["benchmark_version"]:
        raise ValueError("Benchmark versions differ")
    left, right = first["metadata"], second["metadata"]
    if left["execution_mode"] != right["execution_mode"] or left.get("document_format") != right.get("document_format"):
        raise ValueError("Execution modes or document formats differ")
    def case_counts(export):
        return Counter(row["scenario_id"] for row in export["runs"])

    counts = case_counts(first)
    if not counts or counts != case_counts(second):
        raise ValueError("Scenario or repeat coverage differs")
    live = left["execution_mode"] in LIVE_MODES
    def status(key):
        if left.get(key) is None or right.get(key) is None:
            return "unknown"
        return "same" if left[key] == right[key] else "different"

    keys = ["hardware", "timing_scope", "worker_count_scope", "graph_step_scope", "model", "model_configuration",
            "python", "packages", "provider", "retrieval_configuration"]
    if live:
        keys += ["model_revision", "warm_cold", "ollama_server_version", "seed"]
    conditions = {key: status(key) for key in keys}
    return {
        "benchmark_version": first["benchmark_version"], "execution_mode": left["execution_mode"],
        "scenario_repeats": dict(sorted(counts.items())), "conditions": conditions,
        "timing_comparable_as_controlled": live and all(conditions[key] == "same" for key in (
            "hardware", "timing_scope", "model", "model_configuration", "model_revision", "warm_cold", "ollama_server_version")),
        "worker_counts_comparable": conditions["worker_count_scope"] == "same",
        "graph_steps_comparable": conditions["graph_step_scope"] == "same",
        "runs": [
            {"architecture": meta["architecture"], "source_version": meta.get("source_version"),
             "model": meta.get("model"), "summary": aggregate(export["runs"])}
            for meta, export in ((left, first), (right, second))
        ],
    }


def _read_export(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(EVALUATION_DIR.resolve()) or resolved.suffix.lower() != ".json":
        raise ValueError("Place comparison JSON exports inside evaluation/")
    with resolved.open(encoding="utf-8") as stream:
        return json.load(stream)


def _operator_text(raw):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .,:/@_()+-]{0,119}", raw) or "://" in raw:
        raise argparse.ArgumentTypeError("Use a short version or hardware label without a URL")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="offline")
    parser.add_argument("--scenario", action="append", metavar="ID", help="Repeat to select multiple cases")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama", help="Model provider for live runs")
    parser.add_argument("--allow-openai", action="store_true", help="Authorize OpenAI requests containing only this suite's synthetic data and public guidance")
    parser.add_argument("--max-openai-requests", type=int, help="Required positive request cap for a live OpenAI benchmark")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--document-format", choices=("pdf", "txt", "scan"), default="txt",
                        help="Use scan for a synthetic image-only PDF")
    parser.add_argument("--label", help="Name the CSV and JSON export under evaluation/results")
    parser.add_argument("--source-version", type=_operator_text)
    parser.add_argument("--model-revision", type=_operator_text, help="Operator-recorded Ollama model digest/tag revision")
    parser.add_argument("--hardware", type=_operator_text, help="Operator-supplied CPU/GPU/RAM description")
    parser.add_argument("--model-state", choices=("warm", "cold", "mixed"))
    parser.add_argument("--ollama-server-version", type=_operator_text)
    parser.add_argument("--compare", nargs=2, metavar=("ROUTER_JSON", "PEER_JSON"))
    args = parser.parse_args(argv)

    if args.compare:
        try:
            comparison = compare_exports(*(_read_export(path) for path in args.compare))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        print(json.dumps(comparison, indent=2, allow_nan=False))
        return 0
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.label and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.label):
        parser.error("Use a short alphanumeric --label with optional dashes or underscores")
    budget = None
    if args.provider == "openai":
        if args.mode not in LIVE_MODES or not args.allow_openai or not args.max_openai_requests or args.max_openai_requests < 1:
            parser.error("OpenAI requires a live mode, --allow-openai and a positive --max-openai-requests")
        if problem := cloud_configuration_issue(_selected_model(args)):
            parser.error(ERROR_MESSAGES[problem])
        budget = RequestBudget(args.max_openai_requests)
    elif args.allow_openai or args.max_openai_requests is not None:
        parser.error("OpenAI consent and budget options require --provider openai")
    elif args.mode in LIVE_MODES:
        if OLLAMA_HOST != "http://127.0.0.1:11434":
            parser.error("Live measurement requires the local Ollama endpoint")
        model = _selected_model(args)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,119}", model) or "://" in model:
            parser.error("Use a valid local Ollama model tag")
    elif args.model_revision or args.model_state or args.ollama_server_version:
        parser.error("Model revision, server version and warm/cold labels apply to live runs")
    suite = load_suite()
    scenarios = [row for row in suite["scenarios"] if args.mode in row["modes"]]
    if args.scenario:
        requested = set(args.scenario)
        scenarios = [row for row in scenarios if row["id"] in requested]
        if {row["id"] for row in scenarios} != requested:
            parser.error("One or more scenario IDs are unavailable in the selected mode")

    if not RESULTS_DIR.resolve().is_relative_to(PROJECT_ROOT.resolve() / "evaluation" / "results"):
        parser.error("The results directory must be inside the local evaluation directory")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_label = args.label or f"{args.mode}-{uuid4().hex[:12]}"
    if any((RESULTS_DIR / f"{run_label}.{extension}").exists() for extension in ("csv", "json")):
        parser.error("An export with this label already exists")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from src.evaluation.router import run_case

    metadata = _metadata(args)
    ai = AIOptions(provider=args.provider, budget=budget)
    rows = []
    for repeat in range(1, args.repeat + 1):
        for spec in scenarios:
            timestamp = datetime.now(timezone.utc).isoformat()
            result = run_case(suite, spec, args.mode, RESULTS_DIR, document_format=args.document_format, ai=ai)
            retrieval = result.get("retrieval_configuration") or {}
            rows.append({
                "run_id": uuid4().hex, "benchmark_version": suite["benchmark_version"], "scenario_id": spec["id"],
                "repeat_index": repeat, "timestamp": timestamp,
                **{key: metadata[key] for key in ("architecture", "source_version", "provider", "model", "execution_mode", "document_format")},
                "retrieval_mode": retrieval.get("mode"), "embedding_model": retrieval.get("embedding_model"),
                **result,
            })
    payload = _pack_runs(rows, metadata, suite["benchmark_version"])
    paths = export_results(RESULTS_DIR, run_label, payload)
    print(f"{sum(row['task_success'] for row in rows)}/{len(rows)} cases passed in {args.mode} mode.")
    for row in rows:
        failed = [key for key, passed in row["assertions"].items() if not passed]
        if failed:
            print(f"{row['scenario_id']} run {row['repeat_index']} failed checks {', '.join(failed)}")
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))
    return 0 if all(row["task_success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
