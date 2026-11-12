"""Offline R014 scenarios or a live Ollama run through the unchanged graph."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT
from src.core.graph import build_graph
from src.core.state import UserEvent, new_run

SAMPLES = Path(__file__).resolve().parents[1] / "data/synthetic_samples"
CERTIFICATE = dict(document_type="student_enrollment_certificate", holder_name="Alex Example",
                   student_id="TEST-0001", institution="Example University", issue_date="2026-10-01")


def run_workflow(event: UserEvent, *, model: str = DEFAULT_OLLAMA_MODEL,
                 host: str = DEFAULT_OLLAMA_HOST, timeout: float = DEFAULT_OLLAMA_TIMEOUT) -> dict:
    """Return a JSON-compatible report, retaining the actual executed node path."""
    path, final = [], {}
    for kind, value in build_graph().stream(
        new_run(event, "demo-student"), stream_mode=["updates", "values"],
        config={"configurable": {"document_model": model, "document_host": host,
                                  "document_timeout": timeout}},
    ):
        if kind == "updates":
            path.extend(value)
        else:
            final = value
    return {
        "mode_note": "R014 only; planner and approval are MOCKED. No official approval.",
        "graph_path": path,
        "documents": [d.model_dump(mode="json") for d in final.get("documents", [])],
        "extracted_facts": final.get("extracted_facts", {}),
        "validation_issues": [e.model_dump(mode="json") for e in final.get("validation_errors", [])],
        "rule_results_and_provenance": [r.model_dump(mode="json") for r in final.get("relevant_requirements") or []],
        "final_response": final["final_response"],
    }


def run_offline_demo() -> list[dict]:
    """Real files, extraction, validation and graph; only Ollama is simulated.

    PASS/FAIL denotes whether the test expectation was met, not document approval.
    """
    scenarios = [
        ("Complete certificate", "student_enrollment.txt", CERTIFICATE, None, "supplied_and_accepted"),
        ("No certificate supplied", None, None, None, "missing"),
        ("Incomplete certificate", "enrollment_missing.txt",
         {**CERTIFICATE, "student_id": None, "issue_date": None}, None, "manual_review_required"),
        ("Ambiguous date", "enrollment_ambiguous.txt",
         {**CERTIFICATE, "issue_date": None, "ambiguous_fields": ["issue_date"]}, None, "manual_review_required"),
        ("Impossible date", "enrollment_invalid_date.txt",
         {**CERTIFICATE, "issue_date": None, "ambiguous_fields": ["issue_date"]}, None, "manual_review_required"),
        ("Unsupported input", "unsupported.exe", None, None, None),
        ("Ollama unavailable", "student_enrollment.txt", None,
         httpx.ConnectError("Simulated offline Ollama"), "manual_review_required"),
    ]
    reports = []
    for title, filename, payload, error, expected in scenarios:
        event = (UserEvent(kind="upload", file_path=str(SAMPLES / filename)) if filename else
                 UserEvent(kind="message", text="Check my student certificate"))
        with patch("processor.Client.chat", side_effect=error, return_value=SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)))):
            report = run_workflow(event)
        rules = report["rule_results_and_provenance"]
        actual = rules[0]["evaluation_result"] if rules else None
        success = actual == expected and bool(report["validation_issues"]) == (expected != "supplied_and_accepted")
        report.update(scenario=title, test_result="PASS" if success else "FAIL",
                      expected_rule_result=expected, execution_mode="OFFLINE: Ollama mocked")
        print(json.dumps(report, indent=2))
        if not success:
            raise AssertionError(f"Demo expectation failed: {title}")
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_path", nargs="?", type=Path, help="PDF/TXT input for a LIVE Ollama run")
    parser.add_argument("--offline", action="store_true", help="Run deterministic scenarios without Ollama")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--timeout", type=float, default=DEFAULT_OLLAMA_TIMEOUT)
    args = parser.parse_args()
    if args.offline:
        if args.file_path:
            parser.error("Do not provide a file path with --offline")
        run_offline_demo()
        return
    if not args.file_path:
        parser.error("Provide a file path for live processing or use --offline")
    report = run_workflow(UserEvent(kind="upload", file_path=str(args.file_path)),
                          model=args.model, host=args.host, timeout=args.timeout)
    report["execution_mode"] = "LIVE: real Ollama"
    print(json.dumps(report, indent=2))
    if report["validation_issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
