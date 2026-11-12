"""Run, inspect, or resume an R014 session in a local SQLite checkpoint file."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT
from scripts.run_demo import CERTIFICATE, SAMPLES
from src.core.state import UserEvent, new_run
from src.memory.checkpointer import persistent_graph, session_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_path", nargs="?", type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--db", type=Path, default=Path(".local/checkpoints.sqlite"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inspect", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--offline", action="store_true", help="Use the synthetic enrollment fixture; mock only Ollama")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--timeout", type=float, default=DEFAULT_OLLAMA_TIMEOUT)
    args = parser.parse_args()
    if args.file_path and (args.inspect or args.resume or args.offline):
        parser.error("A file path is only used for a new live run.")
    if not (args.file_path or args.inspect or args.resume or args.offline):
        parser.error("Provide a file, --offline, --inspect, or --resume.")
    try:
        config = session_config(args.session)
    except ValueError as error:
        parser.error(str(error))
    config["configurable"].update(document_model=args.model, document_host=args.host,
                                    document_timeout=args.timeout)
    with persistent_graph(args.db) as graph:
        if args.resume and not graph.get_state(config).values:
            parser.error("No checkpoint exists for this session.")
        path = []
        if not args.inspect:
            event_path = SAMPLES / "student_enrollment.txt" if args.offline else args.file_path
            run_input = None if args.resume else new_run(
                UserEvent(kind="upload", file_path=str(event_path)), user_id=args.session)
            mock = (patch("processor.Client.chat", return_value=SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(CERTIFICATE)))) if args.offline else nullcontext())
            with mock:
                for update in graph.stream(run_input, config=config, stream_mode="updates"):
                    path.extend(update)
        snapshot = graph.get_state(config)
        state = snapshot.values
        report = {
            "session_id": args.session, "graph_path_this_execution": path,
            "pending_nodes": list(snapshot.next),
            "mode_note": "R014 only; planner and approval MOCKED; no reflection implemented.",
            "execution_mode": "OFFLINE: Ollama mocked" if args.offline else "INSPECT/RESUME" if args.inspect or args.resume else "LIVE",
            "documents": [d.model_dump(mode="json") for d in state.get("documents", [])],
            "extracted_facts": state.get("extracted_facts", {}),
            "validation_issues": [i.model_dump(mode="json") for i in state.get("validation_errors", [])],
            "rule_results_and_provenance": [r.model_dump(mode="json") for r in state.get("relevant_requirements") or []],
            "final_response": state.get("final_response", "No stored state for this session."),
        }
        print(json.dumps(report, indent=2, default=str))
        if not args.inspect and state.get("validation_errors"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
