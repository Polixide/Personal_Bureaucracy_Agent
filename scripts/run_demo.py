"""Synthetic graph demo. Default is offline; --live explicitly calls Ollama."""
import argparse
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from langgraph.types import Command
from config import PROJECT_ROOT, DATABASE_PATH
from src.core.graph import build_graph
from src.core.state import ActionPlan, PlanStep, UserEvent, new_run
from src.document_processing.schemas import StudentCertificate
from src.memory.checkpointer import sqlite_checkpointer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true",help="Use the configured local Ollama model")
    parser.add_argument("--thread-id", default="demo-user", help="Stable identity for the persisted Digital Twin")
    parser.add_argument("--show-state", action="store_true", help="Read the saved state without executing or resuming nodes")
    parser.add_argument("--persistence-demo", action="store_true", help="Close/reopen SQLite before reviewing the saved plan")
    args=parser.parse_args()
    config={"configurable":{"thread_id":args.thread_id},"recursion_limit":60}
    event=UserEvent(kind="upload",file_path=str(PROJECT_ROOT / "data/synthetic_samples/student_enrollment.txt"))
    with ExitStack() as stack:
        checkpoint = stack.enter_context(sqlite_checkpointer())
        graph = build_graph(checkpoint)
        snapshot = graph.get_state(config)
        print("Database:", DATABASE_PATH)
        print("Thread:", args.thread_id)
        if args.show_state:
            if not snapshot.values:
                print("No saved state for this thread.")
                return
            print("Saved documents:", len(snapshot.values.get("documents", [])))
            print("Saved fact groups:", len(snapshot.values.get("extracted_facts", {})))
            print("Status:", snapshot.values.get("run_status"))
            print("Pending nodes:", ", ".join(snapshot.next) or "none")
            print("Approval decision:", snapshot.values.get("approval_decision"))
            return
        if not args.live:
            # Mock only the model transport. Extraction, schemas, adapters,
            # requirements, validator, graph and approval are all real.
            info=StudentCertificate(document_type="student_enrollment_certificate",holder_name="Alex Example",student_id="TEST-0001",institution="Example University",issue_date="2026-10-01")
            plan=ActionPlan(actions=[PlanStep(step_id="confirm_scope",action="Confirm Sabanci incoming exchange status and requested stay dates",priority=1,reason="Scope and dates are not yet known",related_rule_id="R001",action_type="information")],summary="Gather missing application information")
            response=lambda model:SimpleNamespace(message=SimpleNamespace(content=model.model_dump_json()))
            client=stack.enter_context(patch("src.document_processing.processor.Client"))
            client.return_value.chat.side_effect=[
                SimpleNamespace(message=SimpleNamespace(content='{"document_type":"student_enrollment_certificate"}')),
                response(info),
            ]
            planner=stack.enter_context(patch("src.agents.planner.Client"))
            planner.return_value.chat.return_value=response(plan)
        if snapshot.next:
            # Reading/reopening never resumes a pending approval or executes nodes.
            if snapshot.next != ("human_approval",):
                parser.exit(1, "Thread has unfinished non-approval work; inspect it with --show-state.\n")
            result = dict(snapshot.values)
        else:
            result=graph.invoke(new_run(event,args.thread_id),config)
        if args.persistence_demo:
            before = graph.get_state(config).values
            stack.close()
            checkpoint = stack.enter_context(sqlite_checkpointer())
            graph = build_graph(checkpoint)
            after = graph.get_state(config).values
            if after != before:
                raise RuntimeError("Checkpoint changed after reopening")
            result = dict(after)
            print("SQLite closed/reopened: saved documents, facts and plan restored.")
        print("Documents:", ", ".join(d.doc_type or "unknown" for d in result.get("documents", [])))
        results=result.get("relevant_requirements") or []
        print("Requirements:", ", ".join(f"{status}={sum(r.status == status for r in results)}" for status in ("passed", "failed", "missing_information")))
        print("Repair attempts:", result.get("retry_count", 0))
        if graph.get_state(config).next == ("human_approval",):
            print("Validated proposed plan:")
            for action in result["candidate_plan"]: print(" -",action.action)
            approved=input("Approve this plan? [y/N] ").strip().lower() == "y"
            result=graph.invoke(Command(resume=approved),config)
        print(result["final_response"])

if __name__ == "__main__": main()
