from types import SimpleNamespace
from unittest.mock import patch
import pytest
from langgraph.types import Command
from src.core.graph import build_graph, supervisor
from src.core.state import ActionPlan, PlanStep, UserEvent, new_run
from src.memory.reflection import reflection
from config import MAX_REFLECTION_ATTEMPTS


def plan():
    return ActionPlan(actions=[PlanStep(step_id="ask",action="Provide entry details",priority=1,reason="Information needed",related_rule_id="R003")],summary="Gather information")

def request(): return new_run(UserEvent(kind="message",text="Help with my application"),"synthetic")

def test_valid_plan_pauses_for_approval_and_resumes():
    graph=build_graph(); config={"configurable":{"thread_id":"approval"}}
    with patch("src.agents.planner.create_action_plan",return_value=plan()):
        result=graph.invoke(request(),config)
    assert result.get("__interrupt__")
    assert "approved_plan" not in result
    result=graph.invoke(Command(resume=True),config)
    assert result["run_status"] == "APPROVED"
    assert result["approved_plan"].actions[0].step_id == "ask"
    assert not result.get("completed_steps")

def test_denial_blocks_state_update():
    graph=build_graph(); config={"configurable":{"thread_id":"denial"}}
    with patch("src.agents.planner.create_action_plan",return_value=plan()): graph.invoke(request(),config)
    result=graph.invoke(Command(resume=False),config)
    assert result["run_status"] == "APPROVAL_DENIED"
    assert "approved_plan" not in result

def test_three_invalid_repairs_then_stop():
    bad=plan(); bad.actions[0].reason="R003 failed"
    graph=build_graph(); config={"configurable":{"thread_id":"exhausted"},"recursion_limit":60}
    with patch("src.agents.planner.create_action_plan",return_value=bad) as planner:
        result=graph.invoke(request(),config)
    assert result["run_status"] == "HUMAN_REVIEW_REQUIRED"
    assert result["retry_count"] == MAX_REFLECTION_ATTEMPTS
    assert planner.call_count == 4
    assert len(result["repair_history"]) == 3
    assert "approved_plan" not in result

def test_valid_repair_revalidated_and_feedback_supplied():
    bad=plan(); bad.actions[0].reason="R003 failed"
    with patch("src.agents.planner.create_action_plan",side_effect=[bad,plan()]) as planner:
        result=build_graph().invoke(request(),{"configurable":{"thread_id":"repair"}})
    assert result["__interrupt__"]
    assert result["retry_count"] == 1
    assert planner.call_args.args[0]["repair_feedback"][0].code == "UNKNOWN_AS_FAILURE"
    assert not result["validation_errors"]

def test_malformed_output_is_bounded():
    with patch("src.agents.planner.Client") as client:
        client.return_value.chat.return_value=SimpleNamespace(message=SimpleNamespace(content="not JSON"))
        result=build_graph().invoke(request(),{"configurable":{"thread_id":"malformed"},"recursion_limit":60})
    assert result["run_status"] == "HUMAN_REVIEW_REQUIRED"
    assert client.return_value.chat.call_count == 4

def test_reflection_does_not_mutate_or_self_approve():
    state={"retry_count":0,"validation_errors":[],"candidate_plan":[]}
    result=reflection(state)
    assert state["retry_count"] == 0
    assert result["retry_count"] == 1
    assert "approval_decision" not in result

def test_supervisor_skips_unneeded_workers():
    assert supervisor({"relevant_requirements":[]})["next_step"] == "planner"
    assert supervisor({})["next_step"] == "rules_engine"
    assert supervisor({"run_status":"REPAIRING"})["next_step"] == "planner"


def test_upload_graph_preserves_observations_and_runs_workers_once():
    from config import PROJECT_ROOT
    from src.document_processing.schemas import StudentCertificate
    info=StudentCertificate(document_type="student_enrollment_certificate",holder_name="Alex Example",student_id="TEST-0001",institution="Example University",issue_date="2026-10-01")
    with patch("src.workers.document_analyzer.process_document",return_value=info) as processor, patch("src.agents.planner.create_action_plan",return_value=plan()) as planner:
        event=UserEvent(kind="upload",file_path=str(PROJECT_ROOT / "data/synthetic_samples/student_enrollment.txt"))
        graph=build_graph(); config={"configurable":{"thread_id":"upload"}}
        result=graph.invoke(new_run(event,"synthetic"),config)
    assert processor.call_count == planner.call_count == 1
    assert result["documents"][0].doc_type == "student_certificate"
    assert result["extracted_facts"]
    assert result["__interrupt__"]
    graph.invoke(Command(resume=True),config)
    with patch("src.agents.planner.create_action_plan",return_value=plan()), patch("src.workers.document_analyzer.process_document") as processor:
        result=graph.invoke(request(),config)
    processor.assert_not_called()
    assert len(result["documents"]) == 1
    assert result["extracted_facts"]


def test_gate0_rejection_does_not_call_planner():
    with patch("src.agents.planner.create_action_plan") as planner:
        result=build_graph().invoke(new_run(UserEvent(kind="message",text=""),"test"),{"configurable":{"thread_id":"empty"}})
    planner.assert_not_called()
    assert result["run_status"] == "INVALID_INPUT"
