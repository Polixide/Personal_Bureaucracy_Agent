"""Whole-graph V2 integration: offline tools, exact approval and persisted Twin."""
from copy import deepcopy
from datetime import datetime,timezone
from unittest.mock import patch
import pytest
from langgraph.types import Command
from src.core.graph import build_graph,tool_execute
from src.core.state import UserEvent,new_run,ProcedureState
from src.memory.checkpointer import sqlite_checkpointer,initialize_procedure_thread
from src.tools.schemas import ToolCall
from src.tools.registry import execute_tool
from test_procedure import ready_bundle,RECORDED_AT


def run_tool(graph,config,name,arguments=None,call_id="synthetic-call"):
    call=ToolCall(call_id=call_id,name=name,arguments=arguments or {})
    return graph.invoke(new_run(UserEvent(kind="run_tool",tool_call=call.model_dump(mode="json")),"synthetic-user"),config)


def test_state_chat_bypasses_planner_and_preserves_existing_assessment(ready_bundle):
    graph,config,old=ready_bundle
    with patch("src.agents.planner.create_action_plan") as planner:
        state=graph.invoke(new_run(UserEvent(kind="assistant_message",text="What documents am I missing?"),"synthetic-user"),config)
    planner.assert_not_called()
    assert state["assistant_messages"][-1]["role"]=="assistant"
    assert state["assistant_diagnostics"]["llm_calls"]==0
    for key in ("procedure","documents","verification_records","relevant_requirements"):
        assert state[key]==old[key]


@pytest.mark.parametrize("approved",[True,False])
def test_calendar_interrupt_resumes_with_explicit_approval_only(tmp_path,approved):
    db=tmp_path/"approval.db";config={"configurable":{"thread_id":"calendar"}}
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver);initialize_procedure_thread(graph,config)
        state=run_tool(graph,config,"create_calendar_reminder",{"title":"Personal preparation target","start_at":"2030-01-15T10:00:00+03:00"})
        assert state["__interrupt__"] and graph.get_state(config).next==("tool_approval",)
        assert not state.get("calendar_reminders")
        pending=deepcopy(state["pending_tool_proposal"])
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        assert graph.get_state(config).next==("tool_approval",)
        state=graph.invoke(Command(resume=approved),config)
        assert state["pending_tool_proposal"] is None
        assert state["tool_outcomes"][-1]["status"]==("SUCCEEDED" if approved else "REJECTED")
        assert bool(state.get("calendar_reminders")) is approved
        assert state["procedure"]==ProcedureState()
        assert not graph.get_state(config).next
        assert "state_updates" not in state["tool_outcomes"][-1]


def test_tool_execution_replay_does_not_duplicate_reminder():
    graph=build_graph();config={"configurable":{"thread_id":"replay"}}
    initialize_procedure_thread(graph,config)
    arguments={"title":"Personal reminder","start_at":"2030-01-15T10:00:00+03:00"}
    run_tool(graph,config,"create_calendar_reminder",arguments)
    first=graph.invoke(Command(resume=True),config)
    second=run_tool(graph,config,"create_calendar_reminder",arguments)
    assert len(second["calendar_reminders"])==1
    assert second["last_tool_outcome"]["replayed"] is True
    assert not graph.get_state(config).next


def test_exact_preview_change_blocks_approved_action():
    call=ToolCall(call_id="calendar",name="create_calendar_reminder",arguments={"title":"Before","start_at":"2030-01-15T10:00:00+03:00"})
    preview=execute_tool({},call,adapters={})
    pending={"proposal_id":call.call_id,"tool_call":call.model_dump(mode="json"),"preview":preview.diff}
    pending["tool_call"]["arguments"]["title"]="Changed after preview"
    result=tool_execute({"pending_tool_proposal":pending,"tool_approval_decision":True})
    assert not result.get("calendar_reminders")
    assert result["last_tool_outcome"]["failure_category"]=="STALE_APPROVAL_PREVIEW"


def test_unknown_tool_and_approval_argument_are_rejected_without_execution():
    graph=build_graph();config={"configurable":{"thread_id":"invalid"}}
    initialize_procedure_thread(graph,config)
    state=run_tool(graph,config,"not_a_tool")
    assert state["last_tool_outcome"]["failure_category"]=="TOOL_SELECTION_ERROR"
    state=run_tool(graph,config,"create_calendar_reminder",{"title":"Bad","start_at":"2030-01-15T10:00:00+03:00","approved":True},call_id="bad-args")
    assert state["last_tool_outcome"]["failure_category"]=="TOOL_SCHEMA_ERROR"
    assert not state.get("calendar_reminders")


def test_sandbox_round_trip_changes_only_simulated_environment(ready_bundle):
    from src.core.procedure import advance_procedure_if_ready
    graph,config,old=ready_bundle
    graph.update_state(config,advance_procedure_if_ready(old,recorded_at=RECORDED_AT),as_node="respond")
    state=run_tool(graph,config,"simulate_fill_eikamet_field",{"field_id":"passport_number"})
    assert state["sandbox_form"]["fields"]["passport_number"]["value"]=="P1234567"
    assert state["documents"]==old["documents"] and state["verification_records"]==old["verification_records"]
    assert state["residence_permit_context"]==old["residence_permit_context"]
    assert state["last_tool_outcome"]["diff"]["passport_number"]["before"] is None
    invalid=run_tool(graph,config,"simulate_fill_eikamet_field",{"field_id":"passport_number","value":"FABRICATED"},call_id="bad-source")
    assert invalid["last_tool_outcome"]["status"]=="FAILED"
    assert invalid["sandbox_form"]==state["sandbox_form"]
