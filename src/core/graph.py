"""Workstream A orchestration, integrated workers and bounded graph repairs."""
from langgraph.graph import END, START, StateGraph
from config import MAX_REFLECTION_ATTEMPTS
from src.core.state import AgentState, CHECKPOINT_TYPES
from src.workers.document_analyzer import document_analyzer
from src.workers.rules_engine import rules_engine
from src.agents.planner import planner
from src.guardrails.gate_0 import gate_0
from src.guardrails.gate_1 import gate_1
from src.guardrails.human_approval import human_approval
from src.memory.reflection import reflection
from src.memory.state_update import state_update


def supervisor(state):
    if state.get("event") and state["event"].kind == "assistant_message": decision = "assistant_worker"
    elif state.get("event") and state["event"].kind == "run_tool": decision = "tool_dispatch"
    elif state.get("run_status") == "REPAIRING": decision = "planner"
    elif any(d.status == "pending" for d in state.get("documents", [])): decision = "document_analyzer"
    elif state.get("knowledge_snapshot") and state.get("relevant_requirements") and any(r.rule_id not in {rule["rule_id"] for rule in state["knowledge_snapshot"]["rules"]} for r in state["relevant_requirements"]): decision="rules_engine"
    elif state.get("requirements_knowledge_version") is not None and state.get("knowledge_snapshot") and state.get("event").kind not in {"assistant_message","run_tool"} and state.get("requirements_knowledge_version")!=(state["knowledge_snapshot"].get("version")): decision="rules_engine"
    elif state.get("relevant_requirements") is None: decision = "rules_engine"
    elif state.get("event") and state["event"].kind in {"analyze_document", "confirm_information", "resolve_conflict", "record_progress", "reevaluate", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction"}: decision = "respond"
    else: decision = "planner"
    return {"next_step": decision}


def respond(state):
    status = state.get("run_status", "RUNNING")
    if state.get("plan_origin")=="stage_complete" and not state.get("validation_errors"):status="REQUIREMENTS_EVALUATED"
    if state.get("event") and state["event"].kind in {"assistant_message", "run_tool"} and not state.get("validation_errors"):
        messages = state.get("assistant_messages", [])
        answer = state.get("final_response") or (messages[-1].get("content", "") if messages else "Tool observation recorded.")
        return {"run_status":status, "final_response":answer}
    if any(e.code == "PLANNER_UNAVAILABLE" for e in state.get("validation_errors", [])):
        status = "PLANNING_UNAVAILABLE"
    if status == "PLAN_INVALID" and (state.get("retry_count", 0) >= MAX_REFLECTION_ATTEMPTS or any(e.code=="UNCHANGED_INVALID_PLAN" for e in state.get("validation_errors", []))):
        status = "HUMAN_REVIEW_REQUIRED"
    if state.get("validation_errors"):
        lines = [status] + [f"[{e.code}] {e.message}" for e in state["validation_errors"]]
    elif state.get("event") and state["event"].kind in {"analyze_document", "confirm_information", "resolve_conflict", "record_progress", "reevaluate", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction"}:
        message = ("Review the outstanding document information." if status == "INFORMATION_NEEDS_CONFIRMATION"
                   else "Your saved information has been checked.")
        lines = [status, message]
    else:
        lines = [status, state.get("plan_summary") or "Proposed next actions (no official action executed):"]
        lines += [f"{i}. {step.action}" for i,step in enumerate(state.get("candidate_plan") or [], 1)]
    return {"run_status":status, "final_response":"\n".join(lines)}


def route_check(state): return "fail" if state.get("validation_errors") else "pass"


def route_gate_1(state):
    if state.get("plan_origin")=="stage_complete" and state.get("candidate_plan")==[] and not state.get("validation_errors"):return "exhausted"  # Existing respond edge; no empty approval.
    if any(e.code in {"PLANNER_UNAVAILABLE","UNCHANGED_INVALID_PLAN"} for e in state.get("validation_errors", [])): return "exhausted"
    if not state.get("validation_errors"): return "pass"
    return "exhausted" if state.get("retry_count",0) >= MAX_REFLECTION_ATTEMPTS else "repair"



_ALLOWED_V2_UPDATES = {"assistant_messages","assistant_events","assistant_diagnostics","tool_proposals","tool_outcomes",
    "sandbox_form","calendar_reminders","email_drafts","final_response","run_status"}


def _safe_v2_updates(updates):
    # Tool/model proposals cannot write verified context, documents or procedure.
    if set(updates) - _ALLOWED_V2_UPDATES:
        raise ValueError("Assistant/tool update attempted to alter canonical verified facts")
    return updates


def _pending(state):
    terminal = {o.get("call_id") for o in state.get("tool_outcomes",[]) if o.get("status") in {"SUCCEEDED","FAILED","REJECTED"}}
    return next((p for p in reversed(state.get("tool_proposals",[])) if p.get("status")=="APPROVAL_REQUIRED" and p.get("proposal_id") not in terminal), None)


def assistant_worker(state):
    from src.agents.assistant import assistant_turn
    updates = _safe_v2_updates(assistant_turn(state, state["event"].text,copilot=True))
    pending = _pending({**state,**updates})
    return {**updates, "pending_tool_proposal":pending, "run_status":"AWAITING_TOOL_APPROVAL" if pending else "ASSISTANT_RESPONDED"}


def tool_dispatch(state):
    from src.tools.registry import execute_tool
    from src.tools.schemas import ToolCall
    call = ToolCall.model_validate(state["event"].tool_call)
    observation = execute_tool(state, call)
    raw = observation.model_dump(mode="json", exclude={"state_updates"})
    if observation.status == "APPROVAL_REQUIRED":
        proposal={"proposal_id":call.call_id,"tool_call":call.model_dump(mode="json"),"status":"APPROVAL_REQUIRED","preview":observation.diff,"observation":raw}
        proposals=[p for p in state.get("tool_proposals",[]) if p.get("proposal_id")!=call.call_id]+[proposal]
        return {"tool_proposals":proposals,"pending_tool_proposal":proposal,"last_tool_outcome":raw,"run_status":"AWAITING_TOOL_APPROVAL"}
    updates = _safe_v2_updates(observation.state_updates)
    outcomes=updates.get("tool_outcomes") or [*state.get("tool_outcomes",[]),raw]
    return {**updates,"tool_outcomes":outcomes,"last_tool_outcome":outcomes[-1],
        "pending_tool_proposal":None,"run_status":"TOOL_COMPLETED" if observation.status=="SUCCEEDED" else "TOOL_FAILED",
        "final_response":str(observation.data.get("message") or observation.status)}


def route_tool_approval(state):
    return "approve" if state.get("pending_tool_proposal") else "respond"


def tool_approval(state):
    from langgraph.types import interrupt
    proposal=state.get("pending_tool_proposal")
    if not proposal: raise ValueError("No pending tool proposal")
    decision=interrupt({"kind":"tool_approval","proposal":proposal,"message":"Approve only this exact tool action; no governmental action is authorized."})
    if type(decision) is not bool: raise ValueError("Tool approval requires an explicit boolean decision")
    return {"tool_approval_decision":decision}


def tool_execute(state):
    from src.tools.registry import execute_tool,reject_tool
    from src.tools.schemas import ToolCall,ToolObservation
    proposal=state["pending_tool_proposal"]
    call=ToolCall.model_validate(proposal["tool_call"])
    if state.get("tool_approval_decision") is True:
        preview=execute_tool(state,call,approved=False)
        if preview.status=="SUCCEEDED" and preview.replayed:
            observation=preview
        elif preview.status!="APPROVAL_REQUIRED" or preview.diff!=proposal.get("preview",{}):
            observation=ToolObservation(call_id=call.call_id,name=call.name,status="FAILED",
                data={"message":"The action changed after preview; prepare and approve a fresh proposal."},
                failure_category="STALE_APPROVAL_PREVIEW",fingerprint=preview.fingerprint)
        else:
            observation=execute_tool(state,call,approved=True)
    else:
        observation=reject_tool(state,call)
    raw=observation.model_dump(mode="json",exclude={"state_updates"})
    updates=_safe_v2_updates(observation.state_updates)
    outcomes=updates.get("tool_outcomes") or [*state.get("tool_outcomes",[]),raw]
    messages=[*state.get("assistant_messages",[]),{"role":"assistant","content":"The proposed action was rejected; no external action was executed." if observation.status=="REJECTED" else str(observation.data.get("message") or "The approved action outcome is recorded.")}]
    return {**updates,"assistant_messages":messages,"tool_outcomes":outcomes,"last_tool_outcome":outcomes[-1],
        "pending_tool_proposal":None,"run_status":"TOOL_REJECTED" if observation.status=="REJECTED" else "TOOL_COMPLETED" if observation.status=="SUCCEEDED" else "TOOL_FAILED",
        "final_response":messages[-1]["content"]}


def build_graph(checkpointer=None):
    # CLI applications supply sqlite_checkpointer(); the default stays isolated
    # and in-memory for ephemeral callers and existing offline unit tests.
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    builder = StateGraph(AgentState)
    for node in [gate_0,supervisor,document_analyzer,rules_engine,planner,gate_1,reflection,human_approval,state_update,respond,assistant_worker,tool_dispatch,tool_approval,tool_execute]: builder.add_node(node)
    builder.add_edge(START,"gate_0")
    builder.add_conditional_edges("gate_0",route_check,{"pass":"supervisor","fail":"respond"})
    builder.add_conditional_edges("supervisor",lambda s:s["next_step"], {n:n for n in ["document_analyzer","rules_engine","planner","respond","assistant_worker","tool_dispatch"]})
    builder.add_edge("document_analyzer","supervisor")
    builder.add_edge("rules_engine","supervisor")
    builder.add_edge("planner","gate_1")
    builder.add_conditional_edges("gate_1",route_gate_1,{"pass":"human_approval","repair":"reflection","exhausted":"respond"})
    builder.add_edge("reflection","supervisor")
    builder.add_conditional_edges("human_approval",route_check,{"pass":"state_update","fail":"respond"})
    builder.add_edge("state_update","respond")
    builder.add_conditional_edges("assistant_worker",route_tool_approval,{"approve":"tool_approval","respond":"respond"})
    builder.add_conditional_edges("tool_dispatch",route_tool_approval,{"approve":"tool_approval","respond":"respond"})
    builder.add_edge("tool_approval","tool_execute")
    builder.add_edge("tool_execute","respond")
    builder.add_edge("respond",END)
    return builder.compile(checkpointer=checkpointer)
