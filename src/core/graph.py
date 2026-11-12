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
    if state.get("run_status") == "REPAIRING": decision = "planner"
    elif any(d.status == "pending" for d in state.get("documents", [])): decision = "document_analyzer"
    elif state.get("relevant_requirements") is None: decision = "rules_engine"
    else: decision = "planner"
    return {"next_step": decision}


def respond(state):
    status = state.get("run_status", "RUNNING")
    if status == "PLAN_INVALID" and state.get("retry_count", 0) >= MAX_REFLECTION_ATTEMPTS:
        status = "HUMAN_REVIEW_REQUIRED"
    if state.get("validation_errors"):
        lines = [status] + [f"[{e.code}] {e.message}" for e in state["validation_errors"]]
    else:
        lines = [status, "Proposed next actions (no official action executed):"]
        lines += [f"{i}. {step.action}" for i,step in enumerate(state.get("candidate_plan") or [], 1)]
    return {"run_status":status, "final_response":"\n".join(lines)}


def route_check(state): return "fail" if state.get("validation_errors") else "pass"


def route_gate_1(state):
    if not state.get("validation_errors"): return "pass"
    return "exhausted" if state.get("retry_count",0) >= MAX_REFLECTION_ATTEMPTS else "repair"


def build_graph(checkpointer=None):
    # CLI applications supply sqlite_checkpointer(); the default stays isolated
    # and in-memory for ephemeral callers and existing offline unit tests.
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    builder = StateGraph(AgentState)
    for node in [gate_0,supervisor,document_analyzer,rules_engine,planner,gate_1,reflection,human_approval,state_update,respond]: builder.add_node(node)
    builder.add_edge(START,"gate_0")
    builder.add_conditional_edges("gate_0",route_check,{"pass":"supervisor","fail":"respond"})
    builder.add_conditional_edges("supervisor",lambda s:s["next_step"], {n:n for n in ["document_analyzer","rules_engine","planner"]})
    builder.add_edge("document_analyzer","supervisor")
    builder.add_edge("rules_engine","supervisor")
    builder.add_edge("planner","gate_1")
    builder.add_conditional_edges("gate_1",route_gate_1,{"pass":"human_approval","repair":"reflection","exhausted":"respond"})
    builder.add_edge("reflection","supervisor")
    builder.add_conditional_edges("human_approval",route_check,{"pass":"state_update","fail":"respond"})
    builder.add_edge("state_update","respond")
    builder.add_edge("respond",END)
    return builder.compile(checkpointer=checkpointer)
