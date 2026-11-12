"""LangGraph orchestration for Milestone 1 (Workstream A).

Topology (Orchestrator-Workers + deterministic gates):
    START -> gate_0 -> supervisor
    supervisor -> document_analyzer | rules_engine | planner   (decided from the state)
    document_analyzer, rules_engine -> supervisor              (workers report back)
    planner -> gate_1                                          (every plan is validated)
    gate_0, gate_1, human_approval: PASS -> next node, FAIL -> respond
    human_approval -> state_update -> respond -> END

Milestone 1 stops safely on FAIL. Milestone 2 will route gate_1's FAIL to a
bounded Reflection/Replan node that returns to the supervisor.

Run the mock scenarios:  python -m src.core.graph
"""
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.core.state import AgentState, NextStep
from src.workers.document_analyzer import document_analyzer
from src.workers.rules_engine import rules_engine
from src.core.stubs import (gate_0, gate_1, human_approval, planner,
                            state_update)


# ---------------------------------------------------- orchestration nodes --
def supervisor(state: AgentState) -> dict:
    """Central orchestrator: reads the shared state and picks the next worker.

    Milestone 1 uses a deterministic policy (reproducible, zero tokens). The
    decision is written into the state, so the node can later be replaced by an
    LLM call whose structured output is restricted to `NextStep`.
    """
    if any(doc.status == "pending" for doc in state.get("documents", [])):
        decision = "document_analyzer"   # new documents must be read first
    elif state.get("relevant_requirements") is None:
        decision = "rules_engine"        # then checked against the rules
    else:
        decision = "planner"             # facts and requirements ready -> plan
    return {"next_step": decision}


def respond(state: AgentState) -> dict:
    """Single exit point: every run ends here with a plan or a structured error."""
    errors = state.get("validation_errors", [])
    if errors:
        lines = ["Stopped safely. Issues found:"]
        lines += [f"  [{e.code}] {e.message}" for e in errors]
    else:
        lines = ["Next actions:"]
        lines += [f"  {i}. {step.action}" + (f" (by {step.due_date})" if step.due_date else "")
                  for i, step in enumerate(state["candidate_plan"], start=1)]
    lines.append("Demo scope: R014 only. Planner and approval are MOCKED; no official approval is made.")
    for requirement in state.get("relevant_requirements") or []:
        lines.append(f"{requirement.rule_id}: {requirement.evaluation_result} "
                     f"(source: {requirement.source_id})")
    return {"final_response": "\n".join(lines)}


# ---------------------------------------------------------------- routers --
def route_supervisor(state: AgentState) -> NextStep:
    return state["next_step"]


def route_check(state: AgentState) -> Literal["pass", "fail"]:
    """Same PASS/FAIL rule for every checkpoint: no structured errors = PASS."""
    return "fail" if state.get("validation_errors") else "pass"


# ------------------------------------------------------------------ graph --
def build_graph(checkpointer=None):
    """Compile the graph.

    Pass a checkpointer (e.g. SQLite) to persist the Digital Twin across
    sessions, invoking with config={"configurable": {"thread_id": user_id}}.
    """
    builder = StateGraph(AgentState)
    for node in (gate_0, supervisor, document_analyzer, rules_engine, planner,
                 gate_1, human_approval, state_update, respond):
        builder.add_node(node)                       # node name = function name

    builder.add_edge(START, "gate_0")
    builder.add_conditional_edges("gate_0", route_check, {"pass": "supervisor", "fail": "respond"})
    builder.add_conditional_edges("supervisor", route_supervisor,
                                  ["document_analyzer", "rules_engine", "planner"])
    builder.add_edge("document_analyzer", "supervisor")
    builder.add_edge("rules_engine", "supervisor")
    builder.add_edge("planner", "gate_1") # why only the planner is linked with gate 1 ? check
    builder.add_conditional_edges("gate_1", route_check, {"pass": "human_approval", "fail": "respond"})
    builder.add_conditional_edges("human_approval", route_check, {"pass": "state_update", "fail": "respond"})
    builder.add_edge("state_update", "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from scripts.run_demo import run_offline_demo

    run_offline_demo()
