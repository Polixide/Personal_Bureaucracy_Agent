"""Orchestrator: decompose (sections), delegate (workers), synthesize (final answer), verify (Gate 1).

LangGraph cycle:
    START -> decide -> act -> decide ...            (ReAct, at most MAX_STEPS decisions)
                        act -> verify -> END         (Gate 1 passed)
                               verify -> decide      (Gate 1 failed: errors go back, at most MAX_REPAIRS)
                               verify -> END         (still failing: answer flagged "needs human review")
    decide/act -> fallback -> END                    (2 invalid decisions in a row, or out of steps)
                        act -> END                   (ask_user: the turn ends with a question)
"""
from datetime import date
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from config import AGENT_RECURSION_LIMIT, MAX_CONSECUTIVE_FAILURES, MAX_REPAIRS, MAX_STEPS, OUTBOX_DIR
from src.agent.answer import (FINAL_ACTION, AgentResult, Decision, FinalAnswer, StepRecord, decision_schema)
from src.agent.fallback import deterministic_answer
from src.agent.gate1 import build_gate_context, verify
from src.agent.loop import ask_decision, assistant_message, call_key, error_message, observation_message
from src.agent.prompts import ORCHESTRATOR_SYSTEM, orchestrator_intro
from src.agent.worker import DELEGATE_SPEC, run_worker
from src.llm import Backend, short_validation_error, to_json
from src.memory.episodes import lessons_for, record_repairs
from src.models import Issue, Pack
from src.tools.common import ToolContext, ToolSpec
from src.tools.evaluate_rules import current_results
from src.tools.registry import build_registry, run_tool, tools_for
from src.twin.codec import utc_now
from src.twin.models import RunRecord
from src.twin.store import TwinStore


class RunState(TypedDict, total=False):
    """Graph state of one run (plain data only; the context lives on the Orchestrator)."""
    messages: list[dict]
    decision: dict | None
    final: dict | None
    status: str
    steps: int
    failures: int
    llm_calls: int
    repairs: int
    loops: int
    invalid: int
    seen: dict[str, int]
    last_call: str
    records: list[dict]
    gate_issues: list[dict]
    rejected: list[list[dict]]
    goal: str


def repair_message(issues: list[Issue], lessons: list[dict]) -> dict:
    """Gate 1 errors sent back to the model, with past lessons for the same failure types."""
    text = ("GATE 1 REJECTED YOUR final_answer. Fix exactly these problems and send a new final_answer: "
            + to_json([issue.model_dump() for issue in issues]))
    if lessons:
        text += "\nLESSONS FROM PAST REPAIRS OF THE SAME PROBLEMS: " + to_json(lessons)
    return {"role": "user", "content": text}


class Orchestrator:
    """Holds the run context and the graph nodes."""

    def __init__(self, ctx: ToolContext, backend: Backend, registry: dict[str, ToolSpec], lesson_finder=None):
        self.ctx = ctx
        self.backend = backend
        self.tools = tools_for(registry, "orchestrator")
        self.schema = decision_schema(sorted(self.tools) + [FINAL_ACTION])
        self.lesson_finder = lesson_finder

    # ------------------------------------------------------------ state --
    def initial_state(self, goal: str, lessons: list[dict]) -> RunState:
        intro = orchestrator_intro(goal, self.ctx, self.tools, lessons)
        return RunState(messages=[{"role": "system", "content": ORCHESTRATOR_SYSTEM}, {"role": "user", "content": intro}],
                        decision=None, final=None, status="running", steps=0, failures=0, llm_calls=0, repairs=0,
                        loops=0, invalid=0, seen={}, last_call="", records=[], gate_issues=[], rejected=[], goal=goal)

    @staticmethod
    def _record(state: RunState, decision: Decision | None, ok: bool, error: str | None = None) -> list[dict]:
        record = StepRecord(step=state["steps"], thought=decision.thought if decision else "",
                            action=decision.action if decision else "(invalid)", args=decision.args if decision else {},
                            ok=ok, error=error)
        return state["records"] + [record.model_dump(mode="json")]

    # ------------------------------------------------------------ nodes --
    def decide(self, state: RunState) -> dict:
        """Ask the model for the next Decision."""
        result = ask_decision(self.backend, state["messages"], self.schema, "orchestrator")
        update: dict[str, Any] = {"steps": state["steps"] + 1, "llm_calls": state["llm_calls"] + result.attempts}
        if result.ok:
            return {**update, "decision": result.value.model_dump()}
        failed = {**state, **update}
        return {**update, "decision": None, "failures": state["failures"] + 1, "invalid": state["invalid"] + 1,
                "messages": state["messages"] + [error_message(f"Invalid decision: {result.error}")],
                "records": self._record(failed, None, ok=False, error=result.error)}

    def act(self, state: RunState) -> dict:
        """Run the decided tool, accept a final answer, or refuse a repeated call."""
        decision = Decision.model_validate(state["decision"])
        messages = state["messages"] + [assistant_message(decision)]
        if decision.action == FINAL_ACTION:
            return self._accept_final(state, decision, messages)
        key = call_key(decision)
        if key == state["last_call"] or state["seen"].get(key, 0) >= 2:
            error = f"You already called {decision.action} with these arguments. {self._next_step_hint()}"
            return {"messages": messages + [error_message(error)], "decision": None, "failures": state["failures"] + 1,
                    "loops": state["loops"] + 1, "records": self._record(state, decision, ok=False, error="repeated call")}
        observation = run_tool(self.tools, decision.action, decision.args, self.ctx, "orchestrator")
        spec = self.tools.get(decision.action)
        ended = observation.ok and spec is not None and spec.ends_turn
        update = {"messages": messages + [observation_message(observation)], "decision": None,
                  "failures": 0 if observation.ok else state["failures"] + 1,
                  "invalid": state["invalid"] + (0 if observation.ok else 1),
                  "records": self._record(state, decision, observation.ok, observation.error),
                  "status": "needs_user_input" if ended else state["status"]}
        if observation.ok:
            update["seen"] = {**state["seen"], key: state["seen"].get(key, 0) + 1}
            update["last_call"] = key
        return update

    def _next_step_hint(self) -> str:
        """A concrete way out of a loop: the next section that still needs a worker, else the final answer."""
        pack = self.ctx.pack()
        if pack is None:
            return "Call search_procedures with the user's goal, or give the final_answer."
        delegated = {finding.section_id for finding in self.ctx.findings.values()}
        results = current_results(self.ctx)
        for section in sorted(pack.procedure.sections, key=lambda s: s.order):
            still_open = any(r.section == section.id and r.status != "pass" for r in results)
            if section.delegate and still_open and section.id not in delegated:
                return f'Next: call delegate_section with section_id "{section.id}", or give the final_answer.'
        return "Use the results you have and give the final_answer now."

    def _accept_final(self, state: RunState, decision: Decision, messages: list[dict]) -> dict:
        try:
            answer = FinalAnswer.model_validate(decision.args)
        except ValidationError as error:
            text = f"final_answer args invalid: {short_validation_error(error)}"
            return {"messages": messages + [error_message(text)], "decision": None, "failures": state["failures"] + 1,
                    "invalid": state["invalid"] + 1, "records": self._record(state, decision, ok=False, error=text)}
        return {"messages": messages, "decision": None, "final": answer.model_dump(), "failures": 0,
                "records": self._record(state, decision, ok=True)}

    def verify(self, state: RunState) -> dict:
        """Gate 1. Failing answers go back to the model (bounded), then get flagged."""
        answer = FinalAnswer.model_validate(state["final"])
        issues = verify(answer, build_gate_context(self.ctx))
        dumped = [issue.model_dump() for issue in issues]
        if not issues:
            self._on_accepted(state, answer)
            return {"status": "answered", "gate_issues": []}
        if state["repairs"] >= MAX_REPAIRS:
            return {"status": "needs_human_review", "gate_issues": dumped}
        lessons = self.lesson_finder(sorted({i.code for i in issues})) if self.lesson_finder else []
        return {"messages": state["messages"] + [repair_message(issues, lessons)], "final": None,
                "repairs": state["repairs"] + 1, "gate_issues": dumped, "rejected": state["rejected"] + [dumped]}

    def _on_accepted(self, state: RunState, answer: FinalAnswer) -> None:
        """Episodic memory: a repaired answer that passed Gate 1 turns its rejections into lessons."""
        if state["rejected"] and self.ctx.procedure_id in self.ctx.packs:
            record_repairs(self.ctx.store, self.ctx.packs[self.ctx.procedure_id], self.ctx.user_id,
                           state["rejected"], self.ctx.trace_id)

    def fallback(self, state: RunState) -> dict:
        """Deterministic workflow: evaluate all sections and answer with the checklist."""
        answer, findings = deterministic_answer(self.ctx, state.get("goal", ""))
        for finding in findings:
            self.ctx.findings.setdefault(finding.finding_id, finding)
        issues = verify(answer, build_gate_context(self.ctx))
        return {"final": answer.model_dump(), "status": "fallback",
                "gate_issues": [issue.model_dump() for issue in issues]}

    # ----------------------------------------------------------- routing --
    def _out_of_steps(self, state: RunState) -> bool:
        return state["steps"] >= MAX_STEPS + state["repairs"]

    def route_decide(self, state: RunState) -> str:
        if state["decision"] is not None:
            return "act"
        if state["failures"] >= MAX_CONSECUTIVE_FAILURES or self._out_of_steps(state):
            return "fallback"
        return "decide"

    def route_act(self, state: RunState) -> str:
        if state["status"] == "needs_user_input":
            return "end"
        if state.get("final") is not None:
            return "verify"
        if state["failures"] >= MAX_CONSECUTIVE_FAILURES or self._out_of_steps(state):
            return "fallback"
        return "decide"

    @staticmethod
    def route_verify(state: RunState) -> str:
        return "end" if state["status"] in ("answered", "needs_human_review") else "decide"

    def graph(self):
        builder = StateGraph(RunState)
        for name, node in (("decide", self.decide), ("act", self.act), ("verify", self.verify), ("fallback", self.fallback)):
            builder.add_node(name, node)
        builder.add_edge(START, "decide")
        builder.add_conditional_edges("decide", self.route_decide, {"act": "act", "decide": "decide", "fallback": "fallback"})
        builder.add_conditional_edges("act", self.route_act, {"decide": "decide", "verify": "verify",
                                                               "fallback": "fallback", "end": END})
        builder.add_conditional_edges("verify", self.route_verify, {"decide": "decide", "end": END})
        builder.add_edge("fallback", END)
        return builder.compile()

    # ------------------------------------------------------------ result --
    def result(self, state: RunState) -> AgentResult:
        findings = list(self.ctx.findings.values())
        status = state["status"]
        answer = FinalAnswer.model_validate(state["final"]) if state.get("final") else None
        issues = [Issue.model_validate(item) for item in state["gate_issues"]]
        return AgentResult(
            trace_id=self.ctx.trace_id, status=status, procedure_id=self.ctx.procedure_id, answer=answer,
            question=self.ctx.question if status == "needs_user_input" else None, findings=findings,
            gate_issues=issues, first_answer_rejected=bool(state["rejected"]),
            steps=[StepRecord.model_validate(r) for r in state["records"]],
            llm_calls=state["llm_calls"] + sum(f.llm_calls for f in findings), repairs=state["repairs"],
            fallback_used=status == "fallback",
            needs_human_review=status == "needs_human_review" or (status == "fallback" and bool(issues)),
            proposals=list(self.ctx.created_proposals), loops_detected=state["loops"], invalid_calls=state["invalid"])


def make_context(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date, backend: Backend | None,
                 outbox_dir=OUTBOX_DIR) -> ToolContext:
    """Tool context for one run; the active procedure comes from the Twin."""
    active = store.active_procedure(user_id)
    return ToolContext(store=store, packs=packs, user_id=user_id, today=today, backend=backend,
                       procedure_id=active if active in packs else None, outbox_dir=outbox_dir, trace_id=uuid4().hex)


def log_result(store: TwinStore, user_id: str, goal: str, result: AgentResult, started) -> None:
    summary = result.answer.summary if result.answer else (result.question or "")
    store.log_run(RunRecord(trace_id=result.trace_id, user_id=user_id, procedure_id=result.procedure_id, goal=goal,
                            status=result.status, llm_calls=result.llm_calls, steps=len(result.steps),
                            repairs=result.repairs, fallback_used=result.fallback_used,
                            needs_human_review=result.needs_human_review, summary=summary[:500],
                            started_at=started, finished_at=utc_now(), issues=result.gate_issues))


def run_agent(goal: str, store: TwinStore, packs: dict[str, Pack], user_id: str, today: date, backend: Backend,
              outbox_dir=OUTBOX_DIR, lessons: list[dict] | None = None, lesson_finder=None,
              orchestrator_class=None) -> AgentResult:
    """One user turn: run the orchestrator graph, log the run in the Twin, return the result."""
    started = utc_now()
    ctx = make_context(store, packs, user_id, today, backend, outbox_dir)
    registry = build_registry([DELEGATE_SPEC])

    def worker_for_section(section_id: str, focus: str):
        """delegate_section calls this: one worker run on the shared context."""
        return run_worker(section_id, focus, ctx, backend, registry)

    def lessons_for_codes(codes: list[str]) -> list[dict]:
        """Lessons for the failure types of a rejected answer (active procedure read at call time)."""
        return lessons_for(store, ctx.procedure_id, codes)

    ctx.run_worker = worker_for_section
    if lessons is None:
        lessons = lessons_for(store, ctx.procedure_id)
    if lesson_finder is None:
        lesson_finder = lessons_for_codes
    orchestrator = (orchestrator_class or Orchestrator)(ctx, backend, registry, lesson_finder)
    state = orchestrator.graph().invoke(orchestrator.initial_state(goal, lessons),
                                        {"recursion_limit": AGENT_RECURSION_LIMIT})
    result = orchestrator.result(state)
    log_result(store, user_id, goal, result, started)
    return result
