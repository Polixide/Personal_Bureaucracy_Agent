"""The systems compared by the evaluation (run-evals skill):

A  static checklist: the same generic list for everyone (no state, no rules evaluated, no model)
B  single LLM: one call with the goal and the raw facts (no state, no rules, no tools)
D  full agent: orchestrator + ReAct workers + Gate 1 + fallback (src.agent.orchestrator.run_agent)
"""
from datetime import date

from config import MAX_NEXT_ACTIONS
from src.agent.answer import FinalAnswer, NextAction
from src.agent.orchestrator import run_agent
from src.llm import Backend, ask_structured, to_json
from src.models import Pack
from src.twin.store import TwinStore

SINGLE_LLM_SYSTEM = """You help a student with an official procedure. Answer with ONE JSON object:
{"summary": "...", "claims": [{"rule_id": "...", "status": "pass|fail|unknown"}],
 "next_actions": [{"title": "...", "ref": "<rule id>"}]} with at most 5 next actions."""


def static_checklist(pack: Pack) -> FinalAnswer:
    """System A: the generic list of the first requirements, the same for every user."""
    actions = [NextAction(title=rule.description[:240], ref=rule.rule_id) for rule in pack.rules[:MAX_NEXT_ACTIONS]]
    return FinalAnswer(summary=f"General checklist for {pack.procedure.title}. Prepare every document the office "
                               "lists.", claims=[], next_actions=actions)


def single_llm(backend: Backend, pack: Pack, goal: str, facts: dict) -> tuple[FinalAnswer | None, int]:
    """System B: one model call; returns (answer or None, model calls)."""
    user = f"GOAL: {goal}\nPROCEDURE: {pack.procedure.title}\nRULE IDS: {pack.rule_ids()}\nMY FACTS: {to_json(facts)}"
    result = ask_structured(backend, [{"role": "system", "content": SINGLE_LLM_SYSTEM}, {"role": "user", "content": user}],
                            FinalAnswer, "baseline_single_llm")
    return (result.value if result.ok else None), result.attempts


def full_agent(store: TwinStore, packs: dict[str, Pack], user_id: str, goal: str, today: date, backend: Backend, outbox):
    """System D: the agent."""
    return run_agent(goal, store, packs, user_id, today, backend, outbox_dir=outbox)
