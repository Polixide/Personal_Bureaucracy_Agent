"""The chat assistant on typical messages, live (needs Ollama): which model answers best and how fast.

    python -m eval.chat_eval --models qwen2.5:3b qwen3:4b --out docs/eval_chat.md
    python -m eval.chat_eval --provider openai --model gpt-6.1-terra --effort low --out docs/eval_chat_openai.md

Each scenario starts from synthetic facts, sends one message and checks, by code, what must happen:
the reply passed Gate 1 (not replaced by the plan), the expected tool effect (a fact recorded unconfirmed,
a proposal created, a procedure selected), and the expected source cited. The facts and the expected effects
are fixed here; nothing is compared with a model's wording. Also measured: the sources shown under the reply and
whether every one is a real source of the packs or of the web corpus (citation validity), the reflection rounds,
and the tokens and estimated cost of each turn (usage log of src/llm.py).
"""
from eval import utf8_console
import argparse
import json
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from config import DATABASE_PATH
from src.agent.chat import run_chat
from src.rag.index import search, search_corpus
from src.llm import (USAGE, OllamaBackend, add_model_arguments, agent_backend, agent_model_label, missing_key,
                     model_argument_problem, usage_summary)
from src.pack import load_all_packs
from src.rag.crawl import load_corpus
from src.twin import service
from src.twin.store import TwinStore

TODAY = date(2026, 11, 20)
PASSPORT_TOO_SHORT = {"passport_expiry_date": "2027-07-15", "requested_permit_end_date": "2027-06-01"}


class ChatScenario(BaseModel):
    scenario_id: str
    procedure_id: str | None
    facts: dict = Field(default_factory=dict)
    message: str
    expect_fact: str | None = None          # a field recorded (unconfirmed) from the message
    expect_proposal: bool = False
    expect_procedure: str | None = None
    expect_source: str | None = None        # a source shown under the reply (cited by the model, or linked by code)
    expect_rule: str | None = None          # a rule id the reply must talk about (claim, action or text)


SCENARIOS = [
    ChatScenario(scenario_id="C1-where-am-i", procedure_id="ikamet_first", facts=PASSPORT_TOO_SHORT,
                 message="Explain where I am, step by step", expect_rule="R006"),
    ChatScenario(scenario_id="C2-states-a-fact", procedure_id="tc_kimlik_renewal",
                 message="I lost my ID card last week", expect_fact="renewal_reason"),
    ChatScenario(scenario_id="C3-asks-for-a-checklist", procedure_id="ikamet_first", facts=PASSPORT_TOO_SHORT,
                 message="Can you prepare a checklist for me?", expect_proposal=True),
    ChatScenario(scenario_id="C4-procedure-without-pack", procedure_id=None,
                 message="How do I get a tax number as a foreigner?", expect_source="WEB_IRO_EN_HOW_OBTAIN_TAX_ID"),
    ChatScenario(scenario_id="C5-names-a-procedure", procedure_id=None,
                 message="I need to renew my TC kimlik, it expired", expect_procedure="tc_kimlik_renewal"),
    ChatScenario(scenario_id="C6-why-does-it-fail", procedure_id="ikamet_first", facts=PASSPORT_TOO_SHORT,
                 message="Why is my passport a problem?", expect_rule="R006"),
    ChatScenario(scenario_id="C7-unknown-date-not-invented", procedure_id="ikamet_first", facts=PASSPORT_TOO_SHORT,
                 message="When did I enter Turkey, and is my 90-day limit ok?"),
    ChatScenario(scenario_id="C8-question-for-the-sources", procedure_id="tc_kimlik_renewal",
                 message="Do I need an appointment to apply for the new card?"),
]


class ChatScore(BaseModel):
    scenario_id: str
    model: str
    status: str
    first_try: bool
    expected_effects: bool
    seconds: float
    llm_calls: int
    sources: int = 0                 # sources shown under the reply
    citations_valid: bool = True     # every shown source exists in the packs or the web corpus
    reflections: int = 0             # correction rounds (Gate 1 or self-review)
    tokens: int | None = None
    cost_usd: float | None = None
    notes: list[str] = Field(default_factory=list)


def prepare(store: TwinStore, packs: dict, user_id: str, scenario: ChatScenario) -> None:
    if scenario.procedure_id is None:
        return
    pack = packs[scenario.procedure_id]
    store.set_active_procedure(user_id, pack.id)
    for name, value in scenario.facts.items():
        service.record_app_value(store, pack, user_id, name, value)


def effects(store: TwinStore, packs: dict, user_id: str, scenario: ChatScenario, reply) -> list[str]:
    """What is missing among the expected effects ([] = all there)."""
    missing = []
    if scenario.expect_fact and scenario.expect_fact not in {fact["field"] for fact in reply.facts}:
        missing.append(f"no fact {scenario.expect_fact} recorded")
    if scenario.expect_proposal and not reply.proposals:
        missing.append("no proposal created")
    if scenario.expect_procedure and store.active_procedure(user_id) != scenario.expect_procedure:
        missing.append(f"procedure is {store.active_procedure(user_id)}")
    if scenario.expect_source and scenario.expect_source not in {s["source_id"] for s in reply.sources}:
        missing.append(f"{scenario.expect_source} not cited")
    if scenario.expect_rule and scenario.expect_rule not in reply.text and not any(
            step.get("action") == "final_answer" and scenario.expect_rule in str(step.get("args")) for step in reply.steps):
        missing.append(f"{scenario.expect_rule} not mentioned")
    return missing


def known_sources(packs: dict) -> set[str]:
    """Every source id a reply may cite: the packs' official sources and the web corpus."""
    return {source.source_id for pack in packs.values() for source in pack.sources} | {
        page.source_id for page in load_corpus()}


def run_scenario(store: TwinStore, packs: dict, model: str, scenario: ChatScenario, backend=None,
                 label: str = "") -> ChatScore:
    """One message. `backend` is the model that answers (default: Ollama with `model`)."""
    user_id = f"{scenario.scenario_id}-{model}"
    prepare(store, packs, user_id, scenario)
    started, mark = perf_counter(), USAGE.mark()
    reply = run_chat(scenario.message, store, packs, user_id, TODAY, backend or OllamaBackend(model=model),
                     embedder=OllamaBackend(), outbox_dir=Path(tempfile.gettempdir()) / "chat_eval_outbox")
    usage = usage_summary(USAGE.since(mark))
    missing = effects(store, packs, user_id, scenario, reply)
    repairs = sum(1 for step in reply.steps if not step["ok"])
    shown = {source["source_id"] for source in reply.sources}
    tokens = None if usage["input_tokens"] is None else usage["input_tokens"] + usage["output_tokens"]
    return ChatScore(scenario_id=scenario.scenario_id, model=label or model, status=reply.status,
                     first_try=reply.status == "answered" and repairs == 0, expected_effects=not missing,
                     seconds=round(perf_counter() - started, 1), llm_calls=reply.llm_calls, sources=len(shown),
                     citations_valid=shown <= known_sources(packs), reflections=len(reply.reflections),
                     tokens=tokens, cost_usd=usage["cost_usd"],
                     notes=missing + [f"{i['code']}" for i in reply.gate_issues]
                     + [f"{step['action']}: {(step.get('error') or '')[:80]}" for step in reply.steps if not step["ok"]])


def copy_vectors(source: Path, store: TwinStore) -> int:
    """Reuse the embedding cache of the app's Twin (only rag_vectors: no user data is copied)."""
    if not source.is_file():
        return 0
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    rows = connection.execute("SELECT procedure_id, chunk_id, model, source_id, text_hash, vector FROM rag_vectors").fetchall()
    connection.close()
    for scope, chunk_id, model, source_id, text_hash, vector in rows:
        store.put_vector(scope, chunk_id, model, source_id, text_hash, json.loads(vector))
    return len(rows)


def warm_up(store: TwinStore, packs: dict) -> None:
    """Embed every passage once before timing (otherwise the first scenario pays for the indexing)."""
    embedder = OllamaBackend()
    for pack in packs.values():
        search(store, pack, embedder, "warm up", 1)
    search_corpus(store, embedder, "warm up", 1)


def table(scores: list[ChatScore]) -> str:
    lines = ["| scenario | model | status | first try | expected effects | seconds | model calls | notes |",
             "|---|---|---|---|---|---|---|---|"]
    for s in scores:
        lines.append(f"| {s.scenario_id} | {s.model} | {s.status} | {'yes' if s.first_try else 'no'} | "
                     f"{'yes' if s.expected_effects else 'no'} | {s.seconds} | {s.llm_calls} | {'; '.join(s.notes)} |")
    lines += ["", "| model | answered (Gate 1 passed) | first try | expected effects | citations valid | "
              "reflection rounds | seconds per turn | tokens per turn | cost |", "|---|---|---|---|---|---|---|---|---|"]
    for model in dict.fromkeys(s.model for s in scores):
        mine = [s for s in scores if s.model == model]
        tokens = [s.tokens for s in mine if s.tokens is not None]
        costs = [s.cost_usd for s in mine if s.cost_usd is not None]
        lines.append(f"| {model} | {sum(s.status == 'answered' for s in mine)}/{len(mine)} | "
                     f"{sum(s.first_try for s in mine)}/{len(mine)} | {sum(s.expected_effects for s in mine)}/{len(mine)} | "
                     f"{sum(s.citations_valid for s in mine)}/{len(mine)} | {sum(s.reflections for s in mine)} | "
                     f"{sum(s.seconds for s in mine) / len(mine):.1f} | "
                     f"{round(sum(tokens) / len(tokens)) if tokens else 'n/a'} | "
                     f"{f'${sum(costs):.4f}' if costs and len(costs) == len(mine) else 'n/a'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b", "qwen3:4b"], help="Ollama models to compare")
    parser.add_argument("--out", type=Path, default=None)
    add_model_arguments(parser)                      # --provider openai --model ... --effort ...: a cloud model
    args = parser.parse_args()
    cloud = args.provider in ("openai", "anthropic")
    problem = (missing_key(args.provider) or model_argument_problem(args.provider, args.effort)) if cloud else ""
    if problem:
        parser.error(problem)
    packs = load_all_packs()
    models = [args.model or ""] if cloud else ([args.model] if args.model else args.models)
    scores = []
    with tempfile.TemporaryDirectory() as folder:
        with TwinStore(Path(folder) / "chat_eval.sqlite") as store:
            copy_vectors(DATABASE_PATH, store)
            warm_up(store, packs)
            for model in models:
                backend = agent_backend("chat", args.provider, model or None, args.effort) if cloud else None
                label = agent_model_label("chat", args.provider, model or None, args.effort) if cloud else model
                for scenario in SCENARIOS:
                    score = run_scenario(store, packs, model or label, scenario, backend, label)
                    print(score.model_dump_json(), flush=True)
                    scores.append(score)
    report = table(scores)
    print(report)
    if args.out:
        args.out.write_text("# Chat assistant: live evaluation\n\nGenerated by `python -m eval.chat_eval`. "
                            f"{len(SCENARIOS)} messages per model, synthetic facts, {TODAY.isoformat()}.\n\n" + report,
                            encoding="utf-8")


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
