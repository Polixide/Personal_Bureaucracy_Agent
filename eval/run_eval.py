"""Scenario evaluation of A (static checklist), B (single LLM) and D (full agent).

python -m eval.run_eval                          offline: D uses eval.policy.RuleFollowingPolicy, B is skipped
python -m eval.run_eval --live                   Ollama (LLM_MODEL / EMBED_MODEL from config.py) for B and D
python -m eval.run_eval --live --provider openai the cloud model for B and D (OPENAI_API_KEY in the environment)
Options: --k 3 (runs per scenario, for pass^k and pass@k), --out docs/eval_results.md, --procedure <id>

Scenarios come from every pack's eval/scenarios.json (expected rule verdicts; never edited to improve a
score) plus eval/goal_scenarios.json (procedure identification). Each run uses a fresh in-memory Twin.
"""
from eval import utf8_console
import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from time import perf_counter

from config import EMBED_MODEL, LLM_MODEL, PROJECT_ROOT
from eval.memory_eval import run_memory_checks
from eval.metrics import RunMeasure, agent_behaviour, classify_failure, measure, system_row
from eval.policy import RuleFollowingPolicy
from eval.systems import full_agent, single_llm, static_checklist
from src.agent.gate1 import GateContext, build_gate_context
from src.llm import USAGE, Backend, agent_backend, agent_model_label, missing_key, usage_summary
from src.models import Pack
from src.pack import load_all_packs, load_scenarios
from src.tools.common import ToolContext
from src.twin import service
from src.twin.store import TwinStore

USER = "eval_user"
GOALS_FILE = Path(__file__).with_name("goal_scenarios.json")
DEFAULT_OUT = PROJECT_ROOT / "docs" / "eval_results.md"


def seeded_store(pack: Pack, scenario: dict) -> TwinStore:
    """A fresh Twin with the scenario's facts confirmed and its procedure active."""
    store = TwinStore(":memory:")
    for name, value in scenario["facts"].items():
        service.record_app_value(store, pack, USER, name, value)
    store.set_active_procedure(USER, pack.id)
    return store


def gate_for(store: TwinStore, packs: dict[str, Pack], procedure_id: str | None, today: date, refs: list[str]) -> GateContext:
    ctx = ToolContext(store=store, packs=packs, user_id=USER, today=today, procedure_id=procedure_id)
    ctx.created_proposals = list(refs)
    return build_gate_context(ctx)


def run_a(pack: Pack, packs: dict, scenario: dict) -> RunMeasure:
    today = date.fromisoformat(scenario["today"])
    started = perf_counter()
    answer = static_checklist(pack)
    with seeded_store(pack, scenario) as store:
        gate = gate_for(store, packs, pack.id, today, [])
        return measure(scenario["scenario_id"], "A", answer, scenario["expected"]["rule_status"], gate, 0, perf_counter() - started)


def run_b(pack: Pack, packs: dict, scenario: dict, backend: Backend) -> RunMeasure:
    today = date.fromisoformat(scenario["today"])
    started, mark = perf_counter(), USAGE.mark()
    answer, calls = single_llm(backend, pack, scenario["goal"], scenario["facts"])
    with seeded_store(pack, scenario) as store:
        gate = gate_for(store, packs, pack.id, today, [])
        return measure(scenario["scenario_id"], "B", answer, scenario["expected"]["rule_status"], gate, calls,
                       perf_counter() - started, usage=usage_summary(USAGE.since(mark)))


def run_d(pack: Pack, packs: dict, scenario: dict, backend: Backend) -> RunMeasure:
    today = date.fromisoformat(scenario["today"])
    started, mark = perf_counter(), USAGE.mark()
    with seeded_store(pack, scenario) as store, tempfile.TemporaryDirectory() as outbox:
        result = full_agent(store, packs, USER, scenario["goal"], today, backend, Path(outbox))
        refs = [f.finding_id for f in result.findings] + result.proposals
        gate = gate_for(store, packs, pack.id, today, refs)
        return measure(scenario["scenario_id"], "D", result.answer, scenario["expected"]["rule_status"], gate,
                       result.llm_calls, perf_counter() - started, agent=result, usage=usage_summary(USAGE.since(mark)))


def run_goal(packs: dict, scenario: dict, backend: Backend) -> dict:
    """Procedure identification from a goal alone (fresh user, no active procedure)."""
    today = date.fromisoformat(scenario["today"])
    with TwinStore(":memory:") as store, tempfile.TemporaryDirectory() as outbox:
        result = full_agent(store, packs, USER, scenario["goal"], today, backend, Path(outbox))
    expected = scenario["expected_procedure"]
    ok = result.procedure_id == expected and (expected is not None or not (result.answer and result.answer.claims))
    return {"scenario_id": scenario["scenario_id"], "expected": expected, "selected": result.procedure_id,
            "status": result.status, "ok": ok}


def evaluate(packs: dict[str, Pack], backend: Backend, k: int, live: bool, live_backend: Backend | None = None) -> dict:
    """Run every system k times on every scenario; return measurements and identification results."""
    runs: list[RunMeasure] = []
    for pack in packs.values():
        for scenario in load_scenarios(pack):
            for _ in range(k):
                runs.append(run_a(pack, packs, scenario))
                if live:
                    runs.append(run_b(pack, packs, scenario, live_backend or backend))
                runs.append(run_d(pack, packs, scenario, backend))
    goals = json.loads(GOALS_FILE.read_text(encoding="utf-8"))
    # With --procedure only some packs are loaded; skip goals that expect a pack that is not loaded.
    goals = [g for g in goals if g["expected_procedure"] is None or g["expected_procedure"] in packs]
    identification = [run_goal(packs, scenario, backend) for scenario in goals]
    memory = []
    if "ikamet_first" in packs:   # the memory scenarios use that pack's facts
        with tempfile.TemporaryDirectory() as folder:
            memory = run_memory_checks(Path(folder), packs)
    return {"runs": runs, "identification": identification, "memory": memory}


# --------------------------------------------------------------- report --
COLUMNS = ["system", "runs", "verdict_accuracy", "date_errors", "invented", "valid_answer_rate", "pass^k", "pass@k",
           "avg_llm_calls", "avg_seconds", "p95_seconds", "avg_tokens", "cost_usd"]


def table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows]
    return lines


def report(outcome: dict, k: int, live: bool, model: str = "") -> str:
    runs = outcome["runs"]
    systems = ["A", "B", "D"] if live else ["A", "D"]
    rows = [system_row(system, [r for r in runs if r.system == system], k) for system in systems]
    if not live:
        rows.insert(1, {"system": "B", "runs": "not run offline (needs --live)"})
    d_runs = [r for r in runs if r.system == "D"]
    if not model:
        model = f"{LLM_MODEL} / {EMBED_MODEL}" if live else "offline: eval.policy.RuleFollowingPolicy (no language model)"
    lines = [f"# Evaluation results ({date.today().isoformat()})", "", f"- Mode: {'live' if live else 'offline'}; "
             f"models: {model}; k = {k} runs per scenario.",
             "- A = static checklist, B = single LLM (no state/rules/tools), D = orchestrator + ReAct workers + Gate 1.",
             "- Offline D measures the pipeline with a rule-following scripted policy, not model quality.",
             "- pass^k: scenarios where all k runs succeed; pass@k: chance that at least one of k runs succeeds. "
             "Tokens and cost come from the providers' usage reports (n/a offline or without a price).",
             "- reflection_recovery_rate: answers refused by Gate 1 that the model then repaired (not the fallback).",
             ""]
    lines += table(rows, COLUMNS) + ["", "## Agent behaviour (D)", ""]
    lines += table([agent_behaviour(d_runs)], list(agent_behaviour(d_runs)))
    lines += ["", "## Procedure identification (D)", ""]
    lines += table(outcome["identification"], ["scenario_id", "expected", "selected", "status", "ok"])
    memory = outcome.get("memory", [])
    lines += ["", "## Memory across sessions (Twin closed and reopened; offline policy)", ""]
    lines += [f"- {sum(c.ok for c in memory)}/{len(memory)} checks passed"]
    lines += [f"  - FAILED {c.scenario}: {c.name}" for c in memory if not c.ok]
    failed = [r for r in d_runs if not r.success]
    lines += ["", "## Scenarios where D failed", ""]
    lines += [f"- {r.scenario_id}: {classify_failure(r)} (gate: {', '.join(r.gate_codes) or 'passed'}, "
              f"accuracy {r.accuracy})" for r in failed] or ["- none"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="use the local Ollama model for B and D")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--procedure", default=None, help="evaluate only this pack")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "anthropic"],
                        help="with --live: which model runs B and D")
    args = parser.parse_args(argv)
    if args.live and missing_key(args.provider):
        parser.error(missing_key(args.provider))
    packs = load_all_packs()
    if args.procedure:
        packs = {args.procedure: packs[args.procedure]}
    backend: Backend = agent_backend("team", args.provider) if args.live else RuleFollowingPolicy()
    model = f"{agent_model_label('team', args.provider)} / {EMBED_MODEL}" if args.live else ""
    text = report(evaluate(packs, backend, args.k, args.live), args.k, args.live, model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
