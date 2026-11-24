"""Speed benchmark: how long each part of the agent takes, so a slow run can be blamed on the right part.

    python -m eval.speed                      offline parts (no model); a few seconds
    python -m eval.speed --live               adds the model parts (Ollama running with LLM_MODEL,
                                              EMBED_MODEL and VISION_MODEL of config.py installed)
    python -m eval.speed --live --out docs/eval_speed.md

Offline parts are code the agent always runs: loading packs, routing a goal, opening a Twin with a user's
facts, evaluating the rules, keyword retrieval, and a whole agent run driven by the scripted offline policy
(orchestration, tools and Gate 1, minus the model). Live parts are the model calls: the first chat call (it
loads the model) and warm ones, embedding a question, indexing a pack, reading synthetic Kimlik documents,
and a whole Kimlik agent run. Every item is timed on its own; the report gives median and 95th percentile.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from config import EMBED_MODEL, LLM_MODEL, PROJECT_ROOT, VISION_MODEL
from eval.policy import RuleFollowingPolicy
from eval.routing_eval import GOALS_FILE, select
from eval.run_eval import USER, seeded_store
from eval.systems import full_agent
from src.models import Pack
from src.pack import load_all_packs, load_scenarios
from src.rag.index import chunk_vectors, pack_chunks, rank
from src.rag.lexical import Bm25Index
from src.rules import evaluate_rules
from src.tools.common import ToolContext
from src.twin.store import TwinStore

KIMLIK = ("tc_kimlik_first", "tc_kimlik_renewal")
QUESTION = "I lost my ID card, what should I do?"


class Timing(BaseModel):
    part: str
    samples: int
    median_ms: float
    p95_ms: float
    note: str = ""


def summarize(part: str, times_ms: list[float], note: str = "") -> Timing:
    ordered = sorted(times_ms)
    return Timing(part=part, samples=len(ordered), median_ms=round(statistics.median(ordered), 2),
                  p95_ms=round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2), note=note)


def time_each(items: Iterable[Any], work: Callable[[Any], object], repeat: int = 1) -> list[float]:
    """Milliseconds of `work(item)` for every item, `repeat` times."""
    times = []
    for _ in range(repeat):
        for item in items:
            started = perf_counter()
            work(item)
            times.append((perf_counter() - started) * 1000)
    return times


def kimlik_scenarios(packs: dict[str, Pack]) -> list[tuple[Pack, dict]]:
    return [(packs[pid], scenario) for pid in KIMLIK for scenario in load_scenarios(packs[pid])]


def facts_of(packs: dict[str, Pack], pack: Pack, scenario: dict) -> dict:
    with seeded_store(pack, scenario) as store:
        ctx = ToolContext(store=store, packs=packs, user_id=USER, today=date.today(), procedure_id=pack.id)
        return ctx.view().fact_map()


def run_agent_once(packs: dict[str, Pack], pack: Pack, scenario: dict, backend) -> Any:
    """One whole agent run (system D) on a Twin seeded with the scenario's facts."""
    with seeded_store(pack, scenario) as store, tempfile.TemporaryDirectory() as outbox:
        return full_agent(store, packs, USER, scenario["goal"], date.fromisoformat(scenario["today"]), backend,
                          Path(outbox))


def offline_timings(repeat: int) -> list[Timing]:
    """The deterministic parts, on the real packs, the routing goals and the 12 Kimlik scenarios."""
    packs = load_all_packs()
    goals = [g["goal"] for g in json.loads(GOALS_FILE.read_text(encoding="utf-8"))["goals"]]
    scenarios = kimlik_scenarios(packs)
    facts = [(pack, facts_of(packs, pack, scenario)) for pack, scenario in scenarios]
    indexes = {pid: Bm25Index(pack_chunks(packs[pid])) for pid in KIMLIK}
    return [
        summarize("load all packs", time_each([None], lambda _: load_all_packs(), repeat), f"{len(packs)} packs"),
        summarize("route one goal", time_each(goals, lambda g: select(packs, g), repeat), f"{len(goals)} goals"),
        summarize("open a Twin with a user's facts", time_each(scenarios, lambda ps: seeded_store(*ps).close(), repeat),
                  "12 Kimlik scenarios, in memory"),
        summarize("evaluate all rules", time_each(facts, lambda pf: evaluate_rules(*pf), repeat),
                  "14-15 rules per Kimlik pack"),
        summarize("build keyword index", time_each(KIMLIK, lambda pid: Bm25Index(pack_chunks(packs[pid])), repeat),
                  "one Kimlik pack"),
        summarize("keyword search (BM25)", time_each(KIMLIK, lambda pid: indexes[pid].rank(QUESTION, 3), repeat),
                  "one question"),
        summarize("whole agent run, scripted policy",
                  time_each(scenarios, lambda ps: run_agent_once(packs, *ps, RuleFollowingPolicy()), max(1, repeat // 10)),
                  "Kimlik scenarios; orchestration + tools + Gate 1, no model"),
    ]


def live_timings(repeat: int) -> list[Timing]:
    """Model calls on the local Ollama models of config.py."""
    from src.llm import OllamaBackend
    backend = OllamaBackend()
    packs = load_all_packs()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    message = [{"role": "user", "content": "Reply with JSON: the answer is 'ok'."}]
    chat = lambda _: backend.chat(message, schema, "speed_probe")   # noqa: E731
    timings = [summarize("first chat call (loads the model)", time_each([None], chat), LLM_MODEL),
               summarize("chat call, warm", time_each([None], chat, repeat), f"{LLM_MODEL}, tiny JSON reply")]
    timings += embedding_timings(backend, packs, repeat)
    timings += document_timings(packs)
    scenarios = kimlik_scenarios(packs)
    timings.append(summarize("whole Kimlik agent run, live", time_each(scenarios[:2], lambda ps: run_agent_once(packs, *ps, backend)),
                             f"{LLM_MODEL}; scenarios {', '.join(s['scenario_id'] for _, s in scenarios[:2])}"))
    return timings


def embedding_timings(backend, packs: dict[str, Pack], repeat: int) -> list[Timing]:
    """Index one Kimlik pack from an empty cache, then embed-and-rank single questions."""
    pack = packs["tc_kimlik_renewal"]
    chunks = pack_chunks(pack)
    with TwinStore(":memory:") as store:
        index = time_each([None], lambda _: chunk_vectors(store, pack.id, backend, chunks))
        search = time_each([None], lambda _: rank(store, pack.id, backend, chunks, QUESTION, 3), repeat)
    return [summarize("index one Kimlik pack (embeddings)", index, f"{EMBED_MODEL}, {len(chunks)} passages"),
            summarize("embedding search, one question", search, EMBED_MODEL)]


def document_timings(packs: dict[str, Pack]) -> list[Timing]:
    """Read the synthetic Kimlik documents of eval/document_samples.py (text, photo, two-in-one PDFs)."""
    from eval.document_samples import synthetic_samples
    from src.tools.extract_document import extract_file
    timings = []
    with tempfile.TemporaryDirectory() as folder:
        for sample in synthetic_samples(Path(folder)):
            if "kimlik" not in sample.sample_id and "nufus" not in sample.sample_id:
                continue
            ms = time_each([None], lambda _: extract_file(sample.file, packs[sample.procedure_id]))
            timings.append(summarize(f"read document {sample.sample_id}", ms, f"{LLM_MODEL} + {VISION_MODEL}"))
    return timings


def hardware() -> str:
    """Machine description for the report (Apple chip name when available)."""
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        chip = out.stdout.strip() or chip
    return f"{chip}, {platform.system()} {platform.release()}, Python {platform.python_version()}"


def report(offline: list[Timing], live: list[Timing] | None) -> str:
    lines = ["# Speed benchmark", "", f"Hardware: {hardware()}. Date: {date.today().isoformat()}.", ""]
    for title, rows in (("Offline parts (no model)", offline), ("Model parts (local Ollama)", live)):
        if rows is None:
            lines += [f"## {title}", "", "Not run (use --live with Ollama running).", ""]
            continue
        lines += [f"## {title}", "", "| part | samples | median ms | p95 ms | note |", "|---|---|---|---|---|"]
        lines += [f"| {t.part} | {t.samples} | {t.median_ms} | {t.p95_ms} | {t.note} |" for t in rows] + [""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.speed")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeat", type=int, default=20, help="repetitions of the fast parts")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = report(offline_timings(args.repeat), live_timings(max(3, args.repeat // 4)) if args.live else None)
    print(text)
    if args.out:
        (args.out if args.out.is_absolute() else PROJECT_ROOT / args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
