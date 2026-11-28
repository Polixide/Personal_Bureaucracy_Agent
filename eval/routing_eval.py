"""Procedure routing: does a goal select the right pack, or none when it must not?

    python -m eval.routing_eval [--out docs/eval_routing.md]

Goals are in eval/routing_goals.json with a 'dev' split (used while fixing the matcher) and a 'held_out' split
(written before the fix and scored only after it). Deterministic and offline: search_procedures needs no model.
Counts: correct, wrong pack, picked a pack when none was expected (false start), and no pack when one was
expected (missed, including ties where the agent asks).
"""
from eval import utf8_console
import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from config import PROJECT_ROOT
from src.models import Pack
from src.pack import load_all_packs
from src.tools.search_procedures import rank

GOALS_FILE = PROJECT_ROOT / "eval" / "routing_goals.json"


class Routed(BaseModel):
    split: str
    goal: str
    expected: str | None
    selected: str | None
    candidates: list[tuple[str, int]]

    @property
    def outcome(self) -> str:
        if self.selected == self.expected:
            return "correct"
        if self.expected is None:
            return "false start"
        return "missed" if self.selected is None else "wrong pack"


def select(packs: dict[str, Pack], goal: str) -> tuple[str | None, list[tuple[str, int]]]:
    """The same decision as search_procedures: the single best match, or None on no match or a tie."""
    matches = rank(packs, goal)
    tie = len(matches) > 1 and matches[0].score == matches[1].score
    return (matches[0].procedure_id if matches and not tie else None,
            [(m.procedure_id, m.score) for m in matches])


def route_all(packs: dict[str, Pack], path: Path = GOALS_FILE) -> list[Routed]:
    results = []
    for item in json.loads(path.read_text(encoding="utf-8"))["goals"]:
        selected, candidates = select(packs, item["goal"])
        results.append(Routed(split=item["split"], goal=item["goal"], expected=item["expected"],
                              selected=selected, candidates=candidates))
    return results


def report(results: list[Routed]) -> str:
    lines = ["# Procedure routing", "", "| split | goals | correct | wrong pack | false start | missed |",
             "|---|---|---|---|---|---|"]
    for split in ("dev", "held_out"):
        subset = [r for r in results if r.split == split]
        counts = {o: sum(r.outcome == o for r in subset) for o in ("correct", "wrong pack", "false start", "missed")}
        lines.append(f"| {split} | {len(subset)} | {counts['correct']} | {counts['wrong pack']} | "
                     f"{counts['false start']} | {counts['missed']} |")
    wrong = [r for r in results if r.outcome != "correct"]
    if wrong:
        lines += ["", "| split | goal | expected | selected | scores |", "|---|---|---|---|---|"]
        lines += [f"| {r.split} | {r.goal} | {r.expected} | {r.selected} | {r.candidates} |" for r in wrong]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.routing_eval")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = report(route_all(load_all_packs()))
    print(text)
    if args.out:
        (args.out if args.out.is_absolute() else PROJECT_ROOT / args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main(sys.argv[1:]))
