"""Retrieval evaluation of embedding models on the packs' saved official texts.

python -m eval.rag_eval                                  EMBED_MODEL from config.py
python -m eval.rag_eval --models bge-m3 qwen3-embedding:0.6b --out docs/eval_rag.md

Queries are built from the packs themselves, so nobody hand-picks easy questions:
- every rule's description (English) is a query;
- a hit is a passage from one of the rule's source_ids; when the rule's notes carry a quote
  (Quote: "..."), the passage must also contain the start of that quote (a stricter check).
Turkish sources with English descriptions make most ID-card queries cross-lingual, like real users.
Each model starts from an empty in-memory vector cache, so the indexing time is measured too.
"""
import argparse
import re
import sys
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel

from config import EMBED_MODEL, PROJECT_ROOT
from src.llm import OllamaBackend
from src.models import Pack, Rule
from src.pack import load_all_packs
from src.rag.drafts import comparable
from src.rag.index import Hit, chunk_vectors, pack_chunks, search
from src.twin.store import TwinStore

QUOTE = re.compile(r"Quote: \"(.{20,}?)(?:\.\.\.|\")")
QUOTE_START_CHARS = 40
TOP_K = 3


class Query(BaseModel):
    procedure_id: str
    rule_id: str
    text: str
    sources: list[str]
    quote_start: str = ""


class ModelScore(BaseModel):
    model: str
    queries: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    index_seconds: float
    seconds_per_query: float
    hit_at_3_by_procedure: dict[str, float] = {}


def rule_queries(pack: Pack, rule: Rule) -> Query | None:
    """One query per rule with saved source texts; None when the rule cites no saved text."""
    saved = {path.stem for path in (pack.root / "sources_text").glob("*.txt")}
    sources = [source for source in rule.source_ids if source in saved]
    if not sources:
        return None
    found = QUOTE.search(rule.notes)
    start = comparable(found.group(1))[:QUOTE_START_CHARS] if found else ""
    return Query(procedure_id=pack.id, rule_id=rule.rule_id, text=rule.description, sources=sources, quote_start=start)


def all_queries(packs: dict[str, Pack]) -> list[Query]:
    queries = []
    for pack in packs.values():
        queries.extend(query for rule in pack.rules if (query := rule_queries(pack, rule)) is not None)
    return queries


def is_hit(query: Query, hit: Hit, passages: dict[str, str]) -> bool:
    """Right source, and (when the rule has a quote) the whole passage holding that quote."""
    if hit.source_id not in query.sources:
        return False
    return not query.quote_start or query.quote_start in comparable(passages[hit.chunk_id])


def rank_of_first_hit(query: Query, hits: list[Hit], passages: dict[str, str]) -> int | None:
    return next((index + 1 for index, hit in enumerate(hits) if is_hit(query, hit, passages)), None)


def score_model(model: str, packs: dict[str, Pack], queries: list[Query]) -> ModelScore:
    """Index every pack, then run every query, with one embedding model."""
    backend = OllamaBackend(embed_model=model)
    with TwinStore(":memory:") as store:
        started = perf_counter()
        for pack in packs.values():
            chunk_vectors(store, pack, backend, pack_chunks(pack))
        index_seconds = perf_counter() - started
        ranks, started = [], perf_counter()
        for query in queries:
            pack = packs[query.procedure_id]
            passages = {chunk.chunk_id: chunk.text for chunk in pack_chunks(pack)}
            hits = search(store, pack, backend, query.text, k=TOP_K)
            ranks.append(rank_of_first_hit(query, hits, passages))
        query_seconds = perf_counter() - started
    count = max(len(queries), 1)
    return ModelScore(model=model, queries=len(queries), hit_at_1=round(sum(r == 1 for r in ranks) / count, 3),
                      hit_at_3=round(sum(r is not None for r in ranks) / count, 3),
                      mrr=round(sum(1 / r for r in ranks if r) / count, 3), index_seconds=round(index_seconds, 1),
                      seconds_per_query=round(query_seconds / count, 3),
                      hit_at_3_by_procedure=by_procedure(queries, ranks))


def by_procedure(queries: list[Query], ranks: list[int | None]) -> dict[str, float]:
    """hit@3 per pack (English İkamet pages and Turkish NVİ pages behave differently)."""
    totals: dict[str, list[int]] = {}
    for query, rank in zip(queries, ranks):
        totals.setdefault(query.procedure_id, []).append(1 if rank is not None else 0)
    return {procedure: round(sum(hits) / len(hits), 3) for procedure, hits in sorted(totals.items())}


def table(scores: list[ModelScore], packs: dict[str, Pack]) -> str:
    """The Markdown report: overall scores, then hit@3 per procedure."""
    chunks = sum(len(pack_chunks(pack)) for pack in packs.values())
    lines = [f"# Retrieval evaluation ({len(packs)} packs, {chunks} passages, top-{TOP_K})", "",
             "| model | queries | hit@1 | hit@3 | MRR | index s | s/query |", "|---|---|---|---|---|---|---|"]
    for s in scores:
        lines.append(f"| {s.model} | {s.queries} | {s.hit_at_1} | {s.hit_at_3} | {s.mrr} | {s.index_seconds} | "
                     f"{s.seconds_per_query} |")
    procedures = sorted({p for s in scores for p in s.hit_at_3_by_procedure})
    lines += ["", "hit@3 by procedure:", "", "| model | " + " | ".join(procedures) + " |",
              "|---|" + "---|" * len(procedures)]
    for s in scores:
        lines.append(f"| {s.model} | " + " | ".join(str(s.hit_at_3_by_procedure.get(p, "")) for p in procedures) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    """Score each model, print the report and optionally save it."""
    parser = argparse.ArgumentParser(prog="python -m eval.rag_eval")
    parser.add_argument("--models", nargs="+", default=[EMBED_MODEL])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    packs = load_all_packs()
    queries = all_queries(packs)
    report = table([score_model(model, packs, queries) for model in args.models], packs)
    print(report)
    if args.out:
        out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
