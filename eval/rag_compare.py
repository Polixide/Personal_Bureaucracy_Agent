"""Compare keyword (BM25), embedding and hybrid retrieval on the packs' official texts.

    python -m eval.rag_compare --out docs/eval_rag_compare.md          live: EMBED_MODEL via Ollama
    python -m eval.rag_compare --offline                               pipeline check, fake embedder

Two query sets:
- rules:     every rule's description (eval/rag_eval.py); a hit is a passage of the rule's sources holding
             the start of the rule's quote. Measures "find the text behind a requirement".
- questions: procedures/<id>/eval/rag_questions.json, user-style questions the rules do not decide, each
             with the source passages that answer it. Measures what the chat uses RAG for.

Every retriever ranks the same chunks (src/rag/index.py). Quality, over the top 5:
- hit@1/3/5: a correct passage is in the first 1 / 3 / 5;
- MRR@5: mean of 1 / rank of the first correct passage (0 when none);
- precision@5: share of the 5 passages that are correct;
- recall@5: share of the labelled sources (a question can have several) that the top 5 reaches;
- nDCG@5: the correct passages weighted by their rank (1 / log2(rank + 1)), divided by the best possible order.
Precision, recall and nDCG come from Andrea's retrieval metrics (feature/marco-integration-review).
Speed: index time per retriever, and the median and 95th-percentile time per query. In --offline mode the embedding ranks come from a bag-of-words fake, so only
the BM25 numbers and the pipeline are meaningful.
"""
from eval import utf8_console
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel

from config import EMBED_MODEL, PROJECT_ROOT
from eval.rag_eval import all_queries
from src.llm import Backend, OllamaBackend
from src.models import Pack
from src.pack import load_all_packs
from src.rag.drafts import comparable
from src.rag.index import Chunk, chunk_vectors, pack_chunks, rank
from src.rag.lexical import Bm25Index, fuse
from src.twin.store import TwinStore

K = 5                  # deepest rank scored (hit@5, MRR@5)
FUSION_DEPTH = 20      # how many results of each list the hybrid fuses
RETRIEVERS = ("bm25", "embedding", "hybrid")


class Case(BaseModel):
    """One query with what counts as a correct passage."""
    query_set: str                    # "rules" or "questions"
    procedure_id: str
    case_id: str
    text: str
    lang: str = "en"
    phrases: dict[str, str]           # source_id -> folded phrase the passage must hold ("" = any passage)


class Result(BaseModel):
    case: Case
    ranks: dict[str, int | None]      # retriever -> rank of the first correct passage (1..K) or None
    seconds: dict[str, float]         # retriever -> query time
    correct: dict[str, list[int]] = {}          # retriever -> ranks (1..K) of every correct passage in the top K
    sources_found: dict[str, int] = {}          # retriever -> labelled sources reached in the top K
    relevant_total: int = 0                     # correct passages in the whole pack (for nDCG's best order)


def rule_cases(packs: dict[str, Pack]) -> list[Case]:
    """The rule-description queries of eval/rag_eval.py, in this file's Case form."""
    return [Case(query_set="rules", procedure_id=q.procedure_id, case_id=q.rule_id, text=q.text,
                 phrases={source: q.quote_start for source in q.sources}) for q in all_queries(packs)]


def question_cases(packs: dict[str, Pack]) -> list[Case]:
    """The hand-labelled questions of every pack that has eval/rag_questions.json."""
    cases = []
    for pack in packs.values():
        path = pack.root / "eval" / "rag_questions.json"
        if not path.is_file():
            continue
        for item in json.loads(path.read_text(encoding="utf-8"))["questions"]:
            phrases = {answer["source_id"]: comparable(answer["contains"]) for answer in item["answers"]}
            cases.append(Case(query_set="questions", procedure_id=pack.id, case_id=item["id"],
                              text=item["question"], lang=item.get("lang", "en"), phrases=phrases))
    return cases


def is_correct(case: Case, chunk: Chunk) -> bool:
    """From one of the case's sources, and holding that source's phrase."""
    phrase = case.phrases.get(chunk.source_id)
    return phrase is not None and phrase in comparable(chunk.text)


def first_correct(case: Case, ranked: list[Chunk]) -> int | None:
    return next((i + 1 for i, chunk in enumerate(ranked[:K]) if is_correct(case, chunk)), None)


def correct_ranks(case: Case, ranked: list[Chunk]) -> list[int]:
    """Ranks (1..K) of every correct passage in the top K."""
    return [i + 1 for i, chunk in enumerate(ranked[:K]) if is_correct(case, chunk)]


def sources_reached(case: Case, ranked: list[Chunk]) -> int:
    """How many of the case's labelled sources have a correct passage in the top K."""
    return len({chunk.source_id for chunk in ranked[:K] if is_correct(case, chunk)})


def ndcg(ranks: list[int], relevant_total: int) -> float:
    """Binary-relevance nDCG@K: DCG of the correct ranks over the DCG of the best possible top K."""
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(K, relevant_total) + 1))
    return sum(1 / math.log2(rank + 1) for rank in ranks) / ideal if ideal else 0.0


class PackRetrievers:
    """The three retrievers over one pack's chunks, with their index times."""

    def __init__(self, pack: Pack, store: TwinStore, backend: Backend):
        self.pack, self.store, self.backend = pack, store, backend
        self.chunks = pack_chunks(pack)
        started = perf_counter()
        self.bm25 = Bm25Index(self.chunks)
        self.bm25_index_seconds = perf_counter() - started
        started = perf_counter()
        chunk_vectors(store, pack.id, backend, self.chunks)      # fills the (fresh) vector cache
        self.embedding_index_seconds = perf_counter() - started

    def run(self, case: Case) -> Result:
        """Rank once per retriever and time it; the hybrid's time includes both of its inputs."""
        started = perf_counter()
        lexical = [chunk for chunk, _ in self.bm25.rank(case.text, FUSION_DEPTH)]
        bm25_seconds = perf_counter() - started
        started = perf_counter()
        semantic = [chunk for chunk, _ in rank(self.store, self.pack.id, self.backend, self.chunks, case.text,
                                               FUSION_DEPTH)]
        embedding_seconds = perf_counter() - started
        started = perf_counter()
        hybrid = [chunk for chunk, _ in fuse([semantic, lexical], K)]
        fuse_seconds = perf_counter() - started
        lists = {"bm25": lexical, "embedding": semantic, "hybrid": hybrid}
        return Result(case=case, ranks={name: first_correct(case, ranked) for name, ranked in lists.items()},
                      seconds={"bm25": bm25_seconds, "embedding": embedding_seconds,
                               "hybrid": bm25_seconds + embedding_seconds + fuse_seconds},
                      correct={name: correct_ranks(case, ranked) for name, ranked in lists.items()},
                      sources_found={name: sources_reached(case, ranked) for name, ranked in lists.items()},
                      relevant_total=sum(is_correct(case, chunk) for chunk in self.chunks))


def evaluate(packs: dict[str, Pack], backend: Backend, cases: list[Case]) -> tuple[list[Result], dict]:
    """Run every case against its pack; return the results and the index times per retriever."""
    results, index = [], {"bm25": 0.0, "embedding": 0.0, "hybrid": 0.0}
    with TwinStore(":memory:") as store:
        for pack_id in sorted({case.procedure_id for case in cases}):
            retrievers = PackRetrievers(packs[pack_id], store, backend)
            index["bm25"] += retrievers.bm25_index_seconds
            index["embedding"] += retrievers.embedding_index_seconds
            results += [retrievers.run(case) for case in cases if case.procedure_id == pack_id]
    index["hybrid"] = index["bm25"] + index["embedding"]
    return results, index


def scores(results: list[Result], retriever: str) -> dict:
    """hit@1/3/5, MRR@5, precision/recall/nDCG@5 and query time percentiles for one retriever over some results."""
    ranks = [r.ranks[retriever] for r in results]
    times = sorted(r.seconds[retriever] * 1000 for r in results)
    count = max(len(results), 1)
    return {"n": len(results),
            **{f"hit@{n}": round(sum(1 for x in ranks if x is not None and x <= n) / count, 3) for n in (1, 3, 5)},
            "mrr": round(sum(1 / x for x in ranks if x) / count, 3),
            "precision@5": round(sum(len(r.correct.get(retriever, [])) / K for r in results) / count, 3),
            "recall@5": round(sum(r.sources_found.get(retriever, 0) / max(len(r.case.phrases), 1)
                                  for r in results) / count, 3),
            "ndcg@5": round(sum(ndcg(r.correct.get(retriever, []), r.relevant_total) for r in results) / count, 3),
            "p50_ms": round(statistics.median(times), 2) if times else 0.0,
            "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 2) if times else 0.0}


def report(results: list[Result], index: dict, model: str, offline: bool) -> str:
    """The Markdown report: overall per query set, by procedure, by language, and the misses."""
    title = "offline: fake bag-of-words embedder" if offline else f"embedding model {model}"
    lines = [f"# Retrieval: keyword vs embedding vs hybrid ({title})", "",
             f"Top-{K}; hybrid = reciprocal rank fusion of the top {FUSION_DEPTH} of each. Index time is for all "
             "packs in the run, from an empty cache.", ""]
    for query_set in ("rules", "questions"):
        subset = [r for r in results if r.case.query_set == query_set]
        if subset:
            lines += overall_table(query_set, subset, index) + by_group(subset, "procedure_id") + [""]
    questions = [r for r in results if r.case.query_set == "questions"]
    if questions:
        lines += by_group(questions, "lang") + [""] + misses(questions)
    return "\n".join(lines) + "\n"


def overall_table(query_set: str, subset: list[Result], index: dict) -> list[str]:
    lines = [f"## {query_set} ({len(subset)} queries)", "",
             "| retriever | hit@1 | hit@3 | hit@5 | MRR@5 | precision@5 | recall@5 | nDCG@5 | query p50 ms | "
             "query p95 ms | index s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in RETRIEVERS:
        s = scores(subset, name)
        lines.append(f"| {name} | {s['hit@1']} | {s['hit@3']} | {s['hit@5']} | {s['mrr']} | {s['precision@5']} | "
                     f"{s['recall@5']} | {s['ndcg@5']} | {s['p50_ms']} | {s['p95_ms']} | {round(index[name], 2)} |")
    return lines + [""]


def by_group(subset: list[Result], attribute: str) -> list[str]:
    """hit@3 per procedure (or per question language)."""
    groups = sorted({getattr(r.case, attribute) for r in subset})
    lines = [f"hit@3 by {attribute}:", "", "| retriever | " + " | ".join(f"{g} (n={sum(getattr(r.case, attribute) == g for r in subset)})" for g in groups) + " |",
             "|---|" + "---|" * len(groups)]
    for name in RETRIEVERS:
        cells = [str(scores([r for r in subset if getattr(r.case, attribute) == g], name)["hit@3"]) for g in groups]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def misses(subset: list[Result]) -> list[str]:
    """Questions some retriever misses in the top 3, to read by hand."""
    lines = ["## Questions missed in the top 3", "", "| id | question | " + " | ".join(RETRIEVERS) + " |",
             "|---|---|" + "---|" * len(RETRIEVERS)]
    for r in subset:
        if any(r.ranks[name] is None or r.ranks[name] > 3 for name in RETRIEVERS):
            cells = [str(r.ranks[name] or "-") for name in RETRIEVERS]
            lines.append(f"| {r.case.case_id} | {r.case.text} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.rag_compare")
    parser.add_argument("--model", default=EMBED_MODEL, help="embedding model (live mode)")
    parser.add_argument("--offline", action="store_true", help="fake embedder: checks the pipeline, BM25 is real")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    packs = load_all_packs()
    if args.offline:
        from tests.fakes import ScriptedBackend
        backend = ScriptedBackend()
    else:
        backend = OllamaBackend(embed_model=args.model)
    results, index = evaluate(packs, backend, rule_cases(packs) + question_cases(packs))
    text = report(results, index, args.model, args.offline)
    print(text)
    if args.out:
        (args.out if args.out.is_absolute() else PROJECT_ROOT / args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main(sys.argv[1:]))
