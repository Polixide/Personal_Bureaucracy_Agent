"""Keyword retrieval (src/rag/lexical.py), the Kimlik gold questions and eval/rag_compare.py, offline."""
import json

import pytest

from eval.rag_compare import (RETRIEVERS, Case, evaluate, first_correct, question_cases, report, rule_cases,
                              scores)
from src.pack import load_all_packs
from src.rag.drafts import comparable
from src.rag.index import Chunk, pack_chunks
from src.rag.lexical import Bm25Index, fuse, tokens
from tests.fakes import ScriptedBackend

PACKS = load_all_packs()
KIMLIK = ("tc_kimlik_first", "tc_kimlik_renewal")


def chunk(chunk_id, text, source="S"):
    return Chunk(chunk_id=chunk_id, source_id=source, text=text)


# --------------------------------------------------------------- tokens / BM25 --
def test_tokens_fold_turkish_letters_and_cut_suffixes():
    assert tokens("KAYBI kaybettim Kimlik KARTI İçin") == ["kaybi", "kaybe", "kimli", "karti", "icin"]
    assert tokens("ı İ ş ğ ü ö ç") == []                 # single letters are dropped


def test_bm25_prefers_the_passage_with_the_rare_query_word():
    index = Bm25Index([chunk("a", "kimlik kartı başvurusu randevu"), chunk("b", "kimlik kartı kaybı halinde bildirim"),
                       chunk("c", "kimlik kartı fotoğraf")])
    assert [c.chunk_id for c, _ in index.rank("kimlik kaybı", 3)][0] == "b"
    # Known limit of prefix stemming: consonant softening (kayıp -> kaybı) gives different prefixes.
    assert tokens("kayıp") != tokens("kaybı")
    assert index.rank("passport", 3) == []                # no shared word: no result, not a random one


def test_fusion_rewards_agreement_and_keeps_k():
    a, b, c = chunk("a", "x"), chunk("b", "y"), chunk("c", "z")
    fused = fuse([[a, b, c], [b, c, a]], k=2)
    assert [ch.chunk_id for ch, _ in fused] == ["b", "a"]  # b is 2nd and 1st; a is 1st and 3rd


# ------------------------------------------------------------ gold questions --
@pytest.mark.parametrize("pack_id", KIMLIK)
def test_every_gold_phrase_is_in_its_cited_source(pack_id):
    """The labels must come from the saved official text, not from memory."""
    pack = PACKS[pack_id]
    data = json.loads((pack.root / "eval" / "rag_questions.json").read_text(encoding="utf-8"))
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids)) >= 15
    for question in data["questions"]:
        for answer in question["answers"]:
            assert pack.source(answer["source_id"]) is not None, (question["id"], answer["source_id"])
            text = (pack.root / "sources_text" / f"{answer['source_id']}.txt").read_text(encoding="utf-8")
            assert comparable(answer["contains"]) in comparable(text), question["id"]


@pytest.mark.parametrize("pack_id", KIMLIK)
def test_every_gold_question_is_answerable_by_some_chunk(pack_id):
    """Chunking must not cut every answer phrase in half, or the question could never be a hit."""
    chunks = pack_chunks(PACKS[pack_id])
    for case in [c for c in question_cases(PACKS) if c.procedure_id == pack_id]:
        assert any(first_correct(case, [c]) == 1 for c in chunks), case.case_id


# --------------------------------------------------------------- comparison --
def test_comparison_runs_every_case_with_every_retriever():
    cases = rule_cases(PACKS) + question_cases(PACKS)
    results, index = evaluate(PACKS, ScriptedBackend(), cases)
    assert len(results) == len(cases) and {r.case.query_set for r in results} == {"rules", "questions"}
    for name in RETRIEVERS:
        s = scores(results, name)
        assert 0 <= s["hit@1"] <= s["hit@3"] <= s["hit@5"] <= 1 and s["p50_ms"] <= s["p95_ms"]
        assert index[name] >= 0
    text = report(results, index, "fake", offline=True)
    # one query per rule: 30 + 8 + 15 + 14 (67 since ikamet_first gained R030, migration_plan section 10)
    assert "## rules (67 queries)" in text and "## questions (31 queries)" in text


def test_a_correct_passage_needs_the_source_and_the_phrase():
    case = Case(query_set="questions", procedure_id="p", case_id="q", text="t",
                phrases={"S": comparable("Kartı kaybetmesi halinde")})
    assert first_correct(case, [chunk("1", "x", "S"), chunk("2", "kartı KAYBETMESİ halinde durumu", "S")]) == 2
    assert first_correct(case, [chunk("3", "Kartı kaybetmesi halinde", "OTHER")]) is None


def test_ndcg_precision_and_recall_on_a_known_ranking():
    import math
    from eval.rag_compare import K, ndcg
    assert ndcg([1], 1) == 1.0                                  # the only correct passage first: best order
    assert ndcg([2], 1) == pytest.approx(1 / math.log2(3))      # one place lower
    assert ndcg([], 3) == 0.0 and ndcg([1, 2], 0) == 0.0
    assert K == 5


def test_scores_report_precision_recall_and_ndcg_for_every_retriever():
    cases = question_cases(PACKS)[:4]
    results, _ = evaluate(PACKS, ScriptedBackend(), cases)
    for name in RETRIEVERS:
        s = scores(results, name)
        assert 0 <= s["precision@5"] <= 1 and 0 <= s["recall@5"] <= 1 and 0 <= s["ndcg@5"] <= 1
