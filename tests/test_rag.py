"""src/rag/index.py + search_sources: chunking, cached vectors, and 10 queries with expected sources.

The embedder is tests/fakes.py:hash_embedding (bag of words), so results are deterministic offline.
Each expected set lists the sources that state the requirement (from the rules' source_ids).
The texts are pinned in tests/data/placeholder_sources/ (the summaries the pack had before its official
pages were downloaded), so a change on a government website cannot change these results.
"""
from datetime import date
from pathlib import Path

import pytest

from fakes import ScriptedBackend
from config import RAG_EMBED_BATCH
from src.pack import load_all_packs
from src.rag.index import chunk_text, pack_chunks, search
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool

PACKS = load_all_packs()
PINNED_TEXTS = Path(__file__).parent / "data" / "placeholder_sources" / "ikamet_first"
PACK = PACKS["ikamet_first"].model_copy(update={"root": PINNED_TEXTS})


def sources_of(rule_id: str) -> set[str]:
    return set(PACK.rule(rule_id).source_ids)


QUERIES = [
    ("How many days must my passport stay valid after the permit end date?", "R006"),
    ("Is an Airbnb booking accepted as address proof, or do I need a rental agreement or dorm certificate?", "R017"),
    ("Does the biometric photo have to be ICAO compliant and recent?", "R009"),
    ("Where should I track my application status after IRO submission?", "R029"),
    ("Do I have to declare sufficient regular financial means?", "R012"),
    ("Which health insurance evidence is accepted: SGK provision or private health insurance?", "R016"),
    ("Who makes the final evaluation and approval decision, Migration Management or IRO?", "R028"),
    ("Must scanned documents be sent to IRO for confirmation before physical submission?", "R021"),
    ("Can I use an application form from a previous month or window?", "R025"),
    ("Is single-entry visa fee evidence needed?", "R011"),
]


@pytest.mark.parametrize("query,rule_id", QUERIES, ids=[rule_id for _, rule_id in QUERIES])
def test_query_returns_a_source_that_states_the_requirement(store, query, rule_id):
    hits = search(store, PACK, ScriptedBackend(), query, k=3)
    assert hits and hits[0].source_id in sources_of(rule_id), [h.source_id for h in hits]
    assert rule_id in hits[0].text


def test_chunks_respect_the_size_limit_and_keep_source_ids():
    chunks = pack_chunks(PACK)
    assert chunks and all(len(c.text) <= 800 for c in chunks)
    assert {c.source_id for c in chunks} == {p.stem for p in (PACK.root / "sources_text").glob("*.txt")}
    assert chunk_text("a" * 1700, max_chars=800) == ["a" * 800, "a" * 800, "a" * 100]


def test_vectors_are_cached_in_the_twin(store):
    """The first search embeds every passage once (in small batches), later searches only the question."""
    backend = ScriptedBackend()
    search(store, PACK, backend, "passport validity")
    *passage_batches, first_question = backend.embed_calls
    search(store, PACK, backend, "address proof")
    assert sum(len(batch) for batch in passage_batches) == len(pack_chunks(PACK))
    assert all(len(batch) <= RAG_EMBED_BATCH for batch in passage_batches) and len(passage_batches) > 1
    assert len(first_question) == 1 and backend.embed_calls[len(passage_batches) + 1:] == [["address proof"]]


def test_qwen3_embedding_queries_get_the_instruction_but_passages_do_not(store):
    backend = ScriptedBackend()
    backend.embed_model = "qwen3-embedding:0.6b"
    search(store, PACK, backend, "passport validity")
    *passages, question = backend.embed_calls
    assert question[0].startswith("Instruct: ") and question[0].endswith("Query: passport validity")
    assert not any(text.startswith("Instruct: ") for batch in passages for text in batch)


def test_search_sources_tool_cites_and_never_decides(store):
    ctx = ToolContext(store=store, packs=PACKS, user_id="u", today=date(2026, 11, 18), procedure_id=PACK.id,
                      backend=ScriptedBackend())
    output = run_tool(build_registry(), "search_sources", {"query": "passport valid 60 days", "k": 2}, ctx, "worker").output
    assert len(output["hits"]) <= 2 and output["hits"][0]["title"]
    assert "never decide" in output["note"]
    assert not run_tool(build_registry(), "search_sources", {"query": "x"}, ctx, "worker").ok


def test_search_without_embedding_model_is_an_error(store):
    ctx = ToolContext(store=store, packs=PACKS, user_id="u", today=date(2026, 11, 18), procedure_id=PACK.id)
    assert "No embedding model" in run_tool(build_registry(), "search_sources", {"query": "passport"}, ctx, "worker").error


def test_a_passage_is_never_cut_inside_a_word_or_a_link():
    """Seen live: a passage cut at 600 characters gave the model, and the user, a broken URL."""
    from src.rag.index import passage_text
    text = "word " * 119 + "https://dijital.gib.gov.tr/foreigners/kimlikNoBasvuru and the rest"
    passage = passage_text(text)
    assert passage.startswith(text[:600]) and passage.endswith("kimlikNoBasvuru")
