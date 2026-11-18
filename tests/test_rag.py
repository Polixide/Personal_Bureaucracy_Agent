"""src/rag/index.py + search_sources: chunking, cached vectors, and 10 queries with expected sources.

The embedder is tests/fakes.py:hash_embedding (bag of words), so results are deterministic offline.
Each expected set lists the sources that state the requirement (from the rules' source_ids).
"""
from datetime import date

import pytest

from fakes import ScriptedBackend
from src.pack import load_all_packs
from src.rag.index import chunk_text, pack_chunks, search
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool

PACKS = load_all_packs()
PACK = PACKS["ikamet_first"]


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
    backend = ScriptedBackend()
    search(store, PACK, backend, "passport validity")
    first_batch = len(backend.embed_calls[0])
    search(store, PACK, backend, "address proof")
    assert first_batch == len(pack_chunks(PACK))
    assert [len(call) for call in backend.embed_calls[1:]] == [1, 1]


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
