"""eval/rag_eval.py: queries come from the packs' rules, and a hit needs the rule's quoted passage."""
from eval.rag_eval import all_queries, by_procedure, is_hit
from src.pack import load_all_packs
from src.rag.index import Hit

PACKS = load_all_packs()


def test_every_rule_with_a_saved_source_is_a_query():
    queries = all_queries(PACKS)
    assert {q.procedure_id for q in queries} == set(PACKS)
    quoted = [q for q in queries if q.procedure_id.startswith("tc_kimlik")]
    assert quoted and all(q.quote_start for q in quoted)


def test_a_hit_needs_the_right_source_and_the_quoted_passage():
    [query] = [q for q in all_queries(PACKS) if q.rule_id == "TCF009"]
    passage = "x " * 300 + "On beş yaşını tamamlayan kişilerin şahsen başvurmaları esastır kimlik kartı başvurusunda"
    right = Hit(source_id=query.sources[0], title="t", chunk_id="c1", text=passage[:600], score=0.9)
    assert is_hit(query, right, {"c1": "Kimlik kartı başvurusunda; on beş yaşını tamamlayan kişilerin şahsen"})
    assert not is_hit(query, right, {"c1": "another paragraph of the same page"})
    wrong_source = right.model_copy(update={"source_id": "SOMETHING_ELSE"})
    assert not is_hit(query, wrong_source, {"c1": "Kimlik kartı başvurusunda; on beş yaşını tamamlayan"})


def test_hit_rate_is_reported_per_procedure():
    first = [q for q in all_queries(PACKS) if q.procedure_id == "tc_kimlik_first"][:2]
    renewal = [q for q in all_queries(PACKS) if q.procedure_id == "tc_kimlik_renewal"][:1]
    assert by_procedure(first + renewal, [1, None, 3]) == {"tc_kimlik_first": 0.5, "tc_kimlik_renewal": 1.0}
