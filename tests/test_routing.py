"""Procedure routing with aliases and qualifiers (search_procedures.score, eval/routing_eval.py)."""
import pytest

from eval.routing_eval import route_all, select
from src.pack import load_all_packs
from src.tools.search_procedures import score

PACKS = load_all_packs()
RESULTS = route_all(PACKS)


@pytest.mark.parametrize("goal", ["I lost my passport", "I need to renew my passport", "My visa expired",
                                  "It is my first day at university", "my phone is broken"])
def test_a_qualifier_alone_selects_nothing(goal):
    """'lost', 'renew', 'expired', 'first', 'broken' describe what happened, not which procedure."""
    assert all(score(pack, goal) == 0 for pack in PACKS.values())


def test_qualifiers_decide_between_sibling_packs():
    assert select(PACKS, "I lost my TC kimlik")[0] == "tc_kimlik_renewal"
    assert select(PACKS, "I want to get my first Turkish ID card")[0] == "tc_kimlik_first"
    assert select(PACKS, "I want to renew my residence permit")[0] == "ikamet_extension"
    assert select(PACKS, "How do I get my first residence permit?")[0] == "ikamet_first"


def test_no_alias_appears_twice_in_a_pack_and_no_qualifier_is_also_an_alias():
    for pack in PACKS.values():
        aliases, qualifiers = pack.procedure.aliases, pack.procedure.qualifiers
        assert len(aliases) == len(set(aliases)) and len(qualifiers) == len(set(qualifiers)), pack.id
        assert not set(aliases) & set(qualifiers), pack.id


def test_every_dev_goal_routes_correctly():
    wrong = [(r.goal, r.expected, r.selected) for r in RESULTS if r.split == "dev" and r.outcome != "correct"]
    assert not wrong


def test_held_out_goals_never_start_the_wrong_procedure():
    """The held-out goals were written before the fix. Some still tie (the agent asks), which is safe;
    starting a wrong or out-of-scope procedure is not. Do not tune the packs on these goals."""
    held_out = [r for r in RESULTS if r.split == "held_out"]
    assert held_out and not [r.goal for r in held_out if r.outcome in ("wrong pack", "false start")]
