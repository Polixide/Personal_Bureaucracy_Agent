"""eval/memory_eval.py: the Twin remembers across restarts (Neda's multi-session scenarios M1-M3)."""
import pytest

from eval.memory_eval import run_memory_checks
from src.pack import load_all_packs

CHECKS = None


def checks(folder):
    global CHECKS
    if CHECKS is None:
        CHECKS = run_memory_checks(folder, load_all_packs())
    return CHECKS


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return checks(tmp_path_factory.mktemp("memory"))


def test_every_memory_check_passes(results):
    failed = [f"{c.scenario}: {c.name}" for c in results if not c.ok]
    assert failed == []


def test_all_three_scenarios_ran(results):
    assert {c.scenario for c in results} == {"M1", "M2", "M3"} and len(results) == 10
