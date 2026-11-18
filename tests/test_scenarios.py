"""Synthetic scenarios (data/synthetic_samples/scenarios.json) through the harness."""
import copy

import pytest

from src.evaluation.harness import evaluate, format_report, load_scenarios, run_scenario

SCENARIOS = load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["scenario_id"] for s in SCENARIOS])
def test_scenario(scenario, tmp_path):
    result = run_scenario(scenario, tmp_path / "twin.sqlite")
    failed = [f"session {c.session} {c.name}: expected {c.expected!r}, got {c.actual!r}"
              for c in result.checks if not c.ok]
    assert result.error is None, result.error
    assert not failed, "\n".join(failed)


def test_scenario_ids_and_users_are_unique():
    assert len({s["scenario_id"] for s in SCENARIOS}) == len(SCENARIOS)
    assert len({s["user_id"] for s in SCENARIOS}) == len(SCENARIOS)


def test_harness_detects_a_wrong_expectation(tmp_path):
    """The harness must be able to fail: a scenario with a wrong expectation is reported."""
    scenario = copy.deepcopy(SCENARIOS[0])
    scenario["sessions"][0]["expect"]["blockers"] = ["SOMETHING_ELSE"]
    result = run_scenario(scenario, tmp_path / "twin.sqlite")
    assert not result.ok
    assert [c.name for c in result.checks if not c.ok] == ["blockers"]


def test_harness_reports_crashes_instead_of_raising(tmp_path):
    scenario = copy.deepcopy(SCENARIOS[0])
    scenario["sessions"][0]["event"] = {"kind": "fax"}               # invalid UserEvent
    result = run_scenario(scenario, tmp_path / "twin.sqlite")
    assert not result.ok and "ValidationError" in result.error


def test_evaluation_report(tmp_path):
    report = evaluate(SCENARIOS, workdir=tmp_path)
    assert report["scenarios_passed"] == report["scenarios_total"] == len(SCENARIOS)
    assert report["failures_caught"] == report["failures_expected"] > 0
    assert report["false_alarms"] == 0
    assert "Scenarios passed" in format_report(report)
