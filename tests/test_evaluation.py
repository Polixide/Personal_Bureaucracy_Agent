"""Curated M1 scenarios and multi-session restarts, run through the evaluation harness.

Run: python -m pytest tests/test_evaluation.py -v      (or: python -m scripts.run_eval)
Scenario definitions: src/evaluation/scenarios.py. Every run uses the real graph,
rules engine, Gate 1, repair loop, approval interrupt, SQLite checkpoint and twin
record; only model transport is mocked. The application is restarted between the
approval pause and the decision, and the twin record is compared with the
checkpoint after reopening.

APPROVED means the proposed next-action plan was explicitly approved, never that
the residence permit or an official action was approved/executed.
"""
import copy

import pytest

from src.evaluation.harness import (MULTI_SESSION, evaluate, format_report, run_curated,
                                    run_multi_session)
from src.evaluation.scenarios import SCENARIOS


def failures(result):
    lines = [f"{c.name}: expected {c.expected!r}, got {c.actual!r}"
             for c in result.checks if not c.ok]
    return "\n".join([result.error] if result.error else lines)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario["id"])
def test_curated_end_to_end_scenario(tmp_path, scenario):
    result = run_curated(scenario, tmp_path)
    assert result.checks and result.ok, failures(result)


@pytest.mark.parametrize("scenario", MULTI_SESSION, ids=lambda scenario: scenario.__name__)
def test_multi_session_scenario(tmp_path, scenario):
    result = run_multi_session(scenario, tmp_path)
    assert result.checks and result.ok, failures(result)


def test_harness_detects_a_wrong_expectation(tmp_path):
    """The harness must be able to fail: a wrong expected blocker list is reported."""
    scenario = copy.deepcopy(SCENARIOS[1])
    scenario["blockers"] = ["R999"]
    result = run_curated(scenario, tmp_path)
    assert not result.ok
    assert [c.name for c in result.checks if not c.ok] == ["blockers"]


def test_harness_reports_crashes_instead_of_raising(tmp_path):
    scenario = copy.deepcopy(SCENARIOS[0])
    scenario["changes"] = {"not_a_context_field": True}          # ResidencePermitContext forbids it
    result = run_curated(scenario, tmp_path)
    assert not result.ok and "ValidationError" in result.error


def test_evaluation_report():
    report = evaluate()
    assert report["scenarios_passed"] == report["scenarios_total"] == len(SCENARIOS) + len(MULTI_SESSION)
    assert report["rules_correct"] == report["rules_total"] == 29 * len(SCENARIOS)
    assert report["invalid_plans_caught"] == report["invalid_plans_expected"] > 0
    assert report["false_alarms"] == 0
    assert report["lessons_written"] > 0
    assert "Rule statuses correct" in format_report(report)
