"""Deterministic checks do not need Ollama."""

import pytest

from src.plan import ActionPlan, PlanAction
from src.plan_validator import validate_plan
from src.requirement_engine import RuleResult
from src.state import State


def result(status="failed", rule_id="R003"):
    return RuleResult(rule_id=rule_id, passed=False if status == "failed" else None,
                      status=status, error_code=None, message="Synthetic result")


def action(**changes):
    return PlanAction(**dict({"action": "Review passport renewal options", "priority": 1,
                              "reason": "Resolve the passport validity requirement.",
                              "related_rule_id": "R003"}, **changes))


def check(actions, results=None, summary="Proposed next steps.", state=None):
    return validate_plan(state or State(user_goal="Apply"),
                         [result()] if results is None else results,
                         ActionPlan(actions=actions, summary=summary))


def codes(validation):
    return {error.code for error in validation.errors}


def test_valid_plan():
    assert check([action()]).valid


def test_unknown_rule():
    validation = check([action(related_rule_id="R999")])
    assert "UNKNOWN_RULE_ID" in codes(validation)
    assert validation.errors[0].action_index == 0


def test_ignored_failure():
    assert "FAILED_RULE_IGNORED" in codes(check([]))


def test_unknown_is_not_a_failure():
    unknown = [result("missing_information")]
    assert check([action(action="Provide passport expiry date", reason="Information is unknown.")], unknown).valid
    assert check([], unknown).valid
    validation = check([action(reason="R003 failed; passport is invalid.")], unknown)
    assert "UNKNOWN_AS_FAILURE" in codes(validation)


def test_unknown_failure_claim_in_summary():
    validation = check([], [result("missing_information")], summary="R003 has failed")
    assert "UNKNOWN_AS_FAILURE" in codes(validation)


@pytest.mark.parametrize("text", ["Submit the application", "Finalize the application",
                                  "Submit updated document set to IRO"])
def test_premature_submission(text):
    validation = check([action(), action(action=text, priority=2)])
    assert "PREMATURE_SUBMISSION" in codes(validation)


def test_submission_without_failed_requirements():
    assert check([action(action="Submit the application", related_rule_id=None)], []).valid


@pytest.mark.parametrize("text", ["I submitted the application", "Appointment has been booked",
                                  "The fee is paid", "Successfully renewed the passport"])
def test_completion_claim(text):
    assert "COMPLETION_CLAIM" in codes(check([action(action=text)]))


def test_summary_completion_claim():
    assert "COMPLETION_CLAIM" in codes(check([action()], summary="We submitted the application."))


def test_duplicate_and_unordered_priorities():
    for priorities in ([1, 1], [2, 1]):
        assert "AMBIGUOUS_ORDER" in codes(check([action(priority=p) for p in priorities]))


def test_mutated_invalid_priority():
    proposed = action()
    proposed.priority = 0
    assert "INVALID_PRIORITY" in codes(check([proposed]))


def test_deadline_must_be_grounded():
    proposed = action(deadline="2026-11-15")
    assert "UNSUPPORTED_DEADLINE" in codes(check([proposed]))
    assert check([proposed], state=State(user_goal="Apply", deadlines={"submission": "2026-11-15"})).valid
