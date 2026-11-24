"""Gate 1 checks one by one, and the fallback answer passing Gate 1 on every pack scenario."""
import json
from datetime import date

import pytest

from src.agent.answer import Claim, FinalAnswer, NextAction
from src.agent.fallback import deterministic_answer
from src.agent.gate1 import GateContext, build_gate_context, dates_in, verify, verify_text
from src.pack import load_all_packs, load_pack, load_scenarios
from src.rules import evaluate_rules
from src.tools.common import ToolContext

PACK = load_pack("ikamet_first")
FACTS = {"passport_expiry_date": date(2027, 7, 15), "requested_permit_end_date": date(2027, 6, 1),
         "current_date": date(2026, 11, 18)}


def gate(facts=FACTS, refs=(), pack=PACK):
    return GateContext(pack=pack, results=evaluate_rules(pack, facts), values=dict(facts), refs=set(refs))


def answer(summary="Your passport validity is too short (R006).", claims=(("R006", "fail"),), actions=(("Renew the passport", "R006"),)):
    return FinalAnswer(summary=summary, claims=[Claim(rule_id=r, status=s) for r, s in claims],
                       next_actions=[NextAction(title=t, ref=ref) for t, ref in actions])


def codes(issues):
    return {issue.code for issue in issues}


def test_a_correct_answer_passes():
    assert verify(answer(), gate()) == []


def test_claim_must_equal_the_latest_evaluation():
    assert "CLAIM_MISMATCH" in codes(verify(answer(claims=[("R006", "pass")]), gate()))
    assert "CLAIM_UNKNOWN_RULE" in codes(verify(answer(claims=[("R006", "fail"), ("R999", "pass")]), gate()))


def test_failing_rule_must_be_addressed():
    assert "MISSING_FAILED_RULE" in codes(verify(answer(claims=[], actions=[("Check R001", "R001")]), gate()))


@pytest.mark.parametrize("text", ["Your application has been submitted.", "The fee was paid.",
                                  "I submitted the form for you.", "Your appointment is booked."])
def test_completion_claims_are_rejected(text):
    assert "COMPLETION_CLAIM" in codes(verify(answer(summary=text + " R006 fails."), gate()))


def test_what_the_user_reported_may_be_repeated():
    assert verify(answer(summary="You reported that your scans were sent to IRO. R006 fails."), gate()) == []


def test_approval_promises_are_rejected():
    assert "APPROVAL_CLAIM" in codes(verify(answer(summary="Your permit will be approved. R006 fails."), gate()))
    assert "APPROVAL_CLAIM" in codes(verify(answer(summary="Approval guaranteed for your permit."), gate()))


def test_unknown_is_not_failure():
    text = "R006 fails. R009 failed because the photo is missing."
    assert "UNKNOWN_AS_FAILURE" in codes(verify(answer(summary=text), gate()))


def test_dates_must_come_from_facts_or_rules():
    assert verify(answer(summary="Your passport expires on 2027-07-15; R006 fails."), gate()) == []
    rule_date = "It must be valid until at least 31.07.2027 (60 days after 2027-06-01), so R006 fails."
    assert verify(answer(summary=rule_date), gate()) == []
    invented = codes(verify(answer(summary="Apply before 2026-12-01. R006 fails."), gate()))
    assert "UNSUPPORTED_DATE" in invented


def test_date_parsing_formats():
    assert dates_in("on 2027-07-15 and 31.07.2027 and 01/02/2027, not 2027-02-30") == [
        date(2027, 7, 15), date(2027, 7, 31), date(2027, 2, 1)]


def test_actions_must_reference_rules_findings_or_proposals():
    assert "UNGROUNDED_ACTION" in codes(verify(answer(actions=[("Renew", "R006"), ("Do something", "X-1")]), gate()))
    assert verify(answer(actions=[("Renew", "R006"), ("See finding", "F-passport_photo")]),
                  gate(refs={"F-passport_photo"})) == []


def test_invented_rule_ids_in_text_are_rejected():
    assert "INVENTED_RULE_ID" in codes(verify(answer(summary="R006 fails and R099 needs a stamp."), gate()))


def test_unverified_rules_must_be_labelled(tmp_path):
    rules = json.loads((PACK.root / "rules.json").read_text(encoding="utf-8"))
    for rule in rules:
        if rule["rule_id"] == "R006":
            rule["status"] = "unverified"
    pack = PACK.model_copy(update={"rules": [type(PACK.rules[0]).model_validate(r) for r in rules]})
    assert "UNVERIFIED_NOT_LABELLED" in codes(verify(answer(), gate(pack=pack)))
    labelled = answer(summary="R006 (unverified rule) fails: your passport validity is too short.")
    assert verify(labelled, gate(pack=pack)) == []


def test_without_procedure_claims_are_rejected():
    no_pack = GateContext(pack=None)
    assert verify(FinalAnswer(summary="No verified procedure is available."), no_pack) == []
    assert "CLAIM_UNKNOWN_RULE" in codes(verify(answer(), no_pack))


def test_text_checks_for_drafts():
    assert "COMPLETION_CLAIM" in codes(verify_text("I have submitted my documents.", gate()))
    assert verify_text("Could you check my documents before I submit them?", gate()) == []


# --------------------------------------------- fallback passes Gate 1 --
SCENARIOS = load_scenarios(PACK)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["scenario_id"] for s in SCENARIOS])
def test_fallback_answer_passes_gate_1_and_matches_expected_verdicts(store, scenario):
    from src.twin import service
    for name, value in scenario["facts"].items():
        service.record_app_value(store, PACK, "u", name, value)
    ctx = ToolContext(store=store, packs=load_all_packs(), user_id="u", today=date.fromisoformat(scenario["today"]),
                      procedure_id="ikamet_first")
    fallback, findings = deterministic_answer(ctx)
    ctx.findings.update({f.finding_id: f for f in findings})
    assert verify(fallback, build_gate_context(ctx)) == []
    expected = {rid: status for rid, status in scenario["expected"]["rule_status"].items() if status != "pass"}
    assert {c.rule_id: c.status for c in fallback.claims} == expected
    assert len(fallback.next_actions) <= 5
