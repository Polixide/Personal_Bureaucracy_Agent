"""The second Procedure Pack (ikamet_extension), added with zero changes in src/.

Per rule: one pass, one fail, one missing case. Plus: scenarios, procedure identification, the generic
document path (a residence permit card has no typed schema), and unverified-rule labelling.
"""
from datetime import date

import pytest

from eval.policy import RuleFollowingPolicy
from fakes import ScriptedBackend
from src.agent.answer import Claim, FinalAnswer
from src.agent.gate1 import GateContext, verify
from src.agent.orchestrator import run_agent
from src.pack import coerce_fact, load_all_packs, load_pack, load_scenarios, validate_pack_dir
from src.rules import evaluate_rules
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool
from src.twin import service

PACK = load_pack("ikamet_extension")
PACKS = load_all_packs()
TODAY = date(2027, 1, 10)


def verdicts(raw: dict) -> dict[str, str]:
    facts = {name: coerce_fact(PACK, name, value) for name, value in raw.items()}
    return {r.rule_id: r.status for r in evaluate_rules(PACK, facts)}


RULE_CASES = [
    ("EXT001", {"application_date": "2027-01-10", "current_permit_expiry_date": "2027-02-15"}, {"application_date": "2027-02-20"}),
    ("EXT002", {"residence_permit_card_present": True}, {"residence_permit_card_present": False}),
    ("EXT003", {"exchange_student": True}, {"exchange_student": False}),
    ("EXT004", {"passport_expiry_date": "2027-09-29", "requested_permit_end_date": "2027-07-31"}, {"passport_expiry_date": "2027-09-28"}),
    ("EXT005", {"insurance_end_date": "2027-07-31", "requested_permit_end_date": "2027-07-31"}, {"insurance_end_date": "2027-07-30"}),
    ("EXT006", {"application_channel": "https://e-ikamet.goc.gov.tr/"}, {"application_channel": "embassy"}),
    ("EXT007", {"application_form_signed": True}, {"application_form_signed": False}),
    ("EXT008", {"approval_status": "pending"}, {"approval_status": "guaranteed"}),
]


def test_pack_is_valid_and_every_rule_is_unverified():
    assert validate_pack_dir("ikamet_extension") == []
    assert PACK.procedure.status == "unverified" and all(r.status == "unverified" for r in PACK.rules)
    assert [row[0] for row in RULE_CASES] == PACK.rule_ids()


@pytest.mark.parametrize("rule_id,passing,change", RULE_CASES, ids=[r[0] for r in RULE_CASES])
def test_rule_pass_fail_missing(rule_id, passing, change):
    assert verdicts(passing)[rule_id] == "pass"
    assert verdicts({**passing, **change})[rule_id] == "fail"
    assert verdicts({})[rule_id] == "unknown"


SCENARIOS = load_scenarios(PACK)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["scenario_id"] for s in SCENARIOS])
def test_scenarios_match_their_expected_verdicts(scenario):
    assert verdicts(scenario["facts"]) == scenario["expected"]["rule_status"]


@pytest.mark.parametrize("goal,expected", [("I need to extend my İkamet", "ikamet_extension"),
                                           ("How can I renew my residence permit?", "ikamet_extension"),
                                           ("I want to apply for İkamet", "ikamet_first")])
def test_the_goal_selects_the_right_procedure(store, goal, expected):
    ctx = ToolContext(store=store, packs=PACKS, user_id="u", today=TODAY)
    output = run_tool(build_registry(), "search_procedures", {"query": goal}, ctx, "orchestrator").output
    assert output["selected"] == expected


def test_a_document_type_without_typed_schema_goes_through_the_generic_path(store, tmp_path):
    card = tmp_path / "card.txt"
    card.write_text("SYNTHETIC RESIDENCE PERMIT CARD\nName: Alex Example\nValid until: 15.02.2027\n", encoding="utf-8")
    service.register_upload(store, "u", "card", str(card), PACK.id)
    backend = ScriptedBackend([{"fields": [
        {"field": "holder_name", "value": "Alex Example", "page": 1, "quote": "Name: Alex Example"},
        {"field": "valid_until", "value": "15.02.2027", "page": 1, "quote": "Valid until: 15.02.2027"}]}])
    ctx = ToolContext(store=store, packs=PACKS, user_id="u", today=TODAY, procedure_id=PACK.id, backend=backend)
    assert run_tool(build_registry(), "extract_document", {"doc_id": "card", "doc_type": "residence_permit_card"}, ctx, "worker").ok
    view = ctx.view()
    assert view.values["current_permit_expiry_date"] == date(2027, 2, 15)
    assert view.derived["residence_permit_card_present"] is True


def test_answers_must_say_the_rules_are_unverified():
    facts = {"exchange_student": True}
    gate = GateContext(pack=PACK, results=evaluate_rules(PACK, facts), values=facts)
    plain = FinalAnswer(summary="You are still an exchange student.", claims=[Claim(rule_id="EXT003", status="pass")])
    assert "UNVERIFIED_NOT_LABELLED" in {i.code for i in verify(plain, gate)}
    labelled = plain.model_copy(update={"summary": "You are still an exchange student (unverified rule)."})
    assert verify(labelled, gate) == []


def test_the_full_agent_handles_the_new_procedure_offline(store, tmp_path):
    scenario = next(s for s in SCENARIOS if s["scenario_id"] == "X03-insurance-too-short")
    for name, value in scenario["facts"].items():
        service.record_app_value(store, PACK, "u", name, value)
    result = run_agent("I need to extend my İkamet", store, PACKS, "u", TODAY, RuleFollowingPolicy(), outbox_dir=tmp_path)
    assert result.status == "answered" and result.procedure_id == "ikamet_extension"
    assert {c.rule_id: c.status for c in result.answer.claims} == {"EXT005": "fail"}
    assert "unverified" in result.answer.summary
