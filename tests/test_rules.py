"""src/rules.py: operators, three-valued logic, and pass / fail / missing for every İkamet rule.

Pattern for a new rule (add-rule skill): add one row to RULE_CASES with facts that pass and a
small change that makes it fail. The missing case is the empty fact map.
"""
from datetime import date

import pytest

from src.models import Condition
from src.pack import coerce_fact, load_pack
from src.rules import all_of, any_of, evaluate_condition, evaluate_rules, required_rules

PACK = load_pack("ikamet_first")


def verdict(rule_id: str, raw_facts: dict) -> str:
    facts = {name: coerce_fact(PACK, name, value) for name, value in raw_facts.items()}
    return next(r for r in evaluate_rules(PACK, facts) if r.rule_id == rule_id).status


def cond(data: dict) -> Condition:
    return Condition.model_validate(data)


# ----------------------------------------------------------- three-valued --
@pytest.mark.parametrize("values,expected", [
    ([True, True], True), ([True, None], None), ([False, None], False), ([None, None], None), ([True, False], False)])
def test_all_of_truth_table(values, expected):
    assert all_of(values) is expected


@pytest.mark.parametrize("values,expected", [
    ([False, False], False), ([False, None], None), ([True, None], True), ([None, None], None), ([True, False], True)])
def test_any_of_truth_table(values, expected):
    assert any_of(values) is expected


def test_not_keeps_unknown_unknown():
    condition = cond({"op": "not", "arg": {"op": "is_true", "field": "x"}})
    assert evaluate_condition(condition, {}, set()) is None
    assert evaluate_condition(condition, {"x": True}, set()) is False


def test_missing_fields_are_collected_even_when_another_branch_decides():
    condition = cond({"op": "all", "args": [{"op": "is_true", "field": "a"}, {"op": "is_true", "field": "b"}]})
    missing: set[str] = set()
    assert evaluate_condition(condition, {"a": False}, missing) is False
    assert missing == {"b"}


@pytest.mark.parametrize("op,left,right,expected", [
    ("eq", 3, 3, True), ("eq", 3, 4, False), ("lt", 3, 4, True), ("lte", 4, 4, True),
    ("gt", 5, 4, True), ("gte", 3, 4, False), ("in", "b", ["a", "b"], True), ("in", "c", ["a", "b"], False)])
def test_comparisons_with_constant(op, left, right, expected):
    condition = cond({"op": op, "field": "x", "value": right})
    assert evaluate_condition(condition, {"x": left}, set()) is expected


def test_other_field_with_offset_days_on_dates():
    condition = cond({"op": "gte", "field": "a", "other_field": "b", "offset_days": 60})
    assert evaluate_condition(condition, {"a": date(2027, 7, 31), "b": date(2027, 6, 1)}, set()) is True
    assert evaluate_condition(condition, {"a": date(2027, 7, 30), "b": date(2027, 6, 1)}, set()) is False
    missing: set[str] = set()
    assert evaluate_condition(condition, {"a": date(2027, 7, 30)}, missing) is None
    assert missing == {"b"}


def test_date_constant_given_as_iso_text():
    condition = cond({"op": "lt", "field": "d", "value": "2027-01-01"})
    assert evaluate_condition(condition, {"d": date(2026, 12, 31)}, set()) is True


def test_is_true_does_not_accept_truthy_non_booleans():
    condition = cond({"op": "is_true", "field": "x"})
    assert evaluate_condition(condition, {"x": 1}, set()) is False


@pytest.mark.parametrize("bad", [
    {"op": "all", "args": []}, {"op": "not"}, {"op": "eq", "field": "x"},
    {"op": "eq", "field": "x", "value": 1, "other_field": "y"}, {"op": "in", "field": "x", "value": "a"},
    {"op": "is_true"}, {"op": "between", "field": "x", "value": 1}])
def test_malformed_conditions_are_rejected(bad):
    with pytest.raises(ValueError):
        cond(bad)


# -------------------------------------------------- every rule, 3 cases --
RULE_CASES = [
    ("R001", {"exchange_student": True, "planned_stay_months": 3}, {"planned_stay_months": 2}),
    ("R002", {"permit_type": "short_term"}, {"permit_type": "student"}),
    ("R003", {"is_in_turkiye": True, "entered_with_passport": True}, {"entered_with_passport": False}),
    ("R004", {"application_date": "2026-11-10", "legal_stay_expiry_date": "2026-11-20"}, {"application_date": "2026-11-21"}),
    ("R005", {"application_channel": "https://e-ikamet.goc.gov.tr/"}, {"application_channel": "unofficial"}),
    ("R006", {"passport_expiry_date": "2027-07-31", "requested_permit_end_date": "2027-06-01"}, {"passport_expiry_date": "2027-07-30"}),
    ("R007", {"application_form_generated": True, "application_form_signed": True}, {"application_form_signed": False}),
    ("R008", {"passport_id_page_present": True, "turkey_entry_page_present": True, "visa_required": False}, {"turkey_entry_page_present": False}),
    ("R009", {"photo_age_months": 6, "biometric_photo": True, "icao_compliant": True}, {"photo_age_months": 7}),
    ("R010", {"fee_payment_required": True, "fee_receipt_present": True}, {"fee_receipt_present": False}),
    ("R011", {"single_entry_visa_fee_required": False}, {"single_entry_visa_fee_required": True, "single_entry_visa_fee_receipt_present": False}),
    ("R012", {"financial_capacity_declared": True}, {"financial_capacity_declared": False}),
    ("R013", {"exchange_program_evidence_present": True}, {"exchange_program_evidence_present": False}),
    ("R014", {"student_certificate_present": True}, {"student_certificate_present": False}),
    ("R015", {"insurance_exempt": False, "insurance_valid_for_ikamet": True, "insurance_start_date": "2026-09-01",
              "insurance_end_date": "2027-06-30", "requested_start_date": "2026-09-15", "requested_end_date": "2027-06-15"},
     {"insurance_end_date": "2027-05-31"}),
    ("R016", {"insurance_evidence_type": "sgk_provision"}, {"insurance_evidence_type": "travel_only"}),
    ("R017", {"address_proof_type": "rental_agreement"}, {"address_proof_type": "airbnb"}),
    ("R018", {"turkey_address_present": True, "phone_present": True, "email_present": True}, {"email_present": False}),
    ("R019", {"declaration_control_document_present": True, "declaration_control_document_signed": True}, {"declaration_control_document_signed": False}),
    ("R020", {"all_core_documents_present": True}, {"all_core_documents_present": False}),
    ("R021", {"scans_sent_to_iro": True, "iro_preverification_confirmed": True}, {"scans_sent_to_iro": False}),
    ("R022", {"online_application_completed": True, "fingerprint_step_completed": True, "gi_number_present": True}, {"fingerprint_step_completed": False}),
    ("R023", {"online_application_completed": True, "iro_preverification_confirmed": True, "gi_number_present": True,
              "all_core_documents_present": True}, {"all_core_documents_present": False}),
    ("R024", {"current_date": "2026-11-11", "iro_window_start": "2026-11-11", "iro_window_end": "2026-11-12"}, {"current_date": "2026-11-10"}),
    ("R025", {"application_form_window_id": "nov", "current_iro_window_id": "nov"}, {"current_iro_window_id": "dec"}),
    ("R026", {"missing_required_documents_count": 0}, {"missing_required_documents_count": 2}),
    ("R027", {"approval_status": "pending"}, {"approval_status": "guaranteed"}),
    ("R028", {"decision_authority": "Migration Management"}, {"decision_authority": "Sabancı IRO"}),
    ("R029", {"submitted_to_iro": True, "tracking_channel": "e_ikamet"}, {"tracking_channel": "email"}),
    ("R030", {"address_proof_type": "title_deed", "utility_bill_present": True},
     {"address_proof_type": "title_deed", "utility_bill_present": False}),
]


def test_every_rule_has_a_case():
    assert [row[0] for row in RULE_CASES] == PACK.rule_ids()


@pytest.mark.parametrize("rule_id,passing,change", RULE_CASES, ids=[row[0] for row in RULE_CASES])
def test_rule_passes(rule_id, passing, change):
    assert verdict(rule_id, passing) == "pass"


@pytest.mark.parametrize("rule_id,passing,change", RULE_CASES, ids=[row[0] for row in RULE_CASES])
def test_rule_fails(rule_id, passing, change):
    assert verdict(rule_id, {**passing, **change}) == "fail"


@pytest.mark.parametrize("rule_id,passing,change", RULE_CASES, ids=[row[0] for row in RULE_CASES])
def test_rule_unknown_when_information_is_missing(rule_id, passing, change):
    result = next(r for r in evaluate_rules(PACK, {}) if r.rule_id == rule_id)
    assert result.status == "unknown" and result.missing_fields


# ------------------------------------------------------- results shape --
def test_results_carry_sources_section_and_verification_status():
    result = evaluate_rules(PACK, {}, section="passport_photo")[0]
    assert result.section == "passport_photo"
    assert result.source_ids and result.rule_status == "verified"


def test_blocked_by_lists_prerequisites_that_do_not_pass():
    r020 = next(r for r in evaluate_rules(PACK, {"all_core_documents_present": True}) if r.rule_id == "R020")
    assert r020.status == "pass"
    assert "R007" in r020.blocked_by


def test_required_rules_include_transitive_prerequisites():
    needed = required_rules(PACK, ["R021"])
    assert {"R021", "R020", "R007", "R005"} <= set(needed)


def test_evaluation_does_not_change_the_fact_map():
    facts = {"is_in_turkiye": True}
    evaluate_rules(PACK, facts)
    assert facts == {"is_in_turkiye": True}
