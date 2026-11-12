"""Explicit R001-R029 checks with three-valued, short-circuit-safe logic."""
from src.core.state import AgentState, ResidencePermitContext, Requirement
from src.workers.rule_loader import load_rules


def all_of(*values):
    if False in values: return False
    return None if None in values else True


def any_of(*values):
    if True in values: return True
    return None if None in values else False


def evaluate_requirements(state: AgentState) -> list[Requirement]:
    c = ResidencePermitContext.model_validate(state.get("residence_permit_context", {}))
    missing = set()
    def value(name):
        result = getattr(c, name)
        if result is None:
            missing.add(name)
        return result
    def eq(name, expected):
        v = value(name)
        return None if v is None else v == expected
    def cmp(left, right, op):
        a, b = value(left), value(right)
        return None if a is None or b is None else op(a, b)
    def within(name, allowed):
        v = value(name)
        return None if v is None else v in allowed
    def number(name, op):
        v = value(name)
        return None if v is None else op(v)
    def yes(*names): return all_of(*(eq(n, True) for n in names))
    checks = {
        "R001": lambda: all_of(eq("exchange_student", True), number("planned_stay_months", lambda v: v >= 3)),
        "R002": lambda: eq("permit_type", "short_term"),
        "R003": lambda: yes("is_in_turkiye", "entered_with_passport"),
        "R004": lambda: cmp("application_date", "legal_stay_expiry_date", lambda a,b: a <= b),
        "R005": lambda: eq("application_channel", "https://e-ikamet.goc.gov.tr/"),
        "R006": lambda: cmp("passport_expiry_date", "requested_permit_end_date", lambda a,b: (a-b).days >= 60),
        "R007": lambda: yes("application_form_generated", "application_form_signed"),
        "R008": lambda: all_of(yes("passport_id_page_present", "turkey_entry_page_present"), any_of(eq("visa_required", False), eq("visa_page_present", True))),
        "R009": lambda: all_of(number("photo_age_months", lambda v: 0 <= v <= 6), yes("biometric_photo", "icao_compliant")),
        "R010": lambda: any_of(eq("fee_payment_required", False), eq("fee_receipt_present", True)),
        "R011": lambda: any_of(eq("single_entry_visa_fee_required", False), eq("single_entry_visa_fee_receipt_present", True)),
        "R012": lambda: eq("financial_capacity_declared", True),
        "R013": lambda: eq("exchange_program_evidence_present", True),
        "R014": lambda: eq("student_certificate_present", True),
        "R015": lambda: any_of(eq("insurance_exempt", True), all_of(eq("insurance_valid_for_ikamet", True), cmp("insurance_start_date", "requested_start_date", lambda a,b: a <= b), cmp("insurance_end_date", "requested_end_date", lambda a,b: a >= b))),
        "R016": lambda: within("insurance_evidence_type", ["bilateral_social_security", "sgk_provision", "sgk_application", "private_health_insurance"]),
        "R017": lambda: within("address_proof_type", ["dorm_certificate", "rental_agreement", "notarized_commitment"]),
        "R018": lambda: yes("turkey_address_present", "phone_present", "email_present"),
        "R019": lambda: yes("declaration_control_document_present", "declaration_control_document_signed"),
        "R020": lambda: eq("all_core_documents_present", True),
        "R021": lambda: yes("scans_sent_to_iro", "iro_preverification_confirmed"),
        "R022": lambda: yes("online_application_completed", "fingerprint_step_completed", "gi_number_present"),
        "R023": lambda: yes("online_application_completed", "iro_preverification_confirmed", "gi_number_present", "all_core_documents_present"),
        "R024": lambda: all_of(cmp("current_date", "iro_window_start", lambda a,b: a >= b), cmp("current_date", "iro_window_end", lambda a,b: a <= b)),
        "R025": lambda: cmp("application_form_window_id", "current_iro_window_id", lambda a,b: a == b),
        "R026": lambda: number("missing_required_documents_count", lambda v: v == 0),
        "R027": lambda: None if value("approval_status") is None else value("approval_status") != "guaranteed",
        "R028": lambda: eq("decision_authority", "Migration Management"),
        "R029": lambda: any_of(eq("submitted_to_iro", False), eq("tracking_channel", "e_ikamet")),
    }
    results = []
    for rule in load_rules():
        missing.clear()
        passed = checks[rule["rule_id"]]()
        results.append(Requirement(rule_id=rule["rule_id"], description=rule["description"],
            passed=passed, status="missing_information" if passed is None else "passed" if passed else "failed",
            message="Missing information: " + ", ".join(sorted(missing)) if passed is None else rule["description"],
            missing_fields=sorted(missing) if passed is None else [],
            error_code=rule["rule_id"] + "_FAILED" if passed is False else None,
            source_ids=rule["source_ids"], severity=rule["severity"]))
    return results


def rules_engine(state: AgentState) -> dict:
    results = evaluate_requirements(state)
    # Assessment must remain accurate even when planning fails or approval is
    # never reached. This records blockers, not completed or approved actions.
    return {"relevant_requirements": results,
            "blockers": [r.rule_id for r in results
                         if r.severity == "blocking" and r.passed is not True]}
