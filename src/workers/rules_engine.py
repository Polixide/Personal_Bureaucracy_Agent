"""Active V2 checks with three-valued, short-circuit-safe logic."""
from src.core.address_proof import RESIDENCE_DOCUMENTS, address_proof_status
from src.core.state import AgentState, ResidencePermitContext, Requirement
from src.workers.rule_loader import load_rules

ACCEPTED_ADDRESS_PROOF_TYPES = RESIDENCE_DOCUMENTS


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
    def scoped_insurance():
        if c.insurance_exempt is True:return True
        targets=[getattr(c,name) for name in ("requested_permit_end_date","program_end_date") if getattr(c,name) is not None]
        if c.requested_permit_end_date is None and c.requested_end_date is not None and "requested_permit_end_date" not in state.get("verification_conflicts",[]):targets.append(c.requested_end_date)  # Legacy requested-period alias.
        if not targets:
            missing.update(("requested_permit_end_date","program_end_date"));return None
        end=value("insurance_end_date")
        return None if end is None else any(end>=target for target in targets)
    def insurance_category():
        category=value("insurance_evidence_type")
        if category=="private_health_insurance":return c.insurance_valid_for_ikamet is not False
        if category in {None,"bilateral_social_security","sgk_provision","sgk_application","other_official"}:
            missing.add("insurance_evidence_type");return None
        return False
    address = address_proof_status(state)

    def address_check():
        if address["ready"]:
            return True
        if address["route"] is None:
            missing.add("address_proof_type")
        elif not address["supported"]:
            return False
        return None

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
        "R013": lambda: eq("exchange_program_evidence_present", True),
        "R014": lambda: eq("student_certificate_present", True),
        "R015": scoped_insurance,
        "R016": insurance_category,
        "R017": address_check,
        "R018": lambda: yes("turkey_address_present", "phone_present", "email_present"),
        "R019": lambda: yes("declaration_control_document_present", "declaration_control_document_signed"),
        "R020": lambda: eq("all_core_documents_present", True),
        "R021": lambda: yes("scans_sent_to_iro", "iro_preverification_confirmed"),
        "R022": lambda: yes("online_application_completed", "fingerprint_step_completed", "gi_number_present"),
        "R023": lambda: yes("online_application_completed", "iro_preverification_confirmed", "gi_number_present", "all_core_documents_present"),
        "R026": lambda: number("missing_required_documents_count", lambda v: v == 0),
        "R028": lambda: eq("decision_authority", "Migration Management"),
    }
    results = []
    for rule in load_rules(state.get("knowledge_snapshot")):
        missing.clear()
        passed = checks[rule["rule_id"]]()
        address_message = address["message"]
        if rule["rule_id"] == "R017" and passed is None and address["parts"]:
            outstanding = ", ".join(part["label"] for part in address["parts"] if part["status"] != "complete")
            address_message = f"Address evidence still needed: {outstanding}. {address_message}"
        results.append(Requirement(rule_id=rule["rule_id"], description=rule["description"],
            passed=passed, status="missing_information" if passed is None else "passed" if passed else "failed",
            message=("Manual review: unsupported insurance alternative" if rule["rule_id"]=="R016" and c.insurance_evidence_type is not None and passed is None else address_message if rule["rule_id"]=="R017" and passed is not True else "Missing information: " + ", ".join(sorted(missing)) if passed is None else rule["description"]),
            missing_fields=sorted(missing) if passed is None else [],
            error_code=rule["rule_id"] + "_FAILED" if passed is False else None,
            source_ids=rule["source_ids"], severity=rule["severity"],
            assessment_policy="Milestone-1 exchange end coverage: requested permit end OR accepted program end; start date excluded" if rule["rule_id"]=="R015" else None))
    return results


def rules_engine(state: AgentState) -> dict:
    updates = {}
    event = state.get("event")
    if event and event.kind in {"analyze_document", "confirm_information", "generate_plan", "resolve_conflict", "record_progress", "reevaluate", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction"}:
        from src.workers.verification import verification_updates
        from src.core.state import ValidationIssue
        try:
            updates = verification_updates(state)
        except ValueError as error:
            return {"relevant_requirements": [], "run_status": "INVALID_INPUT", "validation_errors": [
                ValidationIssue(code="INVALID_INPUT", source="verification", message=str(error))]}
    results = evaluate_requirements({**state, **updates})
    # Assessment must remain accurate even when planning fails or approval is
    # never reached. This records blockers, not completed or approved actions.
    result = {**updates, "requirements_knowledge_version":(state.get("knowledge_snapshot") or {}).get("version",""), "relevant_requirements": results,
              "blockers": [r.rule_id for r in results
                           if r.severity == "blocking" and r.passed is not True]}
    # Procedure sequencing stays outside legal evaluation. Only factual/review
    # events refresh V2 stage-local data; chat, planning and approval do not.
    if state.get("procedure") is not None and event and event.kind in {
        "analyze_document","confirm_information","resolve_conflict","reevaluate",
        "delete_document","replace_document","cancel_document","reprocess_document","remove_correction",
    }:
        from src.core.procedure import procedure_evidence_update
        result.update(procedure_evidence_update({**state,**result}))
    return result
