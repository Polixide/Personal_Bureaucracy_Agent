"""HISTORICAL ONE-OFF (phase 1). It imports milestone-1 modules that were deleted in phase 9, so it only
runs on a checkout of commit 486917b or earlier. Its output is committed and must not be edited by hand.

One-off data conversion: knowledge/ (milestone 1) -> procedures/ikamet_first/ (Procedure Pack).

Run: python -m scripts.convert_knowledge
Nothing here is new knowledge. Every value comes from:
  knowledge/rules.json, dependencies.json, sources.json, scope.json  (rules, sources, scope)
  src/core/state.py ResidencePermitContext                           (field names and types, aliases)
  src/workers/verification.py, src/agents/planning_policy.py         (questions, document mappings)
  src/guardrails/gate_0.py, gate_1.py                                (scope test, stage prerequisites)
  tests/test_evaluation.py                                           (scenarios)
The condition trees below are the `deterministic_condition` strings written as JSON.
"""
import json
import sys
import types
import typing
from datetime import date

from config import KNOWLEDGE_DIR, PROCEDURES_DIR, PROJECT_ROOT
from src.agents.planning_policy import FIELD_REQUESTS
from src.core.state import ResidencePermitContext
from src.workers.verification import CONTEXT_QUESTIONS

PACK_ID = "ikamet_first"
OUT = PROCEDURES_DIR / PACK_ID


def leaf(op: str, field: str, **extra) -> dict:
    return {"op": op, "field": field, **extra}


def yes(field: str) -> dict:
    return leaf("is_true", field)


def all_of(*args: dict) -> dict:
    return {"op": "all", "args": list(args)}


def any_of(*args: dict) -> dict:
    return {"op": "any", "args": list(args)}


CONDITIONS = {
    "R001": all_of(yes("exchange_student"), leaf("gte", "planned_stay_months", value=3)),
    "R002": leaf("eq", "permit_type", value="short_term"),
    "R003": all_of(yes("is_in_turkiye"), yes("entered_with_passport")),
    "R004": leaf("lte", "application_date", other_field="legal_stay_expiry_date"),
    "R005": leaf("eq", "application_channel", value="https://e-ikamet.goc.gov.tr/"),
    "R006": leaf("gte", "passport_expiry_date", other_field="requested_permit_end_date", offset_days=60),
    "R007": all_of(yes("application_form_generated"), yes("application_form_signed")),
    "R008": all_of(yes("passport_id_page_present"), yes("turkey_entry_page_present"),
                   any_of(leaf("eq", "visa_required", value=False), yes("visa_page_present"))),
    "R009": all_of(leaf("gte", "photo_age_months", value=0), leaf("lte", "photo_age_months", value=6),
                   yes("biometric_photo"), yes("icao_compliant")),
    "R010": any_of(leaf("eq", "fee_payment_required", value=False), yes("fee_receipt_present")),
    "R011": any_of(leaf("eq", "single_entry_visa_fee_required", value=False),
                   yes("single_entry_visa_fee_receipt_present")),
    "R012": yes("financial_capacity_declared"),
    "R013": yes("exchange_program_evidence_present"),
    "R014": yes("student_certificate_present"),
    "R015": any_of(yes("insurance_exempt"),
                   all_of(yes("insurance_valid_for_ikamet"),
                          leaf("lte", "insurance_start_date", other_field="requested_start_date"),
                          leaf("gte", "insurance_end_date", other_field="requested_end_date"))),
    "R016": leaf("in", "insurance_evidence_type",
                 value=["bilateral_social_security", "sgk_provision", "sgk_application", "private_health_insurance"]),
    "R017": leaf("in", "address_proof_type", value=["dorm_certificate", "rental_agreement", "notarized_commitment"]),
    "R018": all_of(yes("turkey_address_present"), yes("phone_present"), yes("email_present")),
    "R019": all_of(yes("declaration_control_document_present"), yes("declaration_control_document_signed")),
    "R020": yes("all_core_documents_present"),
    "R021": all_of(yes("scans_sent_to_iro"), yes("iro_preverification_confirmed")),
    "R022": all_of(yes("online_application_completed"), yes("fingerprint_step_completed"), yes("gi_number_present")),
    "R023": all_of(yes("online_application_completed"), yes("iro_preverification_confirmed"),
                   yes("gi_number_present"), yes("all_core_documents_present")),
    "R024": all_of(leaf("gte", "current_date", other_field="iro_window_start"),
                   leaf("lte", "current_date", other_field="iro_window_end")),
    "R025": leaf("eq", "application_form_window_id", other_field="current_iro_window_id"),
    "R026": leaf("eq", "missing_required_documents_count", value=0),
    "R027": {"op": "not", "arg": leaf("eq", "approval_status", value="guaranteed")},
    "R028": leaf("eq", "decision_authority", value="Migration Management"),
    "R029": any_of(leaf("eq", "submitted_to_iro", value=False), leaf("eq", "tracking_channel", value="e_ikamet")),
}

EXTRA_NOTES = {
    "R009": "Lower bound 0 reproduces the milestone-1 engine (a photo age cannot be negative).",
    "R015": "The milestone-1 app also used a code-only 'scoped' variant; it was dropped on 2026-11-18 "
            "(docs/migration_plan.md).",
    "R024": "current_date is the evaluation date (derived 'today'); the window is iro_window_start..iro_window_end.",
}

SECTIONS = [
    ("eligibility", "Eligibility and legal stay", ["R001", "R002", "R003", "R004"]),
    ("passport_photo", "Passport, entry pages and photo", ["R006", "R008", "R009"]),
    ("application_fees", "Online application, form and fees", ["R005", "R007", "R010", "R011", "R012"]),
    ("student_insurance", "Student status and health insurance", ["R013", "R014", "R015", "R016"]),
    ("address_declaration", "Address, contact and declaration", ["R017", "R018", "R019"]),
    ("iro_submission", "IRO pre-verification and submission", ["R020", "R021", "R022", "R023", "R024", "R025", "R026"]),
    ("guardrails", "Authority and after submission", ["R027", "R028", "R029"]),
]

# Stage prerequisites hard-coded in src/guardrails/gate_1.py (execution types).
STAGES = [
    {"id": "official_application", "title": "Online e-İkamet application", "requires": ["R001", "R002", "R003", "R004", "R005", "R006"]},
    {"id": "fingerprint", "title": "Fingerprint / GI step", "requires": ["R005", "R007"]},
    {"id": "iro_preverification", "title": "Send scans to IRO for pre-verification", "requires": ["R020"]},
    {"id": "iro_submission", "title": "Physical file submission at IRO", "requires": ["R023", "R024", "R025", "R026"]},
]

# Scope test hard-coded in src/guardrails/gate_0.py: a known FALSE value is out of scope.
SCOPE_CONDITIONS = [
    leaf("eq", "institution", value="Sabancı University"),
    leaf("eq", "applicant_type", value="incoming_exchange"),
    leaf("eq", "application_kind", value="first"),
    leaf("eq", "permit_type", value="short_term"),
    leaf("gte", "age", value=18),
    yes("exchange_student"),
    leaf("gte", "planned_stay_months", value=3),
]

# Aliases from ResidencePermitContext.normalize_scope_value.
ALIASES = {
    "institution": {"sabanci university": "Sabancı University", "sabancı university": "Sabancı University"},
    "applicant_type": {"incoming_exchange": "incoming_exchange", "incoming exchange": "incoming_exchange",
                       "incoming-exchange": "incoming_exchange"},
    "permit_type": {"short_term": "short_term", "short term": "short_term", "short-term": "short_term"},
    "application_kind": {"first": "first"},
}

# Document mappings from src/workers/verification.py MAPPINGS (document field -> context fields).
DOCUMENTS = [
    ("student_enrollment_certificate", "Student certificate (öğrenci belgesi) issued by the university", [
        ("holder_name", "holder_name"), ("student_id", "student_id"), ("institution", "institution"),
        ("student_certificate_issue_date", "issue_date"), ("exchange_student", "exchange_student"),
        ("program_start_date", "program_start_date"), ("program_end_date", "program_end_date")]),
    ("health_insurance_certificate", "Health insurance policy or certificate", [
        ("holder_name", "holder_name"), ("insurance_insurer", "insurer"), ("insurance_policy_number", "policy_number"),
        ("insurance_start_date", "coverage_start_date"), ("insurance_end_date", "coverage_end_date"),
        ("passport_or_foreign_id", "passport_or_foreign_id")]),
    ("ikamet_application_form", "IKAMET / online residence-permit application form", [
        ("holder_name", "applicant_name"), ("passport_or_foreign_id", "passport_or_foreign_id"),
        ("permit_type", "permit_type"), ("application_number", "application_number"),
        ("requested_start_date", "requested_start_date"), ("requested_end_date", "requested_end_date"),
        ("requested_permit_end_date", "requested_end_date")]),
    ("passport_identity_page", "Passport identity page", [
        ("holder_name", "holder_name"), ("passport_or_foreign_id", "passport_number"), ("nationality", "nationality"),
        ("passport_expiry_date", "expiry_date"), ("passport_id_page_present", "identity_page_present")]),
    ("turkiye_entry_stamp", "Latest Türkiye entry-stamp page", [
        ("entry_stamp_present", "entry_stamp_present"), ("turkey_entry_page_present", "entry_stamp_present"),
        ("latest_entry_date", "latest_entry_date")]),
    ("declaration_control_document", "Sabancı declaration control document", [
        ("holder_name", "holder_name"), ("declaration_control_document_present", "document_present"),
        ("declaration_control_document_signed", "signed")]),
    ("card_fee_receipt", "Residence-permit card / document fee payment receipt", [
        ("holder_name", "holder_name"), ("fee_receipt_present", "receipt_present"),
        ("receipt_payment_date", "payment_date"), ("receipt_reference_number", "reference_number"),
        ("receipt_amount", "amount")]),
]
NEEDS_CONFIRMATION = {("turkiye_entry_stamp", "latest_entry_date")}

DERIVED = [
    {"field": "current_date", "op": "today"},
    {"field": "planned_stay_months", "op": "months_between", "args": ["program_start_date", "program_end_date"]},
    {"field": "exchange_program_evidence_present", "op": "true_if_all",
     "args": ["exchange_student", "program_start_date", "program_end_date"]},
    {"field": "student_certificate_present", "op": "document_complete", "doc_type": "student_enrollment_certificate",
     "required": ["holder_name", "student_id", "institution", "issue_date"]},
    {"field": "application_form_generated", "op": "document_complete", "doc_type": "ikamet_application_form",
     "required": ["applicant_name", "passport_or_foreign_id", "requested_start_date", "requested_end_date"]},
]

CONTACTS = [
    {"role": "IRO", "name": "Sabancı University International Relations Office", "email": None},
    {"role": "Migration Management", "name": "Presidency of Migration Management (e-İkamet)", "email": None},
]

# Synthetic preparation sheet made by scripts/make_synthetic_form.py (not an official form).
FORMS = [{"form_id": "preparation_sheet", "title": "Personal preparation sheet (synthetic, not an official form)",
          "template": "forms/preparation_sheet.pdf",
          "fields": {"holder_name": "holder_name", "passport_number": "passport_or_foreign_id",
                     "nationality": "nationality", "passport_expiry": "passport_expiry_date",
                     "permit_start": "requested_start_date", "permit_end": "requested_permit_end_date",
                     "insurance_end": "insurance_end_date", "institution": "institution"}}]

PROCEDURE_ALIASES = ["ikamet", "ikamet izni", "residence permit", "short-term residence permit",
                     "short term residence permit", "first residence permit", "residence permit application",
                     "oturma izni"]


def field_type(annotation) -> str:
    """bool | None -> 'bool' (and so on) for ResidencePermitContext annotations."""
    names = {bool: "bool", int: "int", date: "date", str: "str"}
    for arg in typing.get_args(annotation):
        if arg in names:
            return names[arg]
    raise ValueError(f"unsupported annotation {annotation}")


def build_fields() -> list[dict]:
    questions = {**FIELD_REQUESTS, **CONTEXT_QUESTIONS}
    fields = []
    for name, info in ResidencePermitContext.model_fields.items():
        item = {"name": name, "type": field_type(info.annotation), "question": questions.get(name, "")}
        if name in ALIASES:
            item["aliases"] = ALIASES[name]
        fields.append(item)
    return fields


def build_documents() -> list[dict]:
    documents = []
    for doc_type, description, pairs in DOCUMENTS:
        extracts = []
        for fact, from_field in pairs:
            item = {"field": fact}
            if from_field != fact:
                item["from_field"] = from_field
            if (doc_type, from_field) in NEEDS_CONFIRMATION:
                item["requires_confirmation"] = True
            extracts.append(item)
        documents.append({"doc_type": doc_type, "description": description, "extracts": extracts})
    return documents


def build_procedure() -> dict:
    scope = json.loads((KNOWLEDGE_DIR / "scope.json").read_text(encoding="utf-8"))
    sections = [{"id": sid, "title": title, "order": index + 1, "delegate": sid != "guardrails"}
                for index, (sid, title, _) in enumerate(SECTIONS)]
    return {
        "id": PACK_ID,
        "title": "First short-term residence permit (İkamet) for Sabancı University incoming exchange students",
        "status": "verified", "version": scope["dataset_version"], "aliases": PROCEDURE_ALIASES,
        "scope": (f"{scope['institution']} {scope['user_type']}, {scope['age']}, staying {scope['stay']}, "
                  f"{scope['procedure']} in {scope['country']}."),
        "excluded": scope["excluded"], "scope_conditions": SCOPE_CONDITIONS,
        "sections": sections, "stages": STAGES, "fields": build_fields(), "documents": build_documents(),
        "derived": DERIVED, "contacts": CONTACTS, "forms": FORMS,
    }


def build_rules() -> list[dict]:
    data = json.loads((KNOWLEDGE_DIR / "rules.json").read_text(encoding="utf-8"))
    edges = json.loads((KNOWLEDGE_DIR / "dependencies.json").read_text(encoding="utf-8"))["edges"]
    section_of = {rule_id: sid for sid, _, ids in SECTIONS for rule_id in ids}
    rules = []
    for old in data["rules"]:
        rule_id = old["rule_id"]
        depends = set(old["dependencies"]) | {before for before, after in edges if after == rule_id}
        notes = " ".join(part for part in (old["notes"], EXTRA_NOTES.get(rule_id, "")) if part)
        rules.append({
            "rule_id": rule_id, "name": old["name"], "section": section_of[rule_id],
            "description": old["description"], "category": old["category"], "severity": old["severity"],
            "condition": CONDITIONS[rule_id], "depends_on": sorted(depends), "source_ids": old["source_ids"],
            "version": old["version"], "status": "verified", "volatility": old["volatility"],
            "notes": notes, "legacy_condition": old["deterministic_condition"],
        })
    return rules


def build_sources() -> list[dict]:
    data = json.loads((KNOWLEDGE_DIR / "sources.json").read_text(encoding="utf-8"))
    sources = []
    for old in data["sources"]:
        note = ""
        if old["url"].endswith("-"):
            note = "URL looks truncated in the original dataset; a person must check it."
        sources.append({"source_id": old["source_id"], "title": old["title"], "url": old["url"],
                        "retrieved_at": old["retrieved_at"], "authority": old["authority_level"],
                        "organization": old["organization"], "notes": note})
    return sources


def source_text(source: dict, rules: list[dict]) -> str:
    lines = [f"SOURCE: {source['title']} ({source['organization']})", f"URL: {source['url']}",
             f"RETRIEVED: {source['retrieved_at']}", "",
             "NOTE: This file is NOT a copy of the official page. It was compiled from knowledge/rules.json",
             "(the rule descriptions and notes that cite this source). Replace it with the saved page text.", "",
             "Requirements recorded from this source:"]
    for rule in rules:
        if source["source_id"] in rule["source_ids"]:
            text = f"- {rule['rule_id']} ({rule['name']}): {rule['description']}"
            lines.append(text + (f" {rule['notes']}" if rule["notes"] else ""))
    return "\n".join(lines) + "\n"


STATUS = {"passed": "pass", "failed": "fail", "missing_information": "unknown"}
KINDS = {"01-happy-path": "pass", "02-passport-buffer": "date_conflict", "03-airbnb-address": "fail",
         "04-insurance-coverage": "date_conflict", "05-missing-gi": "fail", "06-expired-legal-stay": "date_conflict",
         "07-unknown-information": "missing"}


def build_scenarios() -> list[dict]:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    import test_evaluation as source
    scenarios = []
    for old in source.SCENARIOS:
        if old["id"] not in KINDS:
            continue
        facts = {} if old.get("empty_context") else dict(source.READY_CONTEXT)
        facts.update(old["changes"])
        today = facts.pop("current_date", "2026-11-11")
        expected = {rule_id: ("unknown" if old.get("empty_context") else "pass") for rule_id in CONDITIONS}
        expected.update({rule_id: STATUS[status] for rule_id, status in old["statuses"].items()})
        scenarios.append({"scenario_id": old["id"], "kind": KINDS[old["id"]], "description": old["purpose"],
                          "goal": "I want to apply for İkamet", "today": today, "facts": facts,
                          "expected": {"procedure_id": PACK_ID, "rule_status": expected}})
    return scenarios


def write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rules = build_rules()
    sources = build_sources()
    write_json(OUT / "procedure.json", build_procedure())
    write_json(OUT / "rules.json", rules)
    write_json(OUT / "sources.json", sources)
    write_json(OUT / "eval" / "scenarios.json", build_scenarios())
    for source in sources:
        path = OUT / "sources_text" / f"{source['source_id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_text(source, rules), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
