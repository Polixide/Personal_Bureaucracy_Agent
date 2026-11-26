"""Deterministic observation acceptance, explicit confirmation and context projection.

Raw Document.info / extracted_facts are never changed here. Source observations
and user statements have different provenance; neither claims official approval.
"""
from copy import deepcopy
from datetime import date
from pathlib import Path
from unicodedata import normalize, category
import json
import re
from src.core.state import ResidencePermitContext, VerifiedInput, ActionPlan, ProcedureState
from src.core.address_proof import (
    ADDRESS_ATTESTATIONS, ADDRESS_REQUIRED_YES, DORM, HOST, HOST_SUPPORT_FIELDS, IRO_PAIR, RESIDENCE_DOCUMENTS,
    compare_host_evidence, host_support_fields, selected_address_method,
)
from src.document_processing.schemas import ADDRESS_EVIDENCE_SCHEMAS, DOCUMENT_SCHEMAS, DocumentInfo
from src.document_processing.evidence import (
    EVIDENCE_DATE_PATTERN, LATEST_ENTRY_LABEL, normalized, normalized_date, normalized_snippet, source_date,
)

IMPORTANT_FIELDS = {
    "student_certificate": ("holder_name", "student_id", "institution", "issue_date"),
    "health_insurance": ("holder_name", "insurer", "policy_number", "coverage_start_date", "coverage_end_date"),
    "ikamet_application_form": ("applicant_name", "passport_or_foreign_id", "requested_start_date", "requested_end_date"),
}
IMPORTANT_FIELDS.update({
    "utility_bill":("holder_name","service_address","billing_date","provider"),
    "address_form":("first_name","last_name","passport_number","declared_address","signature_present"),
    "passport_copy":("holder_name","passport_number","nationality","expiry_date","identity_page_present"),
    "turkiye_entry_stamp":("entry_stamp_present","latest_entry_date"),
    "declaration_control_document":("document_present","signed"),
    "card_fee_receipt":("receipt_present",),
    "tax_id_certificate":("tax_identification_number", "holder_name"),
})
IMPORTANT_FIELDS.update({kind: schema.REQUIRED for kind, schema in ADDRESS_EVIDENCE_SCHEMAS.items()})
MILESTONE_SCOPE = "milestone_1_document_preparation"
MILESTONE_DOCUMENTS = {
    "declaration_control_document":("Declaration control document",("R019",)),
    "ikamet_application_form":("IKAMET / online application form",("R002","R007")),
    "passport_copy":("Passport identity page",("R006",)),
    "turkiye_entry_stamp":("Latest Turkiye entry-stamp page",()),
    "student_certificate":("Student certificate",("R001","R013","R014")),
    "health_insurance":("Health insurance certificate",("R015","R016")),
    "card_fee_receipt":("Card/document-fee payment receipt",("R010",)),
}
PROFILE_TYPES={"utility_bill":"utility_bill","address_form":"address_form","passport_copy":"passport_identity_page","turkiye_entry_stamp":"turkiye_entry_stamp","declaration_control_document":"declaration_control_document","card_fee_receipt":"card_fee_receipt","tax_id_certificate":"tax_id_certificate"}
PROFILE_TYPES.update({kind: kind for kind in ADDRESS_EVIDENCE_SCHEMAS})
STUDENT_FIELDS = ("exchange_student", "program_start_date", "program_end_date")
CONTEXT_QUESTIONS = {
    "is_in_turkiye": "Are you currently in T\u00fcrkiye?",
    "entered_with_passport": "Did you enter T\u00fcrkiye using your passport?",
    "passport_id_page_present": "Have you provided a readable passport identity page?",
    "turkey_entry_page_present": "Have you provided a readable Turkish entry-stamp page?",
    "visa_required": "Was a visa required for your entry?",
    "visa_page_present": "Have you provided the applicable visa page?",
}

# Explicitly declared procedure inputs; no official fact is inferred by this form.
CONTEXT_FIELDS = {**{name:"bool" for name in CONTEXT_QUESTIONS},
    "passport_expiry_date":"date", "requested_permit_end_date":"date",
    "application_date":"date", "legal_stay_expiry_date":"date",
    "insurance_evidence_type":"insurance_type", "insurance_exempt":"bool", "latest_entry_date":"date"}
CONTEXT_QUESTIONS.update({"passport_expiry_date":"Provide the passport expiration date to assess R006",
    "requested_permit_end_date":"What is the requested permit end date?",
    "application_date":"What is your planned application date?",
    "legal_stay_expiry_date":"What is the end date of your current legal stay?",
    "insurance_evidence_type":"Which insurance evidence type do you have?",
    "insurance_exempt":"Do you have confirmed evidence of an insurance exemption?", "latest_entry_date":"Confirm the latest Turkiye entry date (not a legal-stay deadline)"})
CONTEXT_GROUPS={
    "Passport":("is_in_turkiye","entered_with_passport","passport_expiry_date"),
    "Entry stamp / legal stay":("latest_entry_date","legal_stay_expiry_date"),
    "Insurance":("insurance_evidence_type","insurance_exempt"),
    "Application form":("application_date","requested_permit_end_date"),
}
INSURANCE_TYPES=("bilateral_social_security","sgk_provision","sgk_application","private_health_insurance")


# Address selection is available for legacy and guided cases. Photo readiness
# remains a guided-workflow confirmation.
PROCEDURE_CONTEXT_FIELDS = {"biometric_photo":"bool"}
LEGAL_STAY_FIELDS={"nationality":"str", "passport_type":"str", "visa_status":"str", "visa_stay_days":"int", "visa_valid_until":"date", "previous_stay_history":"str", "existing_permit_until":"date"}
CONTEXT_FIELDS.update(LEGAL_STAY_FIELDS)


# Explicit form answers use the same VerifiedInput records as other confirmations.
# They remain accepted after preparation; this is not a second form/fact store.
FORM_CONTEXT_FIELDS={name:"str" for name in ("first_name","last_name","nationality","passport_or_foreign_id","gender","passport_type","passport_issuing_country","passport_issuing_authority","email","turkish_phone","residential_address","reason_of_stay")}
FORM_CONTEXT_FIELDS.update({name:"date" for name in ("birth_date","passport_issue_date","passport_expiry_date","latest_entry_date","requested_start_date","requested_permit_end_date","insurance_end_date")})
FORM_CONTEXT_FIELDS["insurance_evidence_type"]="insurance_type"
FORM_CONTEXT_FIELDS.update({name:"str" for name in ("father_name","mother_name","marital_status","birth_country","contact_method","address_province","address_district","address_locality")})
FORM_CONTEXT_FIELDS.update(form_current_date="date",simulation_declaration_acknowledged="bool")



def declaration_fields(state):
    fields=dict(CONTEXT_FIELDS)
    fields.update({name:"bool" for name in ("fee_payment_required","single_entry_visa_fee_required","single_entry_visa_fee_receipt_present","fingerprint_step_completed","gi_number_present","scans_sent_to_iro","iro_preverification_confirmed","submitted_to_iro","official_decision_received")})
    fields["approval_status"]="str"
    fields["gi_number"]="str"
    fields["address_proof_type"]="str"
    if state.get("procedure") is not None:fields["fee_receipt_present"]="bool"
    procedure=state.get("procedure")
    if procedure is not None:
        fields.update(PROCEDURE_CONTEXT_FIELDS)  # Accepted collection statements remain valid in later stages.
        if ProcedureState.model_validate(procedure).current_stage!='DOCUMENT_COLLECTION':fields.update(FORM_CONTEXT_FIELDS)
    return fields


def clean_name(value):
    text="".join(c for c in normalize("NFKC",str(value)) if category(c)!="Cf")
    text=text.translate(str.maketrans({"\u2018":"'","\u2019":"'","\u02bc":"'","\u2010":"-","\u2011":"-","\u2013":"-"}))
    text=re.sub(r"\s*([-'])\s*",r"\1",text)
    return " ".join(text.replace(","," ").split())


def same_name(a,b):
    return clean_name(a).casefold()==clean_name(b).casefold()


def holder_observations(state, records=None):
    return accepted_sources(state, records).get("holder_name", [])


def holder_signature(rows):
    return json.dumps(sorted((r["document_id"], r["origin"], str(r["value"])) for r in rows), ensure_ascii=True)


def conflict_details(state, *, sources=None):
    """Use the same accepted sources and comparisons as canonical projection."""
    sources = accepted_sources(state) if sources is None else sources
    sources = {**sources, **_host_conflict_sources(state)}
    resolution = state.get("identity_resolutions", {}).get("holder_name", {})
    return {
        name: {
            "observations": rows,
            "reason": "Different accepted values for the same field",
            "resolved": name == "holder_name" and resolution.get("signature") == holder_signature(rows)
                        and bool(resolution.get("value")),
        }
        for name, rows in sources.items()
        if len({comparison_key(name, row["value"]) for row in rows}) > 1
    }


ATTESTATIONS = {
    "health_insurance": ("insurance_valid_for_ikamet",),
    "ikamet_application_form": ("application_form_signed", "online_application_completed"),
    **ADDRESS_ATTESTATIONS,
}
MAPPINGS = {
    "student_certificate": {"holder_name": ("holder_name",), "student_id": ("student_id",), "institution": ("institution",), "issue_date": ("student_certificate_issue_date",), "exchange_student": ("exchange_student",), "program_start_date": ("program_start_date",), "program_end_date": ("program_end_date",)},
    "health_insurance": {"holder_name": ("holder_name",), "insurer": ("insurance_insurer",), "policy_number": ("insurance_policy_number",), "coverage_start_date": ("insurance_start_date",), "coverage_end_date": ("insurance_end_date",), "passport_or_foreign_id": ("passport_or_foreign_id",), "insurance_valid_for_ikamet": ("insurance_valid_for_ikamet",)},
    "ikamet_application_form": {"applicant_name": ("holder_name",), "passport_or_foreign_id": ("passport_or_foreign_id",), "permit_type": ("permit_type",), "application_number": ("application_number",), "requested_start_date": ("requested_start_date",), "requested_end_date": ("requested_end_date", "requested_permit_end_date"), "application_form_signed": ("application_form_signed",), "online_application_completed": ("online_application_completed",)},
}


MAPPINGS.update({
    "utility_bill":{"service_address":("utility_service_address",)},
    "address_form":{"first_name":("first_name",),"last_name":("last_name",),"birth_date":("birth_date",),
                    "passport_number":("passport_or_foreign_id",),"nationality":("nationality",),"declared_address":("residential_address",)},
    "passport_copy":{"first_name":("first_name",),"last_name":("last_name",),"birth_date":("birth_date",),"holder_name":("holder_name",),"passport_number":("passport_or_foreign_id",),"nationality":("nationality",),"expiry_date":("passport_expiry_date",),"identity_page_present":("passport_id_page_present",)},
    "turkiye_entry_stamp":{"entry_stamp_present":("entry_stamp_present","turkey_entry_page_present"),"latest_entry_date":("latest_entry_date",)},
    "declaration_control_document":{"holder_name":("holder_name",),"passport_number":("passport_or_foreign_id",),
        "foreigner_id":("foreigner_id",),"nationality":("nationality",),"application_number":("application_number",),
        "document_present":("declaration_control_document_present",),"signed":("declaration_control_document_signed",)},
    "card_fee_receipt":{"receipt_present":("fee_receipt_present",),"payment_date":("receipt_payment_date",),"reference_number":("receipt_reference_number",),"amount":("receipt_amount",)},
    "tax_id_certificate":{"holder_name":("holder_name",),"passport_number":("passport_or_foreign_id",)},
})
MAPPINGS.update({kind: {"holder_name": ("holder_name",), "declared_address": ("residential_address",)}
                 for kind in RESIDENCE_DOCUMENTS})
DOC_CONTEXT_FIELDS={"health_insurance":{"insurance_evidence_type":"insurance_type"}}
MAPPINGS["health_insurance"]["insurance_evidence_type"]=("insurance_evidence_type",)
OPTIONAL_FIELDS = {
    DORM: ("declared_address",),
    "passport_copy": ("first_name", "last_name", "birth_date"),
    "address_form": ("nationality", "birth_date", "phone", "form_date"),
    "health_insurance": ("passport_or_foreign_id",),
    "ikamet_application_form": ("application_number",),
    "declaration_control_document": ("holder_name", "passport_number", "foreigner_id", "nationality", "application_number"),
    "card_fee_receipt": ("holder_name", "payment_date", "reference_number", "amount", "taxpayer_id"),
    "tax_id_certificate": ("passport_number", "issuing_authority", "issue_date", "document_reference"),
}


def source_identifier_role(document):
    """Use the extracted label of the combined field, never its value's shape."""
    snippet = normalized(document.info.source_snippets.get("passport_or_foreign_id", "")) if document.info else ""
    passport = re.search(r"\b(?:passport|pasaport)\b", snippet)
    foreign = re.search(r"\b(?:foreign(?:er)?\s+(?:id(?:entity)?|identification)|yabanci\s+kimlik|ykn)\b", snippet)
    if passport and not foreign:
        return "passport_number"
    if foreign and not passport:
        return "foreigner_id"
    return "untyped"


def document_field_targets(document, name, entry=None):
    targets = MAPPINGS.get(document.doc_type, {}).get(name, ())
    if name != "passport_or_foreign_id" or not targets:
        return targets
    confirmed = VerifiedInput.model_validate(entry) if entry is not None else None
    role = confirmed.identifier_role if confirmed and confirmed.origin == "user_confirmed" else None
    role = role or source_identifier_role(document)
    return {"passport_number": ("passport_or_foreign_id",), "foreigner_id": ("foreigner_id",)}.get(role, ())


_ASCII_UPPER = str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_NATIONALITY_ALIASES = {
    **dict.fromkeys(("it", "ita", "italy", "italian", "italia", "italiana", "italiano", "italyan"), "ita"),
    **dict.fromkeys(("de", "deu", "germany", "german", "deutschland", "alman", "almanya"), "deu"),
    **dict.fromkeys(("tr", "tur", "turkey", "turkiye", "turkish", "turk"), "tur"),
}


def comparison_key(name, value):
    """Normalize only equivalences supported by the field's own semantics."""
    if value is None:
        return None
    if name in {"holder_name", "first_name", "last_name", "host_name"}:
        return clean_name(value).casefold()
    if name in {"passport_or_foreign_id", "foreigner_id", "application_number"}:
        text = normalize("NFKC", str(value))
        return "".join(char for char in text if not char.isspace() and category(char) != "Cf").translate(_ASCII_UPPER)
    if name == "nationality":
        text = " ".join(normalized(str(value)).split())
        return _NATIONALITY_ALIASES.get(text, text)
    if name in {"residential_address", "utility_service_address", "host_address"}:
        return " ".join(normalize("NFC", str(value)).casefold().split())
    if name.endswith("_date"):
        return normalized_date(str(value).strip()) or value
    if name == "institution":
        return ResidencePermitContext(institution=value).institution
    return value


def _typed_info(document):
    info = document.info
    expected = {"student_certificate": "student_enrollment_certificate", "health_insurance": "health_insurance_certificate", "ikamet_application_form": "ikamet_application_form"}.get(document.doc_type)
    expected=PROFILE_TYPES.get(document.doc_type,expected)
    schema = DOCUMENT_SCHEMAS.get(expected)
    if document.status != "analyzed" or schema is None or not isinstance(info, schema):
        return None
    return info


_IDENTIFIER_FIELDS = {"passport_number", "passport_or_foreign_id", "foreigner_id"}
_EVIDENCE_FLAGS = {"exchange_student", "identity_page_present", "entry_stamp_present", "document_present",
                   "signed", "receipt_present", "signature_present", "residence_permit_purpose"}
_REQUIRED_YES = ADDRESS_REQUIRED_YES | (_EVIDENCE_FLAGS - {"residence_permit_purpose"}) | {
    "application_form_signed", "online_application_completed",
}


def _field_value_problem(info, name, value):
    if value is None:
        return None
    if name in info.DATES and normalized_date(value) is None:
        return "Enter a valid calendar date in YYYY-MM-DD format."
    if name in {"tax_identification_number", "taxpayer_id"}:
        digits = "".join(str(value).split())
        pattern = r"[0-9]{10}" if name == "tax_identification_number" else r"[0-9]{10,11}"
        if not re.fullmatch(pattern, digits):
            return "Enter the identifier shown on the original document."
    if name in _IDENTIFIER_FIELDS:
        token = comparison_key("foreigner_id" if name == "foreigner_id" else "passport_or_foreign_id", value)
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9/-]{2,39}", token) or not any(char.isdigit() for char in token):
            return "Enter an identifier with readable letters and digits from the original."
    if isinstance(value, str) and not value.strip():
        return "Enter a readable value from the original."
    return None


def _reading_problem(info, name, value):
    """Reject detectable reading problems independently of the provider."""
    diagnostics = info.extraction_diagnostics
    if name in diagnostics.get("ocr_uncertain_fields", ()):
        return "OCR reported low confidence for this reading."
    page_number = info.source_pages.get(name)
    page = next((row for row in diagnostics.get("page_quality", ()) if row.get("page_number") == page_number), {})
    method = diagnostics.get("field_read_methods", {}).get(name, page.get("extraction_method"))
    if method is None and diagnostics.get("extraction_route") == "RAPIDOCR":
        method = "rapidocr"
    if method == "rapidocr" and {"malformed_text", "ocr_failed"} & set(page.get("quality_issues", ())):
        return "OCR could not read the source page reliably."
    if (method == "rapidocr" and not page.get("ocr_line_scores_available")
            and "low_ocr_confidence" in page.get("quality_issues", ())):
        return "OCR reported low confidence for the source page."
    if isinstance(value, str) and re.search(r"\[(?:unreadable|unclear|illegible)\]|\ufffd", value, re.I):
        return "Part of this reading is unreadable."
    if (name in {"holder_name", "applicant_name", "first_name", "last_name"}
            and method in {"rapidocr", "multimodal_fallback", "openai_vision"}
            and isinstance(value, str) and re.search(r"\b[A-Z]{2,}[a-z][A-Z]", value)):
        return "Check this name against the original. The reading has unusual capitalization."
    return None


def grounded_values(document):
    """Accept field-specific provenance only after deterministic reading checks."""
    info = _typed_info(document)
    if info is None or info.document_type == "unknown":
        return {}
    values = info.model_dump(mode="json")
    accepted = {}
    for name in info.observation_fields():
        value = values.get(name)
        if (value is None or name in info.ambiguous_fields
                or any(issue.startswith((f"{name}:", f"{name} ")) for issue in info.validation_issues)
                or _field_value_problem(info, name, value) or _reading_problem(info, name, value)):
            continue
        quote, page = info.source_snippets.get(name), info.source_pages.get(name)
        roles = {name, "name"} if name in ("holder_name", "applicant_name") else {name}
        if (quote and page and any(e.field in roles and e.normalized_value == (str(value).lower() if isinstance(value, bool) else value)
                and e.page_number == page and normalized_snippet(e.snippet) == normalized_snippet(quote)
                for e in info.source_evidence)):
            accepted[name] = value
    return accepted


def _accepted_value_problems(info, entries):
    problems = {}
    for name, entry in entries.items():
        if problem := _field_value_problem(info, name, entry.value):
            problems[name] = problem
    if info.PERIOD:
        first, last = info.PERIOD
        start, end = entries.get(first), entries.get(last)
        if (start and end and isinstance(start.value, str) and isinstance(end.value, str)
                and first not in problems and last not in problems and end.value < start.value):
            field = first if start.origin != "user_confirmed" and end.origin == "user_confirmed" else last
            problems[field] = "The end date precedes the start date. Check the period against the original."
    return problems


def _explicit_latest_entry(info):
    quote = info.source_snippets.get("latest_entry_date", "")
    return re.match(rf"^(?:{LATEST_ENTRY_LABEL})\s*:", normalized(quote).strip()) is not None


def _confirmed_records(document, records):
    entries = {}
    for name, raw in records.items():
        entry = VerifiedInput.model_validate(raw)
        if (entry.origin == "user_confirmed" and entry.document_id == document.doc_id
                and entry.field_name == name and entry.confirmed_by and entry.trace_id):
            if name == "passport_or_foreign_id" and entry.identifier_role is None:
                entry = entry.model_copy(update={"identifier_role": source_identifier_role(document)})
            entries[name] = entry
    return entries


def _current_document_records(document, records, trace_id, *, grounded=None):
    entries = _confirmed_records(document, records)
    grounded = grounded_values(document) if grounded is None else grounded
    for name, value in grounded.items():
        if name in entries:
            continue
        if document.doc_type == "turkiye_entry_stamp" and name == "latest_entry_date" and not _explicit_latest_entry(document.info):
            continue
        entries[name] = VerifiedInput(value=value, origin="source_grounded", document_id=document.doc_id,
            field_name=name, source_page=document.info.source_pages[name],
            source_snippet=document.info.source_snippets[name], trace_id=trace_id)
    return entries


def _host_conflict_sources(state):
    if selected_address_method(state) != HOST:
        return {}
    documents = [doc for doc in state.get("documents", ()) if doc.doc_type in
                 {HOST, "utility_bill", "utility_subscription", "host_support"} and _typed_info(doc) is not None]
    primary = [doc for doc in documents if doc.doc_type == HOST]
    if len(primary) != 1:
        return {}
    host, = primary
    saved = state.get("verification_records", {})
    records = {doc.doc_id: _current_document_records(doc, saved.get(doc.doc_id, {}), state.get("trace_id") or "review")
               for doc in documents}
    values = {owner: {name: entry.value for name, entry in entries.items()} for owner, entries in records.items()}
    host_values = values[host.doc_id]
    groups = [
        ("host_utility_current", {"utility_bill", "utility_subscription"}, "holder_name", "service_address"),
        *((field, {"host_support"}, "host_name", "declared_address")
          for field in host_support_fields(host_values.get("host_married"), host_values.get("host_is_turkish"))
          if host_values.get(field) is not True),
    ]
    conflicts = {}
    for attestation, types, name_field, address_field in groups:
        comparisons = [
            (doc, compare_host_evidence(host_values, values[doc.doc_id],
                                        name_field=name_field, address_field=address_field))
            for doc in documents if doc.doc_type in types and values[doc.doc_id].get(attestation) is True
        ]
        if any(all(same is True for same in checks.values()) for _, checks in comparisons):
            continue  # One matching file can supply this supporting item.
        for doc, checks in comparisons:
            for field, host_field, support_field in (("host_name", "host_name", name_field),
                                                     ("host_address", "declared_address", address_field)):
                if checks[field] is not False:
                    continue
                for owner, name in ((host, host_field), (doc, support_field)):
                    entry = records[owner.doc_id][name]
                    conflicts.setdefault(field, {})[owner.doc_id] = {
                        "document_id": owner.doc_id, "document_type": owner.doc_type,
                        "field_name": name, "origin": entry.origin, "value": entry.value,
                    }
    return {field: list(rows.values()) for field, rows in conflicts.items()}


def accepted_sources(state, records=None):
    """Group only active, comparable accepted facts; values keep their provenance."""
    rebuild = records is None
    records = state.get("verification_records", {}) if rebuild else records
    method = selected_address_method({**state, "verification_records": records})
    sources = {}

    def add(target, value, origin, document=None, name=None):
        if value is not None:
            sources.setdefault(target, []).append({
                "document_id": document.doc_id if document else None,
                "document_type": document.doc_type if document else None,
                "field_name": name or target, "origin": origin, "value": value,
            })

    for name, raw in records.get("__declarations__", {}).items():
        entry = VerifiedInput.model_validate(raw)
        if entry.origin == "user_confirmed":
            add(name, entry.value, entry.origin)
    for document in state.get("documents", []):
        if not hasattr(document, "doc_id") or _typed_info(document) is None:
            continue
        kind = document.doc_type
        if kind in RESIDENCE_DOCUMENTS and kind != method:
            continue
        entries = records.get(document.doc_id, {})
        if rebuild:
            entries = _current_document_records(document, entries, state.get("trace_id") or "review")
        entries = {name: VerifiedInput.model_validate(entry) for name, entry in entries.items()}
        problems = _accepted_value_problems(document.info, entries)
        entries = {name: entry for name, entry in entries.items() if name not in problems}
        for name, entry in entries.items():
            if kind == "address_form" and name == "declared_address" and method not in {None, IRO_PAIR}:
                continue
            if kind == "utility_bill" and method not in {None, IRO_PAIR, HOST}:
                continue
            if kind in RESIDENCE_DOCUMENTS and name == "declared_address":
                attestation = ADDRESS_ATTESTATIONS[kind][0]
                if attestation not in entries or VerifiedInput.model_validate(entries[attestation]).value is not True:
                    continue
            for target in document_field_targets(document, name, entry):
                if target != "holder_name":
                    add(target, entry.value, entry.origin, document, name)
        holder_field = "applicant_name" if kind == "ikamet_application_form" else "holder_name"
        holder = entries.get(holder_field)
        holder = VerifiedInput.model_validate(holder) if holder is not None else None
        if kind in {"passport_copy", "address_form"}:
            given, family = (VerifiedInput.model_validate(entries[name]) if name in entries else None
                             for name in ("first_name", "last_name"))
            if (given and family and given.value and family.value
                    and (kind == "address_form" or holder is None
                         or (holder.origin != "user_confirmed" and "user_confirmed" in {given.origin, family.origin}))):
                origin = "user_confirmed" if "user_confirmed" in {given.origin, family.origin} else "source_grounded"
                add("holder_name", f"{given.value} {family.value}", origin, document, "first_name")
                continue
        # Payer names on receipts and utility bills need not be the applicant.
        if holder is not None and (kind == "address_form" or "holder_name" in document_field_targets(document, holder_field)):
            add("holder_name", holder.value, holder.origin, document, holder_field)
    return sources


def cached_conflict_fields(state, *, sources=None):
    """Retain saved flags unless current evidence proves a legacy mixed-ID comparison."""
    cached = set(state.get("verification_conflicts", ()))
    field = "passport_or_foreign_id"
    if field in cached:
        sources = accepted_sources(state) if sources is None else sources
        passports = {comparison_key(field, row["value"]) for row in sources.get(field, ())}
        foreign_ids = {comparison_key("foreigner_id", row["value"]) for row in sources.get("foreigner_id", ())
                       if row["field_name"] == field}
        if len(passports) == 1 and foreign_ids - passports:
            cached.remove(field)
    return cached


def document_editable_fields(document):
    kind = document.doc_type
    names = set(IMPORTANT_FIELDS.get(kind, ())) | set(ATTESTATIONS.get(kind, ()))
    names.update(OPTIONAL_FIELDS.get(kind, ()), DOC_CONTEXT_FIELDS.get(kind, ()))
    if kind == "student_certificate":
        names.update(STUDENT_FIELDS)
    if kind == "ikamet_application_form":
        names.add("permit_type")
    return names


def required_document_fields(state, document):
    kind = document.doc_type
    names = [*IMPORTANT_FIELDS.get(kind, ()), *ATTESTATIONS.get(kind, ())]
    procedure = state.get("procedure")
    stage = procedure.get("current_stage") if isinstance(procedure, dict) else getattr(procedure, "current_stage", None)
    scoped = state.get("planning_scope") == MILESTONE_SCOPE or procedure is not None
    if kind == "utility_bill" and selected_address_method(state) != HOST:
        names.remove("host_utility_current")
    if kind == "student_certificate" and scoped and stage != "DOCUMENT_COLLECTION":
        names += list(STUDENT_FIELDS)
    if kind == "ikamet_application_form" and scoped:
        names.append("permit_type")
    if kind == "health_insurance":
        names.remove("insurance_valid_for_ikamet")
        if scoped:
            names.remove("coverage_start_date")
            names.append("insurance_evidence_type")
    return tuple(name for name in names if name not in HOST_SUPPORT_FIELDS)


def _case_answer(records, name):
    raw = records.get("__declarations__", {}).get(name)
    if raw is not None:
        entry = VerifiedInput.model_validate(raw)
        if (entry.origin == "user_confirmed" and entry.confirmed_by and entry.trace_id
                and entry.document_id == "__declarations__" and entry.field_name == name):
            return entry
    return None


def _required_reading_issue(info, name, entry, observation, accepted_problem):
    if accepted_problem:
        return "invalid", accepted_problem
    if entry is not None and entry.value is not None:
        if name in _REQUIRED_YES and entry.value is not True:
            return "required_yes", "This document needs positive evidence for this requirement."
        return None, None
    if entry is not None and entry.origin == "user_confirmed":
        return "missing", None  # An explicit clear takes precedence over the original reading.
    relevant = [issue for issue in info.validation_issues if issue.startswith((f"{name}:", f"{name} "))]
    if any("precedes" in issue for issue in relevant):
        return "invalid", "The end date precedes the start date. Check the dates against the original."
    if name in info.DATES:
        date_reads = [candidate for candidate in info.source_evidence
                      if candidate.field == name and EVIDENCE_DATE_PATTERN.fullmatch(candidate.original_value)]
        unusable_dates = date_reads and all(
            not (candidate.normalized_value or source_date(candidate.original_value, "dmy")
                 or source_date(candidate.original_value, "mdy"))
            for candidate in date_reads
        )
        if unusable_dates or any(issue.startswith(f"{name}: invalid or implausible date") for issue in relevant):
            return "invalid", "Enter a valid calendar date in YYYY-MM-DD format."
    if name in info.ambiguous_fields:
        return "uncertain", "The document contains unclear or competing readings for this field."
    if problem := _field_value_problem(info, name, observation):
        return "invalid", problem
    if problem := _reading_problem(info, name, observation):
        return "uncertain", problem
    if name == "latest_entry_date" and observation is not None:
        return "declaration", "Confirm that the date shown was your latest entry into Türkiye."
    if name in ATTESTATIONS.get(info.document_type, ()) or name in {"application_form_signed", "online_application_completed"}:
        return "declaration", None
    if observation is not None:
        return "uncertain", "This reading could not be established from the source evidence."
    return "missing", None


def document_review(state, document, *, include_optional=(), full=False, conflicts=None):
    """Derive required fields and readiness from the evidence used by the case."""
    info = _typed_info(document)
    kind = document.doc_type
    can_confirm = info is not None and kind in IMPORTANT_FIELDS
    if full and info is None and isinstance(document.info, DocumentInfo):
        info = document.info
    if info is None or (not full and not can_confirm):
        return {"status": "missing", "fields": [], "can_confirm": False}
    records = state.get("verification_records", {})
    saved = records.get(document.doc_id, {})
    accepted = (_current_document_records(document, saved, state.get("trace_id") or "review")
                if can_confirm else _confirmed_records(document, saved))
    required = required_document_fields(state, document)
    for name in ("latest_entry_date", "insurance_evidence_type"):
        if can_confirm and name in required and name not in accepted:
            if answer := _case_answer(records, name):
                accepted[name] = answer
    accepted_problems = _accepted_value_problems(info, accepted)
    raw = info.model_dump(mode="json")

    applicable_support = set()
    if kind in {HOST, "host_support"}:
        attestations = accepted
        if kind == "host_support":
            primary = next((doc for doc in state.get("documents", ()) if doc.doc_type == HOST and doc.status == "analyzed"), None)
            attestations = _confirmed_records(primary, records.get(primary.doc_id, {})) if primary else {}
        applicable_support = set(host_support_fields(
            getattr(attestations.get("host_married"), "value", None),
            getattr(attestations.get("host_is_turkish"), "value", None),
        ))
    support_needed = kind == "host_support" and not any(
        name in accepted and accepted[name].value is True for name in applicable_support
    )
    conflicts = (conflict_details(state) if conflicts is None else conflicts) if can_confirm else {}
    conflicting = {
        row["field_name"] for detail in conflicts.values() if not detail["resolved"]
        for row in detail["observations"] if row["document_id"] == document.doc_id
    }
    names = [*required, *sorted(name for name in document_editable_fields(document) - set(required)
             if name in include_optional or name in applicable_support or name in conflicting
             or (name not in HOST_SUPPORT_FIELDS and (raw.get(name) is not None or name in accepted)))]
    if kind == "card_fee_receipt" and raw.get("payment_code") is not None:
        names.append("payment_code")
    if full:
        names += sorted(info.observation_fields() | set(accepted) | set(ATTESTATIONS.get(kind, ())))

    fields = []
    for name in dict.fromkeys(names):
        entry = accepted.get(name)
        origin = entry.origin if entry else None
        value = entry.value if entry else None
        observation = raw.get(name)
        status = (origin if origin and value is not None and name not in accepted_problems
                  else "ambiguous" if name in info.ambiguous_fields and origin != "user_confirmed"
                  else "needs_confirmation")
        issue, message = None, None
        if can_confirm and name in conflicting:
            issue = "conflict"
        elif can_confirm and name in required:
            issue, message = _required_reading_issue(info, name, entry, observation, accepted_problems.get(name))
        elif can_confirm and support_needed and name in applicable_support:
            issue = "declaration"
        if issue == "missing" and name in ATTESTATIONS.get(kind, ()) and entry is None:
            issue = "declaration"
        # Prefill a single source reading, or retain a visibly invalid candidate
        # for correction. A saved clear and competing readings stay empty.
        suggestion = value if entry is not None else observation
        if issue in {"missing", "invalid", "uncertain"} and entry is None and suggestion is None:
            readings = {item.original_value for item in info.source_evidence
                        if item.field == name and item.original_value.strip()
                        and not re.search(r"\[(?:unreadable|unclear|illegible)\]", item.original_value, re.I)}
            if len(readings) == 1:
                suggestion = readings.pop()
        combined_identifier = name == "passport_or_foreign_id"
        display_field = observation_field = "resident_name" if name == "holder_name" and kind in RESIDENCE_DOCUMENTS else name
        if name == "holder_name" and kind == "card_fee_receipt":
            display_field = "payer_name"
        if combined_identifier:
            reading_role = source_identifier_role(document)
            accepted_role = (entry.identifier_role if entry else None) or reading_role
            display_field = name if accepted_role == "untyped" else accepted_role
            observation_field = name if reading_role == "untyped" else reading_role
        fields.append({
            "name": name, "display_field": display_field, "observation_field": observation_field,
            "value": value, "observation": observation, "suggestion": suggestion,
            "document_field_absent": kind == "address_form" and name == "passport_number" and observation is None,
            "observation_method": info.extraction_diagnostics.get("field_read_methods", {}).get(name),
            "origin": origin, "status": status, "issue": issue, "message": message,
            "source_page": entry.source_page if entry and entry.source_page else info.source_pages.get(name),
            "from_declaration": bool(entry and entry.document_id == "__declarations__"),
            "identifier_untyped": combined_identifier and value is not None and display_field == name,
            "optional_review": name not in required,
            "requires_yes": name in _REQUIRED_YES,
            "boolean": name in _EVIDENCE_FLAGS or name in ATTESTATIONS.get(kind, ()),
        })
    status = ("conflict" if conflicting else "needs_review" if not can_confirm or any(row["issue"] for row in fields)
              else "complete")
    return {"status": status, "fields": fields, "can_confirm": can_confirm}


def verification_updates(state):
    records = deepcopy(state.get("verification_records", {}))
    event = state["event"]
    documents = state.get("documents", [])
    saved_ids = {document.doc_id for document in documents if document.status != "deleted"}
    records = {key: entries for key, entries in records.items() if key.startswith("__") or key in saved_ids}
    if event.kind=="remove_correction":
        for name in event.confirmed_fields: records.get(event.document_id,{}).pop(name,None)
    if event.kind == "analyze_document":
        for document in documents:
            if Path(document.file_path).resolve() == Path(event.file_path).resolve():
                records[document.doc_id] = {name:value for name,value in records.get(document.doc_id,{}).items() if VerifiedInput.model_validate(value).origin == "user_confirmed"}
    for document in documents:
        if _typed_info(document) is not None:
            records[document.doc_id] = _current_document_records(
                document, records.get(document.doc_id, {}), state["trace_id"]
            )
            continue
        # Failed and pending reads retain corrections for retry, off the active case.
        previous = records.pop(document.doc_id, {})
        if document.status in {"pending", "rejected", "replacement_pending"}:
            saved = _confirmed_records(document, previous)
            if saved:
                records[document.doc_id] = saved
    if event.kind == "confirm_information" and event.document_id == "__declarations__":
        allowed_declarations=declaration_fields(state)
        if set(event.confirmed_fields) - set(allowed_declarations):
            raise ValueError("Unsupported declaration field")
        entries = records.setdefault("__declarations__", {})
        for name,value in event.confirmed_fields.items():
            kind=allowed_declarations[name]
            if value is not None:
                if kind=="str":
                    if not isinstance(value,str):raise ValueError("Declarations require text or unknown")
                    value=value.strip() or None
                if kind=="int" and (type(value) is not int or not 1<=value<=90):raise ValueError("Visa stay days must be 1..90")
                if kind=="bool" and type(value) is not bool: raise ValueError("Declarations require Yes, No or unknown")
                if kind=="insurance_type" and value not in INSURANCE_TYPES: raise ValueError("Choose an accepted insurance evidence type")
                if kind=="date":
                    value=normalized_date(value)
                    if value is None: raise ValueError("Enter a valid ISO date, year 1900..2100")
            entries[name] = VerifiedInput(value=value, origin="user_confirmed", document_id="__declarations__",
                field_name=name, confirmed_by=state["user_id"], trace_id=state["trace_id"])
    elif event.kind == "confirm_information":
        document = next((d for d in documents if d.doc_id == event.document_id), None)
        if document is None or _typed_info(document) is None or document.doc_type not in IMPORTANT_FIELDS:
            raise ValueError("Reprocess a supported document before confirming its information")
        if set(event.confirmed_fields) - document_editable_fields(document):
            raise ValueError("Unsupported confirmation field")
        raw = document.info.model_dump(mode="json")
        observed = {name:value for name,value in event.confirmed_fields.items() if name not in ATTESTATIONS.get(document.doc_type, ()) and name not in DOC_CONTEXT_FIELDS.get(document.doc_type,{})}
        previous = {name: entry.value for name, entry in records[document.doc_id].items()
                    if entry.origin == "user_confirmed" and name in document.info.observation_fields()}
        current = records[document.doc_id]
        usable = {name: entry.value for name, entry in current.items() if name in document.info.observation_fields()}
        validation = {**raw, **previous, **observed}
        for name in document.info.PERIOD or ():
            if name not in observed:
                validation[name] = usable.get(name)
        validation["ambiguous_fields"] = [name for name in raw["ambiguous_fields"]
                                           if name not in previous and name not in observed]
        validation["validation_issues"] = []
        validation["source_pages"] = {}
        validation["source_snippets"] = {}
        checked = type(document.info).model_validate(validation)
        checked_values = checked.model_dump(mode="json")
        if checked.PERIOD and any(name in observed for name in checked.PERIOD) and any("precedes" in issue for issue in checked.validation_issues):
            raise ValueError("End date must not precede start date")
        for name, value in event.confirmed_fields.items():
            role = None
            if name == "passport_or_foreign_id":
                prior = records[document.doc_id].get(name)
                role = (prior.identifier_role if prior else None) or source_identifier_role(document)
            if value is None or value == "":
                records[document.doc_id][name] = VerifiedInput(value=None, origin="user_confirmed", document_id=document.doc_id,
                    field_name=name, confirmed_by=state["user_id"], trace_id=state["trace_id"], identifier_role=role)
                continue
            if name in DOC_CONTEXT_FIELDS.get(document.doc_type,{}):
                if name=="insurance_evidence_type" and value not in INSURANCE_TYPES: raise ValueError("Choose an accepted insurance evidence type")
                accepted=value
            elif name in ATTESTATIONS.get(document.doc_type, ()):
                if type(value) is not bool:
                    raise ValueError(f"{name}: explicitly choose Yes or No")
                accepted = value
            else:
                accepted = checked_values[name]
                if accepted is None or any(issue.startswith((f"{name}:", f"{name} ")) for issue in checked.validation_issues):
                    raise ValueError(f"{name}: enter a valid value (dates use YYYY-MM-DD)")
                if problem := _field_value_problem(checked, name, accepted):
                    raise ValueError(f"{name}: {problem}")
            records[document.doc_id][name] = VerifiedInput(value=accepted, origin="user_confirmed",
                document_id=document.doc_id, field_name=name, confirmed_by=state["user_id"], trace_id=state["trace_id"], identifier_role=role)
    context = ResidencePermitContext.model_validate(state.get("residence_permit_context", {})).model_dump()
    for name in state.get("verified_context_fields", []):
        context[name] = None
    for name, raw in records.get("__declarations__", {}).items():
        if VerifiedInput.model_validate(raw).value is None:
            context[name] = None
    records.pop("__derived__", None)
    sources = accepted_sources(state, records)
    rows = sources.get("holder_name", [])
    resolutions = deepcopy(state.get("identity_resolutions", {}))
    if event.kind == "resolve_conflict":
        choice = event.confirmed_fields.get("holder_name")
        if not choice or not any(row["value"] == choice for row in rows):
            raise ValueError("Select an existing identity observation for explicit review")
        resolutions["holder_name"] = {"value": choice, "signature": holder_signature(rows),
                                        "user_id": state["user_id"], "trace_id": state["trace_id"]}
    details = conflict_details({**state, "verification_records": records,
                                "identity_resolutions": resolutions}, sources=sources)
    conflicts = {name for name, detail in details.items() if not detail["resolved"]}
    review_state = {**state, "verification_records": records, "identity_resolutions": resolutions}
    reviews = {document.doc_id: document_review(review_state, document, conflicts=details)
               for document in documents if document.status != "deleted"}
    proposals = {name: [row["value"] for row in contributors] for name, contributors in sources.items()
                 if name != "holder_name"}
    # Tax assignments are supplementary identity evidence. They can surface a
    # disagreement with a passport source without supplying its canonical value.
    if "passport_or_foreign_id" in proposals:
        passport_sources = [row["value"] for row in sources["passport_or_foreign_id"]
                            if row["document_type"] != "tax_id_certificate"]
        if passport_sources:
            proposals["passport_or_foreign_id"] = passport_sources
        else:
            proposals.pop("passport_or_foreign_id")
    # Address Form name parts corroborate an identity. Projection requires a
    # holder observation from the existing identity sources or explicit review.
    holders = [row["value"] for row in rows
               if row["document_type"] not in {"address_form", "tax_id_certificate", *RESIDENCE_DOCUMENTS}]
    if holders or "holder_name" in details:
        proposals["holder_name"] = holders or [None]
    active = [document for document in documents if _typed_info(document) is not None]
    for document in active:
        entries = records.get(document.doc_id, {})
        problems = _accepted_value_problems(document.info, entries)
        if all(name in entries and entries[name].value is not None and name not in problems
               for name in IMPORTANT_FIELDS.get(document.doc_type, ())):
            if document.doc_type == "student_certificate":
                proposals.setdefault("student_certificate_present", []).append(True)
            if document.doc_type == "ikamet_application_form":
                proposals.setdefault("application_form_generated", []).append(True)
    # Composite pieces must be current, complete and free of conflicting evidence.
    method = selected_address_method({**state, "verification_records": records})
    for kind, target in (("utility_bill", "utility_bill_accepted"), ("address_form", "address_form_accepted")):
        if method not in {None, IRO_PAIR}:
            continue
        matching = [document for document in active if document.doc_type == kind]
        if len(matching) != 1:
            if matching:
                proposals[target] = [None]
            continue
        document, = matching
        proposals[target] = [True if reviews[document.doc_id]["status"] == "complete" else None]
    for actual, presence in (("email", "email_present"), ("turkish_phone", "phone_present"),
                             ("residential_address", "turkey_address_present"), ("gi_number", "gi_number_present")):
        if actual in sources:
            proposals.setdefault(presence, []).append(None if actual in conflicts else True)
            if len(set(proposals[presence])) > 1:
                conflicts.add(presence)
    for name, values in proposals.items():
        value = values[0]
        if name == "holder_name" and details.get(name, {}).get("resolved"):
            value = resolutions[name]["value"]
        if name in {"holder_name", "first_name", "last_name"} and value is not None:
            value = clean_name(value)
        context[name] = None if name in conflicts else value
    progress=deepcopy(state.get("action_progress",{}))
    if event.kind=="record_progress":
        approved=state.get("approved_plan")
        if approved is not None:
            approved=ActionPlan.model_validate(approved.model_dump() if hasattr(approved,"model_dump") else approved)
        if approved is None or event.action_id not in {a.step_id for a in approved.actions}:
            raise ValueError("Only an explicitly approved action can be recorded")
        progress[state["trace_id"]+":"+event.action_id]={"action_id":event.action_id,"status":"declared_completed","user_id":state["user_id"],"trace_id":state["trace_id"],"action":next(a.action for a in approved.actions if a.step_id==event.action_id),"related_rule_id":next(a.related_rule_id for a in approved.actions if a.step_id==event.action_id)}
    # Whole calendar months, subtracting one when the final day precedes the
    # starting day. Academic duration is a planning estimate, never permit dates.
    start,end = context.get("program_start_date"),context.get("program_end_date")
    if start and end:
        start,end = (date.fromisoformat(v) if isinstance(v,str) else v for v in (start,end))
        if end >= start and "program_start_date" not in conflicts and "program_end_date" not in conflicts:
            months=(end.year-start.year)*12+end.month-start.month-(end.day<start.day)
            context["planned_stay_months"]=months
            context["exchange_program_evidence_present"]=True if context.get("exchange_student") is True else None
            proposals["planned_stay_months"]=[months]
            proposals["exchange_program_evidence_present"]=[context["exchange_program_evidence_present"]]
            records["__derived__"]={"planned_stay_months":VerifiedInput(value=months,origin="derived",document_id="__derived__",field_name="planned_stay_months",trace_id=state["trace_id"],derived_from=["program_start_date","program_end_date"],derivation_policy="Complete calendar months of academic exchange; estimate of planned stay, not requested permit duration")}
    if "__derived__" in records:
        records["__derived__"]["exchange_program_evidence_present"] = VerifiedInput(
            value=context["exchange_program_evidence_present"],origin="derived",document_id="__derived__",
            field_name="exchange_program_evidence_present",trace_id=state["trace_id"],
            derived_from=["exchange_student","program_start_date","program_end_date"],
            derivation_policy="Accepted exchange statement and academic period; evidence availability, not authenticity")
    needs_review = any(review["status"] != "complete" for review in reviews.values())
    return {"verification_records": records, "identity_resolutions":resolutions, "action_progress":progress, "verified_context_fields": sorted(proposals),
            "verification_conflicts": sorted(conflicts),
            "residence_permit_context": ResidencePermitContext.model_validate(context),
            "run_status": "INFORMATION_NEEDS_CONFIRMATION" if needs_review or conflicts else "DOCUMENT_ANALYZED"}



def milestone_checklist(state):
    """Product support/completeness, never a replacement for the active checks."""
    results={r.rule_id:r for r in state.get("relevant_requirements") or []}
    context=ResidencePermitContext.model_validate(state.get("residence_permit_context",{}))
    rows=[]
    conflicts = conflict_details(state)
    for kind,(label,rule_ids) in MILESTONE_DOCUMENTS.items():
        documents=[d for d in state.get("documents",[]) if getattr(d,"doc_type",None)==kind and getattr(d,"status",None)=="analyzed"]
        doc_ids = [document.doc_id for document in documents]
        status = "missing"
        if len(documents) > 1:
            status = "ambiguous"
        elif documents:
            document, = documents
            review = document_review(state, document, conflicts=conflicts)
            if review["status"] == "conflict":
                status = "conflict"
            elif any(row["issue"] == "uncertain" and row["status"] == "ambiguous" for row in review["fields"]):
                status = "ambiguous"
            elif review["status"] != "complete":
                status = "needs confirmation"
            elif any(results.get(rid) is None or results[rid].passed is not True for rid in rule_ids):
                status = "needs confirmation"
            elif kind == "turkiye_entry_stamp" and (context.entry_stamp_present is not True or context.latest_entry_date is None):
                status = "needs confirmation"
            elif (cached_conflict_fields(state) - conflicts.keys()) & {
                target for name in document_editable_fields(document)
                for target in document_field_targets(
                    document, name, state.get("verification_records", {}).get(document.doc_id, {}).get(name)
                )
            }:
                status = "conflict"
            else:
                status = "complete"
        rows.append({"document_type":kind,"label":label,"status":status,"document_ids":doc_ids,"rule_ids":list(rule_ids)})
    return rows



def document_lifecycle(state):
    """Canonical graph updates. Files stay intact; checkpoint history remains audit."""
    event=state["event"]
    document=next((d for d in state.get("documents",[]) if d.doc_id==event.document_id),None)
    if document is None: raise ValueError("Document no longer exists")
    if event.kind=="remove_correction":
        if not event.confirmed_fields or set(event.confirmed_fields) - document_editable_fields(document):
            raise ValueError("Choose a supported correction")
        return {}
    if event.kind in {"delete_document","replace_document","cancel_document"}:
        if not event.confirmed: raise ValueError("Explicit confirmation is required")
        if event.kind=="replace_document":
            if document.doc_type == "host_support":
                raise ValueError("Upload new supporting pages and remove only the file you want to replace")
            old=[d for d in state.get("documents",[]) if d.doc_type==document.doc_type and d.doc_id!=document.doc_id and d.status=="analyzed"]
            if document.status!="replacement_pending" and not (document.status=="analyzed" and old): raise ValueError("Choose a pending replacement or an existing duplicate version")
            return {"documents":[d.model_copy(update={"status":"deleted"}) for d in old]+[document.model_copy(update={"status":"analyzed"})],"extracted_facts":{d.doc_id:None for d in old}}
        return {"documents":[document.model_copy(update={"status":"deleted"})],"extracted_facts":{document.doc_id:None}}
    if event.kind=="reprocess_document":
        from config import PROJECT_ROOT
        path=Path(document.file_path).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix.lower() not in {".pdf",".txt"}: raise ValueError("Choose an existing local PDF/TXT")
        if event.document_type_hint is not None and event.document_type_hint not in set(PROFILE_TYPES.values())|{"student_enrollment_certificate","health_insurance_certificate","ikamet_application_form"}: raise ValueError("Choose a supported Milestone-1 profile")
        updates = {"documents": [document.model_copy(update={"status": "pending", "processing_error": None})]}
        records = state.get("verification_records", {})
        if document.doc_id in records:
            # Preserve the role seen at confirmation before a new reading replaces the source label.
            updates["verification_records"] = {
                **records, document.doc_id: _confirmed_records(document, records[document.doc_id]),
            }
        return updates
    raise ValueError("Unsupported lifecycle event")
