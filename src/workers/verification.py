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
from src.core.state import ResidencePermitContext, VerifiedInput, ActionPlan
from src.document_processing.schemas import DOCUMENT_SCHEMAS
from src.document_processing.evidence import normalized_snippet

IMPORTANT_FIELDS = {
    "student_certificate": ("holder_name", "student_id", "institution", "issue_date"),
    "health_insurance": ("holder_name", "insurer", "policy_number", "coverage_start_date", "coverage_end_date"),
    "ikamet_application_form": ("applicant_name", "passport_or_foreign_id", "requested_start_date", "requested_end_date"),
}
IMPORTANT_FIELDS.update({
    "passport_copy":("holder_name","passport_number","nationality","expiry_date","identity_page_present"),
    "turkiye_entry_stamp":("entry_stamp_present","latest_entry_date"),
    "declaration_control_document":("document_present","signed"),
    "card_fee_receipt":("receipt_present",),
})
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
PROFILE_TYPES={"passport_copy":"passport_identity_page","turkiye_entry_stamp":"turkiye_entry_stamp","declaration_control_document":"declaration_control_document","card_fee_receipt":"card_fee_receipt"}
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


def clean_name(value):
    text="".join(c for c in normalize("NFKC",str(value)) if category(c)!="Cf")
    text=text.translate(str.maketrans({"\u2018":"'","\u2019":"'","\u02bc":"'","\u2010":"-","\u2011":"-","\u2013":"-"}))
    text=re.sub(r"\s*([-'])\s*",r"\1",text)
    return " ".join(text.replace(","," ").split())


def same_name(a,b):
    return clean_name(a).casefold()==clean_name(b).casefold()


def holder_observations(state, records=None):
    records=state.get("verification_records",{}) if records is None else records
    rows=[]
    for document in state.get("documents",[]):
        if not hasattr(document,"doc_id"): continue  # Legacy display-only dictionaries require reprocessing.
        if document.status!="analyzed": continue
        values=grounded_values(document)
        raw_field="applicant_name" if document.doc_type=="ikamet_application_form" else "holder_name"
        entry=records.get(document.doc_id,{}).get(raw_field)
        entry=VerifiedInput.model_validate(entry) if entry is not None else None
        if entry and entry.origin=="user_confirmed":
            if entry.value is not None:
                rows.append({"document_id":document.doc_id,"document_type":document.doc_type,"origin":entry.origin,"value":entry.value})
        elif values.get(raw_field):
            rows.append({"document_id":document.doc_id,"document_type":document.doc_type,"origin":"source_grounded","value":values[raw_field]})
    return rows


def holder_signature(rows):
    return json.dumps(sorted((r["document_id"],r["origin"],str(r["value"])) for r in rows),ensure_ascii=True)


def conflict_details(state):
    rows=holder_observations(state)
    conflict=len({clean_name(r["value"]).casefold() for r in rows})>1
    resolution=state.get("identity_resolutions",{}).get("holder_name",{})
    resolved=resolution.get("signature")==holder_signature(rows)
    return {"holder_name":{"observations":rows,"reason":"Distinct identity values in source evidence and/or saved confirmations","resolved":resolved}} if conflict else {}


ATTESTATIONS = {
    "health_insurance": ("insurance_valid_for_ikamet",),
    "ikamet_application_form": ("application_form_signed", "online_application_completed"),
}
MAPPINGS = {
    "student_certificate": {"holder_name": ("holder_name",), "student_id": ("student_id",), "institution": ("institution",), "issue_date": ("student_certificate_issue_date",), "exchange_student": ("exchange_student",), "program_start_date": ("program_start_date",), "program_end_date": ("program_end_date",)},
    "health_insurance": {"holder_name": ("holder_name",), "insurer": ("insurance_insurer",), "policy_number": ("insurance_policy_number",), "coverage_start_date": ("insurance_start_date",), "coverage_end_date": ("insurance_end_date",), "passport_or_foreign_id": ("passport_or_foreign_id",), "insurance_valid_for_ikamet": ("insurance_valid_for_ikamet",)},
    "ikamet_application_form": {"applicant_name": ("holder_name",), "passport_or_foreign_id": ("passport_or_foreign_id",), "permit_type": ("permit_type",), "application_number": ("application_number",), "requested_start_date": ("requested_start_date",), "requested_end_date": ("requested_end_date", "requested_permit_end_date"), "application_form_signed": ("application_form_signed",), "online_application_completed": ("online_application_completed",)},
}


MAPPINGS.update({
    "passport_copy":{"holder_name":("holder_name",),"passport_number":("passport_or_foreign_id",),"nationality":("nationality",),"expiry_date":("passport_expiry_date",),"identity_page_present":("passport_id_page_present",)},
    "turkiye_entry_stamp":{"entry_stamp_present":("entry_stamp_present","turkey_entry_page_present"),"latest_entry_date":("latest_entry_date",)},
    "declaration_control_document":{"holder_name":("holder_name",),"document_present":("declaration_control_document_present",),"signed":("declaration_control_document_signed",)},
    "card_fee_receipt":{"holder_name":("holder_name",),"receipt_present":("fee_receipt_present",),"payment_date":("receipt_payment_date",),"reference_number":("receipt_reference_number",),"amount":("receipt_amount",)},
})
DOC_CONTEXT_FIELDS={"health_insurance":{"insurance_evidence_type":"insurance_type"}}
MAPPINGS["health_insurance"]["insurance_evidence_type"]=("insurance_evidence_type",)
OPTIONAL_FIELDS={"declaration_control_document":("holder_name","passport_number","foreigner_id","nationality","application_number"),"card_fee_receipt":("holder_name","payment_date","reference_number","amount")}


def _typed_info(document):
    info = document.info
    expected = {"student_certificate": "student_enrollment_certificate", "health_insurance": "health_insurance_certificate", "ikamet_application_form": "ikamet_application_form"}.get(document.doc_type)
    expected=PROFILE_TYPES.get(document.doc_type,expected)
    schema = DOCUMENT_SCHEMAS.get(expected)
    if document.status != "analyzed" or schema is None or not isinstance(info, schema):
        return None
    return info


def grounded_values(document):
    """Accept only field-specific, matching source evidence that survived validation."""
    info = _typed_info(document)
    if info is None or info.document_type == "unknown":
        return {}
    values = info.model_dump(mode="json")
    accepted = {}
    for name in info.observation_fields():
        value = values.get(name)
        if value is None or name in info.ambiguous_fields or any(issue.startswith(name) for issue in info.validation_issues):
            continue
        quote, page = info.source_snippets.get(name), info.source_pages.get(name)
        roles = {name, "name"} if name in ("holder_name", "applicant_name") else {name}
        if (quote and page and any(e.field in roles and e.normalized_value == (str(value).lower() if isinstance(value,bool) else value)
                and e.page_number == page and normalized_snippet(e.snippet) == normalized_snippet(quote)
                for e in info.source_evidence)):
            accepted[name] = value
    return accepted


def document_review(state, document):
    """Pure review data shared by UI/tests; not a second state model."""
    info = _typed_info(document)
    if info is None or document.doc_type not in IMPORTANT_FIELDS:
        return {"status": "missing", "fields": [], "can_confirm": False}
    accepted = dict(state.get("verification_records", {}).get(document.doc_id, {}))
    if document.doc_type=="turkiye_entry_stamp" and "latest_entry_date" not in accepted:
        global_date=state.get("verification_records",{}).get("__declarations__",{}).get("latest_entry_date")
        if global_date is not None:accepted["latest_entry_date"]=global_date
    grounded = grounded_values(document)
    raw = info.model_dump(mode="json")
    names = list(IMPORTANT_FIELDS[document.doc_type]) + list(ATTESTATIONS.get(document.doc_type, ()))
    if document.doc_type == "student_certificate" and (state.get("planning_scope")==MILESTONE_SCOPE or any(getattr(info,n,None) is not None for n in STUDENT_FIELDS)):
        names += list(STUDENT_FIELDS)
    if document.doc_type == "ikamet_application_form" and (state.get("planning_scope")==MILESTONE_SCOPE or info.permit_type):
        names += ["permit_type"]
    if document.doc_type=="health_insurance" and state.get("planning_scope")==MILESTONE_SCOPE:
        if raw.get("coverage_start_date") is None:names=[n for n in names if n!="coverage_start_date"]
        names += ["insurance_evidence_type"]
        if "insurance_evidence_type" not in accepted:
            category_record=state.get("verification_records",{}).get("__declarations__",{}).get("insurance_evidence_type")
            if category_record is not None:accepted["insurance_evidence_type"]=category_record
    names += [n for n in OPTIONAL_FIELDS.get(document.doc_type,()) if getattr(info,n,None) is not None]
    fields = []
    for name in names:
        entry = accepted.get(name)
        entry = VerifiedInput.model_validate(entry) if entry is not None else None
        require_confirmation=document.doc_type=="turkiye_entry_stamp" and name=="latest_entry_date"
        origin = entry.origin if entry else "source_grounded" if name in grounded and not require_confirmation else None
        fields.append({"name": name, "value": entry.value if entry else grounded.get(name),
                       "observation": raw.get(name), "origin": origin,
                       "status": origin if origin and (entry.value if entry else grounded.get(name)) is not None else "ambiguous" if name in info.ambiguous_fields else "needs_confirmation",
                       "boolean": name in {"exchange_student","identity_page_present","entry_stamp_present","document_present","signed","receipt_present"} or name in ATTESTATIONS.get(document.doc_type, ())})
    return {"status": "complete" if all(f["origin"] and f["value"] is not None for f in fields) else "needs_review",
            "fields": fields, "can_confirm": True}


def verification_updates(state):
    records = deepcopy(state.get("verification_records", {}))
    event = state["event"]
    documents = state.get("documents", [])
    evidence_conflicts = set()
    active_ids={d.doc_id for d in documents if d.status=="analyzed"}
    records={key:entries for key,entries in records.items() if key.startswith("__") or key in active_ids}
    if event.kind=="reprocess_document" and event.document_type_hint:
        records.pop(event.document_id,None)
    if event.kind=="remove_correction":
        for name in event.confirmed_fields: records.get(event.document_id,{}).pop(name,None)
    if event.kind == "analyze_document":
        for document in documents:
            if Path(document.file_path).resolve() == Path(event.file_path).resolve():
                records[document.doc_id] = {name:value for name,value in records.get(document.doc_id,{}).items() if VerifiedInput.model_validate(value).origin == "user_confirmed"}
    for document in documents:
        if _typed_info(document) is None:
            records.pop(document.doc_id, None)
            continue
        entries = records.setdefault(document.doc_id, {})
        for name, value in grounded_values(document).items():
            if document.doc_type=="turkiye_entry_stamp" and name=="latest_entry_date": continue  # Observation requires explicit confirmation.
            old = entries.get(name)
            if old and VerifiedInput.model_validate(old).origin == "user_confirmed":
                continue  # Same-document explicit correction overrides its immutable observation.
            entries[name] = VerifiedInput(value=value, origin="source_grounded", document_id=document.doc_id,
                field_name=name, source_page=document.info.source_pages[name],
                source_snippet=document.info.source_snippets[name], trace_id=state["trace_id"])
    if event.kind == "confirm_information" and event.document_id == "__declarations__":
        if set(event.confirmed_fields) - set(CONTEXT_FIELDS):
            raise ValueError("Unsupported declaration field")
        entries = records.setdefault("__declarations__", {})
        for name,value in event.confirmed_fields.items():
            kind=CONTEXT_FIELDS[name]
            if value is not None:
                if kind=="bool" and type(value) is not bool: raise ValueError("Declarations require Yes, No or unknown")
                if kind=="insurance_type" and value not in INSURANCE_TYPES: raise ValueError("Choose an accepted insurance evidence type")
                if kind=="date":
                    from src.document_processing.evidence import normalized_date
                    value=normalized_date(value)
                    if value is None: raise ValueError("Enter a valid ISO date, year 1900..2100")
            entries[name] = VerifiedInput(value=value, origin="user_confirmed", document_id="__declarations__",
                field_name=name, confirmed_by=state["user_id"], trace_id=state["trace_id"])
    elif event.kind == "confirm_information":
        document = next((d for d in documents if d.doc_id == event.document_id), None)
        if document is None or _typed_info(document) is None or document.doc_type not in IMPORTANT_FIELDS:
            raise ValueError("Reprocess a supported document before confirming its information")
        allowed = set(IMPORTANT_FIELDS[document.doc_type]) | set(ATTESTATIONS.get(document.doc_type, ())) | set(OPTIONAL_FIELDS.get(document.doc_type,())) | set(DOC_CONTEXT_FIELDS.get(document.doc_type,{}))
        if document.doc_type == "student_certificate": allowed.update(STUDENT_FIELDS)
        if document.doc_type == "ikamet_application_form": allowed.add("permit_type")
        if set(event.confirmed_fields) - allowed:
            raise ValueError("Unsupported confirmation field")
        raw = document.info.model_dump(mode="json")
        observed = {name:value for name,value in event.confirmed_fields.items() if name not in ATTESTATIONS.get(document.doc_type, ()) and name not in DOC_CONTEXT_FIELDS.get(document.doc_type,{})}
        validation = dict(raw)
        validation.update(observed)
        validation["ambiguous_fields"] = [name for name in raw["ambiguous_fields"] if name not in observed]
        validation["validation_issues"] = []
        validation["source_pages"] = {}
        validation["source_snippets"] = {}
        checked = type(document.info).model_validate(validation)
        checked_values = checked.model_dump(mode="json")
        if checked.PERIOD and any(name in observed for name in checked.PERIOD) and any("precedes" in issue for issue in checked.validation_issues):
            raise ValueError("End date must not precede start date")
        for name, value in event.confirmed_fields.items():
            if value is None or value == "":
                records[document.doc_id][name] = VerifiedInput(value=None, origin="user_confirmed", document_id=document.doc_id,
                    field_name=name, confirmed_by=state["user_id"], trace_id=state["trace_id"])
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
                if accepted is None or any(issue.startswith(name) for issue in checked.validation_issues):
                    raise ValueError(f"{name}: enter a valid value (dates use YYYY-MM-DD)")
            records[document.doc_id][name] = VerifiedInput(value=accepted, origin="user_confirmed",
                document_id=document.doc_id, field_name=name, confirmed_by=state["user_id"], trace_id=state["trace_id"])
    context = ResidencePermitContext.model_validate(state.get("residence_permit_context", {})).model_dump()
    for name in state.get("verified_context_fields", []):
        context[name] = None
    proposals = {}
    for name,entry in records.get("__declarations__", {}).items():
        proposals[name] = [VerifiedInput.model_validate(entry).value]
    records.pop("__derived__", None)
    for document in documents:
        entries = {name:VerifiedInput.model_validate(value) for name,value in records.get(document.doc_id, {}).items()}
        for name, targets in MAPPINGS.get(document.doc_type, {}).items():
            if name in entries and entries[name].value is not None:
                for target in targets:
                    proposals.setdefault(target, []).append(entries[name].value)
        if document.status == "analyzed" and document.doc_type in IMPORTANT_FIELDS and all(name in entries and entries[name].value is not None for name in IMPORTANT_FIELDS[document.doc_type]):
            if document.doc_type == "student_certificate":
                proposals.setdefault("student_certificate_present", []).append(True)
            if document.doc_type == "ikamet_application_form":
                proposals.setdefault("application_form_generated", []).append(True)
    conflicts = []
    for name, values in proposals.items():
        # Only controlled scope and person identity comparisons are normalized.
        # Records retain the original accepted value and source provenance.
        if name == "holder_name":
            cleaned = [clean_name(value) for value in values]
            unique = {value.casefold() for value in cleaned}
            projected = cleaned[0]
        elif name == "institution":
            normalized = [ResidencePermitContext(institution=value).institution for value in values]
            unique = set(normalized)
            projected = normalized[0]
        else:
            unique = set(values)
            projected = values[0]
        context[name] = projected if len(unique) == 1 else None
        if len(unique) > 1:
            conflicts.append(name)
    rows=holder_observations(state,records)
    if len({clean_name(r["value"]).casefold() for r in rows})>1:
        evidence_conflicts.add("holder_name")
    else:
        evidence_conflicts.discard("holder_name")
    for name in sorted(evidence_conflicts):
        context[name] = None
        if name not in conflicts: conflicts.append(name)
    resolutions=deepcopy(state.get("identity_resolutions",{}))
    if event.kind=="resolve_conflict":
        choice=event.confirmed_fields.get("holder_name")
        if not choice or not any(r["value"]==choice for r in rows):
            raise ValueError("Select an existing identity observation for explicit review")
        resolutions["holder_name"]={"value":choice,"signature":holder_signature(rows),"user_id":state["user_id"],"trace_id":state["trace_id"]}
    resolution=resolutions.get("holder_name",{})
    if resolution.get("signature")==holder_signature(rows) and resolution.get("value"):
        context["holder_name"]=clean_name(resolution["value"])
        conflicts=[name for name in conflicts if name!="holder_name"]
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
    review_state = {**state, "verification_records": records}
    needs_review = any(document_review(review_state, d)["status"] != "complete" for d in documents)
    return {"verification_records": records, "identity_resolutions":resolutions, "action_progress":progress, "verified_context_fields": sorted(proposals),
            "verification_conflicts": sorted(conflicts),
            "residence_permit_context": ResidencePermitContext.model_validate(context),
            "run_status": "INFORMATION_NEEDS_CONFIRMATION" if needs_review or conflicts else "DOCUMENT_ANALYZED"}



def milestone_checklist(state):
    """Product support/completeness, never a replacement for the 29 legal checks."""
    results={r.rule_id:r for r in state.get("relevant_requirements") or []}
    context=ResidencePermitContext.model_validate(state.get("residence_permit_context",{}))
    rows=[]
    for kind,(label,rule_ids) in MILESTONE_DOCUMENTS.items():
        documents=[d for d in state.get("documents",[]) if getattr(d,"doc_type",None)==kind and getattr(d,"status",None)=="analyzed"]
        status="missing";doc_ids=[]
        for document in documents:
            doc_ids.append(document.doc_id)
            info=_typed_info(document)
            if info is None: candidate="needs confirmation"
            elif any(field["name"] in info.ambiguous_fields and field["status"]!="user_confirmed" for field in document_review(state,document)["fields"]): candidate="ambiguous"
            elif document_review(state,document)["status"]!="complete": candidate="needs confirmation"
            elif any(results.get(rid) is None or results[rid].passed is not True for rid in rule_ids): candidate="needs confirmation"
            elif kind=="turkiye_entry_stamp" and (context.entry_stamp_present is not True or context.latest_entry_date is None): candidate="needs confirmation"
            elif set(state.get("verification_conflicts",[])) & {target for targets in MAPPINGS.get(kind,{}).values() for target in targets}: candidate="ambiguous"
            else: candidate="complete"
            if candidate=="complete": status=candidate;break
            if status=="missing" or candidate=="ambiguous":status=candidate
        rows.append({"document_type":kind,"label":label,"status":status,"document_ids":doc_ids,"rule_ids":list(rule_ids)})
    return rows



def document_lifecycle(state):
    """Canonical graph updates. Files stay intact; checkpoint history remains audit."""
    event=state["event"]
    document=next((d for d in state.get("documents",[]) if d.doc_id==event.document_id),None)
    if document is None: raise ValueError("Document no longer exists")
    if event.kind=="remove_correction":
        allowed=set(IMPORTANT_FIELDS.get(document.doc_type,()))|set(OPTIONAL_FIELDS.get(document.doc_type,()))|set(ATTESTATIONS.get(document.doc_type,()))|set(STUDENT_FIELDS)|set(DOC_CONTEXT_FIELDS.get(document.doc_type,{}))|{"permit_type"}
        if not event.confirmed_fields or set(event.confirmed_fields)-allowed: raise ValueError("Choose a supported correction")
        return {}
    if event.kind in {"delete_document","replace_document","cancel_document"}:
        if not event.confirmed: raise ValueError("Explicit confirmation is required")
        if event.kind=="replace_document":
            old=[d for d in state.get("documents",[]) if d.doc_type==document.doc_type and d.doc_id!=document.doc_id and d.status=="analyzed"]
            if document.status!="replacement_pending" and not (document.status=="analyzed" and old): raise ValueError("Choose a pending replacement or an existing duplicate version")
            return {"documents":[d.model_copy(update={"status":"deleted"}) for d in old]+[document.model_copy(update={"status":"analyzed"})],"extracted_facts":{d.doc_id:None for d in old}}
        return {"documents":[document.model_copy(update={"status":"deleted"})],"extracted_facts":{document.doc_id:None}}
    if event.kind=="reprocess_document":
        from config import PROJECT_ROOT
        path=Path(document.file_path).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix.lower() not in {".pdf",".txt"}: raise ValueError("Choose an existing local PDF/TXT")
        if event.document_type_hint is not None and event.document_type_hint not in set(PROFILE_TYPES.values())|{"student_enrollment_certificate","health_insurance_certificate","ikamet_application_form"}: raise ValueError("Choose a supported Milestone-1 profile")
        return {"documents":[document.model_copy(update={"status":"pending","processing_error":None})]}
    raise ValueError("Unsupported lifecycle event")
