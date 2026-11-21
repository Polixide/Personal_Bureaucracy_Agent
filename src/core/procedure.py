"""Deterministic pre-e-Ikamet collection only; never a legal approval decision."""
from datetime import datetime
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from src.core.state import ProcedureState, StageHistoryEntry, StageStatus, ResidencePermitContext, UserEvent, VerifiedInput, EIkametFieldState, EIkametPreparationState
from src.workers.rules_engine import evaluate_requirements
from src.workers.verification import (
    IMPORTANT_FIELDS, MAPPINGS, declaration_fields, document_review, verification_updates, milestone_checklist,
)

V2_PRE_EIKAMET_INSURANCE_READINESS = "V2_PRE_EIKAMET_INSURANCE_READINESS"
COLLECTION_DOCUMENTS={"passport_copy","turkiye_entry_stamp","student_certificate","health_insurance"}
RELEVANT_FIELDS={
    "holder_name","passport_or_foreign_id","passport_expiry_date","nationality","passport_id_page_present",
    "turkey_entry_page_present","entry_stamp_present","latest_entry_date","student_id","institution",
    "student_certificate_issue_date","exchange_student","program_start_date","program_end_date",
    "insurance_policy_number","insurance_insurer","insurance_end_date","insurance_evidence_type","insurance_exempt",
    "requested_permit_end_date","address_proof_type","biometric_photo","utility_bill_accepted","address_form_accepted",
}
COLLECTION_RULES={"R008","R014","R015","R016","R017"}
COLLECTION_MARKERS={"missing:passport_identity_evidence","missing:latest_turkiye_entry_evidence",
                    "confirmation:latest_entry_date","missing:address_proof","missing:student_certificate",
                    "missing:health_insurance","missing:biometric_photo","missing:insurance_end_target",
                    "confirmation:review_verified_information"}


class CollectionItem(NamedTuple):
    label: str
    document_type: str | None
    context_fields: tuple[str, ...]


COLLECTION_ITEMS = {
    "passport_identity": CollectionItem("Passport identity evidence", "passport_copy",
        ("holder_name", "passport_or_foreign_id", "passport_expiry_date", "nationality", "passport_id_page_present")),
    "latest_entry": CollectionItem("Latest Turkiye entry evidence", "turkiye_entry_stamp",
        ("turkey_entry_page_present", "entry_stamp_present", "latest_entry_date")),
    "address_proof": CollectionItem("Address proof", None, ("address_proof_type","utility_bill_accepted","address_form_accepted")),
    "student_certificate": CollectionItem("Student certificate", "student_certificate",
        ("holder_name", "student_id", "institution", "student_certificate_issue_date", "exchange_student", "program_start_date", "program_end_date")),
    "health_insurance": CollectionItem("Health insurance", "health_insurance",
        ("holder_name", "insurance_policy_number", "insurance_insurer", "insurance_end_date", "insurance_evidence_type", "insurance_exempt", "requested_permit_end_date", "program_end_date")),
    "biometric_photo": CollectionItem("Biometric photo readiness", None, ("biometric_photo",)),
}
CollectionItemStatus = Literal["READY", "MISSING", "NEEDS_CONFIRMATION", "BLOCKED", "CONFLICT"]


class ProcedureAssessment(BaseModel):
    """Transient result: no second requirement store or checkpointed assessment."""
    model_config=ConfigDict(extra="forbid")
    ready: bool
    status: StageStatus
    blockers: list[str] = Field(default_factory=list)
    reason_refs: list[str] = Field(default_factory=list)
    prerequisites: dict[str, CollectionItemStatus] = Field(default_factory=dict)


def _accepted_snapshot(state):
    """Reproject canonical evidence without trusting legacy context flags.

    Only explicit user records survive into this read-only snapshot. Current
    source-grounded values and derived values are rebuilt by the existing
    verifier, so stale source records and arbitrary derived claims cannot pass.
    """
    active={d.doc_id for d in state.get("documents",[]) if getattr(d,"status",None)=="analyzed"}
    declarations=declaration_fields(state)
    records={}
    for owner,entries in state.get("verification_records",{}).items():
        if owner!="__declarations__" and owner not in active:continue
        for field,raw in entries.items():
            entry=VerifiedInput.model_validate(raw)
            if entry.origin!="user_confirmed" or not entry.confirmed_by or not entry.trace_id:
                continue
            if entry.document_id!=owner or entry.field_name!=field:continue
            if owner=="__declarations__" and field not in declarations:continue
            records.setdefault(owner,{})[field]=entry
    snapshot={**state,"verification_records":records,"residence_permit_context":ResidencePermitContext(),
              "verified_context_fields":[],"event":UserEvent(kind="reevaluate"),
              "trace_id":state.get("trace_id") or "procedure-assessment","user_id":state.get("user_id") or "procedure-assessment"}
    updates=verification_updates(snapshot)
    return {**snapshot,**updates}


def _document_ready(snapshot,kind,fields=None):
    for doc in snapshot.get("documents",[]):
        if getattr(doc,"doc_type",None)!=kind or getattr(doc,"status",None)!="analyzed":continue
        rows={row["name"]:row for row in document_review(snapshot,doc)["fields"]}
        required=fields or IMPORTANT_FIELDS[kind]
        if all(name in rows and rows[name]["value"] is not None and rows[name]["status"] in {"source_grounded","user_confirmed"} for name in required):
            return True
    return False


def assess_document_collection(state) -> ProcedureAssessment:
    """Assess intake and its checklist once; never write state or use an LLM."""
    try:
        snapshot = _accepted_snapshot(state)
    except (ValueError, TypeError, AttributeError):
        return ProcedureAssessment(ready=False, status="BLOCKED",
            blockers=["confirmation:review_verified_information"],
            prerequisites={key: "BLOCKED" for key in COLLECTION_ITEMS})
    c = snapshot["residence_permit_context"]
    requirements = {r.rule_id: r for r in evaluate_requirements(snapshot)}
    blockers, reasons = [], []

    def require(passed, refs, reason):
        if passed:
            reasons.append(reason)
        else:
            blockers.extend(refs)

    passport = _document_ready(snapshot, "passport_copy") and c.passport_id_page_present is True
    entry = _document_ready(snapshot, "turkiye_entry_stamp") and c.turkey_entry_page_present is True and c.latest_entry_date is not None
    address = requirements["R017"].passed is True
    student = _document_ready(snapshot, "student_certificate") and requirements["R014"].passed is True
    require(passport, ["R008", "missing:passport_identity_evidence"], "R008:passport_identity_evidence")
    require(entry, ["R008", "missing:latest_turkiye_entry_evidence"], "R008:latest_turkiye_entry_evidence")
    if not entry and c.entry_stamp_present is True and c.latest_entry_date is None:
        blockers.append("confirmation:latest_entry_date")
    require(address, ["R017", "missing:address_proof"], "R017")
    require(student, ["R014", "missing:student_certificate"], "R014")
    # Local V2 policy, unchanged: exemption OR end covers requested/program end.
    # Neither start date nor general R015 decides this procedure prerequisite.
    insurance = c.insurance_exempt is True
    insurance_failed = False
    if insurance:
        reasons.extend((V2_PRE_EIKAMET_INSURANCE_READINESS, "confirmation:insurance_exempt"))
    else:
        policy_present = _document_ready(snapshot, "health_insurance",
            ("holder_name", "insurer", "policy_number", "coverage_end_date"))
        category_ok = requirements["R016"].passed is True
        targets = [d for d in (c.requested_permit_end_date, c.program_end_date) if d is not None]
        coverage_ok = c.insurance_end_date is not None and any(c.insurance_end_date >= d for d in targets)
        require(policy_present, ["R015", "missing:health_insurance"], "evidence:health_insurance")
        require(category_ok, ["R016"], "R016")
        if not targets:
            blockers.append("missing:insurance_end_target")
        require(coverage_ok, ["R015"], V2_PRE_EIKAMET_INSURANCE_READINESS)
        insurance = policy_present and category_ok and coverage_ok
        insurance_failed = (c.insurance_end_date is not None and bool(targets) and not coverage_ok
                            or requirements["R016"].passed is False)
    photo = c.biometric_photo is True
    require(photo, ["missing:biometric_photo"], "confirmation:biometric_photo")
    conflicts = (set(state.get("verification_conflicts", [])) | set(snapshot.get("verification_conflicts", []))) & RELEVANT_FIELDS
    blockers.extend("conflict:" + name for name in sorted(conflicts))
    blockers = list(dict.fromkeys(blockers))
    checks = dict(zip(COLLECTION_ITEMS, (passport, entry, address, student, insurance, photo)))
    failed = {"passport_identity": c.passport_id_page_present is False,
              "latest_entry": c.entry_stamp_present is False or c.turkey_entry_page_present is False,
              "address_proof": requirements["R017"].passed is False,
              "health_insurance": insurance_failed, "biometric_photo": c.biometric_photo is False}
    prerequisites = {}
    for key, item in COLLECTION_ITEMS.items():
        has_observation = any(getattr(d, "doc_type", None) == item.document_type for d in state.get("documents", [])) if item.document_type else False
        prerequisites[key] = ("CONFLICT" if conflicts.intersection(item.context_fields)
            else "READY" if checks[key] else "BLOCKED" if failed.get(key)
            else "NEEDS_CONFIRMATION" if has_observation else "MISSING")
    activity = any(getattr(d, "doc_type", None) in COLLECTION_DOCUMENTS for d in state.get("documents", []))
    accepted = any(getattr(c, name) is not None for name in RELEVANT_FIELDS)
    status = "READY" if not blockers else "BLOCKED" if accepted or conflicts else "IN_PROGRESS" if activity else "NOT_STARTED"
    return ProcedureAssessment(ready=not blockers, status=status, blockers=blockers,
        reason_refs=list(dict.fromkeys(reasons)), prerequisites=prerequisites)


def _collection_blocker(ref):
    return ref in COLLECTION_RULES or ref in COLLECTION_MARKERS or ref.startswith("conflict:") and ref.partition(":")[2] in RELEVANT_FIELDS


def _procedure(state):
    raw=state.get("procedure")
    return ProcedureState.model_validate(raw) if raw is not None else None


def document_collection_update(state):
    """Refresh status/blocker references only, for explicitly initialized V2 threads."""
    procedure=_procedure(state)
    if procedure is None or procedure.current_stage!="DOCUMENT_COLLECTION":return {}
    assessment=assess_document_collection(state)
    blockers=list(dict.fromkeys([ref for ref in procedure.stage_blockers if not _collection_blocker(ref)]+assessment.blockers))
    if procedure.stage_status==assessment.status and procedure.stage_blockers==blockers:return {}
    return {"procedure":procedure.model_copy(update={"stage_status":assessment.status,"stage_blockers":blockers},deep=True)}


def advance_procedure_if_ready(state, *, recorded_at: datetime):
    """Explicit first transition only. Return a graph update, never mutate input.

    recorded_at is the application's recording time, supplied by the caller;
    it is not proof of an official action or a legal application deadline.
    """
    procedure=_procedure(state)
    if procedure is None or procedure.current_stage!="DOCUMENT_COLLECTION":return {}
    assessment=assess_document_collection(state)
    if not assessment.ready:return document_collection_update(state)
    if not state.get("trace_id"):raise ValueError("A procedure transition requires a trace_id")
    if not isinstance(recorded_at,datetime):raise ValueError("Provide the application's recorded_at datetime")
    completed=list(procedure.completed_stages)
    if "DOCUMENT_COLLECTION" not in completed:completed.append("DOCUMENT_COLLECTION")
    entry=StageHistoryEntry(stage="EIKAMET_PREPARATION",status="NOT_STARTED",trace_id=state["trace_id"],
        recorded_at=recorded_at,reason_refs=["completed:DOCUMENT_COLLECTION",*assessment.reason_refs])
    return {"procedure":procedure.model_copy(deep=True,update={"current_stage":"EIKAMET_PREPARATION","stage_status":"NOT_STARTED",
        "completed_stages":completed,"stage_blockers":[ref for ref in procedure.stage_blockers if not _collection_blocker(ref)],
        "stage_history":[*procedure.stage_history,entry], "eikamet_preparation":build_eikamet_preparation(state)})}


class EIkametFieldDefinition(NamedTuple):
    field_id: str
    label: str
    context_field: str | None = None
    document_types: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    conflict_fields: tuple[str, ...] = ()


# Initial preparation targets, not a complete/mandatory official form schema.
# Source: existing knowledge source GOV_SHORT_TERM_FORM_2026 and the local
# Basvuru_Formu-Application_Form.pdf; contact/insurance/period also reuse the
# existing R006/R008/R015/R016/R018 concepts. Unrepresented fields stay missing.
EIKAMET_FIELD_CATALOGUE = (
    EIkametFieldDefinition("first_name", "First name", "first_name", ("passport_copy","address_form"), conflict_fields=("holder_name",)),
    EIkametFieldDefinition("surname", "Surname", "last_name", ("passport_copy","address_form"), conflict_fields=("holder_name",)),
    EIkametFieldDefinition("birth_date", "Birth date", "birth_date", ("passport_copy","address_form")),
    EIkametFieldDefinition("nationality", "Nationality", "nationality", ("passport_copy",)),
    EIkametFieldDefinition("gender", "Gender", "gender"),
    EIkametFieldDefinition("passport_number", "Passport number", "passport_or_foreign_id", ("passport_copy",), ("R008",)),
    EIkametFieldDefinition("passport_type", "Passport type", "passport_type"),
    EIkametFieldDefinition("passport_issue_date", "Passport issue date", "passport_issue_date"),
    EIkametFieldDefinition("passport_expiry_date", "Passport expiry date", "passport_expiry_date", ("passport_copy",), ("R006",)),
    EIkametFieldDefinition("passport_issuing_country", "Passport issuing country", "passport_issuing_country"),
    EIkametFieldDefinition("passport_issuing_authority", "Passport issuing authority", "passport_issuing_authority"),
    EIkametFieldDefinition("email", "Email", "email", rule_ids=("R018",)),
    EIkametFieldDefinition("turkish_phone", "Turkish phone", "turkish_phone", rule_ids=("R018",)),
    EIkametFieldDefinition("turkiye_address", "Turkiye address", "residential_address", ("address_form",), ("R018",)),
    EIkametFieldDefinition("latest_entry_date", "Latest Turkiye entry date", "latest_entry_date", ("turkiye_entry_stamp",), ("R008",)),
    EIkametFieldDefinition("requested_permit_start", "Requested permit start", "requested_start_date", ("ikamet_application_form",), ("R015",)),
    EIkametFieldDefinition("requested_permit_end", "Requested permit end", "requested_permit_end_date", ("ikamet_application_form",), ("R006", "R015")),
    EIkametFieldDefinition("reason_of_stay", "Reason of stay", "reason_of_stay"),
    EIkametFieldDefinition("insurance_category", "Insurance category", "insurance_evidence_type", ("health_insurance",), ("R016",)),
    EIkametFieldDefinition("insurance_end_date", "Insurance end date", "insurance_end_date", ("health_insurance",), ("R015",)),
    # Sabanci short-term guide pp. 3,7,10,14-15: red fields only; p.11 is skipped.
    EIkametFieldDefinition("father_name", "Father's name", "father_name"),
    EIkametFieldDefinition("mother_name", "Mother's name", "mother_name"),
    EIkametFieldDefinition("contact_method", "Preferred contact method", "contact_method"),
    EIkametFieldDefinition("marital_status", "Marital status", "marital_status"),
    EIkametFieldDefinition("birth_country", "Country of birth", "birth_country"),
    EIkametFieldDefinition("address_province", "Province / city", "address_province"),
    EIkametFieldDefinition("address_district", "District", "address_district"),
    EIkametFieldDefinition("address_locality", "Neighborhood / locality", "address_locality"),
    EIkametFieldDefinition("form_current_date", "Form date", "form_current_date"),
    EIkametFieldDefinition("declaration_acknowledged", "Educational declaration acknowledged", "simulation_declaration_acknowledged"),
)


def _input_ref(owner, field):
    escape = lambda value: value.replace("~", "~0").replace("/", "~1")
    return "/verification_records/" + escape(owner) + "/" + escape(field)


def build_eikamet_preparation(state) -> EIkametPreparationState:
    """Read-only projection of accepted facts, referencing their canonical records.

    Never split a holder name, infer contact values from presence flags, use an
    academic period as a requested permit period, or choose conflicting inputs.
    """
    try:
        snapshot = _accepted_snapshot(state)
    except (ValueError, TypeError, AttributeError):
        return EIkametPreparationState(fields={item.field_id: EIkametFieldState(
            field_id=item.field_id, reason_refs=["confirmation:review_verified_information"])
            for item in EIKAMET_FIELD_CATALOGUE})
    context = snapshot["residence_permit_context"].model_dump(mode="json")
    conflicts = set(state.get("verification_conflicts", [])) | set(snapshot.get("verification_conflicts", []))
    docs = {d.doc_id: d for d in snapshot.get("documents", [])}
    fields = {}
    for item in EIKAMET_FIELD_CATALOGUE:
        reasons = ["GOV_SHORT_TERM_FORM_2026", *item.rule_ids]
        relevant = set(item.conflict_fields) | ({item.context_field} if item.context_field else set())
        conflicting = sorted(conflicts & relevant)
        if conflicting:
            fields[item.field_id] = EIkametFieldState(field_id=item.field_id, status="CONFLICT",
                reason_refs=[*reasons, *("conflict:" + name for name in conflicting)])
            continue
        candidates = []
        if item.context_field and context.get(item.context_field) is not None:
            for owner, entries in snapshot["verification_records"].items():
                doc = docs.get(owner)
                if doc is not None and item.document_types and doc.doc_type not in item.document_types:
                    continue
                for name, entry in entries.items():
                    targets = (name,) if owner == "__declarations__" else MAPPINGS.get(getattr(doc, "doc_type", None), {}).get(name, ())
                    if item.context_field not in targets:
                        continue
                    original = state.get("verification_records", {}).get(owner, {}).get(name)
                    if original is None:
                        continue  # A reference must point to an existing canonical record.
                    original = VerifiedInput.model_validate(original)
                    if entry.value is None or original.value != entry.value or original.origin != entry.origin:
                        continue
                    if item.context_field in {'first_name','last_name'}:
                        from src.workers.verification import same_name
                        matches=same_name(entry.value,context[item.context_field])
                    else:matches=str(entry.value)==str(context[item.context_field])
                    if not matches:continue
                    candidates.append((entry.origin != "user_confirmed", owner, name, entry))
        if candidates:
            _, owner, name, entry = sorted(candidates, key=lambda candidate: candidate[:3])[0]
            fields[item.field_id] = EIkametFieldState(field_id=item.field_id,
                status="USER_CONFIRMED" if entry.origin == "user_confirmed" else "VERIFIED",
                source_type={"user_confirmed": "USER", "source_grounded": "DOCUMENT", "derived": "DERIVED"}[entry.origin],
                source_ref=_input_ref(owner, name), reason_refs=reasons)
        else:
            fields[item.field_id] = EIkametFieldState(field_id=item.field_id, reason_refs=reasons)
    return EIkametPreparationState(fields=fields)


def eikamet_field_value(state, prepared_field):
    """Resolve a still-accepted reference; never trust a stale preparation status."""
    current = build_eikamet_preparation(state).fields.get(prepared_field.field_id)
    if current is None or not current.source_ref or current.source_ref != prepared_field.source_ref:
        return None
    _, _, owner, name = current.source_ref.split("/")
    unescape = lambda value: value.replace("~1", "/").replace("~0", "~")
    entry = VerifiedInput.model_validate(state["verification_records"][unescape(owner)][unescape(name)])
    return str(entry.value) if entry.value is not None else None


def procedure_evidence_update(state):
    """Refresh stage-local data on factual events; never advance a stage."""
    procedure = _procedure(state)
    if procedure is None:
        return {}
    if procedure.current_stage == "DOCUMENT_COLLECTION":
        return document_collection_update(state)
    if procedure.current_stage == "EIKAMET_PREPARATION":
        preparation = build_eikamet_preparation(state)
        if preparation != procedure.eikamet_preparation:
            return {"procedure": procedure.model_copy(deep=True, update={"eikamet_preparation": preparation})}
    if procedure.current_stage not in {"DOCUMENT_COLLECTION","EIKAMET_PREPARATION"}:
        assessment=assess_current_stage(state)
        return {"procedure":procedure.model_copy(deep=True,update={"stage_blockers":assessment["blockers"],"stage_status":"READY" if assessment["ready"] else "COMPLETED" if procedure.current_stage=="COMPLETE" else "BLOCKED"})}
    return {}


def entry_intent_update(state, mode, *, user_id, trace_id, recorded_at):
    """Record how a user enters the case without moving or completing any stage."""
    from src.core.state import ProcedureEntryIntent
    return {"procedure_entry_intent": ProcedureEntryIntent(mode=mode, user_id=user_id,
        trace_id=trace_id, recorded_at=recorded_at)}


STAGE_FLOW=("DOCUMENT_COLLECTION","EIKAMET_PREPARATION","EIKAMET_READY","PAYMENT","FINGERPRINT","IRO_REVIEW","SUBMISSION","TRACKING","COMPLETE")
STAGE_CONFIRMATIONS={
    "PAYMENT":("fee_payment_required","single_entry_visa_fee_required","single_entry_visa_fee_receipt_present"),
    "FINGERPRINT":("fingerprint_step_completed","gi_number_present"),
    "IRO_REVIEW":("scans_sent_to_iro","iro_preverification_confirmed"),
    "SUBMISSION":("submitted_to_iro",),
    "TRACKING":("official_decision_received",),
}


def assess_current_stage(state):
    """Procedure prerequisites, separate from legal rule meanings and execution."""
    procedure=_procedure(state)
    if procedure is None:return {"ready":False,"next_stage":None,"blockers":["review:legacy_procedure"]}
    stage=procedure.current_stage
    if stage=="DOCUMENT_COLLECTION":
        result=assess_document_collection(state)
        return {"ready":result.ready,"next_stage":"EIKAMET_PREPARATION","blockers":result.blockers}
    if stage=="COMPLETE":return {"ready":False,"next_stage":None,"blockers":[]}
    snapshot=_accepted_snapshot(state);c=snapshot["residence_permit_context"]
    req={r.rule_id:r for r in evaluate_requirements(snapshot)}
    blockers=[]
    def require(ok,ref):
        if not ok:blockers.append(ref)
    if stage=="EIKAMET_PREPARATION":
        require(_document_ready(snapshot,"ikamet_application_form"),"evidence:generated_application_form")
        require(c.online_application_completed is True,"confirmation:online_application_completed")
        require(c.application_form_signed is True,"confirmation:application_form_signed")
        for rid in ("R004","R006"):
            if req[rid].passed is False:blockers.append(rid)
    elif stage=="EIKAMET_READY":
        require(real_application_ready(state),'evidence:real_application_form')
        require(c.online_application_completed is True,"confirmation:online_application_completed")
    elif stage=="PAYMENT":
        payment=payment_readiness(state)
        blockers.extend(payment['blockers'])
    elif stage=="FINGERPRINT":
        require(real_application_ready(state),'evidence:real_application_form')
        require(req["R022"].passed is True,"R022")
    elif stage=="IRO_REVIEW":
        package=iro_package_readiness(state)
        blockers.extend(package['blockers'])
        require(package['ready'],'evidence:complete_submission_file')
        require(c.scans_sent_to_iro is True and c.iro_preverification_confirmed is True,"R021")
    elif stage=="SUBMISSION":require(c.submitted_to_iro is True,"confirmation:submitted_to_iro")
    elif stage=="TRACKING":
        require(c.official_decision_received is True,"confirmation:official_decision_received")
        require(c.approval_status in {"approved","rejected"},"confirmation:official_decision_status")
    relevant_conflicts=set(state.get("verification_conflicts",[]))|set(snapshot.get("verification_conflicts",[]))
    blockers.extend("conflict:"+name for name in sorted(relevant_conflicts))
    return {"ready":not blockers,"next_stage":STAGE_FLOW[STAGE_FLOW.index(stage)+1],"blockers":blockers}


def advance_current_stage(state, *, recorded_at):
    """One explicit observed transition; never execute or simulate a real action."""
    procedure=_procedure(state)
    if procedure is None:return {}
    if procedure.current_stage=="DOCUMENT_COLLECTION":return advance_procedure_if_ready(state,recorded_at=recorded_at)
    assessment=assess_current_stage(state)
    if not assessment["ready"] or assessment["next_stage"] is None:return {}
    if recorded_at.tzinfo is None:raise ValueError("Stage audit timestamp requires a timezone")
    completed=list(dict.fromkeys([*procedure.completed_stages,procedure.current_stage]))
    if assessment["next_stage"]=="COMPLETE":completed.append("COMPLETE")
    entry=StageHistoryEntry(stage=assessment["next_stage"],status="COMPLETED" if assessment["next_stage"]=="COMPLETE" else "NOT_STARTED",recorded_at=recorded_at,trace_id=state["trace_id"],reason_refs=["user_explicit_transition", "accepted_prerequisites:"+procedure.current_stage])
    return {"procedure":procedure.model_copy(deep=True,update={"current_stage":assessment["next_stage"],"stage_status":entry.status,"completed_stages":completed,"stage_blockers":[],"stage_history":[*procedure.stage_history,entry]})}


# Five presentation steps over the existing nine-stage lifecycle. No legal rule
# or real-world completion is inferred by this mapping.
USER_STEPS=("Documents","e-Ikamet","Payment & Fingerprint","IRO Review","Submission / Tracking")
USER_STEP_INDEX={"DOCUMENT_COLLECTION":0,"EIKAMET_PREPARATION":1,"EIKAMET_READY":1,"PAYMENT":2,"FINGERPRINT":2,"IRO_REVIEW":3,"SUBMISSION":4,"TRACKING":4,"COMPLETE":4}
STAGE_RULE_IDS={
    "DOCUMENT_COLLECTION":{"R001","R003","R008","R009","R013","R014","R015","R016","R017"},
    "EIKAMET_PREPARATION":{"R002","R004","R005","R006","R007","R018"},
    "EIKAMET_READY":{"R007"},"PAYMENT":{"R010","R011"},"FINGERPRINT":{"R022"},
    "IRO_REVIEW":{"R007","R008","R009","R013","R014","R015","R016","R017","R018","R019","R020","R021","R022","R026"},
    "SUBMISSION":{"R023"},"TRACKING":set(),"COMPLETE":set(),
}


def user_step(state):
    procedure=_procedure(state)
    index=USER_STEP_INDEX.get(procedure.current_stage,0) if procedure else 0
    return {"number":index+1,"total":5,"label":USER_STEPS[index],"complete":bool(procedure and procedure.current_stage=="COMPLETE")}


def choose_eikamet_path(state,path):
    if path not in {"UPLOADED_FORM","ASSISTED"}:raise ValueError("Choose an application preparation path")
    procedure=_procedure(state)
    if procedure is None or procedure.current_stage not in {"DOCUMENT_COLLECTION","EIKAMET_PREPARATION","EIKAMET_READY"}:return {}
    if path=='ASSISTED' and procedure.current_stage=='DOCUMENT_COLLECTION':return {}
    return {"procedure":procedure.model_copy(deep=True,update={"eikamet_path":path})}


def application_form_review(state):
    """Compare accepted actual-form observations; never certify a simulation."""
    snapshot=_accepted_snapshot(state)
    docs=[d for d in snapshot.get('documents',[]) if getattr(d,'status',None)=='analyzed' and getattr(d,'doc_type',None)=='ikamet_application_form']
    result={"ready":False,"document_id":None,"matched":[],"accepted":[],"missing":[],"conflicts":list(snapshot.get('verification_conflicts',[])),"confirmations":[]}
    if any(getattr(d,'doc_type',None)=='ikamet_application_form' and getattr(d,'status',None)=='replacement_pending' for d in snapshot.get('documents',[])):
        result['conflicts'].append('pending_application_version')
    if len(docs)!=1:
        result['missing']=['application_form' if not docs else 'select_current_application_form'];return result
    doc=docs[0];result['document_id']=doc.doc_id
    rows={row['name']:row for row in document_review(snapshot,doc)['fields']}
    for name in (*IMPORTANT_FIELDS['ikamet_application_form'],'permit_type'):
        row=rows.get(name,{})
        if row.get('value') is None or row.get('status') not in {'source_grounded','user_confirmed'}:
            result['missing'].append(name);continue
        result['accepted'].append({'field':name,'value':row['value'],'origin':row['origin']})
        target=(MAPPINGS['ikamet_application_form'].get(name) or (name,))[0]
        for other in snapshot.get('documents',[]):
            if other.doc_id==doc.doc_id:continue
            for field,targets in MAPPINGS.get(other.doc_type,{}).items():
                entry=snapshot['verification_records'].get(other.doc_id,{}).get(field)
                if target in targets and entry is not None:
                    from src.workers.verification import same_name
                    equal=same_name(entry.value,row['value']) if target=='holder_name' else str(entry.value).casefold()==str(row['value']).casefold()
                    if equal:result['matched'].append(name)
    c=snapshot['residence_permit_context']
    for name in ('online_application_completed','application_form_signed'):
        if getattr(c,name) is not True:result['confirmations'].append(name)
    rules={r.rule_id:r for r in evaluate_requirements(snapshot)}
    result['conflicts'] += [rid for rid in ('R004','R006') if rules[rid].passed is False]
    result['ready']=not (result['missing'] or result['conflicts'] or result['confirmations'])
    result['matched']=list(dict.fromkeys(result['matched']))
    return result


def continue_after_application_review(state,*,recorded_at):
    """Explicit evidence-based reconstruction, not a Planner/sandbox transition."""
    procedure=_procedure(state)
    if procedure is None or procedure.current_stage not in {'DOCUMENT_COLLECTION','EIKAMET_PREPARATION','EIKAMET_READY'}:return {}
    review=application_form_review(state)
    if not review['ready']:return {}
    collection=assess_document_collection(state)
    intent=state.get('procedure_entry_intent')
    mode=intent.get('mode') if isinstance(intent,dict) else getattr(intent,'mode',None)
    if procedure.current_stage=='DOCUMENT_COLLECTION' and not collection.ready and mode!='EIKAMET_COMPLETED':return {}
    if recorded_at.tzinfo is None:raise ValueError('Recording time requires a timezone')
    completed=list(procedure.completed_stages)
    if collection.ready and 'DOCUMENT_COLLECTION' not in completed:completed.append('DOCUMENT_COLLECTION')
    for stage in ('EIKAMET_PREPARATION','EIKAMET_READY'):
        if stage not in completed:completed.append(stage)
    entry=StageHistoryEntry(stage='PAYMENT',status='NOT_STARTED',trace_id=state['trace_id'],recorded_at=recorded_at,
        reason_refs=['user_explicit_application_review','document:'+review['document_id'],'confirmation:online_application_completed'])
    updated=procedure.model_copy(deep=True,update={'current_stage':'PAYMENT','stage_status':'NOT_STARTED','completed_stages':completed,
        'stage_blockers':[],'stage_history':[*procedure.stage_history,entry],'eikamet_path':'UPLOADED_FORM','reviewed_application_document_id':review['document_id']})
    return {'procedure':updated,'candidate_plan':None,'approved_plan':None,'plan_summary':'','plan_origin':'','validation_errors':[],'run_status':'REQUIREMENTS_EVALUATED'}


def stage_document_checklist(state):
    """Current-stage view of existing assessments, not a second requirement store."""
    procedure=_procedure(state)
    if procedure is None:return milestone_checklist(state)
    stage=procedure.current_stage
    if stage=='DOCUMENT_COLLECTION':
        assessment=assess_document_collection(state)
        codes={'READY':'complete','MISSING':'missing','NEEDS_CONFIRMATION':'needs confirmation','BLOCKED':'needs confirmation','CONFLICT':'ambiguous'}
        return [{'key':key,'label':item.label,'document_type':item.document_type,'status':codes[assessment.prerequisites[key]]} for key,item in COLLECTION_ITEMS.items()]
    if stage in {'EIKAMET_PREPARATION','EIKAMET_READY'}:
        return [row for row in milestone_checklist(state) if row['document_type']=='ikamet_application_form']
    if stage=='PAYMENT':
        rows=[row for row in milestone_checklist(state) if row['document_type']=='card_fee_receipt']
        c=_accepted_snapshot(state)['residence_permit_context']
        if c.single_entry_visa_fee_required is True:rows.append({'label':'Additional visa-fee evidence','document_type':None,'status':'complete' if c.single_entry_visa_fee_receipt_present is True else 'needs confirmation'})
        return rows
    if stage=='FINGERPRINT':
        c=_accepted_snapshot(state)['residence_permit_context']
        return ([{'label':'Real application form review','document_type':'ikamet_application_form','status':'needs confirmation'}] if not real_application_ready(state) else [])+[{'label':'Fingerprint / GI confirmation','document_type':None,'status':'complete' if real_application_ready(state) and c.fingerprint_step_completed is True and c.gi_number_present is True else 'needs confirmation'}]
    if stage=='IRO_REVIEW':
        return iro_package_readiness(state)['items']
    return []



def start_guided_preparation(state,*,recorded_at):
    updates=advance_procedure_if_ready(state,recorded_at=recorded_at)
    if updates and updates['procedure'].current_stage=='EIKAMET_PREPARATION':
        updates.update(candidate_plan=None,approved_plan=None,plan_summary='',plan_origin='',validation_errors=[],run_status='REQUIREMENTS_EVALUATED')
    return updates


def checked_form_answers(state,answers):
    """Reuse typed canonical validation before a UI confirmation is written."""
    from src.workers.verification import FORM_CONTEXT_FIELDS
    from src.document_processing.evidence import normalized_date
    if set(answers)-set(FORM_CONTEXT_FIELDS):raise ValueError('Unsupported preparation answer')
    normalized={}
    for name,value in answers.items():
        if FORM_CONTEXT_FIELDS[name]=='date' and value is not None:
            value=normalized_date(value)
            if value is None:raise ValueError('Enter a valid date as YYYY-MM-DD')
        normalized[name]=value
    if answers.get('reason_of_stay') is not None and not supported_exchange_reason(answers['reason_of_stay']):raise ValueError('Choose the supported student-exchange reason of stay')
    current=ResidencePermitContext.model_validate(state.get('residence_permit_context') or {}).model_dump()
    validated=ResidencePermitContext.model_validate({**current,**normalized})
    for first,last in (('requested_start_date','requested_permit_end_date'),('passport_issue_date','passport_expiry_date')):
        a,b=getattr(validated,first),getattr(validated,last)
        if a and b and b<a:raise ValueError('End date must not be before start date')
    if validated.birth_date and validated.birth_date>datetime.now().date():raise ValueError('Birth date cannot be in the future')
    if validated.first_name and validated.last_name and validated.holder_name:
        from src.workers.verification import same_name
        if not same_name(validated.first_name+' '+validated.last_name,validated.holder_name):raise ValueError('Name differs from the accepted holder identity; review the source documents')
    return {name:validated.model_dump(mode='json')[name] for name in normalized}


# Explicit educational defaults; none is a verified fact or official declaration.
EIKAMET_GUIDE_URL='https://iro.sabanciuniv.edu/sites/iro.sabanciuniv.edu/files/2024-09/SHORT%20TERM%20PERMIT%20APPLICATION.pdf'
EDUCATIONAL_CHOICES={'Application relationship':'I am making my own application','Income selection':'No income / Geliri Yok (educational guide choice)','Reason of stay (guide choice)':'Education within student exchange programmes / ??renci De?i?im Programlar? ?er?evesinde E?itim'}


def address_proof_status(state):
    """Explain the existing R017 alternatives, without creating acceptance flags."""
    snapshot=_accepted_snapshot(state);c=snapshot['residence_permit_context']
    accepted={'dorm_certificate':'Dorm certificate','rental_agreement':'Rental agreement','notarized_commitment':'Notarized undertaking'}
    route=c.address_proof_type
    docs=[d for d in snapshot.get('documents',[]) if getattr(d,'status',None)=='analyzed']
    parts=[]
    if route in accepted:
        ready=next(r for r in evaluate_requirements(snapshot) if r.rule_id=='R017').passed is True
        parts=[{'label':accepted[route],'document_type':None,'status':'complete' if ready else 'needs confirmation','document_ids':[]}]
        return {'route':route,'label':accepted[route],'ready':ready,'parts':parts}
    for kind,label,flag in (('address_form','Address Form','address_form_accepted'),('utility_bill','Utility bill','utility_bill_accepted')):
        present=[d for d in docs if d.doc_type==kind]
        parts.append({'label':label,'document_type':kind,'document_ids':[d.doc_id for d in present],
            'status':'complete' if getattr(c,flag) is True else 'needs confirmation' if present else 'missing'})
    return {'route':'address_form_and_utility_bill' if any(row['document_ids'] for row in parts) else None,
        'label':'Address Form + Utility bill','ready':all(row['status']=='complete' for row in parts),'parts':parts}


def real_application_ready(state):
    return application_form_review(state)['ready']


def payment_readiness(state):
    snapshot=_accepted_snapshot(state);c=snapshot['residence_permit_context']
    receipt=next((row for row in milestone_checklist(snapshot) if row['document_type']=='card_fee_receipt'),None)
    receipt_ready=bool(receipt and receipt['status']=='complete')
    blockers=[]
    if not real_application_ready(state):blockers.append('evidence:real_application_form')
    if not receipt_ready:blockers.append('evidence:card_fee_receipt')
    # UNKNOWN applicability remains UNKNOWN in R011; the normal scoped flow
    # asks about an extra fee only when there is an explicit positive condition.
    if c.single_entry_visa_fee_required is True and c.single_entry_visa_fee_receipt_present is not True:blockers.append('R011')
    return {'ready':not blockers,'receipt_ready':receipt_ready,'additional_fee_required':c.single_entry_visa_fee_required is True,'blockers':blockers}


def iro_package_readiness(state):
    """One source of truth for package UI, attachments, email, plans and gates."""
    snapshot=_accepted_snapshot(state);c=snapshot['residence_permit_context']
    items=milestone_checklist(snapshot)
    address=address_proof_status(state)
    items += [{'key':'address_proof','label':'Address proof','document_type':None,'status':'complete' if address['ready'] else 'needs confirmation','parts':address['parts']},
        {'key':'biometric_photo','label':'Biometric photo readiness','document_type':None,'status':'complete' if c.biometric_photo is True else 'needs confirmation'}]
    missing=[row for row in items if row['status']!='complete']
    blockers=['package:'+row.get('key',row.get('document_type','information')) for row in missing]
    if not real_application_ready(state):blockers.append('evidence:real_application_form')
    if c.fingerprint_step_completed is not True or c.gi_number_present is not True:blockers.append('confirmation:fingerprint_gi')
    if snapshot.get('verification_conflicts'):blockers.extend('conflict:'+key for key in snapshot['verification_conflicts'])
    eligible={row['document_type'] for row in items if row.get('document_type') and row['status']=='complete'}|{part['document_type'] for part in address['parts'] if part.get('document_type') and part['status']=='complete'}
    attachment_ids=[d.doc_id for d in snapshot.get('documents',[]) if d.status=='analyzed' and d.doc_type in eligible]
    return {'ready':not blockers,'ready_count':len(items)-len(missing),'total':len(items),'items':items,'missing':missing,'blockers':blockers,'attachment_ids':attachment_ids}



def case_summary(state):
    """Compact read observation, derived only from current accepted case state."""
    from src.workers.verification import MILESTONE_DOCUMENTS,conflict_details
    snapshot=_accepted_snapshot(state);c=snapshot['residence_permit_context']
    stage=_procedure(state);collection=assess_document_collection(state)
    package=iro_package_readiness(state)
    rows=stage_document_checklist(state)
    docs=[]
    for doc in snapshot.get('documents',[]):
        if doc.status!='analyzed':continue
        review=document_review(snapshot,doc)
        docs.append({'document_id':doc.doc_id,'type':doc.doc_type,'label':MILESTONE_DOCUMENTS.get(doc.doc_type,(doc.doc_type.replace('_',' ').title(),))[0],
            'status':review['status'],'unresolved_fields':[row['name'] for row in review['fields'] if row['status'] not in {'source_grounded','user_confirmed'}]})
    conflicts=[]
    detail=conflict_details(snapshot).get('holder_name')
    if detail and not detail['resolved']:
        conflicts.append({'field':'holder_name','documents':[row['document_type'] for row in detail['observations']],
            'message':'The holder names on the accepted documents do not match. Review the originals and explicitly resolve the identity conflict.'})
    conflicts.extend({'field':name,'message':name.replace('_',' ').capitalize()+' differs between accepted inputs.'} for name in snapshot.get('verification_conflicts',[]) if name!='holder_name')
    current=assess_current_stage(state)
    next_actions=[]
    if conflicts:next_actions=[{'id':'review_conflict','label':'Review the conflicting documents','kind':'review'}]
    elif stage and stage.current_stage=='DOCUMENT_COLLECTION' and collection.ready:
        next_actions=[{'id':'continue_eikamet','label':'Continue to e-Ikamet','kind':'explicit_transition'}]
    else:
        for row in rows:
            if row['status']=='complete':continue
            if row.get('key')=='address_proof' or row.get('label')=='Address proof':
                address=address_proof_status(state)
                parts=[part for part in address['parts'] if part['status']!='complete']
                next_actions += [{'id':'address_'+str(part.get('document_type') or 'confirmation'),'label':('Upload ' if part['status']=='missing' else 'Review ')+part['label'],'kind':'document'} for part in parts]
            else:next_actions.append({'id':row.get('key') or row.get('document_type') or 'current_step','label':('Upload ' if row['status']=='missing' else 'Review ')+row['label'],'kind':'document' if row.get('document_type') else 'confirmation'})
        if not next_actions and 'confirmation:fingerprint_gi' in package['blockers'] and stage and stage.current_stage=='IRO_REVIEW':
            next_actions=[{'id':'confirm_fingerprint','label':'Record actual fingerprint / GI progress','kind':'confirmation'}]
        if not next_actions and current.get('ready') and current.get('next_stage'):
            next_actions=[{'id':'continue_stage','label':'Continue to '+USER_STEPS[USER_STEP_INDEX[current['next_stage']]],'kind':'explicit_transition'}]
    requirements={r.rule_id:r for r in evaluate_requirements(snapshot)}
    return {'stage':stage.current_stage if stage else None,'user_step':user_step(state),
        'completed_stages':list(stage.completed_stages) if stage else [],'stage_readiness':current,
        'collection':{'ready':collection.ready,'ready_count':sum(value=='READY' for value in collection.prerequisites.values()),'total':len(COLLECTION_ITEMS)},
        'current_items':rows,'documents':docs,'conflicts':conflicts,'address_proof':address_proof_status(state),
        'insurance':{'ready':collection.prerequisites['health_insurance']=='READY','coverage_status':requirements['R015'].status,'category_status':requirements['R016'].status},
        'real_application_reviewed':real_application_ready(state),
        'iro_package':{key:value for key,value in package.items() if key!='attachment_ids'},
        'verified_facts':{key:value for key,value in c.model_dump(mode='json').items() if value is not None},
        'next_actions':next_actions[:5]}


EXCHANGE_REASON_VALUES=frozenset({'exchange_student_education','incoming exchange studies','education within student exchange programmes','education within student exchange programs','ogrenci degisim programlari cercevesinde egitim','??renci de?i?im programlar? ?er?evesinde e?itim'})
EXCHANGE_REASON_LABEL='Education within student exchange programmes / ??renci De?i?im Programlar? ?er?evesinde E?itim'


def supported_exchange_reason(value):
    return isinstance(value,str) and ' '.join(value.casefold().split()) in EXCHANGE_REASON_VALUES
