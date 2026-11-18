"""Deterministic pre-e-Ikamet collection only; never a legal approval decision."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.core.state import ProcedureState, StageHistoryEntry, StageStatus, ResidencePermitContext, UserEvent, VerifiedInput
from src.workers.rules_engine import evaluate_requirements
from src.workers.verification import (
    IMPORTANT_FIELDS, declaration_fields, document_review, verification_updates,
)

V2_PRE_EIKAMET_INSURANCE_READINESS = "V2_PRE_EIKAMET_INSURANCE_READINESS"
COLLECTION_DOCUMENTS={"passport_copy","turkiye_entry_stamp","student_certificate","health_insurance"}
RELEVANT_FIELDS={
    "holder_name","passport_or_foreign_id","passport_expiry_date","nationality","passport_id_page_present",
    "turkey_entry_page_present","entry_stamp_present","latest_entry_date","student_id","institution",
    "student_certificate_issue_date","exchange_student","program_start_date","program_end_date",
    "insurance_policy_number","insurance_insurer","insurance_end_date","insurance_evidence_type","insurance_exempt",
    "requested_permit_end_date","address_proof_type","biometric_photo",
}
COLLECTION_RULES={"R008","R014","R015","R016","R017"}
COLLECTION_MARKERS={"missing:passport_identity_evidence","missing:latest_turkiye_entry_evidence",
                    "confirmation:latest_entry_date","missing:address_proof","missing:student_certificate",
                    "missing:health_insurance","missing:biometric_photo","missing:insurance_end_target",
                    "confirmation:review_verified_information"}


class ProcedureAssessment(BaseModel):
    """Transient result: no second requirement store or checkpointed assessment."""
    model_config=ConfigDict(extra="forbid")
    ready: bool
    status: StageStatus
    blockers: list[str] = Field(default_factory=list)
    reason_refs: list[str] = Field(default_factory=list)


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
    """Assess the six intake prerequisites; never write state or use an LLM."""
    try:
        snapshot=_accepted_snapshot(state)
    except (ValueError,TypeError,AttributeError):
        return ProcedureAssessment(ready=False,status="BLOCKED",blockers=["confirmation:review_verified_information"])
    c=snapshot["residence_permit_context"]
    requirements={r.rule_id:r for r in evaluate_requirements(snapshot)}
    blockers=[];reasons=[]
    def require(passed, refs, reason):
        if passed:reasons.append(reason)
        else:blockers.extend(refs)
    passport=_document_ready(snapshot,"passport_copy") and c.passport_id_page_present is True
    entry=_document_ready(snapshot,"turkiye_entry_stamp") and c.turkey_entry_page_present is True and c.latest_entry_date is not None
    require(passport,["R008","missing:passport_identity_evidence"],"R008:passport_identity_evidence")
    require(entry,["R008","missing:latest_turkiye_entry_evidence"],"R008:latest_turkiye_entry_evidence")
    if not entry and c.entry_stamp_present is True and c.latest_entry_date is None:
        blockers.append("confirmation:latest_entry_date")
    require(requirements["R017"].passed is True,["R017","missing:address_proof"],"R017")
    require(_document_ready(snapshot,"student_certificate") and requirements["R014"].passed is True,
            ["R014","missing:student_certificate"],"R014")
    # Explicit V2 policy: exemption OR end covers requested end OR program end.
    # Start date and the general R015 result never participate in this decision.
    exempt=c.insurance_exempt is True
    if exempt:
        reasons.extend((V2_PRE_EIKAMET_INSURANCE_READINESS,"confirmation:insurance_exempt"))
    else:
        policy_fields=("holder_name","insurer","policy_number","coverage_end_date")
        policy_present=_document_ready(snapshot,"health_insurance",policy_fields)
        require(policy_present,["R015","missing:health_insurance"],"evidence:health_insurance")
        require(requirements["R016"].passed is True,["R016"],"R016")
        targets=[d for d in (c.requested_permit_end_date,c.program_end_date) if d is not None]
        if not targets:blockers.append("missing:insurance_end_target")
        require(c.insurance_end_date is not None and any(c.insurance_end_date>=d for d in targets),["R015"],V2_PRE_EIKAMET_INSURANCE_READINESS)
    require(c.biometric_photo is True,["missing:biometric_photo"],"confirmation:biometric_photo")
    conflicts=(set(state.get("verification_conflicts",[]))|set(snapshot.get("verification_conflicts",[]))) & RELEVANT_FIELDS
    blockers.extend("conflict:"+name for name in sorted(conflicts))
    blockers=list(dict.fromkeys(blockers))
    activity=any(getattr(d,"doc_type",None) in COLLECTION_DOCUMENTS for d in state.get("documents",[]))
    accepted=any(getattr(c,name) is not None for name in RELEVANT_FIELDS)
    status="READY" if not blockers else "BLOCKED" if accepted or conflicts else "IN_PROGRESS" if activity else "NOT_STARTED"
    return ProcedureAssessment(ready=not blockers,status=status,blockers=blockers,reason_refs=list(dict.fromkeys(reasons)))


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
        "stage_history":[*procedure.stage_history,entry]})}
