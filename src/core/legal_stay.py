"""Deterministic stay arithmetic and proposed targets; never writes legal facts."""
from datetime import date,datetime,timedelta
import json
from zoneinfo import ZoneInfo
from pydantic import BaseModel,Field,model_validator
from config import KNOWLEDGE_DIR,LOCAL_TIMEZONE,RECOMMENDED_TARGET_BUFFER_DAYS
from src.core.state import ResidencePermitContext


class PriorStay(BaseModel):
    start: date
    end: date
    @model_validator(mode="after")
    def ordered(self):
        if self.end<self.start:raise ValueError("Previous stay end precedes start")
        return self


class LegalStayAssessment(BaseModel):
    status: str
    estimated_or_calculated_end: date | None = None
    recommended_target: date | None = None
    assumptions: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    confidence_type: str = "INSUFFICIENT_INPUT"
    days_remaining: int | None = None


def local_today():
    return datetime.now(ZoneInfo(LOCAL_TIMEZONE)).date()


def assess_legal_stay(state, *, as_of=None):
    """No nationality-only entry+90 shortcut; estimates never populate R004."""
    today=as_of or local_today()
    c=ResidencePermitContext.model_validate(state.get("residence_permit_context") or {})
    if state.get("verification_conflicts") and set(state["verification_conflicts"]) & {"nationality","latest_entry_date","passport_type","visa_status","legal_stay_expiry_date"}:
        return LegalStayAssessment(status="MANUAL_REVIEW",assumptions=["Resolve conflicting entry/identity inputs first."])
    refs=[]
    if c.legal_stay_expiry_date is not None:
        end=c.legal_stay_expiry_date;kind="DECLARED_LIMIT"
        assumptions=["This is the supplied legal-stay end, not an independently verified official decision."]
        refs=["user:legal_stay_expiry_date"]
    elif c.existing_permit_until is not None:
        end=c.existing_permit_until;kind="DECLARED_PERMIT_LIMIT"
        assumptions=["Existing permit dates require individual review; visa-exempt arithmetic is not applied."]
        refs=["user:existing_permit_until"]
    else:
        if c.latest_entry_date is None or c.latest_entry_date>today:
            return LegalStayAssessment(status="UNKNOWN",assumptions=["Confirm the actual latest entry date."])
        if c.passport_type!="ordinary":
            return LegalStayAssessment(status="MANUAL_REVIEW",assumptions=["Confirm ordinary passport type; other types are not implemented."])
        profiles=json.loads((KNOWLEDGE_DIR/"legal_stay_profiles.json").read_text(encoding="utf-8"))["profiles"]
        profile=next((p for p in profiles if (c.nationality or "").strip().casefold() in p["aliases"] and c.visa_status==p["entry_status"]),None)
        if profile:
            allowance=profile["max_days"];refs=profile["source_ids"]
        elif c.visa_status in {"visa","evisa"} and c.visa_stay_days is not None and c.visa_valid_until is not None:
            allowance=min(90,c.visa_stay_days);refs=["MFA_VISA_GENERAL","user:visa_stay_days","user:visa_valid_until"]
        else:
            return LegalStayAssessment(status="MANUAL_REVIEW",assumptions=["Nationality, actual visa/exemption conditions and permitted duration are not fully supported; ask IRO or the authority."])
        assumptions=["Entry and exit days count as stay days; current stay is assumed continuous.","Declared visa/exemption status must be applicable to the actual visit purpose; this calculator does not establish study eligibility."]
        kind="ESTIMATE" if c.previous_stay_history is None else "CALCULATED_FROM_DECLARED_HISTORY"
        if c.previous_stay_history is None:
            history=[];assumptions.append("Previous 180-day stay history is unknown; estimate assumes no previous stays. Confirm history before relying on the result.")
        else:
            try:
                raw=json.loads(c.previous_stay_history)
                if not isinstance(raw,list) or len(raw)>100:raise ValueError("Expected at most 100 stay intervals")
                history=[PriorStay.model_validate(row) for row in raw]
                if any(row.end>=c.latest_entry_date for row in history):raise ValueError("Previous stays must end before latest entry")
            except ValueError:
                return LegalStayAssessment(status="MANUAL_REVIEW",assumptions=["Invalid previous-stay history; use start/end ISO date pairs before the latest entry."])
        previous=set()
        for row in history:
            # Bound work to the relevant lookback; overlapping records count once.
            first=max(row.start,c.latest_entry_date-timedelta(days=179))
            previous.update(first+timedelta(days=i) for i in range(max(0,(row.end-first).days+1)))
        end=c.latest_entry_date-timedelta(days=1)
        for i in range(allowance):
            day=c.latest_entry_date+timedelta(days=i)
            recent=sum(day-timedelta(days=179)<=old<=day for old in previous)+i+1
            if recent>90 or c.visa_valid_until is not None and day>c.visa_valid_until:break
            end=day
    return LegalStayAssessment(status="EXPIRED" if end<today else "ESTIMATE" if kind=="ESTIMATE" else "ASSESSED",
        estimated_or_calculated_end=end,recommended_target=end-timedelta(days=RECOMMENDED_TARGET_BUFFER_DAYS),
        assumptions=assumptions,source_refs=refs,confidence_type=kind,days_remaining=(end-today).days)


def reminder_candidates(state, *, as_of=None):
    """Few stage-relevant suggestions; never create a calendar event here."""
    today=as_of or local_today();assessment=assess_legal_stay(state,as_of=today)
    c=ResidencePermitContext.model_validate(state.get("residence_permit_context") or {})
    result=[]
    for key,title,day,kind,refs in (("recommended_target","Complete residence-permit preparation",assessment.recommended_target,"AGENT_RECOMMENDED_TARGET",assessment.source_refs),("legal_stay","Review legal-stay limit",assessment.estimated_or_calculated_end,"ESTIMATED_LIMIT" if assessment.confidence_type=="ESTIMATE" else "DECLARED_OR_CALCULATED_LIMIT",assessment.source_refs),("insurance_expiry","Review health-insurance expiry",c.insurance_end_date,"INSURANCE_EXPIRY",["R015"]),("passport_buffer","Review passport validity buffer",c.passport_expiry_date-timedelta(days=60) if c.passport_expiry_date else None,"PASSPORT_BUFFER_REVIEW",["R006"])):
        if day is not None and day>=today:
            result.append({"candidate_id":key,"title":title,"date":day.isoformat(),"kind":kind,"source_refs":refs,"requires_approval":True})
    return result[:3]


def insurance_explanation(state):
    c=ResidencePermitContext.model_validate(state.get("residence_permit_context") or {})
    if c.insurance_end_date and c.program_end_date and c.requested_permit_end_date and c.insurance_end_date>=c.program_end_date and c.insurance_end_date<c.requested_permit_end_date:
        return "Insurance satisfies the current academic-period policy, but expires before the requested permit end. Consider an insurance-expiry reminder."
    return None
