from src.workers.rule_loader import load_rules
import threading
from config import TWIN_RECORD_PATH
from src.memory.codec import utc_now
from src.memory.twin_store import TwinStore
from src.memory.twin_sync import record_after_invoke
"""Shared presentation adapters; canonical business decisions stay in core/workers."""
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4
import streamlit as st
from langgraph.types import Command
from pydantic import BaseModel
from config import MAX_REFLECTION_ATTEMPTS, PROJECT_ROOT
from src.core.state import Document, UserEvent, ProcedureState, new_run
from src.core.procedure import (COLLECTION_ITEMS, EIKAMET_FIELD_CATALOGUE, assess_document_collection,
    advance_procedure_if_ready, build_eikamet_preparation, eikamet_field_value, entry_intent_update)
from src.workers.rules_engine import ACCEPTED_ADDRESS_PROOF_TYPES
from src.workers.verification import (document_review, CONTEXT_QUESTIONS, CONTEXT_FIELDS, INSURANCE_TYPES,
    conflict_details, MILESTONE_SCOPE, MILESTONE_DOCUMENTS, CONTEXT_GROUPS, milestone_checklist)
from src.document_processing.schemas import DOCUMENT_SCHEMAS


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

ENTRY_CHOICES = {
    "START_FROM_SCRATCH": "I'm starting from scratch",
    "HAVE_DOCUMENTS": "I already have some documents",
    "EIKAMET_COMPLETED": "I already completed the e-Ikamet application",
    "RESUME": "Resume an existing case",
}

STAGE_LABELS = {
    "DOCUMENT_COLLECTION": "Documents", "EIKAMET_PREPARATION": "e-Ikamet preparation",
    "EIKAMET_READY": "e-Ikamet data review", "PAYMENT": "Payment", "FINGERPRINT": "Fingerprint",
    "IRO_REVIEW": "IRO review", "SUBMISSION": "Submission", "TRACKING": "Tracking", "COMPLETE": "Complete",
}


def ui_run(event, user_id):
    return new_run(event,user_id,scope=MILESTONE_SCOPE)


def save_upload(name: str, content: bytes, thread_id: str, directory: Path = UPLOAD_DIR) -> Path:
    """Store uploads under content/thread-derived names, never a browser-supplied path."""
    extension = Path(name).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("Unsupported file. Choose a PDF or TXT document.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    if not thread_id.strip():
        raise ValueError("Enter a thread ID first.")
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT_ROOT):
        raise ValueError("Uploads must stay inside the project.")
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256(thread_id.encode() + b"\0" + content).hexdigest()
    path = directory / f"{digest}{extension}"
    path.write_bytes(content)
    return path


def jsonable(value):
    """Serialize canonical state values for display, without a second state model."""
    if isinstance(value, BaseModel):
        # Inspect fields recursively: old checkpoints may contain dicts in
        # model-typed slots, which schema serialization can reject or warn about.
        return {name: jsonable(getattr(value, name, None)) for name in type(value).model_fields}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def field(value, name, default=None):
    """Read either checkpoint representation without modifying it."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def display_record(value):
    """Detached display data, including malformed legacy records."""
    data = jsonable(value)
    return data if isinstance(data, dict) else {"legacy_value": data}


def display_records(values):
    if values is None:
        return []
    return [display_record(value) for value in (values if isinstance(values, (list, tuple)) else [values])]


def observation_view(info):
    """Raw observations only; a dict is labelled legacy even if its fields look valid.

    No automatic migration/reconstruction or reassignment into AgentState occurs.
    """
    return display_record(info), not isinstance(info, tuple(DOCUMENT_SCHEMAS.values()))


def status_label(status):
    return {"NOT_STARTED":"Not started","VALIDATED":"Plan validated","APPROVED":"Proposed plan approved","APPROVAL_DENIED":"Proposed plan denied",
        "RUNNING":"Processing","REPAIRING":"Reviewing proposed next steps","DOCUMENT_ANALYZED":"Document analyzed",
        "INFORMATION_NEEDS_CONFIRMATION":"Information needs your confirmation","REQUIREMENTS_EVALUATED":"Information reviewed",
        "PLANNING_UNAVAILABLE":"Optional planning unavailable; your case is saved","PLAN_INVALID":"Proposed next steps need review",
        "HUMAN_REVIEW_REQUIRED":"We need your help to resolve an issue.","AWAITING_TOOL_APPROVAL":"Your approval is required.",
        "TOOL_COMPLETED":"Action prepared successfully.","TOOL_FAILED":"This action needs your attention.","ASSISTANT_RESPONDED":"Ready for your next step"}.get(str(status),str(status).replace('_',' ').capitalize())


def requirement_summary(requirements):
    records = display_records(requirements)
    return {label: sum(r.get("status") == status for r in records) for label, status in
            (("PASS", "passed"), ("FAIL", "failed"), ("missing_information", "missing_information"))}


def blocking_requirement_groups(requirements):
    """Display failures separately from unknown blocking inputs; never change rules."""
    records = [r for r in display_records(requirements) if r.get("severity", "blocking") == "blocking"]
    return {"failed": [r for r in records if r.get("status") == "failed"],
            "missing_information": [r for r in records if r.get("status") == "missing_information"]}


def unknown_requirement_groups(state):
    """Present policy scope separately; retain every UNKNOWN rule and legal status."""
    from src.agents.planner import actionable_needs
    groups={key:[] for key in ("immediate","future","outside_scope","guardrail_only","waiting","advisory")}
    for need in actionable_needs(state):
        if need["status"]=="missing_information":
            groups[need["category"]].append(need["rule_id"])
    return {key:sorted(set(ids)) for key,ids in groups.items()}


def friendly_error(message: str) -> str:
    """Translate transport/extraction failures for display; leave backend decisions intact."""
    text = message.lower()
    if "openai" in text or "cloud" in text:
        return "Optional cloud processing is unavailable. Review any unresolved document fields, check the provider configuration or retry."
    if "ocr" in text and ("timed out" in text or "timeout" in text):
        return "Page OCR timed out. Try a clearer or shorter scanned PDF, then retry."
    if "timed out" in text or "timeout" in text:
        return "The local model timed out. Try a shorter document or retry when Ollama is ready."
    if "rapidocr" in text or "ocr" in text:
        return "Scanned-PDF OCR could not recover readable text. Try a clearer scan and check the project OCR dependencies and available memory."
    if "no extractable text" in text or "contains no text" in text:
        return "No readable text was recovered. Try a clearer scan, a text-based PDF or a TXT document."
    if "cannot read pdf" in text or "filedataerror" in text:
        return "The PDF could not be read. Check that it opens correctly, then upload it again."
    if "password" in text:
        return "Password-protected PDFs are unsupported. Upload an accessible text-based copy."
    if "unsupported" in text:
        return "Unsupported file. Choose a PDF or UTF-8 TXT document."
    if any(word in text for word in ("ollama", "connect", "offline")):
        return "The local model could not run. Start Ollama and check that the model configured in config.py is installed."
    if "empty" in text:
        return "The uploaded document is empty. Choose a readable PDF or TXT file."
    return "Processing could not finish. Check the document and local model, then retry."


def show_error(message):
    st.error(friendly_error(str(message)))
    st.session_state['_last_ui_error']=str(message)  # Display diagnostics only, shown in Technical.


@st.cache_resource
def _run_locks():
    """One lock per profile, shared by every browser tab and device on this server.

    There is one local model. Overlapping runs for a profile queue on it (each
    waits for the others) and leave unfinished checkpoints, so only one runs at a time.
    """
    return {"guard": threading.Lock(), "profiles": {}}

def profile_lock(thread_id: str) -> threading.Lock:
    locks = _run_locks()
    with locks["guard"]:
        return locks["profiles"].setdefault(thread_id, threading.Lock())

def invoke(graph, payload, config):
    lock = profile_lock(config["configurable"]["thread_id"])
    if not lock.acquire(blocking=False):
        st.info("Still processing your last action. This page updates when it finishes.",
                icon=":material/hourglass_top:")
        return False
    started = utc_now()
    try:
        with st.spinner("Processing with the local agent..."):
            graph.invoke(payload, config)
    except Exception as error:
        # UI boundary only: checkpoints remain the source of truth after failure.
        show_error(error)
        return False
    finally:
        try:
            record_run(graph, config, started)
        finally:
            lock.release()
    return True

@st.fragment(run_every=2)
def wait_for_run(thread_id):
    """Shown instead of the views while another tab or device is running this profile."""
    if profile_lock(thread_id).locked():
        st.info("Still processing your last action. This page refreshes by itself when it finishes.",
                icon=":material/hourglass_top:")
    else:
        st.rerun()

def record_run(graph, config, started):
    """Mirror the checkpoint into the twin record. Never blocks the UI: the
    checkpoint is already saved, and the next successful sync heals the record."""
    try:
        with TwinStore(TWIN_RECORD_PATH) as store:
            record_after_invoke(store, graph, config, started)
    except Exception as error:
        st.warning(f"Your progress is saved, but the twin record could not be updated: {error}")



def user_action(step):
    """Readable wording only; preserve the backend action and reason verbatim."""
    record = display_record(step)
    return " ".join(str((record.get("title") or record.get("action", record.get("legacy_value", "Review this proposed step")))).replace("_", " ").split())


def document_card(state, document):
    """Compact, JSON-free view data. Technical observations stay in expanders."""
    if not isinstance(document, Document):
        return {"status": "needs_review", "fields": [], "can_confirm": False}
    return document_review(state, document)


def interaction_state(snapshot):
    approval = snapshot.next == ("human_approval",) and any(field(task,"interrupts",[]) for task in snapshot.tasks)
    return {"approval_pending":approval,"editing_blocked":bool(snapshot.next),
            "message":"A proposed plan is waiting for your decision. Reject it to edit information, or approve it to record the proposal." if approval else "Unfinished graph processing must be retried before factual edits." if snapshot.next else ""}


def procedure_view(state):
    """Read-only procedure display data, separate from the legacy M1 checklist."""
    raw = state.get("procedure")
    if raw is None:
        return {"legacy": True, "can_start": False, "items": [], "blockers": []}
    try:
        procedure = ProcedureState.model_validate(raw)
    except (ValueError, TypeError):
        return {"legacy": True, "can_start": False, "items": [], "blockers": [],
                "error": "Saved procedure data needs explicit review; it has not been migrated."}
    view = {"legacy": False, "stage": procedure.current_stage, "status": procedure.stage_status,
            "completed_stages": list(procedure.completed_stages), "can_start": False,
            "items": [], "blockers": list(procedure.stage_blockers)}
    if procedure.current_stage == "DOCUMENT_COLLECTION":
        assessment = assess_document_collection(state)
        view["can_start"] = assessment.ready
        view["readiness_status"] = assessment.status
        labels = {"READY": "Ready / verified", "MISSING": "Missing", "NEEDS_CONFIRMATION": "Needs confirmation",
                  "BLOCKED": "Blocked", "CONFLICT": "Conflict / blocked"}
        view["items"] = [{"Prerequisite": item.label, "Status": labels[assessment.prerequisites[key]]}
                         for key, item in COLLECTION_ITEMS.items()]
        view["blockers"] = list(dict.fromkeys([*assessment.blockers, *procedure.stage_blockers]))
    elif procedure.current_stage == "EIKAMET_PREPARATION":
        # Fresh display projection also handles older V2 snapshots without a
        # preparation component. Opening the page never writes a migration.
        preparation = build_eikamet_preparation(state)
        view["preparation_counts"] = {"Prepared fields": preparation.prepared_field_count,
            "Missing fields": len(preparation.missing_field_ids), "Conflicts": len(preparation.conflict_field_ids)}
        view["preparation_fields"] = [{"Field": item.label,
            "Status": preparation.fields[item.field_id].status,
            "Value": eikamet_field_value(state, preparation.fields[item.field_id])}
            for item in EIKAMET_FIELD_CATALOGUE]
    return view


def start_eikamet_preparation(graph, config, *, recorded_at):
    thread_id=config["configurable"]["thread_id"]
    lock=profile_lock(thread_id)
    if not lock.acquire(blocking=False):return False
    started=utc_now()
    try:
        snapshot=graph.get_state(config)
        if snapshot.next:return False
        updates=advance_procedure_if_ready(snapshot.values,recorded_at=recorded_at)
        if not updates:return False
        graph.update_state(config,updates,as_node="respond")
        record_run(graph,config,started)
        return field(updates.get("procedure"),"current_stage")=="EIKAMET_PREPARATION"
    finally:lock.release()


def document_information_routes(state, document):
    info = field(document, "info")
    diagnostics = field(info, "extraction_diagnostics", {}) or {}
    route = diagnostics.get("extraction_route") or {"native_text":"NATIVE_TEXT", "rapidocr":"RAPIDOCR", "multimodal_fallback":"OPENAI_MULTIMODAL"}.get(diagnostics.get("extraction_method"))
    routes = [route] if route else []
    if any(field(entry,"origin")=="user_confirmed" for entry in state.get("verification_records",{}).get(field(document,"doc_id"),{}).values()):
        routes.append("USER_CORRECTION")
    return routes


def needs_onboarding(state):
    procedure = state.get("procedure")
    return bool(procedure is not None and not state.get("procedure_entry_intent")
        and field(procedure,"current_stage")=="DOCUMENT_COLLECTION"
        and not state.get("documents") and not state.get("verification_records"))


def save_entry_intent(graph, config, mode, *, recorded_at):
    def update(state):
        result=entry_intent_update(state,mode,user_id=config['configurable']['thread_id'],trace_id=uuid4().hex,recorded_at=recorded_at)
        return result
    return persist_procedure_choice(graph,config,update)


def user_case_view(state):
    """Small stage-specific display model; no rules, raw evidence or raw blockers."""
    detail = procedure_view(state)
    if detail["legacy"]:
        return {"legacy": True, "title":"Review your saved case", "current":"Saved case", "next":None,
            "task":"Your existing documents are available. No procedure stage was inferred.", "items":[], "blockers":[], "progress":[]}
    stage = detail["stage"]
    stages = list(STAGE_LABELS)
    completed = detail["completed_stages"]
    progress = [{"Stage":label, "Status":"Complete" if key in completed else "Current" if key==stage else "Upcoming"}
                for key,label in STAGE_LABELS.items()]
    view = {"legacy":False, "stage":stage, "current":STAGE_LABELS[stage],
        "next":STAGE_LABELS[stages[stages.index(stage)+1]] if stages.index(stage)+1<len(stages) else None,
        "progress":progress, "can_start":detail["can_start"], "items":[], "blockers":[]}
    if stage == "DOCUMENT_COLLECTION":
        view.update(title="Collect and review your documents", task="Upload the documents you have, then review information that needs confirmation.")
        labels = {"Ready / verified":"Ready", "Missing":"Missing", "Needs confirmation":"Needs review", "Blocked":"Needs review", "Conflict / blocked":"Conflict"}
        view["items"] = [{"Document":item["Prerequisite"], "Status":labels[item["Status"]]} for item in detail["items"]]
        for item in view["items"]:
            if item["Status"] != "Ready":
                action = "Resolve conflicting information for" if item["Status"]=="Conflict" else "Review and confirm" if item["Status"]=="Needs review" else "Provide"
                view["blockers"].append(f"{action} {item['Document'].lower()}.")
        if not view["blockers"] and detail["blockers"]:
            view["blockers"] = ["Review the saved case information before continuing."]
    elif stage == "EIKAMET_PREPARATION":
        view.update(title="Prepare your e-Ikamet application", task="Review the prepared fields and the information still needed for the online application.", counts=detail["preparation_counts"])
        groups = {"Ready from your documents":[], "Confirmed by you":[], "Need your confirmation":[], "Need information from you":[], "Conflicting information":[]}
        for item in detail["preparation_fields"]:
            group = {"VERIFIED":"Ready from your documents", "USER_CONFIRMED":"Confirmed by you", "EXTRACTED":"Need your confirmation", "CONFLICT":"Conflicting information"}.get(item["Status"],"Need information from you")
            row = {"Field":item["Field"]}
            if item["Value"] is not None: row["Value"] = item["Value"]
            groups[group].append(row)
        view["field_groups"] = groups
    else:
        view.update(title=STAGE_LABELS[stage], task="Review the accepted prerequisites and record actual outcomes in e-Ikamet. No official action is performed by the assistant.")
    return view


def technical_case_view(state):
    """Detached diagnostics for demos, using the same canonical state."""
    return jsonable({"procedure":state.get("procedure"), "entry_intent":state.get("procedure_entry_intent"),
        "documents":state.get("documents",[]), "verification_records":state.get("verification_records",{}),
        "extracted_facts":state.get("extracted_facts",{}), "context":state.get("residence_permit_context"),
        "requirements":state.get("relevant_requirements"), "blockers":state.get("blockers",[]),
        "validation_errors":state.get("validation_errors",[]), "reflection":state.get("repair_history",[]),
        "planner_diagnostics":state.get("planner_diagnostics",{})})


def user_text(text):
    """Presentation only: internal statuses, rule IDs and course wording stay Technical."""
    value=str(text)
    value=re.sub(r'Milestone[- ]?1 document preparation is complete','Your preparation documents are ready',value,flags=re.I)
    value=re.sub(r'Milestone[- ]?1|\bV2\b','',value,flags=re.I)
    for raw in ('HUMAN_REVIEW_REQUIRED','AWAITING_TOOL_APPROVAL','TOOL_COMPLETED','TOOL_FAILED','LOCAL_ICS','LOCAL_SIMULATION'):
        value=value.replace(raw,{'LOCAL_ICS':'calendar download','LOCAL_SIMULATION':'draft for your email app'}.get(raw,status_label(raw)))
    return re.sub(r'\s+',' ',re.sub(r'\bR\d{3}\b\s*[:,-]?','',value)).strip()


def dispatch_tool(graph, config, name, arguments):
    """Ask the existing graph host to validate/execute a typed call or interrupt."""
    call={"call_id":uuid4().hex,"name":name,"arguments":arguments}
    return dispatch_tool_call(graph,config,call)


def dispatch_tool_call(graph, config, call):
    """Preserve canonical approved call metadata; the host revalidates freshness."""
    return invoke(graph,ui_run(UserEvent(kind="run_tool",tool_call=jsonable(call)),config["configurable"]["thread_id"]),config)


def pending_tool(snapshot):
    return bool("tool_approval" in snapshot.next)


def sandbox_rows(state):
    form=state.get("sandbox_form") or {}
    fields=field(form,"fields",{}) or {}
    rows=[]
    for definition in EIKAMET_FIELD_CATALOGUE:
        item=fields.get(definition.field_id)
        value=field(item,"value",item if isinstance(item,str) else None)
        rows.append({"Field":definition.label,"Value":value or "Not filled",
                     "Status":field(item,"status","FILLED" if value is not None else "MISSING")})
    return rows


def field_group(field_id):
    if field_id in {"first_name","surname","birth_date","nationality","gender"}:return "Personal information"
    if field_id.startswith("passport_"):return "Passport"
    if field_id in {"email","turkish_phone","turkiye_address"}:return "Contact / address"
    if field_id.startswith("insurance_"):return "Insurance"
    return "Residence request"


def application_rows(state):
    preparation=build_eikamet_preparation(state)
    groups={name:[] for name in ("Personal information","Passport","Contact / address","Residence request","Insurance")}
    labels={"VERIFIED":"From your document","USER_CONFIRMED":"Confirmed by you","MISSING":"Missing","EXTRACTED":"Needs confirmation","CONFLICT":"Conflict"}
    for definition in EIKAMET_FIELD_CATALOGUE:
        item=preparation.fields[definition.field_id]
        groups[field_group(definition.field_id)].append({"Field":definition.label,
            "Value":eikamet_field_value(state,item) or "Not available",
            "Source": "Document" if item.source_type=="DOCUMENT" else "Your confirmation" if item.source_type=="USER" else "Derived" if item.source_type=="DERIVED" else "Not available",
            "Status":labels[item.status]})
    return groups


def document_sections(state):
    """Read-only current/later document organization, never a second checklist."""
    view=user_case_view(state)
    current=view.get("items",[])
    active=state.get("documents",[]) or []
    return {"required_now":current,
        "uploaded":[d for d in active if field(d,"status") not in {"deleted","cancelled"}],
        "needs_review":[d for d in active if document_display_status(state,d) not in {"Ready for this step","Complete"}],
        "later":["Generated e-Ikamet form","Payment receipt","Fingerprint / GI confirmation","IRO review and submission evidence"] if view.get("stage")=="DOCUMENT_COLLECTION" else []}


def tool_preview(state):
    proposal=state.get("pending_tool_proposal") or {}
    observation=field(proposal,"observation",{}) or {}
    call=field(proposal,"tool_call",{}) or {}
    return jsonable({"name":field(call,"name",field(observation,"name","Pending action")),
        "arguments":field(call,"arguments",{}) or {},"data":field(observation,"data",{}) or {},
        "diff":field(proposal,"preview",field(observation,"diff",{})) or {},
        "proposal_id":field(proposal,"proposal_id",field(call,"call_id"))})


def record_items(value):
    """Read list- or map-shaped artifact collections defensively for display."""
    if isinstance(value,dict):return list(value.values())
    return list(value or []) if isinstance(value,(list,tuple)) else []


def document_display_status(state, document):
    """Current-step status from the canonical assessment, not the legacy form."""
    status=field(document,"status")
    if status!="analyzed":
        return {"replacement_pending":"Replacement awaiting decision","rejected":"Processing needs attention","pending":"Awaiting analysis"}.get(status,str(status or "Needs review").replace("_"," ").capitalize())
    mapping={"passport_copy":"passport_identity","turkiye_entry_stamp":"latest_entry","student_certificate":"student_certificate","health_insurance":"health_insurance"}
    key=mapping.get(field(document,"doc_type"))
    if state.get("procedure") is not None and key:
        result=assess_document_collection(state).prerequisites.get(key)
        return {"READY":"Ready for this step","MISSING":"Missing information","NEEDS_CONFIRMATION":"Needs review","BLOCKED":"Needs review","CONFLICT":"Conflicting information"}.get(result,"Needs review")
    return document_card(state,document)["status"].replace("_"," ").capitalize()

def run_event(graph, config, event):
    """Send one UserEvent through the graph exactly as before, then refresh."""
    if invoke(graph, ui_run(event, config["configurable"]["thread_id"]), config):
        st.rerun()

def plan_steps(state):
    steps = state.get("candidate_plan")
    approved = state.get("approved_plan")
    if steps is None and approved:
        steps = field(approved, "actions", [])
    return steps

def render_status_bar(state, snapshot):
    """Preserve Neda's three bordered cards; show stage progress, not rule totals."""
    summary=workflow_summary(state)
    step,progress,status=st.columns(3,border=True)
    step.caption('Current step');step.markdown('**'+summary['step_text']+'**')
    progress.caption('Progress');progress.markdown('**'+summary['progress_text']+'**')
    label=status_label(state.get('run_status','NOT_STARTED'))
    if summary['stage']=='DOCUMENT_COLLECTION' and summary['ready_count']==len(summary['items']) and state.get('run_status') in {'HUMAN_REVIEW_REQUIRED','PLAN_INVALID','PLANNING_UNAVAILABLE'}:label='Documents ready; continue to e-Ikamet'
    status.caption('Agent status');status.markdown('**'+label+'**')



DOC_LABELS={**{key:value[0] for key,value in MILESTONE_DOCUMENTS.items()},'utility_bill':'Utility bill','address_form':'Address Form','unknown':'Unrecognized document'}
FIELD_LABELS={'holder_name':'Holder name','applicant_name':'Applicant name','student_id':'Student ID','issue_date':'Issue date','institution':'University',
    'passport_or_foreign_id':'Passport / foreign ID','passport_number':'Passport number','nationality':'Nationality','expiry_date':'Passport expiry date',
    'policy_number':'Policy number','insurer':'Insurer','coverage_start_date':'Coverage start date','coverage_end_date':'Coverage end date','insurance_evidence_type':'Insurance category',
    'first_name':'First name','last_name':'Surname','birth_date':'Birth date','phone':'Phone number','declared_address':'Declared address','service_address':'Service address',
    'signature_present':'Signed Address Form','form_date':'Form date','billing_date':'Billing date','provider':'Utility provider','permit_type':'Permit type',
    'exchange_student':'Exchange student status','program_start_date':'Exchange start date','program_end_date':'Exchange end date','latest_entry_date':'Latest entry date',
    'identity_page_present':'Readable passport identity page','entry_stamp_present':'Entry stamp visible','application_form_signed':'Signed application form',
    'online_application_completed':'Real online application completed','document_present':'Document available','receipt_present':'Payment receipt available','signed':'Signed document',
    'application_number':'Application number','requested_start_date':'Requested start date','requested_end_date':'Requested end date','legal_stay_expiry_date':'Legal stay expiry date',
    'passport_expiry_date':'Passport expiry date','requested_permit_end_date':'Requested permit end date','insurance_end_date':'Insurance end date','residential_address':'Residence address',
    'insurance_policy_number':'Policy number','insurance_insurer':'Insurer','passport_issue_date':'Passport issue date','passport_issuing_country':'Passport issuing country',
    'passport_issuing_authority':'Passport issuing authority','gender':'Gender','turkish_phone':'Contact phone','reason_of_stay':'Reason for stay','gi_number':'GI number'}


def field_label(name):return FIELD_LABELS.get(str(name),str(name).replace('_',' ').capitalize())


def document_label(kind):return DOC_LABELS.get(kind,(kind or 'Document').replace('_',' ').capitalize())


def persist_procedure_choice(graph,config,update_fn):
    """Use the existing profile guard/sync for an explicit deterministic UI choice."""
    lock=profile_lock(config['configurable']['thread_id'])
    if not lock.acquire(blocking=False):return False
    started=utc_now()
    try:
        snapshot=graph.get_state(config)
        if snapshot.next:return False
        updates=update_fn(snapshot.values)
        if not updates:return False
        graph.update_state(config,updates,as_node='respond')
        record_run(graph,config,started)
        return True
    finally:lock.release()


def request_view(thread_key,view,document_id=None):
    # Navigation intent only; apply before the segmented widget is instantiated.
    st.session_state['requested_view_'+thread_key]=view
    if document_id:st.session_state['review_document_'+thread_key]=document_id


def workflow_summary(state):
    from src.core.procedure import user_step,stage_document_checklist
    step=user_step(state)
    stage=field(state.get('procedure'),'current_stage')
    rows=stage_document_checklist(state)
    ready=sum(row['status']=='complete' for row in rows)
    progress=f'{ready} of {len(rows)} prerequisites ready' if rows else 'Follow the next action below'
    if stage=='EIKAMET_PREPARATION' and field(state.get('procedure'),'eikamet_path')=='ASSISTED':
        preparation=build_eikamet_preparation(state)
        progress=f'{preparation.prepared_field_count} of {len(preparation.fields)} accepted preparation values'
        from src.tools.sandbox import validate_form
        if field(state.get('sandbox_form'),'simulated_application_ready') and validate_form(state).valid:progress='Simulation complete; real application still needed'
    if stage=='COMPLETE':progress='Your reported procedure outcome is recorded'
    return {**step,'stage':stage,'step_text':f"Step {step['number']} of 5 - {step['label']}",'progress_text':progress,'items':rows,'ready_count':ready}


def conflict_messages(state):
    """Specific readable blockers; raw IDs and provenance remain in Technical."""
    messages=[]
    detail=conflict_details(state).get('holder_name')
    if detail and not detail['resolved']:
        rows=detail['observations'];passport=next((r for r in rows if r['document_type']=='passport_copy'),None)
        base=passport or rows[0]
        from src.workers.verification import same_name
        for row in rows:
            if row is base or same_name(row['value'],base['value']):continue
            messages.append({'field':'holder_name','document_id':row['document_id'],'label':'Review '+document_label(row['document_type']),
                'message':f"The holder name on {document_label(row['document_type'])} does not match your {'passport identity' if passport else 'other document identity'}."})
    for name in state.get('verification_conflicts',[]):
        if name=='holder_name' and messages:continue
        owners=[]
        from src.workers.verification import MAPPINGS
        for doc in state.get('documents',[]):
            if any(name in targets for targets in MAPPINGS.get(field(doc,'doc_type'),{}).values()):owners.append(doc)
        labels=', '.join(dict.fromkeys(document_label(field(d,'doc_type')) for d in owners))
        messages.append({'field':name,'document_id':field(owners[0],'doc_id') if owners else None,'label':'Review conflicting information',
            'message':f"The {field_label(name).lower()} differs between your accepted information"+(f" ({labels})." if labels else '.')})
    return messages


def primary_action(state):
    from src.core.procedure import assess_current_stage,application_form_review,stage_document_checklist
    conflicts=conflict_messages(state)
    if conflicts:return {'kind':'navigate','view':'documents',**conflicts[0]}
    stage=field(state.get('procedure'),'current_stage')
    if stage=='DOCUMENT_COLLECTION':
        mode=field(state.get('procedure_entry_intent'),'mode')
        if mode=='EIKAMET_COMPLETED':return {'kind':'navigate','view':'application','label':'Review your e-Ikamet application form','message':'Upload the form generated by your real online application. We will reconstruct only supported information.'}
        assessment=assess_document_collection(state)
        if assessment.ready:return {'kind':'start_application','label':'Continue to e-Ikamet','message':'Your initial documents are ready.'}
        labels={'passport_identity':'passport identity page','latest_entry':'latest entry-stamp page','student_certificate':'student certificate','health_insurance':'health insurance','address_proof':'address proof','biometric_photo':'photo readiness'}
        for key in ('passport_identity','latest_entry','student_certificate','health_insurance','address_proof','biometric_photo'):
            status=assessment.prerequisites[key]
            if status=='READY':continue
            item=COLLECTION_ITEMS[key]
            doc=next((d for d in state.get('documents',[]) if field(d,'doc_type')==item.document_type and field(d,'status')=='analyzed'),None) if item.document_type else None
            if key in {'address_proof','biometric_photo'}:
                if key=='address_proof':
                    from src.core.procedure import address_proof_status
                    address=address_proof_status(state)
                    if address['route']=='address_form_and_utility_bill':
                        missing=next((part for part in address['parts'] if part['status']!='complete'),None)
                        if missing:return {'kind':'navigate','view':'documents','label':('Upload ' if missing['status']=='missing' else 'Review ')+missing['label'],'message':missing['label']+' is still required for your Address Form + Utility bill route.'}
                return {'kind':'navigate','view':'documents','label':'Review '+labels[key],'message':'Complete your supported address-proof information or photo readiness confirmation.'}
            if doc:
                review=document_card(state,doc)
                unresolved=next((r for r in review['fields'] if r['status'] not in {'source_grounded','user_confirmed'}),None)
                extra='Confirm '+field_label(unresolved['name']).lower()+'.' if unresolved else 'Review the document and any conflicting or insufficient information.'
                return {'kind':'navigate','view':'documents','document_id':field(doc,'doc_id'),'label':'Review '+labels[key],'message':extra}
            return {'kind':'navigate','view':'documents','label':'Upload '+labels[key],'message':'This document is needed for your current step.'}
    if stage in {'EIKAMET_PREPARATION','EIKAMET_READY'}:
        if application_form_review(state)['ready']:return {'kind':'review_application','label':'Continue to Payment & Fingerprint','message':'Your uploaded application form is reviewed.'}
        if field(state.get('procedure'),'eikamet_path')=='ASSISTED' and field(state.get('sandbox_form'),'simulated_application_ready'):
            return {'kind':'navigate','view':'application','label':'Review your simulation and complete the real application','message':'Simulation complete. Complete the real online application externally, then upload its generated form.'}
        return {'kind':'navigate','view':'application','label':'Continue your e-Ikamet preparation','message':'Review your real application form or prepare a clearly labelled simulation.'}
    if stage in {'PAYMENT','FINGERPRINT','IRO_REVIEW','SUBMISSION'}:
        assessment=assess_current_stage(state)
        if assessment['ready']:
            label={'PAYMENT':'Continue to Fingerprint / GI','FINGERPRINT':'Prepare your IRO package','IRO_REVIEW':'Continue to submission','SUBMISSION':'Continue to tracking'}[stage]
            return {'kind':'advance','label':label,'message':'The accepted prerequisites for this step are ready.'}
        missing=next((r for r in stage_document_checklist(state) if r['status']!='complete' and r.get('document_type')),None)
        return {'kind':'navigate','view':'documents' if missing else 'application','label':(('Upload ' if missing['status']=='missing' else 'Review ')+missing['label']) if missing else {'PAYMENT':'Review payment evidence','FINGERPRINT':'Confirm fingerprint / GI progress','IRO_REVIEW':'Review your IRO package','SUBMISSION':'Record your physical submission'}[stage],
            'message':'Review the information needed for this step. Later checks do not prevent you from completing the current task.'}
    if stage=='TRACKING':return {'kind':'navigate','view':'application','label':'Review tracking and record an official decision','message':'Report only actual updates from the authority.'}
    return None


def validation_message(issue):
    text=str(issue)
    first=text.partition(':')[0]
    if first in FIELD_LABELS:return 'Please review '+field_label(first).lower()+'. We could not verify it confidently from the document.'
    if 'precedes' in text or 'before start' in text:return 'Please check the start and end dates; the end cannot be before the start.'
    return 'Some document information needs review. Confirm unclear values using the original document.'


def user_value(name,value):
    """Presentation labels only; identifiers, observations and state stay intact."""
    if isinstance(value,bool):return 'Yes' if value else 'No'
    controlled={'permit_type','insurance_category','insurance_evidence_type','passport_type','gender','Permit type','Insurance category','Passport type','Gender'}
    names={'short_term':'Short-term residence permit','private_health_insurance':'Private health insurance','sgk_provision':'SGK - manual review','sgk_application':'SGK application - manual review','bilateral_social_security':'Bilateral cover - manual review'}
    if name in controlled:return names.get(value,str(value).replace('_',' ').capitalize())
    return str(value)
