from src.workers.rule_loader import load_rules
import threading
from config import TWIN_RECORD_PATH
from src.memory.codec import utc_now
from src.memory.twin_store import TwinStore
from src.memory.twin_sync import record_after_invoke
"""Shared presentation adapters; canonical business decisions stay in core/workers."""
from dataclasses import replace
from datetime import date, datetime, timezone
from time import perf_counter
from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4
import streamlit as st
from langgraph.types import Command
from langsmith import tracing_context
from pydantic import BaseModel
from config import MAX_REFLECTION_ATTEMPTS, PROJECT_ROOT
from src.core.state import Document, UserEvent, ProcedureState, new_run
from src.core.llm_diagnostics import collect_model_calls
from src.core.model_provider import ERROR_MESSAGES, use_ai_options
from src.ui.model_settings import cloud_documents_available, last_run_metrics, remember_ai_run, ui_ai_options
from src.core.procedure import (COLLECTION_ITEMS, EIKAMET_FIELD_CATALOGUE, assess_document_collection,
    advance_procedure_if_ready, build_eikamet_preparation, eikamet_field_value, eikamet_preparation_inputs, entry_intent_update)
from src.core.address_proof import ADDRESS_PART_LABELS, REVIEW_LABELS, address_proof_status
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


def upload_path(name: str, content: bytes, thread_id: str, directory: Path = UPLOAD_DIR) -> Path:
    """Compute the existing content-addressed destination without touching the file."""
    extension = Path(name).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("Unsupported file. Choose a PDF or TXT document.")
    if not content:
        raise ValueError("The uploaded document is empty.")
    if not thread_id.strip():
        raise ValueError("Enter a profile name first.")
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT_ROOT):
        raise ValueError("Uploads must stay inside the project.")
    digest = sha256(thread_id.encode() + b"\0" + content).hexdigest()
    return directory / f"{digest}{extension}"


def save_upload(name: str, content: bytes, thread_id: str, directory: Path = UPLOAD_DIR) -> Path:
    """Store uploads under content- and profile-derived names."""
    path = upload_path(name, content, thread_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _already_uploaded(state, path):
    return any(field(document, "file_path") == str(path)
               and field(document, "status") in {"analyzed", "pending", "replacement_pending"}
               for document in state.get("documents") or [])


def submit_document_upload(uploaded, state, graph, config, *, save_file, allow_document_cloud=False):
    if uploaded is None:
        st.warning("Choose a PDF or TXT file first.")
        return
    thread_id = config["configurable"]["thread_id"]
    try:
        content = uploaded.getvalue()
        destination = upload_path(uploaded.name, content, thread_id)
        if _already_uploaded(state, destination):
            st.info("This file is already in Documents. Review the saved copy or use Reprocess if you need a fresh analysis.")
            return
        path = save_file(uploaded.name, content, thread_id)
    except (OSError, ValueError) as error:
        show_error(error, thread_id=thread_id)
        return
    event = UserEvent(kind="analyze_document", file_path=str(path), original_filename=uploaded.name)
    if invoke(graph, ui_run(event, thread_id), config, check_upload=True,
              allow_document_cloud=allow_document_cloud):
        return path
    return None


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
        "TOOL_COMPLETED":"Action prepared successfully.","TOOL_FAILED":"This action needs your attention.","ASSISTANT_RESPONDED":"Ready for your next step",
        "INVALID_INPUT":"Information needs review"}.get(str(status),str(status).replace('_',' ').capitalize())


def requirement_summary(requirements):
    records = display_records(requirements)
    return {label: sum(r.get("status") == status for r in records) for label, status in
            (("PASS", "passed"), ("FAIL", "failed"), ("missing_information", "missing_information"))}


def blocking_requirement_groups(requirements):
    """Display failures separately from unknown blocking inputs; never change rules."""
    records = [r for r in display_records(requirements) if r.get("severity", "blocking") == "blocking"]
    return {"failed": [r for r in records if r.get("status") == "failed"],
            "missing_information": [r for r in records if r.get("status") == "missing_information"]}


def unknown_requirement_groups(state, *, needs=None):
    """Present policy scope separately; retain every UNKNOWN rule and legal status."""
    if needs is None:
        from src.agents.planner import actionable_needs
        needs = actionable_needs(state)
    groups={key:[] for key in ("immediate","future","outside_scope","guardrail_only","waiting","advisory")}
    for need in needs:
        if need["status"]=="missing_information":
            groups[need["category"]].append(need["rule_id"])
    return {key:sorted(set(ids)) for key,ids in groups.items()}


def friendly_error(message: str) -> str:
    """Translate transport/extraction failures for display; leave backend decisions intact."""
    text = message.lower()
    if message in ERROR_MESSAGES.values():
        return message
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


def profile_key(thread_id):
    return sha256(thread_id.encode()).hexdigest()[:16]


def show_error(message, *, thread_id=None):
    st.error(friendly_error(str(message)))
    if thread_id:
        st.session_state[f"_last_ui_error_{profile_key(thread_id)}"] = str(message)


@st.cache_resource
def _run_locks():
    """One lock per profile, shared by every browser tab and device on this server.

    There is one local model. Overlapping runs for a profile queue on it (each
    waits for the others) and leave unfinished checkpoints, so only one runs at a time.
    """
    return {"guard": threading.Lock(), "profiles": {}, "planning": {}}

def profile_lock(thread_id: str) -> threading.Lock:
    locks = _run_locks()
    with locks["guard"]:
        return locks["profiles"].setdefault(thread_id, threading.Lock())

def checkpoint_id(snapshot):
    return (field(snapshot, "config", {}) or {}).get("configurable", {}).get("checkpoint_id")


def _run_detailed_plan(config, expected_checkpoint, resume, shared, profile, lock, request, ai):
    from src.core.graph import build_graph
    from src.memory import checkpointer

    notice = None
    metrics = None
    try:
        # The Streamlit render closes its connection. All execution and mirroring
        # for this request use an independent connection owned by this worker.
        with checkpointer.sqlite_checkpointer() as saver:
            graph = build_graph(saver)
            current = graph.get_state(config)
            if checkpoint_id(current) != expected_checkpoint:
                notice = ("info", "The case changed before planning started. Review the current case and try again.", None)
                return
            if resume:
                if (not current.next or any(task.interrupts for task in current.tasks)
                        or field(current.values.get("event"), "kind") != "generate_plan"):
                    notice = ("info", "Review the current case before retrying the interrupted plan.", None)
                    return
                payload = None
            else:
                if current.next:
                    notice = ("info", "Finish the pending action before preparing a new plan.", None)
                    return
                payload = ui_run(UserEvent(kind="generate_plan"), profile)
            started = utc_now()
            timer = perf_counter()
            try:
                with use_ai_options(ai), collect_model_calls() as calls, tracing_context(enabled=False):
                    for update in graph.stream(payload, config, stream_mode="updates"):
                        if "reflection" in update:
                            with shared["guard"]:
                                request["message"] = "Checking the plan and correcting inconsistencies..."
            finally:
                metrics = last_run_metrics(calls, perf_counter() - timer, operation="generate_plan")
                try:
                    with TwinStore(TWIN_RECORD_PATH) as store:
                        record_after_invoke(store, graph, config, started)
                except Exception as error:
                    notice = ("warning", "Your case is saved. The activity log could not be updated.", str(error))
    except Exception as error:
        notice = ("error", "Plan preparation stopped. Your saved information is available. Review your case and retry.",
                  str(error))
    finally:
        with shared["guard"]:
            request.update(message=None, notice=notice, metrics=metrics)
            if notice is None and metrics is None:
                shared["planning"].pop(profile, None)
            lock.release()


def start_detailed_plan(snapshot, config, *, resume=False):
    profile = config["configurable"]["thread_id"]
    expected_checkpoint = checkpoint_id(snapshot)
    shared = _run_locks()
    lock = profile_lock(profile)
    if not lock.acquire(blocking=False):
        st.info("An action is already running for this case.")
        return False

    ai = ui_ai_options(profile_key(profile))
    request = {"message": "Preparing your detailed plan...", "notice": None}
    with shared["guard"]:
        shared["planning"][profile] = request
    try:
        threading.Thread(target=_run_detailed_plan, name="detailed-plan", daemon=True,
                         args=(config, expected_checkpoint, resume, shared, profile, lock, request, ai)).start()
    except Exception as error:
        with shared["guard"]:
            shared["planning"].pop(profile, None)
            lock.release()
        show_error(error, thread_id=profile)
        return False

    key = profile_key(profile)
    st.session_state.pop(f"plan_notice_{key}", None)
    st.session_state[f"open_plan_result_{key}"] = True
    return True


def planning_message(profile):
    shared = _run_locks()
    with shared["guard"]:
        return shared["planning"].get(profile, {}).get("message")


def render_plan_notice(profile, snapshot):
    key = profile_key(profile)
    revision = checkpoint_id(snapshot)
    shared = _run_locks()
    with shared["guard"]:
        request = shared["planning"].get(profile)
        notice = request["notice"] if request else None
        metrics = request.get("metrics") if request else None
        if notice or metrics:
            shared["planning"].pop(profile)
    if metrics:
        remember_ai_run(key, metrics)
    state_key = f"plan_notice_{key}"
    if notice:
        st.session_state[state_key] = (revision, notice)
        if notice[2]:
            st.session_state[f"_last_ui_error_{key}"] = notice[2]
    if saved := st.session_state.get(state_key):
        if len(saved) != 2 or saved[0] != revision:
            st.session_state.pop(state_key)
        else:
            level, message, _ = saved[1]
            getattr(st, level)(message)


def invoke(graph, payload, config, *, check_upload=False, expected_checkpoint=None, allow_document_cloud=False):
    thread_id = config["configurable"]["thread_id"]
    lock = profile_lock(thread_id)
    if not lock.acquire(blocking=False):
        st.info("Still processing your last action. Wait for it to finish.", icon=":material/hourglass_top:")
        return False
    event = field(payload, "event")
    progress = {
        "analyze_document": "Reading this document locally and checking which fields need your review...",
        "reprocess_document": "Reading this document locally again...",
        "generate_plan": "Preparing and checking your next steps...",
        "assistant_message": "Preparing a reply for your case...",
        "confirm_information": "Saving your answers and checking the requirements...",
        "reevaluate": "Checking your saved information...",
        "run_tool": "Preparing your requested action...",
    }
    try:
        key = profile_key(thread_id)
        ai = ui_ai_options(key)
        if allow_document_cloud:
            kind = field(event, "kind")
            target = field(event, "file_path") if kind == "analyze_document" else field(event, "document_id")
            if kind not in {"analyze_document", "reprocess_document"} or not target or not cloud_documents_available(key):
                st.warning("Cloud document analysis is unavailable. Check the OpenAI model in Case settings.")
                return False
            ai = replace(ai, cloud_document=(kind, target))
        if check_upload or expected_checkpoint is not None:
            current = graph.get_state(config)
        if expected_checkpoint is not None and (current.next or checkpoint_id(current) != expected_checkpoint):
            st.info("Your case changed. Review the latest saved information before saving again.")
            st.rerun()
        if check_upload:
            if current.next:
                st.info("Finish the pending action before adding a document.")
                return False
            if _already_uploaded(current.values, field(event, "file_path")):
                st.info("This file is already in Documents. Review the saved copy or use Reprocess.")
                return False
        started = utc_now()
        timer = perf_counter()
        kind = field(event, "kind")
        retrieval_calls = 0
        message = ("Analyzing the authorized file with OpenAI…" if allow_document_cloud
                   else progress.get(kind, "Updating your case..."))
        with use_ai_options(ai), collect_model_calls() as calls, tracing_context(enabled=False):
            try:
                with st.spinner(message):
                    result = graph.invoke(payload, config)
                if kind == "assistant_message" and isinstance(result, dict):
                    retrieval_calls = result.get("assistant_diagnostics", {}).get("retrieval_calls", 0)
            finally:
                if calls or kind in {"generate_plan", "assistant_message", "analyze_document", "reprocess_document"}:
                    remember_ai_run(key, last_run_metrics(calls, perf_counter() - timer, operation=kind,
                                                         retrieval_calls=retrieval_calls))
                record_run(graph, config, started)
    except Exception as error:
        show_error(error, thread_id=thread_id)
        return False
    finally:
        lock.release()
    return True


@st.fragment(run_every=2)
def wait_for_run(thread_id):
    """Show progress while another tab is processing this profile."""
    if profile_lock(thread_id).locked():
        message = planning_message(thread_id) or "Still processing your last action."
        st.info(f"{message} This page refreshes when it finishes.", icon=":material/hourglass_top:")
    else:
        st.rerun()


def record_run(graph, config, started):
    """Mirror the checkpoint into the twin record. Never blocks the UI: the
    checkpoint is already saved, and the next successful sync heals the record."""
    try:
        with TwinStore(TWIN_RECORD_PATH) as store:
            record_after_invoke(store, graph, config, started)
    except Exception as error:
        st.warning("Your saved case is available. The activity log could not be updated.")
        thread_id = config["configurable"]["thread_id"]
        st.session_state[f"_last_ui_error_{profile_key(thread_id)}"] = str(error)



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


def user_text(text, *, keep_lines=False):
    """Present student-facing prose while keeping the saved message intact."""
    value = str(text).replace(
        "Use Generate a stage plan below; Gate 1 and your approval remain required.",
        "Open Next steps to prepare a plan for you to review and approve.",
    )
    value = re.sub(r'Milestone[- ]?1 document preparation is complete',
                   'Your preparation documents are ready', value, flags=re.I)
    value = re.sub(r'Milestone[- ]?1|\bV2\b', '', value, flags=re.I)
    for raw in ('HUMAN_REVIEW_REQUIRED', 'AWAITING_TOOL_APPROVAL', 'TOOL_COMPLETED',
                'TOOL_FAILED', 'LOCAL_ICS', 'LOCAL_SIMULATION'):
        value = value.replace(raw, {'LOCAL_ICS': 'calendar download',
                                    'LOCAL_SIMULATION': 'draft for your email app'}.get(raw, status_label(raw)))
    value = re.sub(r'\bR\d{3}\b[ \t]*[:,-]?', '', value)
    if keep_lines:
        return value.strip()
    return re.sub(r'\s+', ' ', value).strip()


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
        return {"replacement_pending":"Replacement awaiting decision","rejected":"Analysis failed","pending":"Analyzing"}.get(status,str(status or "Needs review").replace("_"," ").capitalize())
    mapping={"passport_copy":"passport_identity","turkiye_entry_stamp":"latest_entry","student_certificate":"student_certificate","health_insurance":"health_insurance"}
    key=mapping.get(field(document,"doc_type"))
    if state.get("procedure") is not None and key:
        result=assess_document_collection(state).prerequisites.get(key)
        return {"READY":"Ready for this step","MISSING":"Missing information","NEEDS_CONFIRMATION":"Needs review","BLOCKED":"Needs review","CONFLICT":"Conflicting information"}.get(result,"Needs review")
    labels = {"complete": "Ready for this step", "needs_review": "Needs review", "conflict": "Conflicting information"}
    return labels.get(document_card(state, document)["status"], "Needs review")

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

DOC_LABELS = {**{key: value[0] for key, value in MILESTONE_DOCUMENTS.items()},
              **ADDRESS_PART_LABELS, "tax_id_certificate": "Tax ID assignment certificate",
              "unknown": "Unrecognized document"}
FIELD_LABELS={'holder_name':'Holder name','applicant_name':'Applicant name','student_id':'Student ID','issue_date':'Issue date','institution':'University',
    'passport_or_foreign_id':'Passport / foreign ID','passport_number':'Passport number','foreigner_id':'Foreigner ID (YKN)',
    'nationality':'Nationality','expiry_date':'Passport expiry date',
    'policy_number':'Policy number','insurer':'Insurer','coverage_start_date':'Coverage start date','coverage_end_date':'Insurance expiry date','insurance_evidence_type':'Insurance category',
    'first_name':'First name','last_name':'Surname','birth_date':'Birth date','phone':'Phone number','declared_address':'Declared address','service_address':'Service address',
    'signature_present':'Signed Address Form','form_date':'Form date','billing_date':'Billing date','provider':'Utility provider','permit_type':'Permit type',
    'exchange_student':'Exchange student status','program_start_date':'Exchange start date','program_end_date':'Exchange end date','latest_entry_date':'Date of entry into Türkiye',
    'identity_page_present':'Readable passport identity page','entry_stamp_present':'Entry stamp visible','application_form_signed':'Signed application form',
    'online_application_completed':'Real online application completed','document_present':'Document available','receipt_present':'Payment receipt available','signed':'Signed document',
    'application_number':'Application number','requested_start_date':'Requested start date','requested_end_date':'Requested end date','legal_stay_expiry_date':'Legal stay expiry date',
    'passport_expiry_date':'Passport expiry date','requested_permit_end_date':'Requested permit end date','insurance_end_date':'Insurance end date','residential_address':'Residence address',
    'insurance_policy_number':'Policy number','insurance_insurer':'Insurer','passport_issue_date':'Passport issue date','passport_issuing_country':'Passport issuing country',
    'passport_issuing_authority':'Passport issuing authority','gender':'Gender','turkish_phone':'Contact phone','reason_of_stay':'Reason for stay','gi_number':'GI number'}


FIELD_LABELS.update(REVIEW_LABELS, host_name="Host's name", landlord_name="Landlord's name",
                    resident_name="Student's name", payer_name="Payer or taxpayer name",
                    tax_identification_number="Tax identification number", taxpayer_id="Taxpayer identifier on receipt",
                    payment_code="Payment code", amount="Paid amount", reference_number="Receipt reference")


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


def editor_draft(scope):
    """Keep pending values in session state outside Streamlit's widget cleanup."""
    return st.session_state.setdefault(f"draft_{scope}", {})


def draft_widget(scope, draft, name, default):
    """Initialize a widget before rendering it and capture each committed edit."""
    key = f"draft_input_{scope}_{name}"
    value = draft.get(name, default)
    if value == default:
        draft.pop(name, None)
    if key not in st.session_state or st.session_state[key] != value:
        st.session_state[key] = value

    def remember():
        changed = st.session_state[key]
        if changed == default:
            draft.pop(name, None)
        else:
            draft[name] = changed

    return {"key": key, "on_change": remember}


def request_view(thread_key, view, document_id=None, *, document_type=None, detail_section=None):
    st.session_state[f"requested_view_{thread_key}"] = view
    if view == "documents":
        if document_id:
            st.session_state[f"review_document_{thread_key}"] = document_id
        if document_type:
            st.session_state[f"review_type_{thread_key}"] = document_type
    if view == "details" and detail_section:
        st.session_state[f"requested_details_{thread_key}"] = detail_section


def workflow_summary(state):
    from src.core.procedure import iro_package_readiness, stage_document_checklist, user_step

    step = user_step(state)
    stage = field(state.get("procedure"), "current_stage")
    collection = assess_document_collection(state) if stage == "DOCUMENT_COLLECTION" else None
    package = iro_package_readiness(state) if stage == "IRO_REVIEW" else None
    rows = package["items"] if package is not None else stage_document_checklist(state, collection=collection)
    ready = sum(row["status"] == "complete" for row in rows)
    fraction = ready / len(rows) if rows else None
    progress = f"{ready} of {len(rows)} items ready for this step" if rows else None
    if package is not None:
        progress = f"IRO package: {ready} of {len(rows)} ready"
    preparation_inputs = None
    if stage in {"EIKAMET_PREPARATION", "EIKAMET_READY"} and field(state.get("procedure"), "eikamet_path") == "ASSISTED":
        from src.tools.sandbox import validate_form

        preparation_inputs = eikamet_preparation_inputs(state)
        preparation, _ = preparation_inputs
        fraction = preparation.prepared_field_count / len(preparation.fields)
        progress = f"{preparation.prepared_field_count} of {len(preparation.fields)} preparation values accepted"
        if field(state.get("sandbox_form"), "simulated_application_ready") and validate_form(state).valid:
            progress = "Simulation complete. The real application is still needed."
    if stage == "COMPLETE":
        progress = "Your reported procedure outcome is recorded"
    return {
        **step, "stage": stage, "step_text": f"Step {step['number']} of 5 - {step['label']}",
        "progress_text": progress, "progress_fraction": fraction, "items": rows, "ready_count": ready,
        "package": package, "preparation_inputs": preparation_inputs, "collection": collection,
    }


def conflict_messages(state):
    """Describe current comparisons using the same evidence as verification."""
    from src.workers.verification import accepted_sources, cached_conflict_fields, comparison_key

    sources = accepted_sources(state)
    current = conflict_details(state, sources=sources)
    messages = []
    names = sorted(set(current) | cached_conflict_fields(state, sources=sources),
                   key=lambda name: (name != "holder_name", name))
    for name in names:
        detail = current.get(name, {})
        if detail.get("resolved"):
            continue
        rows = detail.get("observations", [])
        observations = [{**row, "label": document_label(row["document_type"]) if row["document_id"] else "Your details"}
                        for row in rows]
        label = "Passport number" if name == "passport_or_foreign_id" else field_label(name)
        review = next((row for row in observations if row["document_id"]), None)
        message = f"The {label.lower()} differs between your accepted answers."
        passport = next((row for row in observations if row["document_type"] == "passport_copy"), None)
        base = passport or review
        other = next((row for row in observations if row is not base
                      and comparison_key(name, row["value"]) != comparison_key(name, base["value"])), None) if base else None
        if base and other:
            review = other if other["document_id"] else base
            if name == "holder_name" and other["document_id"]:
                source = "passport identity" if passport else "other document identity"
                message = f"The holder name on {other['label']} does not match your {source}."
            else:
                message = f"{label} conflict between {base['label']} and {other['label']}. Review the original values."
        messages.append({
            "field": name, "document_id": review["document_id"] if review else None,
            "label": "Review " + review["label"] if review else "Review conflicting information",
            "message": message, "observations": observations,
            "related_document_ids": [row["document_id"] for row in observations if row["document_id"]],
        })
    return messages


def address_evidence_action(part):
    missing = part["status"] == "missing"
    kind = part["document_type"]
    upload_labels = {"address_form": "signed Address Form", "utility_bill": "utility bill"}
    label = upload_labels.get(kind, part["label"]) if missing and part["key"] == kind else part["label"]
    return {
        "kind": "navigate", "view": "documents", "document_type": kind,
        "document_id": next(iter(part["document_ids"]), None),
        "label": ("Upload " if missing else "Review ") + label,
        "message": part.get("message") or ("Upload the evidence for your selected address method." if missing
                                           else "Review the outstanding fields against your original document."),
    }



def primary_action(state, *, checklist=None, conflicts=None, collection=None):
    from src.core.procedure import assess_current_stage,application_form_review,stage_document_checklist
    if conflicts is None:
        conflicts=conflict_messages(state)
    if conflicts:return {'kind':'navigate','view':'documents',**conflicts[0]}
    stage=field(state.get('procedure'),'current_stage')
    if stage=='DOCUMENT_COLLECTION':
        mode=field(state.get('procedure_entry_intent'),'mode')
        if mode=='EIKAMET_COMPLETED':return {'kind':'navigate','view':'application','label':'Review your e-Ikamet application form','message':'Upload the form generated by your real online application. We will reconstruct only supported information.'}
        assessment = assess_document_collection(state) if collection is None else collection
        if assessment.ready:return {'kind':'start_application','label':'Continue to e-Ikamet','message':'Your initial documents are ready.'}
        labels={'passport_identity':'passport identity page','latest_entry':'latest entry-stamp page','student_certificate':'student certificate','health_insurance':'health insurance','address_proof':'address proof','biometric_photo':'photo readiness'}
        for key in ('passport_identity','latest_entry','student_certificate','health_insurance','address_proof','biometric_photo'):
            status=assessment.prerequisites[key]
            if status=='READY':continue
            item=COLLECTION_ITEMS[key]
            doc=next((d for d in state.get('documents',[]) if field(d,'doc_type')==item.document_type and field(d,'status')=='analyzed'),None) if item.document_type else None
            if key in {'address_proof','biometric_photo'}:
                if key == "address_proof":
                    address = address_proof_status(state)
                    missing = next((part for part in address["parts"] if part["status"] != "complete"), None)
                    if missing:
                        return address_evidence_action(missing)
                message = ("Choose the evidence you will use to prove your address." if key == "address_proof"
                           else "Tell us whether your biometric photo is ready.")
                return {'kind':'navigate','view':'details','detail_section':'address','label':'Review '+labels[key],'message':message}
            if doc:
                review=document_card(state,doc)
                unresolved = next((row for row in review["fields"] if row["issue"]), None)
                if unresolved:
                    message = unresolved["message"] or f"Check {field_label(unresolved['name']).lower()}."
                    return {"kind": "navigate", "view": "documents", "document_type": item.document_type,
                            "document_id": field(doc, "doc_id"), "label": "Review " + labels[key], "message": message}
                return {"kind": "navigate", "view": "checks", "label": "Check " + labels[key] + " requirements",
                        "message": "The document has been read. Check what the procedure still requires."}
            return {'kind':'navigate','view':'documents','document_type':item.document_type,'label':'Upload '+labels[key],'message':'Use a clear copy so we can read and check the details.'}
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
        rows=stage_document_checklist(state) if checklist is None else checklist
        missing=next((r for r in rows if r['status']!='complete' and r.get('document_type')),None)
        if not missing and stage == "IRO_REVIEW":
            address = address_proof_status(state)
            part = next((row for row in address["parts"] if row["status"] != "complete"), None)
            if part:
                return address_evidence_action(part)
            if not address["ready"]:
                return {"kind": "navigate", "view": "details", "detail_section": "address",
                        "label": "Review address proof", "message": address["message"]}
        if missing:
            label = ("Upload " if missing["status"] == "missing" else "Review ") + missing["label"]
        else:
            label = {
                "PAYMENT": "Review payment evidence", "FINGERPRINT": "Confirm fingerprint / GI progress",
                "IRO_REVIEW": "Review your IRO package", "SUBMISSION": "Record your physical submission",
            }[stage]
        return {
            "kind": "navigate", "view": "documents" if missing else "application", "label": label,
            "document_type": missing["document_type"] if missing else None,
            "message": "Review the information needed for this step. Later checks do not prevent you from completing the current task.",
        }
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
    if isinstance(value,date):
        return value.strftime('%d %b %Y')
    if isinstance(value,str) and str(name).endswith('_date') and re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        try:
            return date.fromisoformat(value).strftime('%d %b %Y')
        except ValueError:
            pass
    controlled={'permit_type','insurance_category','insurance_evidence_type','passport_type','gender','Permit type','Insurance category','Passport type','Gender'}
    names={'short_term':'Short-term residence permit','private_health_insurance':'Private health insurance','sgk_provision':'SGK - manual review','sgk_application':'SGK application - manual review','bilateral_social_security':'Bilateral cover - manual review'}
    if name in controlled:return names.get(value,str(value).replace('_',' ').capitalize())
    return str(value)
