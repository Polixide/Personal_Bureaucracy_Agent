"""Local presentation layer for the existing SQLite-backed Milestone-1 graph."""
from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st
from langgraph.types import Command
from pydantic import BaseModel

from config import MAX_REFLECTION_ATTEMPTS, PROJECT_ROOT, TWIN_RECORD_PATH
from src.core.graph import build_graph
from src.core.state import Document, UserEvent, new_run
from src.workers.verification import document_review, CONTEXT_QUESTIONS, CONTEXT_FIELDS, INSURANCE_TYPES, conflict_details, MILESTONE_SCOPE, MILESTONE_DOCUMENTS, CONTEXT_GROUPS, milestone_checklist
from src.memory.checkpointer import sqlite_checkpointer, reset_thread
from src.memory.codec import utc_now
from src.memory.twin_store import TwinStore
from src.memory.twin_sync import record_after_invoke
from src.document_processing.schemas import DOCUMENT_SCHEMAS


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


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
    return {"NOT_STARTED": "Not started", "VALIDATED": "Plan validated",
            "APPROVED": "Proposed plan approved", "APPROVAL_DENIED": "Proposed plan denied",
            "RUNNING": "Processing", "REPAIRING": "Repairing plan",
            "DOCUMENT_ANALYZED": "Document analyzed; requirements evaluated",
            "INFORMATION_NEEDS_CONFIRMATION": "Information needs confirmation; requirements evaluated",
            "REQUIREMENTS_EVALUATED": "Requirements evaluated",
            "PLANNING_UNAVAILABLE": "Planning unavailable; saved information retained",
            "PLAN_INVALID": "Plan needs repair"}.get(str(status), str(status))


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
    with st.expander("Error details"):
        st.code(str(message))


def invoke(graph, payload, config):
    started = utc_now()
    try:
        with st.spinner("Processing with the local agent. This may take a few minutes..."):
            graph.invoke(payload, config)
    except Exception as error:
        # UI boundary only: checkpoints remain the source of truth after failure.
        show_error(error)
        return False
    finally:
        record_run(graph, config, started)
    return True


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


def review_form(review, document, graph, config, thread_key, disabled):
    if not review["can_confirm"]:
        st.caption("Reprocess this document to review supported information.")
        return
    with st.container():
        st.caption("Confirmed entries are saved as your statements, separately from the original document observations. They do not establish official acceptance.")
        edit_grounded = st.checkbox("Also edit source-grounded values", key=f"edit_{thread_key}_{document.doc_id}", disabled=disabled)
        with st.form(f"review_{thread_key}_{document.doc_id}"):
            inputs = {}
            for item in review["fields"]:
                if item["status"] == "source_grounded" and not edit_grounded:
                    continue
                name = item["name"]
                label = name.replace("_", " ").capitalize()
                value = item["value"]
                if name=="insurance_evidence_type":
                    options=["Not confirmed"]+list(INSURANCE_TYPES)
                    choice=st.selectbox("Insurance evidence category",options,index=options.index(value) if value in options else 0,disabled=disabled)
                    if choice!="Not confirmed":inputs[name]=choice
                elif item["boolean"]:
                    labels = {"insurance_valid_for_ikamet": "I confirm this insurance is valid for residence-permit purposes",
                              "application_form_signed": "I have signed this application form",
                              "online_application_completed": "I have completed the online application"}
                    choices = ["Not confirmed", "Yes", "No"]
                    choice = st.selectbox(labels.get(name, label if name != "exchange_student" else "Is this student on an exchange programme?"), choices, index=0 if value is None else 1 if value else 2, disabled=disabled)
                    inputs[name] = None if choice == "Not confirmed" else choice == "Yes"
                else:
                    if "date" in name:
                        label += " (YYYY-MM-DD)"
                    text = st.text_input(label, value="" if value is None else str(value), disabled=disabled)
                    if item["observation"] is not None and item["status"] not in {"source_grounded", "user_confirmed"}:
                        st.caption(f"Unconfirmed document observation: {item['observation']}")
                    inputs[name] = text.strip() or None
            acknowledged = st.checkbox("I confirm the entered information is accurate", disabled=disabled)
            submit = st.form_submit_button("Confirm information", disabled=disabled)
        if submit:
            if not acknowledged:
                st.warning("Confirm that the entered information is accurate before saving.")
            elif not inputs:
                st.info("The displayed fields are already source-grounded.")
            else:
                event = UserEvent(kind="confirm_information", document_id=document.doc_id, confirmed_fields=inputs)
                if invoke(graph, ui_run(event, config["configurable"]["thread_id"]), config):
                    st.rerun()


def interaction_state(snapshot):
    approval = snapshot.next == ("human_approval",) and any(field(task,"interrupts",[]) for task in snapshot.tasks)
    return {"approval_pending":approval,"editing_blocked":bool(snapshot.next),
            "message":"A proposed plan is waiting for your decision. Reject it to edit information, or approve it to record the proposal." if approval else "Unfinished graph processing must be retried before factual edits." if snapshot.next else ""}


def identity_review(state, graph, config, thread_key, disabled):
    detail=conflict_details(state).get("holder_name")
    if not detail or detail["resolved"]: return
    st.warning("Distinct holder-name evidence needs your review. Original observations will be preserved.")
    rows=detail["observations"]
    for row in rows: st.write(f"{row['value']} | {row['document_type']} | {row['document_id']} | {row['origin']}")
    with st.form(f"identity_{thread_key}_{state.get('trace_id','initial')}"):
        choice=st.selectbox("Select the correct identity after reviewing the documents",range(len(rows)),format_func=lambda i: f"{rows[i]['value']} ({rows[i]['document_type']}, {rows[i]['origin']})",disabled=disabled)
        confirmed=st.checkbox("I explicitly resolve these identity observations",disabled=disabled)
        submit=st.form_submit_button("Save identity decision",disabled=disabled)
    if submit and confirmed:
        event=UserEvent(kind="resolve_conflict",confirmed_fields={"holder_name":rows[choice]["value"]})
        if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config): st.rerun()


def declaration_form(state, graph, config, thread_key, disabled):
    context = state.get("residence_permit_context")
    edit = st.checkbox("Edit saved information", key=f"edit_declarations_{thread_key}")
    situation={"is_in_turkiye","entered_with_passport","legal_stay_expiry_date","application_date"}
    unresolved = [name for name in CONTEXT_QUESTIONS if name in situation and (edit or field(context,name) is None)]
    if any(getattr(d,"doc_type",None)=="turkiye_entry_stamp" and getattr(d,"status",None)=="analyzed" for d in state.get("documents",[])):
        unresolved=[name for name in unresolved if name!="latest_entry_date"]  # Review the page-specific candidate above.
    if not unresolved:
        st.caption("Your current location and passport declarations are recorded.")
        return
    st.caption("Answers are user declarations, not independent legal verification. Legal-stay expiry must be supplied explicitly; nationality is never used to calculate it.")
    for group,names in CONTEXT_GROUPS.items():
        questions=[name for name in names if name in unresolved]
        if not questions:continue
        with st.expander(group+" confirmations",expanded=False):
            with st.form(f"declarations_{thread_key}_{group}_{state.get('trace_id','initial')}"):
                answers={}
                for name in questions:
                    value=field(context,name);kind=CONTEXT_FIELDS[name]
                    if kind=="bool":
                        choice=st.selectbox(CONTEXT_QUESTIONS[name],["Not answered","Yes","No"],index=0 if value is None else 1 if value else 2,disabled=disabled)
                        if choice!="Not answered":answers[name]=choice=="Yes"
                        elif edit and value is not None:answers[name]=None
                    elif kind=="insurance_type":
                        choices=["Not answered"]+list(INSURANCE_TYPES)
                        choice=st.selectbox(CONTEXT_QUESTIONS[name],choices,index=choices.index(value) if value in choices else 0,disabled=disabled)
                        if choice!="Not answered":answers[name]=choice
                        elif edit and value is not None:answers[name]=None
                    else:
                        entered=st.text_input(CONTEXT_QUESTIONS[name]+" (YYYY-MM-DD)",value="" if value is None else str(value),disabled=disabled)
                        if entered.strip():answers[name]=entered.strip()
                        elif edit and value is not None:answers[name]=None
                submit=st.form_submit_button("Save answers and recheck requirements",disabled=disabled)
            if submit:
                if answers:
                    event=UserEvent(kind="confirm_information",document_id="__declarations__",confirmed_fields=answers)
                    if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config):st.rerun()
                else:st.info("Answer at least one question to save.")


CHECKLIST_ICONS = {"complete": ":green[:material/check_circle:]", "needs confirmation": ":orange[:material/rate_review:]",
                   "missing": ":gray[:material/radio_button_unchecked:]", "ambiguous": ":orange[:material/help:]"}
CHECKLIST_LABELS = {"complete": "Ready", "needs confirmation": "Needs your review", "missing": "Missing", "ambiguous": "Unclear"}


def render_checklist(state):
    """Compact two-column checklist; same milestone_checklist() data as before."""
    checklist = milestone_checklist(state)
    if all(row["status"] == "complete" for row in checklist):
        st.success("All supported documents are ready. Later legal steps remain separate.")
    left, right = st.columns(2)
    for index, row in enumerate(checklist):
        (left if index % 2 == 0 else right).markdown(
            f"{CHECKLIST_ICONS[row['status']]} {row['label']}  \n:gray[{CHECKLIST_LABELS[row['status']]}]")
    st.caption("Ready means the document is prepared, not that the permit is approved.")


def lifecycle_button(graph,config,thread_key,doc_id,kind,label,disabled=False,**kwargs):
    if st.button(label,key=f"{kind}_{thread_key}_{doc_id}",disabled=disabled):
        target=kwargs.pop("target_document_id",doc_id)
        event=UserEvent(kind=kind,document_id=target,**kwargs)
        if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config):st.rerun()


def render_document(document,state,graph,config,thread_key,disabled):
    record=display_record(document);doc_id=record.get("doc_id","");kind=record.get("doc_type")
    review=document_card(state,document)
    title=MILESTONE_DOCUMENTS.get(kind,("Unknown document",()))[0]
    status="Replacement awaiting decision" if record.get("status")=="replacement_pending" else review["status"].replace("_"," ")
    with st.container(border=True):
        st.write(f"{title} - {status}")
        if kind=="health_insurance":st.caption("Milestone-1 R015 compares insurance end against requested permit end OR academic program end. Start date is retained but does not determine this scoped result.")
        shared=sum(field(d,"file_path")==record.get("file_path") for d in state.get("documents",[]))>1
        st.caption(Path(record.get("file_path","")).name+(" | logical item from shared upload" if shared else ""))
        if record.get("processing_error"):st.error(friendly_error(str(record["processing_error"])))
        if record.get("status")=="analyzed" and sum(getattr(d,"doc_type",None)==kind and getattr(d,"status",None)=="analyzed" for d in state.get("documents",[]))>1:
            st.warning("Saved duplicate versions exist. Choose the current version explicitly.")
            lifecycle_button(graph,config,thread_key,doc_id,"replace_document","Use this version and replace other versions",disabled=disabled,confirmed=True)
        if record.get("status")=="replacement_pending":
            st.warning("Another current document has this type. Replace it explicitly or cancel this upload; pending evidence is not used.")
            if not disabled:
                lifecycle_button(graph,config,thread_key,doc_id,"replace_document","Replace current document",confirmed=True)
                lifecycle_button(graph,config,thread_key,doc_id,"cancel_document","Cancel",confirmed=True)
        with st.expander("Details / Edit",expanded=False):
            for item in review["fields"]:
                if item["value"] is not None:
                    origin={"source_grounded":"source-grounded","user_confirmed":"confirmed by you"}.get(item["origin"],"observation awaiting confirmation")
                    st.write(f"{item['name'].replace('_',' ').capitalize()}: {item['value']} ({origin})")
            if isinstance(document,Document) and document.status=="analyzed":
                review_form(review,document,graph,config,thread_key,disabled)
                for name,entry in state.get("verification_records",{}).get(doc_id,{}).items():
                    if field(entry,"origin")=="user_confirmed" and not disabled:
                        lifecycle_button(graph,config,thread_key,doc_id+name,"remove_correction","Use original "+name.replace("_"," ")+" observation",confirmed_fields={name:None},target_document_id=doc_id)
            if isinstance(document,Document) and not disabled and document.status!="replacement_pending":
                from src.workers.verification import PROFILE_TYPES
                profiles={**PROFILE_TYPES,"student_certificate":"student_enrollment_certificate","health_insurance":"health_insurance_certificate","ikamet_application_form":"ikamet_application_form"}
                choices=list(profiles)
                selected=st.selectbox("Reclassify / Reprocess as",choices,index=choices.index(kind) if kind in choices else 0,format_func=lambda value:MILESTONE_DOCUMENTS[value][0],key=f"profile_{thread_key}_{doc_id}")
                lifecycle_button(graph,config,thread_key,doc_id,"reprocess_document","Reprocess selected profile",document_type_hint=profiles[selected])
            with st.expander("Technical details",expanded=False):
                st.json(record);st.json(jsonable(state.get("verification_records",{}).get(doc_id,{})))
        checked=st.checkbox("I want to delete this document",key=f"delete_confirm_{thread_key}_{doc_id}_{state.get('trace_id','initial')}",disabled=disabled)
        lifecycle_button(graph,config,thread_key,doc_id,"delete_document","Delete",disabled=disabled or not checked,confirmed=True)


VIEWS = {"next": ":material/flag: Next steps", "documents": ":material/description: Documents",
         "details": ":material/person: Your details", "checks": ":material/fact_check: Checks",
         "technical": ":material/code: Technical"}


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
    """Three numbers that answer 'where am I?'."""
    checklist = milestone_checklist(state)
    ready = sum(row["status"] == "complete" for row in checklist)
    counts = requirement_summary(display_records(state.get("relevant_requirements")))
    evaluated = state.get("relevant_requirements") is not None
    docs, checks, status = st.columns(3, border=True)
    docs.metric("Documents ready", f"{ready} / {len(checklist)}")
    checks.metric("Checks passed", f"{counts['PASS']} / 29" if evaluated else "Not run yet",
                  delta=f"{counts['FAIL']} failing" if evaluated and counts["FAIL"] else None,
                  delta_color="inverse")
    status.caption("Status")             # text, not a metric: long labels must not be cut off
    status.markdown(f"**{status_label(state.get('run_status', 'NOT_STARTED'))}**")


def render_decision(state, snapshot, graph, config, thread_key):
    """A pending approval or unfinished run comes first, before any view."""
    interaction = interaction_state(snapshot)
    if interaction["approval_pending"]:
        steps = plan_steps(state)
        with st.container(border=True):
            st.markdown("#### :material/gavel: A plan is waiting for your decision")
            if steps:
                for index, step in enumerate(display_records(steps), 1):
                    st.markdown(f"**{index}.** {user_action(step)}")
            else:
                st.write(state.get("plan_summary", ""))
            st.caption("Approving records these next steps only. Nothing is submitted, paid or "
                       "sent on your behalf. Deny the plan to edit your information.")
            approve, deny, _ = st.columns([1, 1, 3])
            decision = None
            if approve.button("Approve", type="primary", key=f"approve_{thread_key}_{state.get('trace_id')}"):
                decision = True
            if deny.button("Deny", key=f"deny_{thread_key}_{state.get('trace_id')}"):
                decision = False
            if decision is not None and invoke(graph, Command(resume=decision), config):
                st.rerun()
    elif snapshot.next:
        with st.container(border=True):
            st.warning("Processing stopped before it finished. Retry to continue; editing is locked until then.")
            if st.button("Retry unfinished processing", key=f"retry_{thread_key}") and invoke(graph, None, config):
                st.rerun()


def view_next_steps(state, snapshot, graph, config, thread_key):
    status = str(state.get("run_status", "NOT_STARTED"))
    busy = bool(snapshot.next)
    st.markdown("##### Your documents")
    render_checklist(state)

    st.markdown("##### Recommended next steps")
    if status == "PLANNING_UNAVAILABLE":
        st.warning("The local model could not write a plan. Your documents and answers are saved; try again when Ollama is running.")
    elif status == "HUMAN_REVIEW_REQUIRED":
        st.error("The plan could not be fixed automatically after "
                 f"{MAX_REFLECTION_ATTEMPTS} attempts. Review your information, then generate a new plan.")
    elif status == "APPROVED":
        st.success("You approved this plan. No official action was taken.")
    elif status == "APPROVAL_DENIED":
        st.info("You denied the last plan. Update your information and generate a new one.")
    approved = state.get("approved_plan")
    steps = plan_steps(state)
    origin = state.get("plan_origin") or (state.get("approved_plan_origin") if approved else None)
    if origin == "milestone_completion":
        st.info(state.get("plan_summary") or field(approved, "summary", ""))
    if steps is None:
        st.caption("Upload documents and answer the questions in **Your details**, then generate a plan.")
    for index, step in enumerate(display_records(steps), 1):
        with st.container(border=True):
            text, action = st.columns([5, 2])
            text.markdown(f"**{index}. {user_action(step)}**")
            if step.get("reason"):
                text.caption(step["reason"])
            if step.get("due_date") and status in {"VALIDATED", "APPROVED"}:
                text.caption(f":material/event: Due {step['due_date']}")
            if approved and not busy:
                marked = any(row.get("action_id") == step.get("step_id") and row.get("action") == step.get("action")
                             for row in state.get("action_progress", {}).values())
                if marked:
                    action.markdown(":green[:material/check: Marked done]")
                elif action.button("Mark done", key=f"done_{thread_key}_{step.get('step_id')}",
                                   help="Records your own declaration; requirements still depend on evidence."):
                    run_event(graph, config, UserEvent(kind="record_progress", action_id=step.get("step_id")))
    issues = display_records(state.get("validation_errors"))
    if issues and status not in {"APPROVAL_DENIED"}:
        st.warning(f"{len(issues)} issue(s) with the last plan: {issues[0].get('message', '')}")
    if st.button("Generate plan" if steps is None else "Update plan", type="primary",
                 disabled=busy, key=f"plan_{thread_key}"):
        run_event(graph, config, UserEvent(kind="generate_plan"))


def view_documents(state, snapshot, graph, config, thread_key, thread_id):
    busy = bool(snapshot.next)
    with st.form(f"upload_{thread_key}", border=True):
        uploaded = st.file_uploader("Add a document (PDF or TXT)", type=["pdf", "txt"],
                                    key=f"file_{thread_key}", disabled=busy)
        submit = st.form_submit_button("Upload and analyze", type="primary", disabled=busy)
    if submit:
        if uploaded is None:
            st.warning("Choose a PDF or TXT file first.")
        else:
            path = save_upload(uploaded.name, uploaded.getvalue(), thread_id)
            run_event(graph, config, UserEvent(kind="analyze_document", file_path=str(path)))
    documents = state.get("documents") or []
    if not documents:
        st.caption("No documents yet. Each upload is read by the local model; you can review and correct what it found.")
    for document in documents:
        render_document(document, state, graph, config, thread_key, busy)


def view_details(state, snapshot, graph, config, thread_key):
    busy = bool(snapshot.next)
    identity_review(state, graph, config, thread_key, busy)
    declaration_form(state, graph, config, thread_key, busy)
    derived = state.get("verification_records", {}).get("__derived__", {})
    if derived:
        entry = derived.get("planned_stay_months")
        st.info(f"Estimated academic stay: {field(entry, 'value')} full months (calculated from your programme dates; not the permit period).")
    if state.get("verification_conflicts"):
        st.warning("Some confirmed information conflicts across documents: " + ", ".join(state["verification_conflicts"]))


def view_checks(state, snapshot, graph, config, thread_key):
    requirements = display_records(state.get("relevant_requirements"))
    if state.get("relevant_requirements") is None:
        st.caption("The 29 residence-permit checks run after your first upload or answer.")
    else:
        counts = requirement_summary(requirements)
        passed, failed, unknown = st.columns(3)
        passed.metric("Passed", counts["PASS"])
        failed.metric("Failed", counts["FAIL"])
        unknown.metric("Not enough information yet", counts["missing_information"])
    from src.agents.planner import actionable_needs
    needs = actionable_needs(state)
    immediate_ids = {n["rule_id"] for n in needs if n["category"] == "immediate"}
    groups = blocking_requirement_groups(requirements)
    failures = [r for r in groups["failed"] if r.get("rule_id") in immediate_ids]
    missing = [r for r in groups["missing_information"] if r.get("rule_id") in immediate_ids]
    if failures:
        st.markdown("##### Problems to fix")
        for requirement in failures:
            st.error(requirement.get("description", "Requirement not met"), icon=":material/error:")
    if missing:
        st.markdown("##### Information needed now")
        st.caption("Upload the matching document, or answer the questions in **Your details**.")
        with st.container(border=True):
            for requirement in missing:
                fields = ", ".join(name.replace("_", " ") for name in requirement.get("missing_fields", []))
                st.markdown(f":blue[:material/info:] {requirement.get('description', 'Information needed')}"
                            + (f"  \n:gray[Needs: {fields}]" if fields else ""))
    if requirements and not failures and not missing:
        st.success("Nothing blocking right now. Other checks belong to later stages.")
    if st.button("Re-check", disabled=bool(snapshot.next), key=f"recheck_{thread_key}",
                 help="Re-runs the rule checks on saved information; documents are not re-read."):
        run_event(graph, config, UserEvent(kind="reevaluate"))


def view_technical(state, snapshot):
    """Everything the earlier layout showed inline, in one place."""
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import workflow_stage
    requirements = display_records(state.get("relevant_requirements"))
    unknown_groups = unknown_requirement_groups(state)
    origin = state.get("plan_origin") or (state.get("approved_plan_origin") if state.get("approved_plan") else None)
    st.caption(f"Workflow stage: {workflow_stage(state).replace('_', ' ').title()}"
               f" | Plan source: {({'deterministic_fallback': 'Deterministic fallback', 'milestone_completion': 'Deterministic milestone assessment'}.get(origin, 'Local Ollama model') if origin else 'none')}"
               f" | Repair attempts: {state.get('retry_count', 0)} / {MAX_REFLECTION_ATTEMPTS}")
    st.caption("Not-yet-known checks by scope: " + " | ".join(
        f"{label}: {len(unknown_groups[key])}" for key, label in
        (("immediate", "needed now"), ("future", "later stage"), ("outside_scope", "outside Milestone 1"),
         ("guardrail_only", "safety guardrails"), ("waiting", "waiting on prerequisites"), ("advisory", "advisory"))))
    with st.expander("Rule results (all 29)"):
        st.json({"unknown_by_scope": unknown_groups, "requirements": requirements})
    with st.expander("Workflow priorities and dependencies"):
        st.json(jsonable(actionable_needs(state)))
    with st.expander("Structured plan"):
        st.json(jsonable({"summary": state.get("plan_summary", ""), "candidate_plan": plan_steps(state)}))
    with st.expander("Gate 1 and reflection"):
        st.json(jsonable({"validation_errors": display_records(state.get("validation_errors")),
                          "feedback": state.get("repair_feedback", []), "history": state.get("repair_history", []),
                          "planner_diagnostics": state.get("planner_diagnostics", {}),
                          "action_progress": state.get("action_progress", {})}))
    with st.expander("Digital Twin"):
        st.json(jsonable({"extracted_facts": state.get("extracted_facts", {}),
                          "residence_permit_context": state.get("residence_permit_context", {}),
                          "verification_records": state.get("verification_records", {}),
                          "verification_conflicts": state.get("verification_conflicts", [])}))


def render_state(snapshot, graph, config, thread_key, thread_id):
    state = {**snapshot.values, "planning_scope": MILESTONE_SCOPE}  # Display-only scope; edits go through graph events.
    render_status_bar(state, snapshot)
    render_decision(state, snapshot, graph, config, thread_key)
    view = st.segmented_control("View", list(VIEWS), format_func=VIEWS.get, default="next",
                                key=f"view_{thread_key}", label_visibility="collapsed") or "next"
    if view == "next":
        view_next_steps(state, snapshot, graph, config, thread_key)
    elif view == "documents":
        view_documents(state, snapshot, graph, config, thread_key, thread_id)
    elif view == "details":
        view_details(state, snapshot, graph, config, thread_key)
    elif view == "checks":
        view_checks(state, snapshot, graph, config, thread_key)
    else:
        view_technical(state, snapshot)


def main():
    st.set_page_config(page_title="Residence Permit Assistant", page_icon=":material/badge:", layout="wide")
    st.sidebar.markdown("### :material/badge: Residence Permit Assistant")
    st.sidebar.caption("Sabancı incoming exchange · first short-term residence permit")
    thread_id = st.sidebar.text_input("Your profile name", value="demo-user",
                                      help="Use the same name next time to continue where you left off.").strip()
    if not thread_id:
        st.info("Enter a profile name to begin.")
        return
    thread_key = sha256(thread_id.encode()).hexdigest()[:16]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    try:
        # A fresh connection per rerun avoids caching a closed/shared connection.
        # No canonical state is stored in Streamlit session_state.
        with sqlite_checkpointer() as saver:
            graph = build_graph(saver)
            snapshot = graph.get_state(config)
            st.sidebar.caption(":green[:material/cloud_done:] Saved progress restored" if snapshot.values
                               else "New profile")
            if st.sidebar.button("Refresh", icon=":material/refresh:"):
                st.rerun()
            with st.sidebar.expander("Reset this profile", icon=":material/delete:"):
                st.caption("Deletes this profile's saved progress and record. Uploaded files and other profiles are kept.")
                typed = st.text_input("Type the profile name to confirm", key=f"reset_typed_{thread_key}")
                if st.button("Reset", type="primary", disabled=typed != thread_id, key=f"reset_{thread_key}"):
                    reset_thread(thread_id, typed)
                    for key in list(st.session_state):
                        if thread_key in key and not key.startswith("reset_"): del st.session_state[key]
                    st.rerun()
            st.title("Your residence permit")
            render_state(snapshot, graph, config, thread_key, thread_id)
    except Exception as error:
        show_error(error)
        st.caption("Saved progress stays on this computer. Refresh to try again.")


if __name__ == "__main__":
    main()
