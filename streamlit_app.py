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


def render_checklist(state):
    st.caption("Document preparation only: complete does not mean legal acceptance or an approved residence permit.")
    checklist=milestone_checklist(state)
    labels={"complete":"Complete","needs confirmation":"Needs review","missing":"Missing","ambiguous":"Ambiguous"}
    if all(row["status"]=="complete" for row in checklist):st.success("Supported document preparation complete; later legal steps remain separate.")
    for row in checklist:st.write(f"{row['label']}: **{labels[row['status']]}**")


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
        checked=st.checkbox("Confirm deletion of this logical document",key=f"delete_confirm_{thread_key}_{doc_id}_{state.get('trace_id','initial')}",disabled=disabled)
        lifecycle_button(graph,config,thread_key,doc_id,"delete_document","Delete",disabled=disabled or not checked,confirmed=True)


def render_state(snapshot, graph, config, thread_key):
    state = {**snapshot.values,"planning_scope":MILESTONE_SCOPE}  # Display-only scope; factual edits go through graph events.
    status = str(state.get("run_status", "NOT_STARTED"))
    interaction=interaction_state(snapshot)
    approval_pending=interaction["approval_pending"]
    if interaction["message"]:
        st.info(interaction["message"])
    if approval_pending and st.button("Reject pending plan and unlock editing",key=f"unlock_{thread_key}"):
        if invoke(graph,Command(resume=False),config): st.rerun()
    documents=state.get("documents") or []
    for document in documents:render_document(document,state,graph,config,thread_key,bool(snapshot.next))
    if not documents:st.info("No documents yet. Upload a supported document, analyze it and review its fields.")
    st.subheader("3. Information needing confirmation")
    identity_review(state,graph,config,thread_key,bool(snapshot.next))
    declaration_form(state, graph, config, thread_key, bool(snapshot.next))
    derived = state.get("verification_records", {}).get("__derived__", {})
    if derived:
        entry = derived.get("planned_stay_months")
        st.info(f"Estimated academic stay: {field(entry, 'value')} complete calendar months (derived). This is not the requested permit period.")
    with st.expander("Technical details: Digital Twin", expanded=False):
        st.json(jsonable({"extracted_facts": state.get("extracted_facts", {}),
                          "residence_permit_context": state.get("residence_permit_context", {}),
                          "verification_records": state.get("verification_records", {}),
                          "verification_conflicts": state.get("verification_conflicts", [])}))

    st.subheader("4. Requirement summary")
    st.caption("All 29 legal checks remain evaluated, including later and out-of-scope requirements.")
    requirements = display_records(state.get("relevant_requirements"))
    for column, (label, count) in zip(st.columns(3), requirement_summary(requirements).items()):
        column.metric("UNKNOWN" if label == "missing_information" else label, count)
    if state.get("relevant_requirements") is None:
        st.caption("Requirements have not been evaluated for this run.")
    else:
        st.caption("Requirements evaluated using accepted inputs; unknown information remains unknown.")
    unknown_groups=unknown_requirement_groups(state)
    st.caption("UNKNOWN by workflow scope: " + " | ".join(f"{label}: {len(unknown_groups[key])}" for key,label in (("immediate","needed now"),("future","later stage"),("outside_scope","outside Milestone 1"),("guardrail_only","safety guardrails"),("waiting","waiting on prerequisites"),("advisory","advisory"))))
    st.caption("Outside this milestone does not mean legally not applicable; these checks remain UNKNOWN.")
    with st.expander("Technical details: complete rule results", expanded=False):
        st.json({"unknown_by_scope":unknown_groups,"requirements":requirements})

    st.caption("Current blockers / missing items")
    groups = blocking_requirement_groups(requirements)
    st.write("Blocking failures")
    from src.agents.planner import actionable_needs
    current_ids={n["rule_id"] for n in actionable_needs(state) if n["category"]=="immediate"}
    current_failures=[r for r in groups["failed"] if r.get("rule_id") in current_ids]
    for requirement in current_failures:
        st.error(f"{requirement.get('rule_id')}: {requirement.get('description', 'Requirement not met')}")
    if not current_failures:
        st.caption("No assessed blocking failures.")
    from src.agents.planner import actionable_needs
    needs=actionable_needs(state)
    immediate_ids={n["rule_id"] for n in needs if n["category"]=="immediate"}
    immediate=[r for r in groups["missing_information"] if r.get("rule_id") in immediate_ids]
    st.write("Information needed now")
    for requirement in immediate[:3]:
        st.info(f"{requirement.get('rule_id')}: {requirement.get('message', requirement.get('description', 'Information needed'))}")
    if not immediate:
        st.caption("No recorded missing blocking information.")
    elif len(immediate) > 3:
        st.caption("More missing information is listed in Technical details: complete rule results.")
    with st.expander("Technical details: workflow priorities and dependencies",expanded=False):
        st.json(jsonable(needs))
    with st.expander("Advanced / Technical: requirements",expanded=False):
        st.caption("This recomputes accepted state and deterministic rules only; OCR/LLM extraction is not rerun.")
        if st.button("Re-check requirements",disabled=bool(snapshot.next),key=f"recheck_{thread_key}"):
            if invoke(graph,ui_run(UserEvent(kind="reevaluate"),config["configurable"]["thread_id"]),config): st.rerun()
    if state.get("verification_conflicts"):
        st.warning("Confirmed information conflicts across documents. Review: " + ", ".join(state["verification_conflicts"]))

    st.subheader("5. Recommended next steps")
    if status == "PLANNING_UNAVAILABLE":
        st.warning("Ollama could not generate a plan. Your documents and requirements are saved. Retry Generate / Update Plan when the local model is available.")
    if st.button("Generate / Update Plan", disabled=bool(snapshot.next), key=f"plan_{thread_key}"):
        if invoke(graph, ui_run(UserEvent(kind="generate_plan"), config["configurable"]["thread_id"]), config):
            st.rerun()
    if status == "VALIDATED":
        st.success("Plan validated by Gate 1. This validates the proposed next steps, not the residence permit.")
    origin=state.get("plan_origin") or (state.get("approved_plan_origin") if state.get("approved_plan") else None)
    if origin:
        st.caption("Plan source: " + ({"deterministic_fallback":"Deterministic fallback","milestone_completion":"Deterministic milestone checklist assessment"}.get(origin,"Local Ollama model")))
    from src.agents.planning_policy import workflow_stage
    st.caption("Workflow stage: "+workflow_stage(state).replace("_"," ").title())
    steps = state.get("candidate_plan")
    approved=state.get("approved_plan")
    if steps is None and approved:
        steps=field(approved,"actions",[])
    if origin=="milestone_completion":
        st.info(state.get("plan_summary") or field(approved,"summary",""))
    if steps is None:
        st.info("Review your information, then select Generate / Update Plan when ready.")
    for index, step in enumerate(display_records(steps), 1):
        st.write(f"{index}. {user_action(step)}")
        if step.get("request_kind"):
            st.caption(step["request_kind"].replace("_"," ").title()+" | "+("Available through the controls above" if step.get("available_in_prototype") else "Manual step or external guidance; no official execution"))
        if approved and not snapshot.next:
            marked=any(row.get("action_id")==step.get("step_id") and row.get("action")==step.get("action") for row in state.get("action_progress",{}).values())
            if marked: st.caption("Completion declared by you; requirements still depend on evidence and accepted inputs.")
            elif st.button("Mark action done (declaration only)",key=f"done_{thread_key}_{step.get('step_id')}"):
                event=UserEvent(kind="record_progress",action_id=step.get("step_id"))
                if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config): st.rerun()
        if step.get("related_rule_id"): st.caption(f"{step['related_rule_id']}: {step.get('reason', '')}")
        if step.get("due_date") and status in {"VALIDATED", "APPROVED"}:
            st.caption(f"Due: {step['due_date']}")
    with st.expander("Technical details: structured plan", expanded=False):
        st.json(jsonable({"summary": state.get("plan_summary", ""), "candidate_plan": steps}))

    st.caption("Plan validation / reflection")
    st.caption(f"Repair attempts: {state.get('retry_count',0)} / {MAX_REFLECTION_ATTEMPTS}")
    if status == "HUMAN_REVIEW_REQUIRED":
        st.error("HUMAN_REVIEW_REQUIRED: automatic repair stopped. Review the information and issues.")
    issues = display_records(state.get("validation_errors"))
    if issues:
        st.warning(f"{len(issues)} issue(s) need attention.")
        st.caption(str(issues[0].get("message", "Review technical validation details")))
    with st.expander("Technical details: Gate 1 and reflection", expanded=False):
        st.json(jsonable({"validation_errors": issues, "feedback": state.get("repair_feedback", []), "history": state.get("repair_history", []), "planner_diagnostics":state.get("planner_diagnostics",{}), "action_progress":state.get("action_progress",{})}))
    if snapshot.next and not approval_pending:
        st.warning("This thread has unfinished processing.")
        if st.button("Retry unfinished processing", key=f"retry_{thread_key}") and invoke(graph, None, config):
            st.rerun()

    st.subheader("6. Human approval")
    st.info("Approval accepts only the proposed next steps. It does NOT approve the residence permit or execute official actions.")
    if approval_pending:
        st.write("You are approving these steps:" if steps else "You are approving this preparation assessment; no official action is authorized:")
        if not steps:st.write(state.get("plan_summary",""))
        for index, step in enumerate(display_records(steps), 1):
            st.write(f"{index}. {user_action(step)}")
        approve, deny = st.columns(2)
        decision = None
        if approve.button("Approve", key=f"approve_{thread_key}_{state.get('trace_id')}"):
            decision = True
        if deny.button("Deny", key=f"deny_{thread_key}_{state.get('trace_id')}"):
            decision = False
        if decision is not None and invoke(graph, Command(resume=decision), config):
            st.rerun()
    elif status == "APPROVED":
        st.success("Plan approved by you. No official action was executed.")
    elif status == "APPROVAL_DENIED":
        st.warning("The proposed plan was denied.")


def main():
    st.set_page_config(page_title="Personal Bureaucracy Agent", layout="wide")
    st.title("Personal Bureaucracy Agent")
    st.caption("Milestone 1 | Sabanc\u0131 incoming exchange | First short-term residence permit")
    thread_id = st.sidebar.text_input("Thread ID", value="demo-user").strip()
    st.sidebar.caption("Use the same ID to restore a saved Digital Twin.")
    if not thread_id:
        st.info("Enter a thread ID to begin.")
        return
    thread_key = sha256(thread_id.encode()).hexdigest()[:16]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    try:
        # A fresh connection per rerun avoids caching a closed/shared connection.
        # No canonical state is stored in Streamlit session_state.
        with sqlite_checkpointer() as saver:
            graph = build_graph(saver)
            snapshot = graph.get_state(config)
            st.sidebar.write("Procedure:", snapshot.values.get("active_procedure", "ikamet"))
            st.sidebar.write("Agent status:", status_label(snapshot.values.get("run_status", "NOT_STARTED")))
            if snapshot.values:
                st.info("Existing Digital Twin restored from SQLite")
            if st.sidebar.button("Refresh saved state"):
                st.rerun()
            with st.sidebar.expander("Advanced / Danger Zone",expanded=False):
                st.caption("Reset only the current thread. Uploaded files are retained; other Digital Twins are untouched.")
                typed=st.text_input("Type the current thread ID to confirm reset",key=f"reset_typed_{thread_key}")
                if st.button("Reset this Digital Twin",disabled=typed!=thread_id,key=f"reset_{thread_key}"):
                    reset_thread(thread_id,typed)
                    for key in list(st.session_state):
                        if thread_key in key and not key.startswith("reset_"):del st.session_state[key]
                    st.rerun()
            st.subheader("1. Milestone-1 checklist")
            render_checklist({**snapshot.values,"planning_scope":MILESTONE_SCOPE})
            st.subheader("2. Your documents")
            with st.form(f"upload_{thread_key}"):
                uploaded = st.file_uploader("PDF or TXT document", type=["pdf", "txt"],
                                            key=f"file_{thread_key}", disabled=bool(snapshot.next))
                submit = st.form_submit_button("Analyze document", disabled=bool(snapshot.next))
            if submit:
                if uploaded is None:
                    st.warning("Choose a PDF or TXT file first.")
                else:
                    path = save_upload(uploaded.name, uploaded.getvalue(), thread_id)
                    event = UserEvent(kind="analyze_document", file_path=str(path))
                    if invoke(graph, ui_run(event, thread_id), config):
                        st.rerun()
                    snapshot = graph.get_state(config)
            render_state(snapshot, graph, config, thread_key)
    except Exception as error:
        show_error(error)
        st.caption("Saved checkpoints remain local. Refresh to inspect the thread again.")


if __name__ == "__main__":
    main()
