"""Local presentation layer for the existing SQLite-backed Milestone-1 graph."""
from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st
from langgraph.types import Command
from pydantic import BaseModel

from config import MAX_REFLECTION_ATTEMPTS, PROJECT_ROOT
from src.core.graph import build_graph
from src.core.state import UserEvent, new_run
from src.memory.checkpointer import sqlite_checkpointer
from src.document_processing.schemas import DOCUMENT_SCHEMAS


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


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
            "PLAN_INVALID": "Plan needs repair"}.get(str(status), str(status))


def requirement_summary(requirements):
    records = display_records(requirements)
    return {label: sum(r.get("status") == status for r in records) for label, status in
            (("PASS", "passed"), ("FAIL", "failed"), ("missing_information", "missing_information"))}


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
    try:
        with st.spinner("Processing with the local agent. This may take a few minutes..."):
            graph.invoke(payload, config)
    except Exception as error:
        # UI boundary only: checkpoints remain the source of truth after failure.
        show_error(error)
        return False
    return True


def render_state(snapshot, graph, config, thread_key):
    state = snapshot.values
    status = str(state.get("run_status", "NOT_STARTED"))
    approval_pending = snapshot.next == ("human_approval",) and any(
        field(task, "interrupts", []) for task in snapshot.tasks
    )

    st.subheader("2. Digital Twin / Documents")
    st.caption("Extracted document fields are observations and are not automatically treated as verified legal facts.")
    documents = state.get("documents") or []
    for document in documents if isinstance(documents, (list, tuple)) else [documents]:
        record = display_record(document)
        with st.expander(f"{record.get('doc_type') or 'Pending document'} | {record.get('status', 'Legacy')}"):
            st.caption(f"Document ID: {record.get('doc_id', 'Unavailable')}")
            st.write("Detected type:", record.get("doc_type") or "Not analyzed")
            if isinstance(document, dict):
                st.warning("Legacy document record: display only. Reprocess the document before relying on its observations.")
            if record.get("processing_error"):
                st.error(friendly_error(str(record["processing_error"])))
                st.code(str(record["processing_error"]))
            info = field(document, "info")
            if info is not None:
                observations, legacy = observation_view(info)
                if legacy:
                    st.warning("Legacy document observations: display only, not verified facts. Reprocess this document under the current schema.")
                if observations.get("document_type") == "unknown":
                    st.warning("The document type could not be identified or is unsupported. Provide a supported administrative document.")
                st.write("Observed document type:", observations.get("document_type", "Unavailable"))
                fields = {key: value for key, value in observations.items()
                          if key not in {"document_type", "missing_fields", "ambiguous_fields",
                                         "validation_issues", "selected_pages", "source_pages",
                                         "source_snippets", "source_evidence", "correction_notes"}}
                st.write("Extracted fields")
                st.json(fields)
                st.write("Missing fields:", observations.get("missing_fields") or "None recorded")
                st.write("Ambiguous fields:", observations.get("ambiguous_fields") or "None recorded")
                st.write("Validation issues:", observations.get("validation_issues") or "None recorded")
                st.write("Selected pages:", observations.get("selected_pages") or "Unavailable")
                st.write("Field source pages")
                st.json(observations.get("source_pages") or {})
                st.write("Correction notes:", observations.get("correction_notes") or "None recorded")
                st.write("Source snippets")
                st.json(observations.get("source_snippets") or {})
                st.write("Deterministic source evidence")
                st.json(observations.get("source_evidence") or [])
            elif isinstance(document, dict) or "legacy_value" in record:
                st.json(record)
    if not state.get("documents"):
        st.info("No documents saved for this thread.")
    with st.expander("Extracted facts"):
        st.json(jsonable(state.get("extracted_facts", {})))
    with st.expander("Residence-permit context"):
        st.json(jsonable(state.get("residence_permit_context", {})))

    st.subheader("3. Requirements")
    requirements = state.get("relevant_requirements")
    if requirements is None:
        st.info("Requirements have not been evaluated for this run.")
    else:
        requirements = display_records(requirements)
        for column, (label, count) in zip(st.columns(3), requirement_summary(requirements).items()):
            column.metric(label, count)
        for title, outcome in (("PASS", "passed"), ("FAIL", "failed"), ("missing_information", "missing_information")):
            group = [result for result in requirements if result.get("status") == outcome]
            with st.expander(f"{title} - complete rule details ({len(group)})", expanded=False):
                for result in group:
                    st.write(f"{result.get('rule_id', 'Legacy rule')}: {result.get('description', 'Description unavailable')}")
                    if result.get("message"):
                        st.caption(str(result["message"]))
                    if result.get("error_code"):
                        st.code(str(result["error_code"]))
        unrecognized = [r for r in requirements if r.get("status") not in ("passed", "failed", "missing_information")]
        if unrecognized:
            st.warning("Some legacy requirement results have no recognized status; re-evaluation is recommended.")
            with st.expander("Legacy requirement details"):
                st.json(unrecognized)

    st.subheader("4. Blockers")
    blockers = state.get("blockers") or []
    descriptions = {str(r.get("rule_id")): r.get("description") for r in display_records(requirements)}
    if blockers:
        labels = []
        for rule in blockers if isinstance(blockers, (list, tuple)) else [blockers]:
            description = descriptions.get(str(rule))
            labels.append(f"{rule} \u2014 {description}" if description else str(rule))
        for label in labels[:3]:
            st.warning(label)
        if len(labels) > 3:
            with st.expander(f"All blockers ({len(labels)})"):
                for label in labels:
                    st.write(label)
    else:
        st.write("No recorded blockers.")
    st.caption("Missing information can block an official action without being a confirmed rule failure.")

    st.subheader("5. Proposed Plan")
    if status == "VALIDATED":
        st.info("Plan validated: Gate 1 validated this proposed plan. This does not mean the residence permit is valid or approved.")
    if state.get("plan_summary"):
        st.write(state["plan_summary"])
    steps = state.get("candidate_plan")
    if steps is None:
        st.info("No candidate plan is available.")
    elif not steps:
        st.write("The candidate plan contains no actions.")
    else:
        if status not in {"VALIDATED", "APPROVED"}:
            st.caption("Candidate proposal; consult the validation result below.")
        for index, step in enumerate(display_records(steps), 1):
            st.write(f"{index}. {step.get('action', step.get('legacy_value', 'Legacy action'))}")
            st.caption(f"Reason: {step.get('reason', 'Unavailable')}")
            st.caption(f"Related rule: {step.get('related_rule_id') or 'None'}")
            # Gate 1 owns deadline grounding; the UI does not reimplement it.
            if step.get("due_date") and status in {"VALIDATED", "APPROVED"}:
                st.caption(f"Due date: {step['due_date']}")

    st.subheader("6. Reflection / Validation")
    st.write(f"Repair attempts: {state.get('retry_count', 0)} / {MAX_REFLECTION_ATTEMPTS}")
    if status == "HUMAN_REVIEW_REQUIRED":
        st.error("HUMAN_REVIEW_REQUIRED: automatic repair has stopped. Review the issues before proceeding.")
    issues = display_records(state.get("validation_errors"))
    if issues:
        st.warning(f"Validation needs attention: {len(issues)} issue(s).")
        planner_error = next((issue for issue in issues if issue.get("code") == "PLANNER_ERROR"), None)
        if planner_error:
            st.warning(friendly_error(str(planner_error.get("message", "Ollama error"))))
        with st.expander("Validation error details"):
            for issue in issues:
                message = str(issue.get("message", issue.get("legacy_value", "Legacy issue")))
                st.write(f"[{issue.get('code', 'LEGACY')}] {message}")
    with st.expander("Repair feedback and history"):
        st.json(jsonable({"feedback": state.get("repair_feedback", []), "history": state.get("repair_history", [])}))
    if snapshot.next and not approval_pending:
        st.warning("This thread has unfinished processing. New uploads are paused until it finishes.")
        if st.button("Retry unfinished processing", key=f"retry_{thread_key}"):
            if invoke(graph, None, config):
                st.rerun()

    st.subheader("7. Human Approval")
    st.info("Approval means approval of the proposed plan. It does NOT approve the residence permit and does NOT execute official actions.")
    if approval_pending:
        st.warning("This saved plan is awaiting your decision.")
        approve, deny = st.columns(2)
        decision = None
        if approve.button("Approve", key=f"approve_{thread_key}_{state.get('trace_id')}"):
            decision = True
        if deny.button("Deny", key=f"deny_{thread_key}_{state.get('trace_id')}"):
            decision = False
        if decision is not None:
            if invoke(graph, Command(resume=decision), config):
                st.rerun()
    elif status == "APPROVED":
        st.success("The proposed plan was approved. No official actions were executed.")
    elif status == "APPROVAL_DENIED":
        st.warning("The proposed plan was denied.")
    else:
        st.caption("No plan is currently awaiting approval.")


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
            st.subheader("1. Upload Document")
            with st.form(f"upload_{thread_key}"):
                uploaded = st.file_uploader("PDF or TXT document", type=["pdf", "txt"],
                                            key=f"file_{thread_key}", disabled=bool(snapshot.next))
                submit = st.form_submit_button("Process document & generate next steps", disabled=bool(snapshot.next))
            if submit:
                if uploaded is None:
                    st.warning("Choose a PDF or TXT file first.")
                else:
                    path = save_upload(uploaded.name, uploaded.getvalue(), thread_id)
                    event = UserEvent(kind="upload", file_path=str(path))
                    if invoke(graph, new_run(event, thread_id), config):
                        st.rerun()
                    snapshot = graph.get_state(config)
            render_state(snapshot, graph, config, thread_key)
    except Exception as error:
        show_error(error)
        st.caption("Saved checkpoints remain local. Refresh to inspect the thread again.")


if __name__ == "__main__":
    main()
