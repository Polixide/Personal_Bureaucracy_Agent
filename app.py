"""Streamlit app for the Personal Bureaucracy Agent: streamlit run app.py

Panels: goal box (agent), documents (upload -> extraction), what the Twin knows (confirm, resolve,
enter values), rule status by section, sandbox, proposed actions (approve / deny), report an outcome.
The Twin (SQLite at config.DATABASE_PATH) is the only source of truth; nothing is sent anywhere.
Upload handling and error wording are adapted from the milestone-1 streamlit_app.py.
"""
from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st

from config import DATABASE_PATH, LLM_MODEL, OUTBOX_DIR, PROJECT_ROOT, UPLOAD_DIR
from src.agent.orchestrator import run_agent
from src.llm import OllamaBackend
from src.models import Pack
from src.pack import load_all_packs
from src.rules import evaluate_rules
from src.sandbox.approval import approve_action, deny_action
from src.sandbox.mock_world import run_simulation
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool
from src.twin import service
from src.twin.facts import FactView
from src.twin.store import TwinStore

STATUS_ICONS = {"pass": "✅", "fail": "❌", "unknown": "❔"}
AUTO_TYPE = "detect automatically"


# ---------------------------------------------------------- pure helpers --
def save_upload(name: str, content: bytes, user_id: str, directory: Path = UPLOAD_DIR) -> Path:
    """Store an upload under a content/user-derived name inside the project, never a browser path."""
    extension = Path(name).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("Unsupported file. Choose a PDF or TXT document.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    directory = Path(directory).resolve()
    if not directory.is_relative_to(PROJECT_ROOT):
        raise ValueError("Uploads must stay inside the project.")
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256(user_id.encode() + b"\0" + content).hexdigest()
    path = directory / f"{digest}{extension}"
    path.write_bytes(content)
    return path


def doc_id_for(path: Path) -> str:
    return "doc_" + sha256(str(path).encode()).hexdigest()[:12]


def friendly_error(message: str) -> str:
    """Plain-language explanation of extraction / model problems."""
    text = message.lower()
    if "ocr" in text:
        return "The scanned pages could not be read. Try a clearer scan or a text-based PDF."
    if "timed out" in text or "timeout" in text:
        return "The local model timed out. Retry when Ollama is ready, or enter the values yourself."
    if "no extractable text" in text or "contains no text" in text:
        return "No readable text was found. Try a clearer scan, a text-based PDF or a TXT file."
    if "password" in text:
        return "Password-protected PDFs are not supported."
    if any(word in text for word in ("ollama", "connect", "unavailable")):
        return "The local model is not running. Start Ollama, or enter the values yourself below."
    return "Processing could not finish. Check the document and retry."


def fact_rows(view: FactView) -> list[dict]:
    """Display rows: value, where it came from, whether it is derived."""
    rows = []
    for field in sorted(view.values):
        if field in view.derived:
            rows.append({"field": field, "value": str(view.values[field]), "from": "computed"})
            continue
        newest = max(view.sources[field], key=lambda fact: fact.fact_id or 0)
        rows.append({"field": field, "value": str(view.values[field]), "from": f"{newest.provenance} ({newest.source_ref})"})
    return rows


def rule_rows(pack: Pack, values: dict, section_id: str) -> list[dict]:
    rows = []
    for result in evaluate_rules(pack, values, section_id):
        label = " (unverified)" if result.rule_status == "unverified" else ""
        rows.append({"rule": result.rule_id, "status": f"{STATUS_ICONS[result.status]} {result.status}",
                     "requirement": result.description + label, "missing": ", ".join(result.missing_fields)})
    return rows


# --------------------------------------------------------------- panels --
def tool_context(store: TwinStore, packs: dict, user_id: str, today: date) -> ToolContext:
    active = store.active_procedure(user_id)
    return ToolContext(store=store, packs=packs, user_id=user_id, today=today, backend=OllamaBackend(),
                       procedure_id=active if active in packs else None)


def sidebar(store: TwinStore, packs: dict) -> tuple[str, date]:
    st.sidebar.title("Personal Bureaucracy Agent")
    user_id = st.sidebar.text_input("Your id (one Digital Twin per id)", value="demo-user").strip() or "demo-user"
    choices = ["(let the agent decide)"] + sorted(packs)
    active = store.active_procedure(user_id)
    chosen = st.sidebar.selectbox("Procedure", choices, index=choices.index(active) if active in packs else 0)
    if chosen != choices[0] and chosen != active:
        store.set_active_procedure(user_id, chosen)
    today = st.sidebar.date_input("Today", value=date.today())
    st.sidebar.caption(f"Model: {LLM_MODEL} (local Ollama). Nothing is sent, booked, paid or submitted.")
    with st.sidebar.expander("Danger zone"):
        typed = st.text_input("Type your id to delete your Digital Twin", key="reset")
        if st.button("Delete my Digital Twin", disabled=typed != user_id):
            store.delete_user(user_id)
            st.rerun()
    return user_id, today


def show_result(result: dict) -> None:
    status = result["status"]
    if status == "needs_user_input":
        st.info(f"Question for you: {result['question']}")
        return
    answer = result.get("answer") or {}
    st.write(answer.get("summary", ""))
    if result.get("needs_human_review"):
        st.warning("This answer did not pass every automatic check; a person should review it: "
                   + ", ".join(issue["code"] for issue in result["gate_issues"]))
    if result.get("fallback_used"):
        st.caption("Built from the rules alone (the model was not usable for this answer).")
    for action in answer.get("next_actions", []):
        st.write(f"➡️ {action['title']}  `{action['ref']}`")
    with st.expander("Claims, findings and steps"):
        st.json({"claims": answer.get("claims", []), "findings": result.get("findings", []), "steps": result.get("steps", [])})


def goal_panel(store: TwinStore, packs: dict, user_id: str, today: date) -> None:
    st.header("1. What do you want to do?")
    goal = st.text_area("Your goal", value="I want to apply for İkamet", height=70)
    if st.button("Ask the agent", type="primary"):
        with st.spinner("The agent is working (this can take a few minutes on a laptop)..."):
            result = run_agent(goal, store, packs, user_id, today, OllamaBackend(), outbox_dir=OUTBOX_DIR)
        st.session_state["last_result"] = result.model_dump(mode="json")
    if "last_result" in st.session_state:
        show_result(st.session_state["last_result"])


def documents_panel(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    st.header("2. Your documents")
    types = [AUTO_TYPE] + [d.doc_type for d in pack.procedure.documents]
    with st.form("upload"):
        uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
        doc_type = st.selectbox("Document type", types)
        submitted = st.form_submit_button("Upload and analyze")
    if submitted and uploaded is not None:
        path = save_upload(uploaded.name, uploaded.getvalue(), ctx.user_id)
        doc_id = doc_id_for(path)
        service.register_upload(store, ctx.user_id, doc_id, str(path), pack.id)
        args = {"doc_id": doc_id} if doc_type == AUTO_TYPE else {"doc_id": doc_id, "doc_type": doc_type}
        with st.spinner("Reading the document..."):
            observation = run_tool(build_registry(), "extract_document", args, ctx, "orchestrator")
        if observation.ok and observation.output["error"]:
            st.error(friendly_error(observation.output["error"]))
    for document in store.list_documents(ctx.user_id):
        issues = (document.observation or {}).get("issues", [])
        st.write(f"**{document.doc_type or 'unknown type'}** · {document.status} · `{document.doc_id}`")
        if document.processing_error:
            st.caption(friendly_error(document.processing_error))
        for issue in issues[:3]:
            st.caption(f"⚠️ {issue}")
        if st.button("Delete", key=f"delete_{document.doc_id}"):
            service.delete_document(store, ctx.user_id, pack.id, document.doc_id)
            st.rerun()


def facts_panel(store: TwinStore, pack: Pack, user_id: str, view: FactView) -> None:
    st.header("3. What the Twin knows")
    st.dataframe(fact_rows(view), hide_index=True)
    for fact in view.unconfirmed:
        cols = st.columns([4, 1])
        cols[0].write(f"Waiting for your confirmation: **{fact.field} = {fact.value}** ({fact.source_ref})")
        if cols[1].button("Confirm", key=f"confirm_{fact.fact_id}"):
            store.confirm_fact(user_id, fact.fact_id)
            st.rerun()
    for field, values in view.conflicts.items():
        choice = st.selectbox(f"Sources disagree on {field}. Which value is right?", [str(v) for v in values], key=f"c_{field}")
        if st.button("Use this value", key=f"resolve_{field}"):
            service.resolve_conflict(store, pack, user_id, field, choice)
            st.rerun()
    with st.form("enter_value"):
        field = st.selectbox("Enter or correct a value", [f.name for f in pack.procedure.fields])
        raw = st.text_input("Value (dates YYYY-MM-DD, yes/no for questions)")
        if st.form_submit_button("Save as confirmed"):
            service.record_app_value(store, pack, user_id, field, raw)
            st.rerun()


def status_panel(pack: Pack, view: FactView) -> None:
    st.header("4. Where you are")
    values = view.fact_map()
    for section in sorted(pack.procedure.sections, key=lambda s: s.order):
        rows = rule_rows(pack, values, section.id)
        done = sum(row["status"].endswith("pass") for row in rows)
        with st.expander(f"{section.title}: {done}/{len(rows)} pass"):
            st.dataframe(rows, hide_index=True)


def sandbox_panel(store: TwinStore, pack: Pack, user_id: str, today: date) -> None:
    st.header("5. Try a step in the sandbox")
    if not pack.procedure.stages:
        st.caption("This procedure has no stages to simulate.")
        return
    stage = st.selectbox("Stage", [s.id for s in pack.procedure.stages])
    if st.button("Simulate (nothing is submitted)"):
        outcome, _ = run_simulation(store, pack, user_id, today, stage)
        (st.success if outcome.accepted else st.warning)(outcome.message)


def actions_panel(store: TwinStore, packs: dict, user_id: str, today: date) -> None:
    st.header("6. Proposed actions (your approval is required)")
    for proposal in store.list_proposals(user_id):
        st.write(f"**{proposal.title}** · {proposal.status}" + (f" · needs stage `{proposal.requires_stage}`" if proposal.requires_stage else ""))
        if proposal.note:
            st.caption(proposal.note)
        if proposal.status in ("proposed", "blocked"):
            approve, deny = st.columns(2)
            if approve.button("Approve", key=f"approve_{proposal.action_id}"):
                st.info(approve_action(store, packs, user_id, proposal.action_id, today, OUTBOX_DIR).message)
            if deny.button("Deny", key=f"deny_{proposal.action_id}"):
                deny_action(store, user_id, proposal.action_id)
                st.rerun()
        if proposal.status == "executed" and proposal.artifact_path and Path(proposal.artifact_path).is_file():
            path = Path(proposal.artifact_path)
            st.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name, key=f"dl_{proposal.action_id}")


def outcome_panel(store: TwinStore, pack: Pack, user_id: str) -> None:
    st.header("7. Tell the Twin what happened")
    with st.form("outcome"):
        description = st.text_input("What happened? (e.g. 'IRO confirmed my scans')")
        field = st.selectbox("Which fact does it change?", [f.name for f in pack.procedure.fields])
        raw = st.text_input("New value")
        if st.form_submit_button("Record outcome") and description and raw:
            service.record_outcome(store, pack, user_id, "user_report", description, {field: raw})
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Personal Bureaucracy Agent", layout="wide")
    packs = load_all_packs()
    with TwinStore(DATABASE_PATH) as store:
        user_id, today = sidebar(store, packs)
        try:
            goal_panel(store, packs, user_id, today)
            ctx = tool_context(store, packs, user_id, today)
            pack = ctx.pack()
            if pack is None:
                st.info("Choose a procedure in the sidebar, or ask the agent, to see documents and status.")
                return
            view = ctx.view()
            documents_panel(store, ctx, pack)
            facts_panel(store, pack, user_id, view)
            status_panel(pack, view)
            sandbox_panel(store, pack, user_id, today)
            actions_panel(store, packs, user_id, today)
            outcome_panel(store, pack, user_id)
        except ValueError as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
