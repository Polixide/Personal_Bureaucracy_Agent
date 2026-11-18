"""Streamlit app for the Personal Bureaucracy Agent: streamlit run app.py

The page answers the three questions of the Digital Twin (team presentation, slide 2):
"Where am I?", "What is missing?" and "What should I do next?".

- Top: the goal box. The agent (orchestrator + workers, checked by Gate 1) answers in plain words.
- Tab "Where am I?": progress per section, the uploaded documents, everything the Twin knows.
- Tab "What is missing?": add a document, check uncertain values, see the problems, answer open questions.
- Tab "What next?": actions waiting for approval (with a sandbox check when a step needs one), the
  sandbox, and "something happened" (real -> twin).

The Twin (SQLite at config.DATABASE_PATH) is the only source of truth. Verdicts come from src.rules, never
from the model. Approved artifacts are files in data/outbox/; nothing is sent, booked, paid or submitted.
"""
from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st

from config import DATABASE_PATH, LLM_MODEL, OUTBOX_DIR, PROJECT_ROOT, UPLOAD_DIR
from src.agent.orchestrator import run_agent
from src.llm import OllamaBackend
from src.models import Pack, RuleResult, Section
from src.pack import load_all_packs
from src.rules import evaluate_rules
from src.sandbox.approval import approve_action, deny_action
from src.sandbox.mock_world import SimulationOutcome, run_simulation
from src.tools.common import ToolContext
from src.tools.registry import build_registry, run_tool
from src.twin import service
from src.twin.facts import FactView
from src.twin.models import ProposedAction, TwinDocument
from src.twin.store import TwinStore

AUTO_TYPE = "detect automatically"
NO_PROCEDURE = "(let the agent decide)"
NOT_ANSWERED = "not answered"
STATUS_ICONS = {"pass": "✅", "fail": "❌", "unknown": "❔"}
TOOL_ICONS = {"create_ics": "📅", "draft_email": "✉️", "fill_pdf": "📄", "build_checklist": "☑️"}
ORIGIN_WORDS = {"document": "a document", "user_stated": "you", "tool": "a tool", "derived": "computed"}
OPEN_PROPOSALS = ("proposed", "blocked")
# Fixed names: Streamlit remembers the open tab by name, so a click does not jump back to the first tab.
TAB_NAMES = ["📍 Where am I?", "🧩 What is missing?", "➡️ What next?"]
EARLIEST_DATE = date(1900, 1, 1)
LATEST_DATE = date(2100, 12, 31)


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
    """Stable document id for a stored upload (same file, same id)."""
    return "doc_" + sha256(str(path).encode()).hexdigest()[:12]


def friendly_error(message: str) -> str:
    """Plain-language explanation of extraction / model problems."""
    text = message.lower()
    if "ocr" in text:
        return "The scanned pages could not be read. Try a clearer scan or a text-based PDF."
    if "timed out" in text or "timeout" in text:
        return "The local model timed out. Retry when Ollama is ready, or type the values yourself."
    if "no extractable text" in text or "contains no text" in text:
        return "No readable text was found. Try a clearer scan, a text-based PDF or a TXT file."
    if "password" in text:
        return "Password-protected PDFs are not supported."
    if any(word in text for word in ("ollama", "connect", "unavailable")):
        return "The local model is not running. Start Ollama, or type the values yourself."
    return "Processing could not finish. Check the document and retry."


def friendly_issue(issue: str) -> str:
    """Extraction warnings in plain words: 'insurance_end: value absent ...; discarded' -> a short sentence."""
    field, separator, reason = issue.partition(": ")
    if not separator or " " in field:
        return issue
    if "discarded" in reason or "absent" in reason:
        return f"Could not read the {readable(field)} reliably; please type it in 'What is missing?'."
    if "manual review" in reason or "conflicting" in reason:
        return f"The {readable(field)} is unclear in the document; please check it."
    return f"{readable(field)}: {reason}"


def follow_up_goal(goal: str, question: str, answer: str) -> str:
    """The agent keeps no chat history, so a reply to its question is sent together with the original goal."""
    return f"{goal}\nYou asked: {question}\nMy answer: {answer}"


def readable(name: str) -> str:
    """'passport_expiry_date' -> 'passport expiry date'."""
    return name.replace("_", " ")


def field_question(pack: Pack, field: str) -> str:
    """The pack's question for a field, or its name as a short question when the pack has none."""
    spec = pack.field(field)
    if spec is not None and spec.question:
        return spec.question
    label = readable(field).capitalize()
    return f"{label}?" if spec is not None and spec.type == "bool" else label


def pack_label(pack: Pack) -> str:
    """Procedure title as shown to the user, with a warning for unverified procedures."""
    if pack.procedure.status == "unverified":
        return f"{pack.procedure.title} (rules not verified yet)"
    return pack.procedure.title


def user_sections(pack: Pack) -> list[Section]:
    """Sections about the user's file, in order. Sections with delegate=false are the agent's guardrails."""
    ordered = sorted(pack.procedure.sections, key=lambda section: section.order)
    return [section for section in ordered if section.delegate]


def in_section(results: list[RuleResult], section_id: str) -> list[RuleResult]:
    return [result for result in results if result.section == section_id]


def user_results(pack: Pack, results: list[RuleResult]) -> list[RuleResult]:
    """Verdicts of the user sections only."""
    wanted = {section.id for section in user_sections(pack)}
    return [result for result in results if result.section in wanted]


def count_status(results: list[RuleResult]) -> dict[str, int]:
    return {status: sum(result.status == status for result in results) for status in ("pass", "fail", "unknown")}


def section_icon(results: list[RuleResult]) -> str:
    """✅ everything met, ❌ at least one problem, ⏳ still waiting for information."""
    counts = count_status(results)
    if counts["pass"] == len(results):
        return "✅"
    if counts["fail"]:
        return "❌"
    return "⏳"


def rule_label(result: RuleResult) -> str:
    """The requirement in plain words; unverified rules always say so (CLAUDE.md hard rule)."""
    if result.rule_status == "unverified":
        return f"{result.description} (unverified rule)"
    return result.description


def rule_line(result: RuleResult) -> str:
    """One requirement as a line of text: icon, requirement, and what is missing to decide it."""
    line = f"{STATUS_ICONS[result.status]} {rule_label(result)}"
    if result.missing_fields:
        line += f" *(missing: {', '.join(readable(field) for field in result.missing_fields)})*"
    return line


def fact_origin(view: FactView, field: str) -> str:
    """Where a known value came from, in plain words."""
    if field in view.derived:
        return "computed"
    newest = max(view.sources[field], key=lambda fact: fact.fact_id or 0)
    return ORIGIN_WORDS.get(newest.provenance, newest.provenance)


def fact_rows(view: FactView) -> list[dict]:
    """Everything the Twin knows: value and where it came from."""
    return [{"fact": readable(field), "value": str(view.values[field]), "from": fact_origin(view, field)}
            for field in sorted(view.values)]


def missing_fields(results: list[RuleResult]) -> list[str]:
    """Fields that unknown rules wait for, without duplicates, in rule order."""
    fields: list[str] = []
    for result in results:
        if result.status != "unknown":
            continue
        for field in result.missing_fields:
            if field not in fields:
                fields.append(field)
    return fields


def askable_fields(pack: Pack, missing: list[str]) -> list[str]:
    """Fields the user can answer. A derived field is replaced by the fields it is computed from."""
    derived = {spec.field: spec for spec in pack.procedure.derived}
    asked: list[str] = []
    for field in missing:
        inputs = derived[field].args if field in derived else [field]
        for name in inputs:
            if name not in asked and name not in derived and pack.field(name) is not None:
                asked.append(name)
    return asked


def documents_for(pack: Pack, missing: list[str]) -> list[str]:
    """Descriptions of the documents whose upload would fill some of the missing fields."""
    derived = {spec.field: spec for spec in pack.procedure.derived}
    wanted, doc_types = set(missing), set()
    for field in missing:
        if field in derived:
            wanted.update(derived[field].args)
            doc_types.add(derived[field].doc_type)
    descriptions = []
    for document in pack.procedure.documents:
        extracted = {extract.field for extract in document.extracts}
        if document.doc_type in doc_types or extracted & wanted:
            descriptions.append(document.description)
    return descriptions


def open_item_count(pack: Pack, results: list[RuleResult], view: FactView) -> int:
    """Problems + questions + values waiting for a check: the number on the 'What is missing?' tab."""
    mine = user_results(pack, results)
    questions = askable_fields(pack, missing_fields(mine))
    return count_status(mine)["fail"] + len(questions) + len(view.unconfirmed) + len(view.conflicts)


def sandbox_lines(pack: Pack, outcome: SimulationOutcome) -> list[str]:
    """Why the mock portal refused a step, in plain words."""
    lines = [f"Problem: {rule_label_by_id(pack, rule_id)}" for rule_id in outcome.failing]
    for rule_id, fields in outcome.unknown.items():
        lines.append(f"Missing for '{rule_label_by_id(pack, rule_id)}': {', '.join(readable(f) for f in fields)}")
    return lines


def rule_label_by_id(pack: Pack, rule_id: str) -> str:
    rule = pack.rule(rule_id)
    return rule.description if rule is not None else rule_id


def document_title(pack: Pack, document: TwinDocument) -> str:
    spec = pack.document(document.doc_type) if document.doc_type else None
    return spec.description if spec is not None else (document.doc_type or "Document of unknown type")


def stage_title(pack: Pack | None, stage_id: str) -> str:
    stage = pack.stage(stage_id) if pack is not None else None
    return stage.title if stage is not None else stage_id


# ------------------------------------------------------- page messages --
FLASH_STYLES = {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}


def flash(message: str, kind: str = "info") -> None:
    """Keep a message across the rerun that follows a button click."""
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        FLASH_STYLES[kind](message)


# --------------------------------------------------------------- sidebar --
def sidebar(store: TwinStore, packs: dict[str, Pack]) -> tuple[str, date]:
    """Who you are, which procedure, which day. Returns (user id, today)."""
    st.sidebar.title("Personal Bureaucracy Agent")
    st.sidebar.caption("Always know where you are, what is missing and what to do next.")
    user_id = st.sidebar.text_input("Your id (one Digital Twin per id)", value="demo-user").strip() or "demo-user"
    choose_procedure(store, packs, user_id)
    today = st.sidebar.date_input("Today's date (change it to test deadlines)", value=date.today())
    st.sidebar.caption(f"Local model: {LLM_MODEL}. Nothing is sent, booked, paid or submitted: "
                       "approved items become files you download.")
    delete_twin_box(store, user_id)
    return user_id, today


def choose_procedure(store: TwinStore, packs: dict[str, Pack], user_id: str) -> None:
    active = store.active_procedure(user_id)
    labels = {NO_PROCEDURE: NO_PROCEDURE}
    for procedure_id in sorted(packs):
        labels[procedure_id] = pack_label(packs[procedure_id])
    choices = list(labels)
    index = choices.index(active) if active in packs else 0
    chosen = st.sidebar.selectbox("Procedure", choices, index=index, format_func=labels.get)
    if chosen != NO_PROCEDURE and chosen != active:
        store.set_active_procedure(user_id, chosen)


def delete_twin_box(store: TwinStore, user_id: str) -> None:
    with st.sidebar.expander("Delete my Digital Twin"):
        typed = st.text_input("Type your id to confirm", key="reset")
        if st.button("Delete everything", disabled=typed != user_id):
            store.delete_user(user_id)
            st.session_state.pop("results", None)
            st.rerun()


# ------------------------------------------------------------- the agent --
def agent_box(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date) -> None:
    """Goal box and the agent's last answer for this user."""
    st.title("What do you want to do?")
    with st.form("goal"):
        goal = st.text_input("Your goal, in your own words", value="I want to apply for İkamet")
        asked = st.form_submit_button("Ask the agent", type="primary")
    if asked and goal.strip():
        ask_agent(store, packs, user_id, today, goal.strip())
    last = st.session_state.get("results", {}).get(user_id)
    if last:
        show_answer(store, packs, user_id, today, last)


def ask_agent(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date, goal: str) -> None:
    """One agent turn; the page reloads so every tab shows the Twin after the run."""
    with st.spinner("The agent is working. On a laptop this can take a few minutes..."):
        result = run_agent(goal, store, packs, user_id, today, OllamaBackend(), outbox_dir=OUTBOX_DIR)
    st.session_state.setdefault("results", {})[user_id] = {"goal": goal, "result": result.model_dump(mode="json")}
    st.rerun()


def show_answer(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date, last: dict) -> None:
    result = last["result"]
    with st.container(border=True):
        if result["status"] == "needs_user_input":
            reply_box(store, packs, user_id, today, last["goal"], result["question"])
            return
        answer = result.get("answer") or {}
        st.markdown(answer.get("summary", ""))
        for number, action in enumerate(answer.get("next_actions", []), start=1):
            st.markdown(f"{number}. {action['title']}")
        if result.get("needs_human_review"):
            codes = ", ".join(sorted({issue["code"] for issue in result["gate_issues"]}))
            st.warning(f"This answer did not pass every automatic check ({codes}). Read it with care.")
        if result.get("fallback_used"):
            st.caption("Built from the rules alone: the language model was not usable for this answer.")
        with st.expander("How the agent got there"):
            st.dataframe(step_rows(result), hide_index=True)


def step_rows(result: dict) -> list[dict]:
    """The orchestrator's steps: what it thought, which tool it called, whether the call worked.

    Small models often leave the thought empty; the column is shown only when there is something in it.
    """
    steps = result.get("steps", [])
    show_thoughts = any(step["thought"] for step in steps)
    rows = []
    for step in steps:
        row = {"step": step["step"]}
        if show_thoughts:
            row["thought"] = step["thought"]
        row["tool"] = step["action"]
        row["ok"] = "✅" if step["ok"] else f"❌ {step.get('error') or ''}"
        rows.append(row)
    return rows


def reply_box(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date, goal: str, question: str) -> None:
    st.info(f"The agent asks: {question}")
    with st.form("reply"):
        reply = st.text_input("Your answer")
        if st.form_submit_button("Send") and reply.strip():
            ask_agent(store, packs, user_id, today, follow_up_goal(goal, question, reply.strip()))


# -------------------------------------------------------- tab: where am I --
def where_tab(store: TwinStore, pack: Pack, user_id: str, view: FactView, results: list[RuleResult]) -> None:
    mine = user_results(pack, results)
    met = count_status(mine)["pass"]
    st.progress(met / max(len(mine), 1), text=f"{met} of {len(mine)} requirements met")
    for section in user_sections(pack):
        section_results = in_section(results, section.id)
        met_here = count_status(section_results)["pass"]
        with st.expander(f"{section_icon(section_results)} {section.title} · {met_here}/{len(section_results)}"):
            for result in section_results:
                st.markdown(rule_line(result))
    documents_list(store, pack, user_id)
    with st.expander(f"Everything the Twin knows ({len(view.values)} facts)"):
        st.dataframe(fact_rows(view), hide_index=True)


def documents_list(store: TwinStore, pack: Pack, user_id: str) -> None:
    st.subheader("Your documents")
    documents = store.list_documents(user_id)
    if not documents:
        st.caption("No documents yet. Add them in the tab 'What is missing?'.")
    for document in documents:
        text, remove = st.columns([5, 1])
        text.markdown(f"📄 **{document_title(pack, document)}** · {document.status}")
        if document.processing_error:
            text.caption(friendly_error(document.processing_error))
        for issue in (document.observation or {}).get("issues", [])[:3]:
            text.caption(f"⚠️ {friendly_issue(issue)}")
        if remove.button("Remove", key=f"remove_{document.doc_id}"):
            service.delete_document(store, user_id, pack.id, document.doc_id)
            st.rerun()


# --------------------------------------------------- tab: what is missing --
def missing_tab(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, results: list[RuleResult]) -> None:
    upload_form(store, ctx, pack)
    checks(store, pack, ctx.user_id, view)
    problems(user_results(pack, results))
    open_questions(store, pack, ctx.user_id, results)


def upload_form(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """Real -> twin: a document is read, and its values become facts of the Twin."""
    st.subheader("Add a document")
    labels = {AUTO_TYPE: AUTO_TYPE}
    for document in pack.procedure.documents:
        labels[document.doc_type] = document.description
    with st.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
        doc_type = st.selectbox("Which document is it?", list(labels), format_func=labels.get)
        submitted = st.form_submit_button("Upload and read")
    if submitted and uploaded is not None:
        read_upload(store, ctx, pack, uploaded.name, uploaded.getvalue(), doc_type)


def read_upload(store: TwinStore, ctx: ToolContext, pack: Pack, name: str, content: bytes, doc_type: str) -> None:
    path = save_upload(name, content, ctx.user_id)
    doc_id = doc_id_for(path)
    service.register_upload(store, ctx.user_id, doc_id, str(path), pack.id)
    args = {"doc_id": doc_id} if doc_type == AUTO_TYPE else {"doc_id": doc_id, "doc_type": doc_type}
    with st.spinner("Reading the document..."):
        observation = run_tool(build_registry(), "extract_document", args, ctx, "orchestrator")
    if not observation.ok:
        flash(friendly_error(observation.error or ""), "error")
    elif observation.output["error"]:
        flash(friendly_error(observation.output["error"]), "error")
    else:
        found = sum(len(row["facts"]) for row in observation.output["documents"])
        flash(f"Document read: {found} values found.", "success")
    st.rerun()


def checks(store: TwinStore, pack: Pack, user_id: str, view: FactView) -> None:
    """Values that wait for the user: unconfirmed statements and sources that disagree."""
    if not view.unconfirmed and not view.conflicts:
        return
    st.subheader("Please check")
    for fact in view.unconfirmed:
        text, confirm = st.columns([4, 1])
        origin = ORIGIN_WORDS.get(fact.provenance, fact.provenance)
        text.markdown(f"{readable(fact.field)}: **{fact.value}** (from {origin})")
        if confirm.button("Confirm", key=f"confirm_{fact.fact_id}"):
            store.confirm_fact(user_id, fact.fact_id)
            st.rerun()
    for field, values in view.conflicts.items():
        choice = st.radio(f"Your documents disagree on {readable(field)}. Which value is right?",
                          [str(value) for value in values], key=f"conflict_{field}", horizontal=True)
        if st.button("Use this value", key=f"resolve_{field}"):
            service.resolve_conflict(store, pack, user_id, field, choice)
            st.rerun()


def problems(results: list[RuleResult]) -> None:
    failing = [result for result in results if result.status == "fail"]
    if not failing:
        return
    st.subheader("Problems to fix")
    for result in failing:
        st.error(rule_label(result), icon="❌")


def open_questions(store: TwinStore, pack: Pack, user_id: str, results: list[RuleResult]) -> None:
    """Per section: the questions that would turn unknown requirements into known ones."""
    st.subheader("Open questions")
    asked_any = False
    for section in user_sections(pack):
        missing = missing_fields(in_section(results, section.id))
        fields = askable_fields(pack, missing)
        if not fields:
            continue
        asked_any = True
        with st.expander(f"{section.title} ({len(fields)})"):
            documents = documents_for(pack, missing)
            if documents:
                st.caption("Uploading these documents answers some of them: " + "; ".join(documents))
            answers_form(store, pack, user_id, section.id, fields)
    if not asked_any:
        st.success("No open questions: the Twin has everything the rules need.")


def answers_form(store: TwinStore, pack: Pack, user_id: str, section_id: str, fields: list[str]) -> None:
    with st.form(f"answers_{section_id}"):
        values = {field: field_input(pack, field, f"{section_id}_{field}") for field in fields}
        if st.form_submit_button("Save answers"):
            save_answers(store, pack, user_id, values)


def field_input(pack: Pack, field: str, key: str):
    """One input widget that fits the field type. Returns None when the user left it empty."""
    label = field_question(pack, field)
    kind = pack.field(field).type
    if kind == "bool":
        choice = st.radio(label, [NOT_ANSWERED, "yes", "no"], horizontal=True, key=key)
        return None if choice == NOT_ANSWERED else choice
    if kind == "date":
        return st.date_input(label, value=None, min_value=EARLIEST_DATE, max_value=LATEST_DATE, key=key)
    if kind == "int":
        return st.number_input(label, value=None, step=1, key=key)
    return st.text_input(label, key=key).strip() or None


def save_answers(store: TwinStore, pack: Pack, user_id: str, values: dict) -> None:
    """Typed answers are explicit, so they are stored as confirmed user statements."""
    saved, errors = 0, []
    for field, value in values.items():
        if value is None:
            continue
        raw = value.isoformat() if isinstance(value, date) else value
        try:
            service.record_app_value(store, pack, user_id, field, raw)
            saved += 1
        except ValueError as error:
            errors.append(f"{readable(field)}: {error}")
    if errors:
        flash("Not saved: " + "; ".join(errors), "error")
    if saved:
        flash(f"{saved} answers saved.", "success")
    st.rerun()


# ------------------------------------------------------- tab: what next --
def next_tab(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    approvals(store, ctx)
    sandbox_box(store, pack, ctx.user_id, ctx.today)
    outcome_box(store, pack, ctx.user_id)


def approvals(store: TwinStore, ctx: ToolContext) -> None:
    """Twin -> real: prepared artifacts wait here; approving writes a file, nothing is sent."""
    st.subheader("Waiting for your approval")
    proposals = store.list_proposals(ctx.user_id)
    waiting = [proposal for proposal in proposals if proposal.status in OPEN_PROPOSALS]
    if not waiting:
        st.caption("Nothing is waiting. Ask the agent to prepare a reminder, an e-mail or a filled form.")
    for proposal in waiting:
        proposal_card(store, ctx, proposal)
    if st.button("☑️ Prepare my checklist"):
        observation = run_tool(build_registry(), "build_checklist", {}, ctx, "orchestrator")
        flash("Checklist prepared: approve it to download it." if observation.ok else str(observation.error), "info")
        st.rerun()
    done = [proposal for proposal in proposals if proposal.status not in OPEN_PROPOSALS]
    if done:
        with st.expander(f"Done ({len(done)})"):
            for proposal in done:
                done_row(proposal)


def proposal_card(store: TwinStore, ctx: ToolContext, proposal: ProposedAction) -> None:
    pack = ctx.packs.get(proposal.procedure_id)
    with st.container(border=True):
        st.markdown(f"{TOOL_ICONS.get(proposal.tool, '•')} **{proposal.title}**")
        if proposal.note:
            st.caption(proposal.note)
        if proposal.requires_stage:
            st.caption(f"Needs a successful sandbox check of: {stage_title(pack, proposal.requires_stage)}")
        buttons = st.container(horizontal=True)
        if buttons.button("Approve", key=f"approve_{proposal.action_id}", type="primary"):
            result = approve_action(store, ctx.packs, ctx.user_id, proposal.action_id, ctx.today, OUTBOX_DIR)
            flash(result.message, "success" if result.status == "executed" else "warning")
            st.rerun()
        if buttons.button("Deny", key=f"deny_{proposal.action_id}"):
            flash(deny_action(store, ctx.user_id, proposal.action_id).message)
            st.rerun()
        if proposal.requires_stage and pack is not None:
            if buttons.button("Check in sandbox", key=f"check_{proposal.action_id}"):
                try_stage(store, pack, ctx.user_id, ctx.today, proposal.requires_stage)


def done_row(proposal: ProposedAction) -> None:
    text, download = st.columns([3, 2])
    text.markdown(f"{TOOL_ICONS.get(proposal.tool, '•')} {proposal.title} · {proposal.status}")
    path = Path(proposal.artifact_path) if proposal.artifact_path else None
    if proposal.status == "executed" and path is not None and path.is_file():
        download.download_button("Download", path.read_bytes(), file_name=path.name, key=f"dl_{proposal.action_id}")


def sandbox_box(store: TwinStore, pack: Pack, user_id: str, today: date) -> None:
    """The mock world: applies the step's rules to what the Twin knows. Nothing is submitted."""
    st.subheader("Try an official step in the sandbox")
    st.caption("A mock portal checks the step against the rules and your facts. Nothing is submitted.")
    labels = {stage.id: stage.title for stage in pack.procedure.stages}
    if not labels:
        st.caption("This procedure has no official steps to try.")
        return
    stage_id = st.selectbox("Step", list(labels), format_func=labels.get)
    if st.button("Try this step"):
        try_stage(store, pack, user_id, today, stage_id)


def try_stage(store: TwinStore, pack: Pack, user_id: str, today: date, stage_id: str) -> None:
    outcome, _ = run_simulation(store, pack, user_id, today, stage_id)
    title = stage_title(pack, stage_id)
    if outcome.accepted:
        flash(f"The mock portal would accept '{title}'.", "success")
    else:
        flash(f"The mock portal would refuse '{title}'.\n\n- " + "\n- ".join(sandbox_lines(pack, outcome)), "warning")
    st.rerun()


def outcome_box(store: TwinStore, pack: Pack, user_id: str) -> None:
    """Real -> twin: what happened outside (an office answer, a corrected value) updates the Twin."""
    with st.expander("Something happened, or a value is wrong? Tell the Twin"):
        labels = {field.name: readable(field.name) for field in pack.procedure.fields}
        with st.form("outcome", clear_on_submit=True):
            description = st.text_input("What happened?", placeholder="IRO confirmed my scans")
            field = st.selectbox("Which fact changes?", list(labels), format_func=labels.get)
            raw = st.text_input("New value (dates as YYYY-MM-DD, yes or no for questions)")
            if st.form_submit_button("Save") and description.strip() and raw.strip():
                service.record_outcome(store, pack, user_id, "user_report", description.strip(), {field: raw.strip()})
                flash("Saved. The Twin and the requirements are updated.", "success")
                st.rerun()


# ------------------------------------------------------------------ page --
def tool_context(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date) -> ToolContext:
    active = store.active_procedure(user_id)
    return ToolContext(store=store, packs=packs, user_id=user_id, today=today, backend=OllamaBackend(),
                       procedure_id=active if active in packs else None)


def twin_tabs(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """The three questions of the Digital Twin, one tab each."""
    view = ctx.view()
    results = evaluate_rules(pack, view.fact_map())
    if pack.procedure.status == "unverified":
        st.warning("The rules of this procedure are not verified yet. Check every result with the official office.")
    waiting = len(store.list_proposals(ctx.user_id, "proposed")) + len(store.list_proposals(ctx.user_id, "blocked"))
    where, missing, next_steps = st.tabs(TAB_NAMES, key="tab", on_change="rerun")
    with where:
        where_tab(store, pack, ctx.user_id, view, results)
    with missing:
        st.caption(f"Open items: {open_item_count(pack, results, view)}")
        missing_tab(store, ctx, pack, view, results)
    with next_steps:
        st.caption(f"Waiting for your approval: {waiting}")
        next_tab(store, ctx, pack)


def main() -> None:
    st.set_page_config(page_title="Personal Bureaucracy Agent", page_icon="🗂️")
    packs = load_all_packs()
    with TwinStore(DATABASE_PATH) as store:
        user_id, today = sidebar(store, packs)
        try:
            agent_box(store, packs, user_id, today)
            show_flash()
            ctx = tool_context(store, packs, user_id, today)
            pack = ctx.pack()
            if pack is None:
                st.info("Ask the agent, or choose a procedure in the sidebar, to see where you are.")
                return
            twin_tabs(store, ctx, pack)
        except ValueError as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
