"""Streamlit app for the Personal Bureaucracy Agent: streamlit run app.py

Layout from Neda's interface (status bar, the decision first, tabs); engine from our architecture
(orchestrator + workers checked by Gate 1, the Digital Twin, the rule engine and the sandbox).

- Status bar ("Where am I?"): documents ready, checks passed, the agent's last status.
- Decision card: actions waiting for approval come before anything else.
- Assistant (the main tab): a chat with the assistant (src/agent/chat.py). It opens with the plan computed by
  code from the Twin (src/agent/briefing.py), answers questions in plain words, records what the user says
  (unconfirmed until confirmed), prepares artifacts for approval, and reads documents attached to a message.
  Beside it: the document checklist, a full check by the agent team (orchestrator + workers), and the tools.
- Other tabs: Documents, Your details ("What is missing?"), Checks (every requirement with its official
  source) and Technical (agent steps, Gate 1, run log, lessons, knowledge base).

The Twin (SQLite at config.DATABASE_PATH) is the only source of truth. Verdicts come from src.rules, never
from the model. Approved artifacts are files in data/outbox/; nothing is sent, booked, paid or submitted.
There is one local model, so a profile runs one agent turn or document reading at a time.
"""
import threading
from collections.abc import Callable
from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st

from config import CHAT_MODEL, DATABASE_PATH, LLM_MODEL, OUTBOX_DIR, PROJECT_ROOT, UPLOAD_DIR, VISION_MODEL
from src.agent import orchestrator
from src.agent.briefing import (OPEN_PROPOSALS, ORIGIN_WORDS, askable_fields, briefing_markdown, build_briefing,
                                checklist_rows, count_status, document_status, documents_for, field_question,
                                in_section, missing_fields, readable, user_results, user_sections)
from src.agent.chat import run_chat
from src.llm import OllamaBackend
from src.models import Pack, RuleResult
from src.pack import load_all_packs
from src.rag.crawl import load_corpus
from src.rag.index import pack_chunks
from src.rules import evaluate_rules
from src.sandbox.approval import approve_action, deny_action
from src.sandbox.mock_world import SimulationOutcome, run_simulation
from src.tools.common import ToolContext
from src.tools.open_links import open_links
from src.tools.registry import build_registry, run_tool
from src.twin import service
from src.twin.facts import FactView
from src.twin.models import ChatMessage, ProposedAction, TwinDocument
from src.twin.store import TwinStore

__all__ = ["askable_fields", "checklist_rows", "document_status", "documents_for", "field_question", "missing_fields"]

AUTO_TYPE = "detect automatically"
NO_PROCEDURE = "(let the agent decide)"
NOT_ANSWERED = "not answered"
UPLOAD_TYPES = ["pdf", "txt", "jpg", "jpeg", "png"]
STATUS_ICONS = {"pass": "✅", "fail": "❌", "unknown": "❔"}
TOOL_ICONS = {"create_ics": ":material/event:", "draft_email": ":material/mail:", "fill_pdf": ":material/picture_as_pdf:",
              "build_checklist": ":material/checklist:"}
VIEWS = {"assistant": ":material/forum: Assistant", "documents": ":material/description: Documents",
         "details": ":material/person: Your details", "checks": ":material/fact_check: Checks",
         "technical": ":material/code: Technical"}
CHECKLIST_ICONS = {"complete": ":green[:material/check_circle:]", "needs confirmation": ":orange[:material/rate_review:]",
                   "missing": ":gray[:material/radio_button_unchecked:]", "ambiguous": ":orange[:material/help:]"}
CHECKLIST_LABELS = {"complete": "Ready", "needs confirmation": "Needs your review", "missing": "Missing",
                    "ambiguous": "Unclear"}
RUN_LABELS = {"answered": "Answer ready", "needs_user_input": "The agent asked you a question",
              "needs_human_review": "Answer needs your review", "fallback": "Answer built from the rules alone"}
REPLY_NOTES = {"answered": "Checked by code before it was shown (Gate 1).",
               "fallback": "Written by code from your plan: the model's answer was not usable or did not pass the checks.",
               "needs_human_review": "This answer did not pass every automatic check: read it with care.",
               "needs_user_input": "The agent needs an answer from you."}
FOUND_BY = {"title": "recognised by its title", "mrz": "recognised by its machine-readable zone",
            "keywords": "recognised by its content", "model": "recognised by the language model",
            "vision": "recognised from the image by the vision model", "user": "read as the type you chose"}
SUGGESTIONS = ["Explain where I am, step by step", "What should I do first?", "Why is something blocking me?",
               "Prepare a checklist for me"]
FIRST_SUGGESTIONS = ["I want to apply for my first İkamet", "I lost my TC kimlik", "How do I get a tax number?",
                     "Do I have to register my phone?"]
WELCOME = ("Hi! Tell me what you need — for example *I want to apply for İkamet*, *I lost my TC kimlik* or "
           "*How do I get a tax number?* — and I will tell you what to upload and what to do, step by step.")
FULL_CHECK_GOAL = "Check my whole file section by section and tell me what to do next."
EARLIEST_DATE = date(1900, 1, 1)
LATEST_DATE = date(2100, 12, 31)
BUSY_MESSAGE = "Still processing your last action. This page updates when it finishes."
CHAT_SHOWN = 30


# ---------------------------------------------------------- pure helpers --
def save_upload(name: str, content: bytes, user_id: str, directory: Path = UPLOAD_DIR) -> Path:
    """Store an upload under a content/user-derived name inside the project, never a browser path."""
    extension = Path(name).suffix.lower()
    if extension.lstrip(".") not in UPLOAD_TYPES:
        raise ValueError("Unsupported file. Choose a PDF, a TXT file or a photo (JPG or PNG).")
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
    if any(word in reason for word in ("discarded", "absent", "not present", "not found", "does not exist")):
        return f"Could not read the {readable(field)} reliably; please type it in 'Your details'."
    if "manual review" in reason or "conflicting" in reason:
        return f"The {readable(field)} is unclear in the document; please check it."
    return f"{readable(field)}: {reason}"


def follow_up_goal(goal: str, question: str, answer: str) -> str:
    """The orchestrator keeps no chat history, so a reply to its question is sent with the original goal."""
    return f"{goal}\nYou asked: {question}\nMy answer: {answer}"


def pack_label(pack: Pack) -> str:
    """Procedure title as shown to the user, with a warning for unverified procedures."""
    if pack.procedure.status == "unverified":
        return f"{pack.procedure.title} (rules not verified yet)"
    return pack.procedure.title


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


def open_item_count(pack: Pack, results: list[RuleResult], view: FactView) -> int:
    """Problems + questions + values waiting for a check: what is still open for the user."""
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


def run_label(status: str | None) -> str:
    """The agent's last status in plain words."""
    return RUN_LABELS.get(status or "", "Not started")


def step_rows(result: dict) -> list[dict]:
    """The agent's steps: what it thought, which tool it called, whether the call worked.

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


def pages_text(pages: list[int]) -> str:
    """[1, 2, 3] -> 'pages 1-3'; [2] -> 'page 2'."""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"page {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pages {pages[0]}–{pages[-1]}"
    return "pages " + ", ".join(str(page) for page in pages)


def reading_summary(pack: Pack, name: str, documents: list[TwinDocument]) -> str:
    """What one upload gave, in plain words (shown in the chat after a document is attached)."""
    if not documents:
        return f"I could not read **{name}**."
    if len(documents) == 1 and documents[0].status == "rejected":
        return f"I could not read **{name}**: {friendly_error(documents[0].processing_error or '')}"
    lines = [f"I read **{name}**" + (f" and found {len(documents)} documents in it:" if len(documents) > 1 else ":")]
    for document in documents:
        lines.append("- " + document_line(pack, document))
    lines.append("")
    lines.append("Your plan above is updated.")
    return "\n".join(lines)


def document_line(pack: Pack, document: TwinDocument) -> str:
    """One document of an upload: type, pages, how it was recognised, values read and to confirm."""
    observation = document.observation or {}
    where = pages_text(observation.get("pages", []))
    if document.doc_type in (None, "unknown"):
        return f"❔ {where or 'This file'}: I could not tell which document this is. Choose its type in the Documents tab."
    facts = observation.get("facts", [])
    waiting = sum(1 for fact in facts if fact.get("requires_confirmation"))
    how = FOUND_BY.get(observation.get("found_by", ""), "")
    details = ", ".join(part for part in (where, how) if part)
    line = f"📄 **{document_title(pack, document)}**" + (f" ({details})" if details else "")
    line += f": {len(facts)} value{'' if len(facts) == 1 else 's'} read" + (f", {waiting} to confirm in 'Your details'" if waiting else "")
    return line + "."


# ------------------------------------------------- one run per profile --
@st.cache_resource
def _run_locks() -> dict:
    """One lock per profile, shared by every browser tab and device using this server.

    There is one local model: overlapping runs for a profile queue on it and get slower and slower,
    so a profile runs one agent turn or document reading at a time (Neda's guard).
    """
    return {"guard": threading.Lock(), "profiles": {}}


def profile_lock(user_id: str) -> threading.Lock:
    locks = _run_locks()
    with locks["guard"]:
        return locks["profiles"].setdefault(user_id, threading.Lock())


def run_locked(user_id: str, message: str, work: Callable[[], None]) -> bool:
    """Run `work` with the profile's lock. False when another run holds it or the work failed."""
    lock = profile_lock(user_id)
    if not lock.acquire(blocking=False):
        st.info(BUSY_MESSAGE, icon=":material/hourglass_top:")
        return False
    try:
        with st.spinner(message):
            work()
    except Exception as error:   # UI boundary: what was saved before the failure stays saved
        show_error(error)
        return False
    finally:
        lock.release()
    return True


@st.fragment(run_every=2)
def wait_for_run(user_id: str) -> None:
    """Shown instead of the views while another tab or device is running this profile."""
    if profile_lock(user_id).locked():
        st.info(BUSY_MESSAGE, icon=":material/hourglass_top:")
    else:
        st.rerun()


# ------------------------------------------------------- page messages --
FLASH_STYLES = {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}


def flash(message: str, kind: str = "info") -> None:
    """Keep a message across the rerun that follows a button click."""
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        FLASH_STYLES[kind](message)


def show_error(error: Exception | str) -> None:
    st.error(friendly_error(str(error)))
    with st.expander("Error details"):
        st.code(str(error))


# --------------------------------------------------------------- sidebar --
def sidebar(store: TwinStore, packs: dict[str, Pack]) -> tuple[str, date]:
    """Who you are, which procedure, which day. Returns (user id, today)."""
    st.sidebar.markdown("### :material/badge: Personal Bureaucracy Agent")
    st.sidebar.caption("Always know where you are, what is missing and what to do next.")
    user_id = st.sidebar.text_input("Your profile name", value="demo-user",
                                    help="Use the same name next time to continue where you left off.").strip()
    user_id = user_id or "demo-user"
    st.sidebar.caption(":green[:material/cloud_done:] Saved progress restored" if store.user_exists(user_id)
                       else "New profile")
    choose_procedure(store, packs, user_id)
    today = st.sidebar.date_input("Today's date (change it to test deadlines)", value=date.today())
    if st.sidebar.button("Refresh", icon=":material/refresh:"):
        st.rerun()
    reset_box(store, user_id)
    st.sidebar.caption(f"Local models: {CHAT_MODEL} (chat), {LLM_MODEL} (agent team, documents), {VISION_MODEL} "
                       "(pictures). Nothing is sent, booked, paid or submitted: approved items become files you download.")
    return user_id, today


def choose_procedure(store: TwinStore, packs: dict[str, Pack], user_id: str) -> None:
    """The procedure selector; choosing one makes it the profile's active procedure."""
    active = store.active_procedure(user_id)
    labels = {NO_PROCEDURE: NO_PROCEDURE}
    for procedure_id in sorted(packs):
        labels[procedure_id] = pack_label(packs[procedure_id])
    choices = list(labels)
    index = choices.index(active) if active in packs else 0
    chosen = st.sidebar.selectbox("Procedure", choices, index=index, format_func=labels.get)
    if chosen != NO_PROCEDURE and chosen != active:
        store.set_active_procedure(user_id, chosen)


def reset_box(store: TwinStore, user_id: str) -> None:
    """Forget the conversation, or delete this profile's Twin after the user types the profile name."""
    with st.sidebar.expander("Reset", icon=":material/delete:"):
        if st.button("Clear the conversation", disabled=profile_lock(user_id).locked()):
            store.clear_chat(user_id)
            st.rerun()
        st.caption("Deleting the profile removes its Digital Twin: documents, answers, actions, conversation, run log. "
                   "Uploaded files and other profiles are kept.")
        typed = st.text_input("Type the profile name to confirm", key="reset")
        if st.button("Delete this profile", type="primary", disabled=typed != user_id or profile_lock(user_id).locked()):
            store.delete_user(user_id)
            st.session_state.pop("results", None)
            st.rerun()


# ------------------------------------------------------------ status bar --
def status_bar(store: TwinStore, pack: Pack, user_id: str, view: FactView, results: list[RuleResult]) -> None:
    """Three numbers that answer 'Where am I?'."""
    rows = checklist_rows(pack, store.list_documents(user_id), view)
    mine = user_results(pack, results)
    counts = count_status(mine)
    docs, checks, status = st.columns(3, border=True)
    ready = sum(row["status"] == "complete" for row in rows)
    docs.metric("Documents ready", f"{ready} / {len(rows)}")
    checks.metric("Checks passed", f"{counts['pass']} / {len(mine)}",
                  delta=f"{counts['fail']} failing" if counts["fail"] else None, delta_color="inverse")
    status.caption("Agent")   # text, not a metric: long labels must not be cut off
    status.markdown(f"**{run_label(last_run_status(store, user_id, pack.id))}**")


def last_run_status(store: TwinStore, user_id: str, procedure_id: str) -> str | None:
    runs = [run for run in store.list_runs(user_id) if run.procedure_id == procedure_id]
    return runs[-1].status if runs else None


# ------------------------------------------------------- decision card --
def decision_card(store: TwinStore, ctx: ToolContext) -> None:
    """Actions waiting for approval come first, before any tab."""
    waiting = [proposal for proposal in store.list_proposals(ctx.user_id) if proposal.status in OPEN_PROPOSALS]
    if not waiting:
        return
    with st.container(border=True):
        st.markdown("#### :material/gavel: Waiting for your decision")
        for proposal in waiting:
            proposal_row(store, ctx, proposal)
        st.caption("Approving writes a file you can download. Nothing is submitted, paid or sent on your behalf.")


def proposal_row(store: TwinStore, ctx: ToolContext, proposal: ProposedAction) -> None:
    """One action waiting for approval, with Approve / Deny and, if needed, a sandbox check."""
    pack = ctx.packs.get(proposal.procedure_id)
    st.markdown(f"{TOOL_ICONS.get(proposal.tool, ':material/task:')} **{proposal.title}**")
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


# --------------------------------------------------------- view: assistant --
def view_assistant(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, results: list[RuleResult]) -> None:
    """The chat with the assistant, and beside it the documents and the tools."""
    chat_side, tools_side = st.columns([3, 2], gap="large")
    briefing = build_briefing(pack, store.list_documents(ctx.user_id), store.list_proposals(ctx.user_id), view, results)
    with chat_side:
        chat_panel(store, ctx, pack, view, briefing_markdown(briefing),
                   "Your plan, computed from your documents and answers. It updates by itself.", SUGGESTIONS)
    with tools_side:
        st.markdown("##### Your documents")
        checklist(pack, store.list_documents(ctx.user_id), view)
        st.markdown("##### Full check by the agent team")
        full_check_box(store, ctx)
        st.markdown("##### Prepare and try")
        prepare_box(store, ctx, pack)
        outcome_box(store, pack, ctx.user_id)
        done_list(store, ctx.user_id)


def chat_panel(store: TwinStore, ctx: ToolContext, pack: Pack | None, view: FactView | None, opening: str,
               opening_note: str, suggestions: list[str]) -> None:
    """The opening message (the plan, or a welcome), the conversation, suggestions and the message box."""
    history = store.list_chat(ctx.user_id)
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        st.markdown(opening)
        if opening_note:
            st.caption(opening_note)
    if len(history) > CHAT_SHOWN:
        with st.expander(f"Earlier messages ({len(history) - CHAT_SHOWN})"):
            for message in history[:-CHAT_SHOWN]:
                st.markdown(f"**{'You' if message.role == 'user' else 'Assistant'}:** {message.content}")
    unconfirmed = {fact.fact_id for fact in view.unconfirmed} if view is not None else set()
    for message in history[-CHAT_SHOWN:]:
        chat_bubble(store, ctx.user_id, message, unconfirmed)
    chosen = suggestion_buttons(suggestions)
    prompt = st.chat_input("Ask a question, tell me something new, or attach a document", key="chat_input",
                           accept_file=True, file_type=UPLOAD_TYPES)
    if chosen:
        ask_assistant(store, ctx, chosen)
    elif prompt:
        handle_prompt(store, ctx, pack, prompt)


def suggestion_buttons(suggestions: list[str]) -> str | None:
    """One-click questions; returns the clicked one."""
    row = st.container(horizontal=True)
    for number, text in enumerate(suggestions):
        if row.button(text, key=f"suggest_{number}", type="tertiary", icon=":material/chat_bubble:"):
            return text
    return None


def chat_bubble(store: TwinStore, user_id: str, message: ChatMessage, unconfirmed: set[int]) -> None:
    """One message; an assistant reply also shows its sources, facts to confirm and how it was checked."""
    avatar = ":material/person:" if message.role == "user" else ":material/support_agent:"
    with st.chat_message(message.role, avatar=avatar):
        st.markdown(message.content)
        if message.role != "assistant":
            return
        details = message.details or {}
        for fact in details.get("facts", []):
            if fact.get("fact_id") in unconfirmed:
                text, button = st.columns([4, 1])
                text.markdown(f":orange[:material/rate_review:] {readable(fact['field'])}: **{fact['value']}** "
                              "(from you, not counted until you confirm it)")
                if button.button("Confirm", key=f"chat_confirm_{message.message_id}_{fact['fact_id']}"):
                    store.confirm_fact(user_id, fact["fact_id"])
                    st.rerun()
        if details.get("proposals"):
            st.caption(":material/gavel: Prepared for your approval: see 'Waiting for your decision' above.")
        sources = details.get("sources", [])
        if sources:
            st.caption("Sources: " + " · ".join(f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
                                                for s in sources))
        if details.get("status") in REPLY_NOTES:
            st.caption(REPLY_NOTES[details["status"]])


def handle_prompt(store: TwinStore, ctx: ToolContext, pack: Pack | None, prompt) -> None:
    """A message from the chat box: attached files are read first, then the text goes to the assistant."""
    files = list(getattr(prompt, "files", None) or [])
    text = (getattr(prompt, "text", None) or (prompt if isinstance(prompt, str) else "") or "").strip()
    if files and pack is None:
        flash("Tell me first which procedure this is for (or choose it in the sidebar), so I know what to look for.",
              "warning")
        files = []

    def work() -> None:
        for upload in files:
            read_in_chat(store, ctx, pack, upload.name, upload.getvalue())
        if text:
            chat_turn(store, ctx, text)

    if (files or text) and run_locked(ctx.user_id, "Working on it...", work):
        st.rerun()
    elif not files and not text:
        st.rerun()


def ask_assistant(store: TwinStore, ctx: ToolContext, message: str) -> None:
    """One chat turn under the profile lock, then a page reload."""
    if run_locked(ctx.user_id, "The assistant is working...", lambda: chat_turn(store, ctx, message)):
        st.rerun()


def chat_turn(store: TwinStore, ctx: ToolContext, message: str) -> None:
    """Run the assistant on one message, showing what it does while it works."""
    with st.status("The assistant is working...", expanded=True) as status:
        reply = run_chat(message, store, ctx.packs, ctx.user_id, ctx.today, OllamaBackend(model=CHAT_MODEL),
                         outbox_dir=OUTBOX_DIR, embedder=OllamaBackend(), progress=status.write)
        status.update(label="Answer ready", state="complete", expanded=False)
    st.session_state.setdefault("results", {})[ctx.user_id] = {"goal": message, "result": reply.model_dump(mode="json")}


def read_in_chat(store: TwinStore, ctx: ToolContext, pack: Pack, name: str, content: bytes) -> None:
    """A document attached in the chat: read it, then say in the chat what it gave."""
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="user", content=f"📎 {name}", procedure_id=pack.id))
    documents = read_document(store, ctx, pack, name, content, AUTO_TYPE)
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="assistant", procedure_id=pack.id,
                                       content=reading_summary(pack, name, documents)))


def full_check_box(store: TwinStore, ctx: ToolContext) -> None:
    """The orchestrator with its workers checks every section; the answer is posted in the chat."""
    st.caption("The orchestrator gives each section to a worker, checks their findings and writes one answer "
               "(Gate 1 verifies it). Slower than the chat: a few minutes on a laptop.")
    if st.button("Run a full check", icon=":material/groups:"):
        if run_locked(ctx.user_id, "The agent team is checking your file...", lambda: full_check(store, ctx)):
            st.rerun()


def full_check(store: TwinStore, ctx: ToolContext) -> None:
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="user", content="🔍 Full check by the agent team",
                                       procedure_id=ctx.procedure_id))
    result = orchestrator.run_agent(FULL_CHECK_GOAL, store, ctx.packs, ctx.user_id, ctx.today, OllamaBackend(),
                                    outbox_dir=OUTBOX_DIR)
    st.session_state.setdefault("results", {})[ctx.user_id] = {"goal": FULL_CHECK_GOAL,
                                                               "result": result.model_dump(mode="json")}
    text = (result.question or "") if result.status == "needs_user_input" else agent_text(result)
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="assistant", content=text, procedure_id=result.procedure_id,
                                       trace_id=result.trace_id, details={"status": result.status,
                                                                          "proposals": result.proposals}))


def agent_text(result) -> str:
    """The orchestrator's answer as chat text: summary, then numbered next actions."""
    answer = result.answer
    if answer is None:
        return "The agent team could not finish. Try again when Ollama is ready."
    lines = [answer.summary]
    lines += [f"{number}. {action.title}" for number, action in enumerate(answer.next_actions, start=1)]
    return "\n".join(lines)


def checklist(pack: Pack, documents: list[TwinDocument], view: FactView) -> None:
    """Document checklist (Neda's layout)."""
    rows = checklist_rows(pack, documents, view)
    if not rows:
        st.caption("This procedure has no documents to upload.")
        return
    if all(row["status"] == "complete" for row in rows):
        st.success("All documents are ready. Later official steps remain separate.")
    for row in rows:
        st.markdown(f"{CHECKLIST_ICONS[row['status']]} {row['label']} · :gray[{CHECKLIST_LABELS[row['status']]}]")
    st.caption("Attach documents in the chat (several in one PDF is fine) or in the Documents tab. "
               "Ready means read and checked, not that the application is approved.")


def prepare_box(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """Twin -> real: prepare a checklist (approval needed), or try an official step in the mock portal."""
    if st.button("Prepare my checklist", icon=":material/checklist:"):
        observation = run_tool(build_registry(), "build_checklist", {}, ctx, "orchestrator")
        flash("Checklist prepared: approve it above to download it." if observation.ok else str(observation.error))
        st.rerun()
    sandbox_box(store, pack, ctx.user_id, ctx.today)


def sandbox_box(store: TwinStore, pack: Pack, user_id: str, today: date) -> None:
    """The mock world: applies the step's rules to what the Twin knows. Nothing is submitted."""
    labels = {stage.id: stage.title for stage in pack.procedure.stages}
    if not labels:
        st.caption("This procedure has no official step to try.")
        return
    st.caption("A mock portal checks an official step against the rules and your facts. Nothing is submitted.")
    stage_id = st.selectbox("Step", list(labels), format_func=labels.get, label_visibility="collapsed")
    if st.button("Try this step", icon=":material/science:"):
        try_stage(store, pack, user_id, today, stage_id)


def try_stage(store: TwinStore, pack: Pack, user_id: str, today: date, stage_id: str) -> None:
    """Run one official step in the mock portal and explain the result after the rerun."""
    outcome, _ = run_simulation(store, pack, user_id, today, stage_id)
    title = stage_title(pack, stage_id)
    if outcome.accepted:
        flash(f"The mock portal would accept '{title}'.", "success")
    else:
        flash(f"The mock portal would refuse '{title}'.\n\n- " + "\n- ".join(sandbox_lines(pack, outcome)), "warning")
    st.rerun()


def outcome_box(store: TwinStore, pack: Pack, user_id: str) -> None:
    """Real -> twin: what happened outside (an office answer, a corrected value) updates the Twin."""
    with st.expander("Something happened, or a value is wrong? Tell the Twin", icon=":material/campaign:"):
        labels = {field.name: readable(field.name) for field in pack.procedure.fields}
        with st.form("outcome", clear_on_submit=True):
            description = st.text_input("What happened?", placeholder="The office confirmed my documents")
            field = st.selectbox("Which fact changes?", list(labels), format_func=labels.get)
            raw = st.text_input("New value (dates as YYYY-MM-DD, yes or no for questions)")
            if st.form_submit_button("Save") and description.strip() and raw.strip():
                service.record_outcome(store, pack, user_id, "user_report", description.strip(), {field: raw.strip()})
                flash("Saved. The Twin and the requirements are updated.", "success")
                st.rerun()


def done_list(store: TwinStore, user_id: str) -> None:
    """Approved and denied actions; approved files can be downloaded."""
    done = [proposal for proposal in store.list_proposals(user_id) if proposal.status not in OPEN_PROPOSALS]
    if not done:
        return
    with st.expander(f"Approved and denied actions ({len(done)})", icon=":material/inventory_2:"):
        for proposal in done:
            text, download = st.columns([3, 2])
            text.markdown(f"{TOOL_ICONS.get(proposal.tool, ':material/task:')} {proposal.title} · {proposal.status}")
            path = Path(proposal.artifact_path) if proposal.artifact_path else None
            if proposal.status == "executed" and path is not None and path.is_file():
                download.download_button("Download", path.read_bytes(), file_name=path.name,
                                         key=f"dl_{proposal.action_id}")
            if proposal.status == "executed":
                for label, url in open_links(proposal):     # opens the user's own app; nothing is sent
                    download.link_button(label, url)


# -------------------------------------------------------- view: documents --
def view_documents(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView) -> None:
    """Upload form and one card per document of the profile."""
    upload_form(store, ctx, pack)
    documents = [document for document in store.list_documents(ctx.user_id) if document.status != "deleted"]
    if not documents:
        st.caption("No documents yet. Each upload is read by the local models; you can review and correct what they found.")
    for document in documents:
        document_card(store, ctx, pack, view, document)


def upload_form(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """Real -> twin: a document is read, and its values become facts of the Twin."""
    labels = {AUTO_TYPE: AUTO_TYPE}
    for document in pack.procedure.documents:
        labels[document.doc_type] = document.description
    with st.form("upload", clear_on_submit=True, border=True):
        uploaded = st.file_uploader("Add a document (PDF, TXT or a photo). One PDF may hold several documents.",
                                    type=UPLOAD_TYPES)
        doc_type = st.selectbox("Which document is it?", list(labels), format_func=labels.get)
        submitted = st.form_submit_button("Upload and analyze", type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Choose a file first.")
        else:
            read_upload(store, ctx, pack, uploaded.name, uploaded.getvalue(), doc_type)


def read_upload(store: TwinStore, ctx: ToolContext, pack: Pack, name: str, content: bytes, doc_type: str) -> None:
    """Read an upload under the profile lock and say what it gave."""
    def work() -> None:
        documents = read_document(store, ctx, pack, name, content, doc_type)
        rejected = [document for document in documents if document.status == "rejected"]
        flash(reading_summary(pack, name, documents), "error" if rejected else "success")

    if run_locked(ctx.user_id, "Reading the document...", work):
        st.rerun()


def read_document(store: TwinStore, ctx: ToolContext, pack: Pack, name: str, content: bytes, doc_type: str,
                  pages: list[int] | None = None) -> list[TwinDocument]:
    """Save the upload, register it in the Twin, run extract_document; returns the documents found in it."""
    path = save_upload(name, content, ctx.user_id)
    doc_id = doc_id_for(path)
    service.register_upload(store, ctx.user_id, doc_id, str(path), pack.id)
    return run_extraction(store, ctx, doc_id, doc_type, pages)


def run_extraction(store: TwinStore, ctx: ToolContext, doc_id: str, doc_type: str,
                   pages: list[int] | None = None) -> list[TwinDocument]:
    args: dict = {"doc_id": doc_id}
    if doc_type != AUTO_TYPE:
        args["doc_type"] = doc_type
    if pages:
        args["pages"] = pages
    observation = run_tool(build_registry(), "extract_document", args, ctx, "orchestrator")
    if not observation.ok:
        raise ValueError(observation.error or "The document could not be read")
    ids = [row["doc_id"] for row in observation.output["documents"]]
    return [document for document in (store.get_document(ctx.user_id, i) for i in ids) if document is not None]


def document_card(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, document: TwinDocument) -> None:
    """Compact card; the values found, 'read again as' and the delete button sit in the expander."""
    observation = document.observation or {}
    with st.container(border=True):
        st.markdown(f":material/description: **{document_title(pack, document)}** · {document.status}")
        where = pages_text(observation.get("pages", []))
        how = FOUND_BY.get(observation.get("found_by", ""), "")
        st.caption(" · ".join(part for part in (Path(document.file_path).name, where, how) if part))
        if document.processing_error:
            st.error(friendly_error(document.processing_error))
        for note in observation.get("notes", [])[:2]:
            st.caption(f":material/info: {note}")
        for issue in observation.get("issues", [])[:3]:
            st.caption(f":material/warning: {friendly_issue(issue)}")
        with st.expander("Details / Edit"):
            document_values(store, ctx.user_id, view, document)
            read_again(store, ctx, pack, document)
            checked = st.checkbox("I want to delete this document", key=f"delete_confirm_{document.doc_id}")
            if st.button("Delete", key=f"delete_{document.doc_id}", disabled=not checked):
                service.delete_document(store, ctx.user_id, pack.id, document.doc_id)
                st.rerun()


def read_again(store: TwinStore, ctx: ToolContext, pack: Pack, document: TwinDocument) -> None:
    """Wrong or unknown type: the user chooses the type and the same pages are read again."""
    labels = {spec.doc_type: spec.description for spec in pack.procedure.documents}
    with st.form(f"again_{document.doc_id}"):
        chosen = st.selectbox("Read it again as", list(labels), format_func=labels.get)
        if st.form_submit_button("Read again"):
            pages = (document.observation or {}).get("pages") or None

            def work() -> None:
                flash(reading_summary(pack, Path(document.file_path).name,
                                      run_extraction(store, ctx, document.doc_id, chosen, pages)), "success")

            if run_locked(ctx.user_id, "Reading the document again...", work):
                st.rerun()


def document_values(store: TwinStore, user_id: str, view: FactView, document: TwinDocument) -> None:
    """The values this document gave the Twin; values that wait for the user can be confirmed here."""
    shown = False
    for field, facts in sorted(view.sources.items()):
        for fact in facts:
            if fact.source_ref == document.doc_id:
                shown = True
                st.write(f"{readable(field).capitalize()}: {fact.value}")
    for fact in view.unconfirmed:
        if fact.source_ref == document.doc_id:
            shown = True
            text, confirm = st.columns([4, 1])
            text.markdown(f"{readable(fact.field).capitalize()}: **{fact.value}** (awaiting your confirmation)")
            if fact.snippet:
                text.caption(f"Quote: {fact.snippet}")
            if confirm.button("Confirm", key=f"confirm_doc_{fact.fact_id}"):
                store.confirm_fact(user_id, fact.fact_id)
                st.rerun()
    if not shown:
        st.caption("No value from this document is used yet. Type missing values in 'Your details'.")


# ---------------------------------------------------------- view: details --
def view_details(store: TwinStore, pack: Pack, user_id: str, view: FactView, results: list[RuleResult]) -> None:
    """What is missing: values to confirm, open questions per section, computed values."""
    checks_to_confirm(store, pack, user_id, view)
    open_questions(store, pack, user_id, results)
    computed = [f"{readable(field)}: {view.values[field]}" for field in sorted(view.derived) if field in view.values
                and field != "current_date"]
    if computed:
        st.info("Computed from your answers: " + "; ".join(computed))


def checks_to_confirm(store: TwinStore, pack: Pack, user_id: str, view: FactView) -> None:
    """Values that wait for the user: unconfirmed statements and sources that disagree."""
    if not view.unconfirmed and not view.conflicts:
        return
    st.markdown("##### Please check")
    for fact in view.unconfirmed:
        text, confirm = st.columns([4, 1])
        origin = ORIGIN_WORDS.get(fact.provenance, fact.provenance)
        text.markdown(f"{readable(fact.field)}: **{fact.value}** (from {origin})")
        if fact.snippet:
            text.caption(f"Quote: {fact.snippet}")
        if confirm.button("Confirm", key=f"confirm_{fact.fact_id}"):
            store.confirm_fact(user_id, fact.fact_id)
            st.rerun()
    for field, values in view.conflicts.items():
        st.warning(f"Your documents disagree on {readable(field)}. The original values are kept.")
        choice = st.radio("Which value is right?", [str(value) for value in values], key=f"conflict_{field}",
                          horizontal=True)
        if st.button("Save my decision", key=f"resolve_{field}"):
            service.resolve_conflict(store, pack, user_id, field, choice)
            st.rerun()


def open_questions(store: TwinStore, pack: Pack, user_id: str, results: list[RuleResult]) -> None:
    """Per section: the questions that would turn unknown requirements into known ones.

    A field used by several sections is asked once, in the first of them.
    """
    st.markdown("##### Open questions")
    asked: set[str] = set()
    for section in user_sections(pack):
        missing = missing_fields(in_section(results, section.id))
        fields = [field for field in askable_fields(pack, missing) if field not in asked]
        if not fields:
            continue
        asked.update(fields)
        with st.expander(f"{section.title} ({len(fields)})"):
            documents = documents_for(pack, missing)
            if documents:
                st.caption("Uploading these documents answers some of them: " + "; ".join(documents))
            answers_form(store, pack, user_id, section.id, fields)
    if not asked:
        st.success("No open questions: the Twin has everything the rules need.")


def answers_form(store: TwinStore, pack: Pack, user_id: str, section_id: str, fields: list[str]) -> None:
    with st.form(f"answers_{section_id}"):
        values = {field: field_input(pack, field, f"{section_id}_{field}") for field in fields}
        if st.form_submit_button("Save answers and recheck"):
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


# ----------------------------------------------------------- view: checks --
def view_checks(pack: Pack, results: list[RuleResult]) -> None:
    """Where am I in detail: counts, problems, every rule per section, the official sources."""
    mine = user_results(pack, results)
    counts = count_status(mine)
    passed, failed, unknown = st.columns(3)
    passed.metric("Passed", counts["pass"])
    failed.metric("Failed", counts["fail"])
    unknown.metric("Not enough information yet", counts["unknown"])
    problems(mine)
    for section in user_sections(pack):
        section_results = in_section(results, section.id)
        met_here = count_status(section_results)["pass"]
        with st.expander(f"{section_icon(section_results)} {section.title} · {met_here}/{len(section_results)}"):
            for result in section_results:
                st.markdown(rule_line(result))
    sources_box(pack)


def problems(results: list[RuleResult]) -> None:
    """Failed blocking rules first, then advice (advisory rules that are not met)."""
    blocking = [r for r in results if r.status == "fail" and r.severity == "blocking"]
    advice = [r for r in results if r.status == "fail" and r.severity != "blocking"]
    if blocking:
        st.markdown("##### Problems to fix")
        for result in blocking:
            st.error(rule_label(result), icon=":material/error:")
    if advice:
        st.markdown("##### Recommendations")
        for result in advice:
            st.warning(rule_label(result), icon=":material/lightbulb:")
    if results and not blocking and not any(r.status == "unknown" for r in results):
        st.success("Nothing blocking right now.")


def sources_box(pack: Pack) -> None:
    """The official pages the rules come from (the knowledge base of the procedure)."""
    with st.expander(f"Official sources ({len(pack.sources)})", icon=":material/menu_book:"):
        for source in pack.sources:
            saved = " · text saved" if source.content_sha256 else ""
            st.markdown(f"[{source.title}]({source.url})  \n:gray[{source.organization or source.authority} · "
                        f"retrieved {source.retrieved_at or 'never'}{saved}]")


# -------------------------------------------------------- view: technical --
def view_technical(store: TwinStore, pack: Pack, user_id: str, view: FactView, results: list[RuleResult]) -> None:
    """Everything a developer or examiner wants to see, in one place."""
    last = st.session_state.get("results", {}).get(user_id)
    with st.expander("How the agent got there (last answer)", expanded=bool(last)):
        if last:
            st.caption(f"Message: {last['goal']}")
            st.dataframe(step_rows(last["result"]), hide_index=True)
            if last["result"].get("gate_issues"):
                st.json(last["result"]["gate_issues"])
        else:
            st.caption("No agent answer in this session yet.")
    with st.expander("Run log (memory across sessions)"):
        runs = [run for run in store.list_runs(user_id) if run.procedure_id == pack.id]
        st.dataframe([{"when": run.started_at.strftime("%Y-%m-%d %H:%M"), "status": run.status, "steps": run.steps,
                       "llm calls": run.llm_calls, "repairs": run.repairs, "passed": run_passed(run)}
                      for run in runs], hide_index=True)
    with st.expander("Lessons from validated repairs (episodic memory)"):
        st.dataframe([{"failure": e.failure_type, "rule": e.rule_or_source_id, "lesson": e.corrective_action}
                      for e in store.find_episodes(pack.id, limit=20)], hide_index=True)
    with st.expander(f"Everything the Twin knows ({len(view.values)} facts)"):
        st.dataframe(fact_rows(view), hide_index=True)
    with st.expander("All rule results"):
        st.json([result.model_dump() for result in results])
    with st.expander("Knowledge base"):
        st.caption(f"{len(pack_chunks(pack))} passages from {len(pack.sources)} official sources of this procedure. "
                   f"Update with: python -m src.rag fetch {pack.id}")
        st.caption(f"Web corpus for other procedures: {len(load_corpus())} pages (python -m src.rag crawl).")


def run_passed(run) -> str:
    """'passed/total' of the rule results saved with a run ('' for runs logged before results were kept)."""
    if not run.results:
        return ""
    return f"{sum(result.status == 'pass' for result in run.results)}/{len(run.results)}"


# ------------------------------------------------------------------ page --
def tool_context(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date) -> ToolContext:
    active = store.active_procedure(user_id)
    return ToolContext(store=store, packs=packs, user_id=user_id, today=today, backend=OllamaBackend(),
                       procedure_id=active if active in packs else None)


def no_procedure_page(store: TwinStore, ctx: ToolContext) -> None:
    """Before a procedure is chosen: the chat finds the procedure, or answers from the web corpus."""
    st.title("What do you want to do?")
    decision_card(store, ctx)
    chat_panel(store, ctx, None, None, WELCOME, "", FIRST_SUGGESTIONS)
    st.caption("Procedures with a full checklist: " + "; ".join(pack_label(p) for p in ctx.packs.values())
               + ". For anything else the assistant answers from official pages, without checking your documents.")


def procedure_page(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """Status bar, the decision card, then the chosen tab."""
    view = ctx.view()
    results = evaluate_rules(pack, view.fact_map())
    st.title(pack.procedure.title)
    if pack.procedure.status == "unverified":
        st.warning("The rules of this procedure are not verified yet. Check every result with the official office.")
    status_bar(store, pack, ctx.user_id, view, results)
    decision_card(store, ctx)
    chosen = st.segmented_control("View", list(VIEWS), format_func=VIEWS.get, key="view",
                                  default=None if "view" in st.session_state else "assistant",
                                  label_visibility="collapsed") or "assistant"
    if chosen == "assistant":
        view_assistant(store, ctx, pack, view, results)
    elif chosen == "documents":
        view_documents(store, ctx, pack, view)
    elif chosen == "details":
        view_details(store, pack, ctx.user_id, view, results)
    elif chosen == "checks":
        view_checks(pack, results)
    else:
        view_technical(store, pack, ctx.user_id, view, results)


def main() -> None:
    """Build the page: sidebar, then the waiting banner, the no-procedure page or the procedure page."""
    st.set_page_config(page_title="Personal Bureaucracy Agent", page_icon=":material/badge:", layout="wide")
    packs = load_all_packs()
    with TwinStore(DATABASE_PATH) as store:
        user_id, today = sidebar(store, packs)
        try:
            show_flash()
            ctx = tool_context(store, packs, user_id, today)
            pack = ctx.pack()
            if profile_lock(user_id).locked():
                st.title(pack.procedure.title if pack else "What do you want to do?")
                wait_for_run(user_id)
                return
            if pack is None:
                no_procedure_page(store, ctx)
            else:
                procedure_page(store, ctx, pack)
        except ValueError as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
