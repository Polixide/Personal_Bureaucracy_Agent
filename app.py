"""Streamlit app for the Personal Bureaucracy Agent: streamlit run app.py

Designed to be simple: one thing at a time.
- On top, the journey: the official steps of the procedure, with "you are here", and one progress bar.
- Assistant (the main view): the chat with the assistant (src/agent/chat.py) in a fixed-size window that scrolls,
  and beside it the "Now" box: the ONE next step of the plan computed by code (src/agent/briefing.py), with the
  control it needs right there (Yes / No, a date, a choice, upload this document, approve this draft, check this
  value) and "Skip for now". Below: what comes next, what is ready (open it in Gmail or Google Calendar), and what
  the agent can prepare.
- Documents: upload, and each document of the procedure with what was read from it (values to check are confirmed
  there).
- Details: for examiners and developers: every requirement with its source, the sandbox, how the agent works
  (ReAct, Gate 1, reflection), the last answer step by step, history, the Twin's facts and the knowledge base.

The Twin (SQLite at config.DATABASE_PATH) is the only source of truth. Verdicts come from src.rules, never from the
model. Nothing is sent, booked, paid or submitted: an approved e-mail opens in the user's Gmail and an approved
reminder in their Google Calendar, already filled in, and the user presses Send or Save (src/tools/open_links.py).
The sidebar chooses the model: Local (Ollama) or OpenAI (Claude too when its key is set). A cloud model sees the
state, never the pages, unless the sidebar switch lets it read the documents.
"""
import threading
from collections.abc import Callable
from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st

from config import (CHAT_MAX_REPAIRS, CHAT_MAX_STEPS, CLOUD_MAX_REQUESTS, CLOUD_PRICES_AS_OF, DATABASE_PATH,
                    DOCUMENT_PROVIDER, EMBED_MODEL, LLM_PROVIDER, MAX_REPAIRS, MAX_STEPS, OUTBOX_DIR, PROJECT_ROOT,
                    PROVIDER_LABELS, UPLOAD_DIR, UPLOAD_TYPES as ALLOWED_UPLOADS, WORKER_STEPS)
from src.agent import gate0, orchestrator
from src.agent.actions import Suggestion, suggestions
from src.agent.briefing import (OPEN_PROPOSALS, ORIGIN_WORDS, Briefing, Step, askable_fields, build_briefing,
                                checklist_rows, count_status, document_status, documents_for, field_question,
                                in_section, missing_fields, readable, user_results, user_sections)
from src.agent.chat import review_enabled, run_chat
from src.llm import (USAGE, agent_backend, agent_model_label, document_backend, document_model_label,
                     documents_in_cloud, is_cloud, local_backend, missing_key, usage_summary)
from src.models import DocumentSpec, Pack, RuleResult
from src.pack import load_all_packs
from src.rag.crawl import load_corpus
from src.rag.index import pack_chunks, search
from src.rules import evaluate_rules
from src.sandbox.approval import approve_action, deny_action
from src.sandbox.mock_world import SimulationOutcome, run_simulation
from src.tools.common import ToolContext
from src.tools.open_links import open_links
from src.tools.registry import build_registry, run_tool
from src.twin import service
from src.twin.facts import FactView
from src.twin.models import ChatMessage, Fact, ProposedAction, TwinDocument
from src.twin.store import TwinStore

__all__ = ["askable_fields", "checklist_rows", "document_status", "documents_for", "field_question", "missing_fields"]

AUTO_TYPE = "detect automatically"
NO_PROCEDURE = "(let the agent decide)"
NOT_ANSWERED = "not answered"
UPLOAD_TYPES = list(ALLOWED_UPLOADS)
STATUS_ICONS = {"pass": "✅", "fail": "❌", "unknown": "❔"}
TOOL_ICONS = {"create_ics": ":material/event:", "draft_email": ":material/mail:", "fill_pdf": ":material/picture_as_pdf:",
              "build_checklist": ":material/checklist:"}
VIEWS = {"assistant": ":material/forum: Assistant", "documents": ":material/description: Documents",
         "details": ":material/tune: Details"}
CHECKLIST_ICONS = {"complete": ":green[:material/check_circle:]", "needs confirmation": ":orange[:material/rate_review:]",
                   "missing": ":gray[:material/radio_button_unchecked:]", "ambiguous": ":orange[:material/help:]"}
CHECKLIST_LABELS = {"complete": "Ready", "needs confirmation": "Check the values", "missing": "Not uploaded yet",
                    "ambiguous": "Unclear"}
KIND_WORDS = {"decide": "Your approval", "fix": "A problem to fix", "confirm": "Check what I read",
              "upload": "A document", "answer": "A question", "improve": "A recommendation", "try": "Next official step"}
RUN_LABELS = {"answered": "Answer ready", "needs_user_input": "The agent asked you a question",
              "needs_human_review": "Answer needs your review", "fallback": "Answer built from the rules alone",
              "invalid_input": "Message not sent (Gate 0)"}
REPLY_NOTES = {"answered": "✓ Checked by code",
               "fallback": "Written by code from your plan (the model's answer did not pass the checks)",
               "needs_human_review": "Read with care: not every automatic check passed",
               "needs_user_input": "The agent needs an answer from you",
               "invalid_input": "Gate 0: not sent to the model"}
FOUND_BY = {"title": "recognised by its title", "mrz": "recognised by its machine-readable zone",
            "keywords": "recognised by its content", "model": "recognised by the language model",
            "vision": "recognised from the image by the vision model", "user": "read as the type you chose"}
SUGGESTIONS = ["What should I do now?", "Explain my situation in simple words",
               "Write an e-mail to IRO about my next step"]
FIRST_SUGGESTIONS = ["I want to apply for my first İkamet", "I lost my TC kimlik", "How do I get a tax number?"]
WELCOME = ("Hi! Tell me what you need — for example *I want to apply for İkamet*, *I lost my TC kimlik* or "
           "*How do I get a tax number?* — and I will guide you one step at a time.")
FULL_CHECK_GOAL = "Check my whole file section by section and tell me what to do next."
EARLIEST_DATE = date(1900, 1, 1)
LATEST_DATE = date(2100, 12, 31)
BUSY_MESSAGE = "Still processing your last action. This page updates when it finishes."
CHAT_SHOWN = 60          # messages in the scrolling window; older ones in an expander
CHAT_HEIGHT = 600        # pixels: the chat window keeps this size and scrolls
COMING_UP = 3            # steps shown under "Now"
READY_SHOWN = 4
MAX_CHOICE_BUTTONS = 5
HOW_IT_WORKS = """
**0. Gate 0 checks what you send** (`src/agent/gate0.py`), before any model call: a message must not be empty or
too long; a file must be a supported type, not empty, not too large, really what its extension says, and a PDF of
a reasonable length. The model's own tool calls pass the same kind of gate: every call is validated with Pydantic
in `src/tools/registry.py` and an invalid call never runs.

**1. The state is computed by code.** Your documents are read by {reader} (the text, scans and photos); code checks
every value against the page and keeps it as a *fact* of your Digital Twin. The rule engine
(`src/rules.py`) evaluates every requirement with three-valued logic: pass, fail, or unknown (information missing).
No model decides whether a rule is met, and no model computes a date.

**2. The plan is built by code** (`src/agent/briefing.py`) and shown one step at a time in the *Now* box: approve
what the agent prepared, fix what fails, check what the models were not sure about, upload what is missing, answer
the questions, then the next official step.

**3. The assistant is a ReAct agent** (`src/agent/chat.py`, on the LangGraph graph of `src/agent/orchestrator.py`):
*decide* (the model picks one tool and its arguments, validated with Pydantic) → *act* (code runs the tool) →
*observe* (the result goes back to the model) → … → `final_answer`. At most {chat_steps} steps per chat turn.

**4. Gate 1 verifies every answer** (`src/agent/gate1.py`): statuses copied from the rule engine, no invented dates
or rule ids, no "submitted / approved / paid" claims, refs that exist, unverified rules labelled.

**5. Reflection.** A refused answer goes back to the model with the structured errors and is rewritten. Then the
model re-reads its own answer against the state (*self-review*: {review}); code keeps only criticisms that quote
words really in the answer. At most {chat_repairs} rewrites, then the plan written by code is shown. Document reading
has its own reflection: a value whose quote is not on the page is read a second time.

**6. Orchestrator-Workers** ("Full check by the agent team"): the orchestrator gives each section to a worker (a small
ReAct loop, at most {worker_steps} steps), reads their findings and writes one answer (at most {max_steps} steps,
{max_repairs} Gate 1 repairs).

**7. Tools that act on the real world only propose.** `build_checklist`, `fill_pdf`, `create_ics`, `draft_email`
create a *proposal*. After you approve it, an e-mail opens in your Gmail and a reminder in your Google Calendar,
already filled in: you press Send or Save. Nothing is sent, booked, paid or submitted by the agent.

**8. Official sources (RAG).** The rules cite official pages; their text is searched by meaning (`{embed}`, local), so
the assistant can quote the source of an answer and answer questions outside the rules. It never decides a rule.

**9. The same knowledge for other agents (MCP).** `python -m src.integrations.mcp.server` is a read-only Model Context
Protocol server: any MCP client can find the procedure for a goal and search the official sources of a pack.
"""


# ---------------------------------------------------------- pure helpers --
def save_upload(name: str, content: bytes, user_id: str, directory: Path = UPLOAD_DIR) -> Path:
    """Store an upload under a content/user-derived name inside the project, never a browser path."""
    refused = gate0.check_upload(name, content)          # Gate 0: type, size, real content, pages
    if refused:
        raise ValueError(refused[0].message)
    extension = Path(name).suffix.lower()
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
    if any(word in text for word in ("api key", "authentication", "credentials", "401")):
        return "The cloud model refused the API key. Check OPENAI_API_KEY / ANTHROPIC_API_KEY, or choose Local (Ollama)."
    if "ollama" in text:
        return "The local model is not running. Start Ollama, or type the values yourself."
    if any(word in text for word in ("claude", "openai", "connect", "unavailable")):
        return "The cloud model is not reachable. Check the internet connection, or choose Local (Ollama)."
    return "Processing could not finish. Check the document and retry."


def friendly_issue(issue: str) -> str:
    """Extraction warnings in plain words: 'insurance_end: value absent ...; discarded' -> a short sentence."""
    field, separator, reason = issue.partition(": ")
    if not separator or " " in field:
        return issue
    if any(word in reason for word in ("discarded", "absent", "not present", "not found", "does not exist")):
        return f"Could not read the {readable(field)} reliably; the assistant will ask you for it."
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
    """One requirement as a line of text: icon, requirement, and what is still needed to decide it."""
    line = f"{STATUS_ICONS[result.status]} {rule_label(result)}"
    if result.missing_fields:
        line += f" *(waiting for: {', '.join(readable(field) for field in result.missing_fields)})*"
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


def shown_value(pack: Pack, field: str, value) -> str:
    """A value as the user knows it: the label of a choice ("Rental agreement in my name") instead of its code."""
    spec = pack.field(field)
    labels = {stored: label for label, stored in spec.choices.items()} if spec is not None else {}
    return labels.get(value, str(value))


def open_item_count(briefing: Briefing) -> int:
    """What is still open for the user: the steps of the plan (approve, fix, confirm, upload, answer...)."""
    return len(briefing.steps)


def step_key(step: Step) -> str:
    return f"{step.kind}:{step.ref}"


def remaining_steps(briefing: Briefing, skipped: list[str]) -> list[Step]:
    """The plan without the steps the user skipped for now (skipped ones come back at the end)."""
    kept = [step for step in briefing.steps if step_key(step) not in skipped]
    return kept + [step for step in briefing.steps if step_key(step) in skipped]


def journey(pack: Pack, results: list[RuleResult]) -> list[tuple[str, str]]:
    """The official steps in order with done / now / later (a step is done when its rules pass and so did the one
    before it)."""
    status = {result.rule_id: result.status for result in results}
    states, current_seen = [], False
    for stage in pack.procedure.stages:
        passed = all(status.get(rule_id) == "pass" for rule_id in stage.requires)
        if passed and not current_seen:
            states.append((stage.title, "done"))
        elif not current_seen:
            states.append((stage.title, "now"))
            current_seen = True
        else:
            states.append((stage.title, "later"))
    return states


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


def reflection_summary(rounds: list[dict]) -> str:
    """'Gate 1 found 1 problem → rewritten; self-review: nothing to correct' (one line under a reply)."""
    parts = []
    for item in rounds:
        found = len(item.get("issues", []))
        if not found:
            parts.append(f"{item['by']}: nothing to correct")
        elif "not corrected" in item["by"]:
            parts.append(f"{item['by']}: {found} doubt{'' if found == 1 else 's'}")
        else:
            parts.append(f"{item['by']} found {found} problem{'' if found == 1 else 's'} → rewritten")
    return "; ".join(parts)


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
    lines.append("Your plan is updated.")
    return "\n".join(lines)


def document_line(pack: Pack, document: TwinDocument) -> str:
    """One document of an upload: type, pages, how it was recognised, values read and to confirm."""
    observation = document.observation or {}
    where = pages_text(observation.get("pages", []))
    if document.doc_type in (None, "unknown"):
        return f"❔ {where or 'This file'}: I could not tell which document this is. Choose its type in Documents."
    facts = observation.get("facts", [])
    waiting = sum(1 for fact in facts if fact.get("requires_confirmation"))
    how = FOUND_BY.get(observation.get("found_by", ""), "")
    details = ", ".join(part for part in (where, how) if part)
    line = f"📄 **{document_title(pack, document)}**" + (f" ({details})" if details else "")
    line += f": {len(facts)} value{'' if len(facts) == 1 else 's'} read" + (f", {waiting} to confirm" if waiting else "")
    return line + "."


def proposal_preview(proposal: ProposedAction) -> str:
    """What approving this proposal would write, in Markdown (so the user knows what they approve)."""
    payload = proposal.payload or {}
    if proposal.tool == "draft_email":
        return (f"**To:** {payload.get('to_name', '')} <{payload.get('to_email', '')}>  \n"
                f"**Subject:** {payload.get('subject', '')}\n\n{payload.get('body', '')}")
    if proposal.tool == "create_ics":
        return f"**Calendar event:** {proposal.title}  \n**Date:** {payload.get('date', '')} (all day)"
    if proposal.tool == "fill_pdf":
        values = "\n".join(f"- {name}: {value}" for name, value in (payload.get("values") or {}).items())
        missing = ", ".join(readable(name) for name in payload.get("missing") or []) or "none"
        return f"**Form fields filled:**\n{values or '- none yet'}\n\n**Left empty (not known yet):** {missing}"
    if proposal.tool == "build_checklist":
        return "\n".join((payload.get("markdown") or "").splitlines()[:40])
    return proposal.title


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


def ask_in_chat(prompt: str) -> None:
    """Button callback: open the assistant and send it this message."""
    st.session_state["view"] = "assistant"
    st.session_state["pending_prompt"] = prompt


def skip_step(key: str) -> None:
    """Button callback: put a step at the end of the plan for now."""
    skipped = st.session_state.setdefault("skipped", [])
    if key not in skipped:
        skipped.append(key)


# --------------------------------------------------------------- sidebar --
def sidebar(store: TwinStore, packs: dict[str, Pack]) -> tuple[str, date]:
    """Who you are and which procedure; demo settings are folded away. Returns (user id, today)."""
    st.sidebar.markdown("### :material/badge: Personal Bureaucracy Agent")
    st.sidebar.caption("Your assistant for Turkish paperwork, one step at a time.")
    user_id = st.sidebar.text_input("Your profile name", value="demo-user",
                                    help="Use the same name next time to continue where you left off.").strip()
    if gate0.check_profile(user_id):                       # Gate 0: an empty or endless name
        st.sidebar.caption("Profile names have 1 to 60 characters: using demo-user.")
        user_id = "demo-user"
    choose_procedure(store, packs, user_id)
    model_choice()
    with st.sidebar.expander("Settings", icon=":material/settings:"):
        today = st.date_input("Today's date (change it to test deadlines)", value=date.today())
        if st.button("Refresh", icon=":material/refresh:"):
            st.rerun()
        reset_box(store, user_id)
    return user_id, today


def provider() -> str:
    """The provider chosen in the sidebar and usable now (a cloud provider without its key falls back to Ollama)."""
    return st.session_state.get("provider_in_use", "ollama")


def documents_cloud() -> bool:
    """The sidebar switch: the cloud model reads the documents (only with a cloud provider)."""
    return bool(st.session_state.get("docs_cloud", False))


def provider_options() -> list[str]:
    """Ollama and OpenAI always; Claude only when it is configured (its key, or LLM_PROVIDER=anthropic)."""
    options = ["ollama", "openai"]
    if LLM_PROVIDER == "anthropic" or not missing_key("anthropic"):
        options.append("anthropic")
    return options


def model_choice() -> None:
    """Sidebar: which model runs the assistant and the agent team, and what leaves the computer."""
    options = provider_options()
    start = LLM_PROVIDER if LLM_PROVIDER in options else "ollama"
    chosen = st.sidebar.segmented_control("AI model", options, default=start, format_func=PROVIDER_LABELS.get,
                                          key="provider") or start
    problem = missing_key(chosen)
    if problem:
        st.sidebar.warning(f"{PROVIDER_LABELS[chosen]} is not ready. {problem} The local model answers meanwhile.")
        chosen = "ollama"
    st.session_state["provider_in_use"] = chosen
    if is_cloud(chosen):
        if "docs_cloud" not in st.session_state:
            st.session_state["docs_cloud"] = DOCUMENT_PROVIDER == "cloud"
        st.sidebar.toggle("The cloud model also reads my documents", key="docs_cloud",
                          help="Page images and text are sent to the provider; its OCR replaces the local one. "
                               "Code still checks every value against the page.")
    models_note()


def models_note() -> None:
    """Which models do what, and what leaves the computer."""
    st.sidebar.caption(f"Assistant: {agent_model_label('chat', provider())} · Agent team: "
                       f"{agent_model_label('team', provider())} · Documents: "
                       f"{document_model_label(provider(), documents_cloud())}.")
    if documents_in_cloud(provider(), documents_cloud()):
        st.sidebar.caption("Your documents are read in the cloud: their page images and text are sent to the provider.")
    elif is_cloud(provider()):
        st.sidebar.caption("The assistant runs in the cloud: it receives your state (requirements, values already "
                           "read), never the pages of your documents.")


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
    if st.button("Clear the conversation", disabled=profile_lock(user_id).locked()):
        store.clear_chat(user_id)
        st.rerun()
    typed = st.text_input("Type the profile name to delete it", key="reset")
    if st.button("Delete this profile", type="primary", disabled=typed != user_id or profile_lock(user_id).locked()):
        store.delete_user(user_id)
        st.session_state.pop("results", None)
        st.rerun()


# ------------------------------------------------------------- journey --
def journey_bar(pack: Pack, results: list[RuleResult], briefing: Briefing) -> None:
    """The official steps with "you are here", and one progress bar."""
    states = journey(pack, results)
    if states:
        columns = st.columns(len(states))
        for number, (column, (title, state)) in enumerate(zip(columns, states), start=1):
            if state == "done":
                column.markdown(f":green[:material/check_circle: {number}. {title}]")
            elif state == "now":
                column.markdown(f":blue[:material/radio_button_checked: **{number}. {title}**]  \n:blue[you are here]")
            else:
                column.markdown(f":gray[:material/radio_button_unchecked: {number}. {title}]")
    total = max(briefing.documents_total, 1)
    st.progress(briefing.documents_ready / total,
                text=f"{briefing.documents_ready} of {briefing.documents_total} documents ready · "
                     f"{briefing.checks_passed} of {briefing.checks_total} requirements met")


# --------------------------------------------------------- view: assistant --
def view_assistant(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, briefing: Briefing) -> None:
    """The chat beside the Now box (one step at a time), what is ready, and what the agent can prepare."""
    chat_side, plan_side = st.columns([3, 2], gap="large")
    with chat_side:
        chat_panel(store, ctx, view, opening_message(briefing), SUGGESTIONS)
    with plan_side:
        steps = remaining_steps(briefing, st.session_state.get("skipped", []))
        now_box(store, ctx, pack, view, steps)
        coming_up(steps)
        ready_box(store, ctx.user_id)
        prepare_box(store, ctx, pack, view)


def opening_message(briefing: Briefing) -> str:
    """The assistant's first message: where you are, and how to use it."""
    return (f"Hi! I'm your assistant for **{briefing.title}**. {briefing.where}.\n\n"
            "The **Now** box on the right always shows the one thing to do next. You can also ask me anything, "
            "tell me something new, ask me to write an e-mail or set a reminder, or attach a document with **+**.")


def chat_panel(store: TwinStore, ctx: ToolContext, view: FactView | None, opening: str, suggestions_: list[str]) -> None:
    """The conversation in a scrolling window of fixed size, then the suggestions and the message box."""
    history = store.list_chat(ctx.user_id)
    unconfirmed = {fact.fact_id for fact in view.unconfirmed} if view is not None else set()
    if len(history) > CHAT_SHOWN:
        with st.expander(f"Earlier messages ({len(history) - CHAT_SHOWN})"):
            for message in history[:-CHAT_SHOWN]:
                st.markdown(f"**{'You' if message.role == 'user' else 'Assistant'}:** {message.content}")
    with st.container(height=CHAT_HEIGHT, border=True, autoscroll=True):
        with st.chat_message("assistant", avatar=":material/support_agent:"):
            st.markdown(opening)
        for message in history[-CHAT_SHOWN:]:
            chat_bubble(store, ctx.user_id, message, unconfirmed)
    chosen = st.session_state.pop("pending_prompt", None) or suggestion_buttons(suggestions_)
    prompt = st.chat_input("Ask, tell me something, or attach a document", key="chat_input",
                           accept_file=True, file_type=UPLOAD_TYPES)
    if chosen:
        ask_assistant(store, ctx, chosen)
    elif prompt:
        handle_prompt(store, ctx, ctx.pack(), prompt)


def suggestion_buttons(suggestions_: list[str]) -> str | None:
    """One-click questions; returns the clicked one."""
    row = st.container(horizontal=True)
    for number, text in enumerate(suggestions_):
        if row.button(text, key=f"suggest_{number}", type="tertiary", icon=":material/chat_bubble:"):
            return text
    return None


def chat_bubble(store: TwinStore, user_id: str, message: ChatMessage, unconfirmed: set[int]) -> None:
    """One message; an assistant reply also shows facts to confirm, sources, and how it was checked."""
    avatar = ":material/person:" if message.role == "user" else ":material/support_agent:"
    with st.chat_message(message.role, avatar=avatar):
        st.markdown(message.content)
        if message.role != "assistant":
            return
        details = message.details or {}
        for fact in details.get("facts", []):
            if fact.get("fact_id") in unconfirmed:
                fact_to_confirm(store, user_id, message, fact)
        if details.get("proposals"):
            st.caption(":material/gavel: Prepared for you: approve it in the **Now** box.")
        reply_footer(details)


def fact_to_confirm(store: TwinStore, user_id: str, message: ChatMessage, fact: dict) -> None:
    """A fact the assistant recorded from the user's words: it counts only after this Confirm."""
    text, button = st.columns([4, 1])
    text.markdown(f":orange[:material/rate_review:] {readable(fact['field'])}: **{fact['value']}** "
                  "(from you, not counted until you confirm it)")
    if button.button("Confirm", key=f"chat_confirm_{message.message_id}_{fact['fact_id']}"):
        store.confirm_fact(user_id, fact["fact_id"])
        st.rerun()


def reply_footer(details: dict) -> None:
    """Sources, the check by code and the reflection rounds of one reply, in one small line."""
    sources = details.get("sources", [])
    if sources:
        st.caption("Sources: " + " · ".join(f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
                                            for s in sources))
    note = REPLY_NOTES.get(details.get("status", ""), "")
    model = f" · {details['model']}" if details.get("model") else ""
    rounds = details.get("reflections") or []
    if rounds:
        with st.expander(f"{note}{model} · reflection: {reflection_summary(rounds)}"):
            for item in rounds:
                for issue in item.get("issues", []):
                    st.markdown(f"- *{item['by']}*: {issue}")
    elif note or model:
        st.caption(note + model)


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
        reply = run_chat(message, store, ctx.packs, ctx.user_id, ctx.today, agent_backend("chat", provider()),
                         outbox_dir=OUTBOX_DIR, embedder=local_backend(), progress=status.write,
                         model_label=agent_model_label("chat", provider()))
        status.update(label="Answer ready", state="complete", expanded=False)
    if reply.status == "invalid_input":              # Gate 0 refused it: nothing reached the model or the Twin
        flash(f"Gate 0: {reply.text}", "warning")
        return
    st.session_state.setdefault("results", {})[ctx.user_id] = {"goal": message, "result": reply.model_dump(mode="json")}


def read_in_chat(store: TwinStore, ctx: ToolContext, pack: Pack, name: str, content: bytes) -> None:
    """A document attached in the chat: read it, then say in the chat what it gave."""
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="user", content=f"📎 {name}", procedure_id=pack.id))
    documents = read_document(store, ctx, pack, name, content, AUTO_TYPE)
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="assistant", procedure_id=pack.id,
                                       content=reading_summary(pack, name, documents)))


# ------------------------------------------------------------- now box --
def now_box(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, steps: list[Step]) -> None:
    """The ONE next step of the plan, with the control it needs right here."""
    with st.container(border=True):
        st.markdown("#### :material/bolt: Now")
        if not steps:
            st.success("Nothing to do right now. Ask the assistant what comes next, or wait for the office.")
            return
        step = steps[0]
        st.caption(f"{KIND_WORDS[step.kind]} · {len(steps)} step{'' if len(steps) == 1 else 's'} left")
        STEP_VIEWS[step.kind](store, ctx, pack, view, step)


def decide_step(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, step: Step) -> None:
    """The agent prepared something: read it, then approve or deny."""
    proposal = store.get_proposal(step.ref)
    if proposal is None:
        return
    st.markdown(f"{TOOL_ICONS.get(proposal.tool, ':material/task:')} **{proposal.title}**")
    with st.container(height=220, border=True):
        st.markdown(proposal_preview(proposal))
    buttons = st.container(horizontal=True)
    if buttons.button("Approve", key=f"approve_{proposal.action_id}", type="primary", icon=":material/check:"):
        result = approve_action(store, ctx.packs, ctx.user_id, proposal.action_id, ctx.today, OUTBOX_DIR)
        flash(result.message + (" Open it from 'Ready for you'." if result.status == "executed" else ""),
              "success" if result.status == "executed" else "warning")
        st.rerun()
    if buttons.button("Deny", key=f"deny_{proposal.action_id}"):
        flash(deny_action(store, ctx.user_id, proposal.action_id).message)
        st.rerun()
    if proposal.requires_stage and buttons.button("Check in sandbox", key=f"check_{proposal.action_id}"):
        try_stage(store, pack, ctx.user_id, ctx.today, proposal.requires_stage)


def confirm_step(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, step: Step) -> None:
    """A value a model was not sure about (or documents that disagree): the user decides."""
    if step.ref in view.conflicts:
        choice = st.radio(f"Your documents disagree on the {readable(step.ref)}. Which one is right?",
                          [str(value) for value in view.conflicts[step.ref]], key=f"conflict_{step.ref}")
        if st.button("Save my choice", key=f"resolve_{step.ref}", type="primary"):
            service.resolve_conflict(store, pack, ctx.user_id, step.ref, choice)
            st.rerun()
        return
    fact = next((fact for fact in view.unconfirmed if fact.field == step.ref), None)
    if fact is not None:
        value_check(store, pack, ctx.user_id, fact)


def value_check(store: TwinStore, pack: Pack, user_id: str, fact: Fact) -> None:
    """One value to check: what it is, where it comes from, Yes / Change."""
    document = store.get_document(user_id, fact.source_ref or "") if fact.source_ref else None
    origin = f"read from your {document_title(pack, document)}" if document else "you told me in the chat"
    st.markdown(f"{field_question(pack, fact.field)}  \n### {shown_value(pack, fact.field, fact.value)}")
    st.caption(origin + (f" · “{fact.snippet}”" if fact.snippet else ""))
    buttons = st.container(horizontal=True)
    if buttons.button("Yes, that's right", key=f"confirm_{fact.fact_id}", type="primary", icon=":material/check:"):
        store.confirm_fact(user_id, fact.fact_id)
        st.rerun()
    with buttons.popover("Change it", icon=":material/edit:"):
        with st.form(f"correct_{fact.fact_id}", border=False):
            new = field_input(pack, fact.field, f"correct_value_{fact.fact_id}")
            if st.form_submit_button("Save") and new is not None:
                save_correction(store, pack, user_id, fact, new)


def save_correction(store: TwinStore, pack: Pack, user_id: str, fact: Fact, value) -> None:
    """The corrected value counts; the unsure one is closed."""
    raw = value.isoformat() if isinstance(value, date) else value
    try:
        service.correct_unconfirmed(store, pack, user_id, fact.fact_id, raw)
        flash(f"Saved: {readable(fact.field)} = {raw}.", "success")
    except ValueError as error:
        flash(f"Not saved: {error}", "error")
    st.rerun()


def fix_step(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, step: Step) -> None:
    """A requirement that fails (or a recommendation): what it is, its source, and the assistant to ask how."""
    rule = pack.rule(step.ref)
    text = rule_label_by_id(pack, step.ref) + (" (unverified rule)" if rule and rule.status == "unverified" else "")
    st.markdown(f":red[:material/error:] {text}" if step.kind == "fix" else f":orange[:material/lightbulb:] {text}")
    if rule is not None:
        st.caption("Source: " + source_links(pack, rule.source_ids))
    buttons = st.container(horizontal=True)
    buttons.button("Ask the assistant how", key=f"how_{step.ref}", type="primary", on_click=ask_in_chat,
                   args=(f"How do I fix this: {text}?",))
    buttons.button("Skip for now", key=f"skip_{step_key(step)}", on_click=skip_step, args=(step_key(step),))


def source_links(pack: Pack, source_ids: list[str]) -> str:
    links = []
    for source_id in source_ids:
        source = pack.source(source_id)
        links.append(f"[{source.title}]({source.url})" if source is not None else source_id)
    return " · ".join(links) or "none"


def upload_step(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, step: Step) -> None:
    """A document to upload, right here (or with + in the chat)."""
    spec = pack.document(step.ref)
    st.markdown(f"**Upload your {spec.description}**")
    st.caption(step.detail.replace("It gives:", "It tells me:"))
    uploaded = st.file_uploader("File", type=UPLOAD_TYPES, key=f"upload_{step.ref}", label_visibility="collapsed")
    buttons = st.container(horizontal=True)
    if uploaded is not None and buttons.button("Read it", key=f"read_{step.ref}", type="primary"):
        read_upload(store, ctx, pack, uploaded.name, uploaded.getvalue(), step.ref)
    buttons.button("I don't have it yet", key=f"skip_{step_key(step)}", on_click=skip_step, args=(step_key(step),))


def answer_step(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, step: Step) -> None:
    """One question, with the input that fits it: Yes / No buttons, choice buttons, a date, a number or text."""
    spec = pack.field(step.ref)
    st.markdown(f"**{field_question(pack, step.ref)}**")
    if spec.type == "bool" or (spec.choices and len(spec.choices) <= MAX_CHOICE_BUTTONS):
        options = {"Yes": "yes", "No": "no"} if spec.type == "bool" else spec.choices
        for number, (label, value) in enumerate(options.items()):
            if st.button(label, key=f"answer_{step.ref}_{number}", width="stretch"):
                save_answers(store, pack, ctx.user_id, {step.ref: value})
    else:
        with st.form(f"answer_{step.ref}", border=False):
            value = field_input(pack, step.ref, f"answer_value_{step.ref}", label_visibility="collapsed")
            if st.form_submit_button("Save", type="primary") and value is not None:
                save_answers(store, pack, ctx.user_id, {step.ref: value})
    st.button("I don't know yet", key=f"skip_{step_key(step)}", on_click=skip_step, args=(step_key(step),),
              type="tertiary")


def try_step(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, step: Step) -> None:
    """Nothing blocks: try the official step in the mock portal first."""
    st.markdown(f"Nothing blocks **{stage_title(pack, step.ref)}**. Try it in the sandbox, then do it for real.")
    if st.button("Try it in the sandbox", key=f"try_{step.ref}", type="primary", icon=":material/science:"):
        try_stage(store, pack, ctx.user_id, ctx.today, step.ref)


STEP_VIEWS = {"decide": decide_step, "fix": fix_step, "improve": fix_step, "confirm": confirm_step,
              "upload": upload_step, "answer": answer_step, "try": try_step}


def coming_up(steps: list[Step]) -> None:
    """The next few steps after Now (read only), and the whole plan folded away."""
    if len(steps) <= 1:
        return
    st.markdown("**Then:** " + " · ".join(short_step(step) for step in steps[1:1 + COMING_UP]))
    with st.expander(f"The whole plan ({len(steps)} steps)"):
        for number, step in enumerate(steps, start=1):
            st.markdown(f"{number}. {step.title}")
        st.caption("Computed by code from your documents and answers; it updates by itself.")


def short_step(step: Step) -> str:
    title = step.title.removeprefix("Upload, if you have it: ").removeprefix("Fix: ").removeprefix("Recommended: ")
    return title if len(title) <= 60 else title[:57] + "…"


def ready_box(store: TwinStore, user_id: str) -> None:
    """Approved items: open them in Gmail / Google Calendar (already filled in), or download the file."""
    done = [p for p in store.list_proposals(user_id) if p.status == "executed"][-READY_SHOWN:]
    if not done:
        return
    with st.container(border=True):
        st.markdown("##### :material/inventory_2: Ready for you")
        for proposal in reversed(done):
            st.markdown(f"{TOOL_ICONS.get(proposal.tool, ':material/task:')} {proposal.title}")
            row = st.container(horizontal=True)
            for number, (label, url) in enumerate(open_links(proposal)):
                row.link_button(label, url, type="primary" if number == 0 else "secondary")
            path = Path(proposal.artifact_path) if proposal.artifact_path else None
            if path is not None and path.is_file():
                row.download_button("Download", path.read_bytes(), file_name=path.name, key=f"dl_{proposal.action_id}")
        st.caption("Gmail and Google Calendar open already filled in: you press Send or Save. The agent never sends.")


def prepare_box(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView) -> None:
    """What the agent can prepare now (each one is a proposal to approve), and the full check by the agent team."""
    with st.expander("The agent can prepare: checklist, form, calendar, e-mail", icon=":material/construction:"):
        found = suggestions(pack, view, ctx.today, store.list_proposals(ctx.user_id))
        for number, suggestion in enumerate(found):
            action_button(store, ctx, suggestion, f"action_{number}")
        st.caption("E-mails open in Gmail and reminders in Google Calendar after you approve them.")
    with st.expander("Full check by the agent team", icon=":material/groups:"):
        full_check_box(store, ctx)


def action_button(store: TwinStore, ctx: ToolContext, suggestion: Suggestion, key: str) -> None:
    """One suggestion as a button: run the tool (a proposal), or ask the assistant to write an e-mail."""
    icon = TOOL_ICONS.get(suggestion.tool, ":material/task:")
    if suggestion.pending:
        st.button(f"{suggestion.label} · waiting for your approval", key=key, icon=icon, disabled=True)
    elif suggestion.ask_assistant:
        st.button(suggestion.label, key=key, icon=icon, help=suggestion.why, on_click=ask_in_chat,
                  args=(suggestion.ask_assistant,))
    elif st.button(suggestion.label, key=key, icon=icon, help=suggestion.why):
        observation = run_tool(build_registry(), suggestion.tool, suggestion.args, ctx, "orchestrator")
        flash("Prepared: approve it in the Now box." if observation.ok else str(observation.error),
              "success" if observation.ok else "warning")
        st.rerun()


def full_check_box(store: TwinStore, ctx: ToolContext) -> None:
    """The orchestrator with its workers checks every section; the answer is posted in the chat."""
    st.caption("The orchestrator gives each section of the procedure to a worker, checks their findings and writes "
               "one answer (Gate 1 verifies it). Slower than the chat.")
    if st.button("Run a full check", icon=":material/groups:"):
        if run_locked(ctx.user_id, "The agent team is checking your file...", lambda: full_check(store, ctx)):
            st.rerun()


def full_check(store: TwinStore, ctx: ToolContext) -> None:
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="user", content="🔍 Full check by the agent team",
                                       procedure_id=ctx.procedure_id))
    result = orchestrator.run_agent(FULL_CHECK_GOAL, store, ctx.packs, ctx.user_id, ctx.today,
                                    agent_backend("team", provider()), outbox_dir=OUTBOX_DIR,
                                    tool_backend=document_backend(provider(), documents_cloud()))
    st.session_state.setdefault("results", {})[ctx.user_id] = {"goal": FULL_CHECK_GOAL,
                                                               "result": result.model_dump(mode="json")}
    text = (result.question or "") if result.status == "needs_user_input" else agent_text(result)
    store.add_chat_message(ChatMessage(user_id=ctx.user_id, role="assistant", content=text, procedure_id=result.procedure_id,
                                       trace_id=result.trace_id, details={"status": result.status,
                                                                          "proposals": result.proposals,
                                                                          "reflections": result.reflections,
                                                                          "model": agent_model_label("team", provider())}))


def agent_text(result) -> str:
    """The orchestrator's answer as chat text: summary, then numbered next actions."""
    answer = result.answer
    if answer is None:
        return "The agent team could not finish. Try again when the model is ready."
    lines = [answer.summary]
    lines += [f"{number}. {action.title}" for number, action in enumerate(answer.next_actions, start=1)]
    return "\n".join(lines)


def field_input(pack: Pack, field: str, key: str, label_visibility: str = "visible"):
    """One input widget that fits the field (yes/no, date, number, a choice, text). None when left empty."""
    spec = pack.field(field)
    label = field_question(pack, field)
    if spec.type == "bool":
        choice = st.radio(label, [NOT_ANSWERED, "yes", "no"], horizontal=True, key=key, label_visibility=label_visibility)
        return None if choice == NOT_ANSWERED else choice
    if spec.type == "date":
        return st.date_input(label, value=None, min_value=EARLIEST_DATE, max_value=LATEST_DATE, key=key,
                             label_visibility=label_visibility)
    if spec.type == "int":
        return st.number_input(label, value=None, step=1, key=key, label_visibility=label_visibility)
    if spec.choices:
        choice = st.radio(label, [NOT_ANSWERED] + list(spec.choices), key=key, label_visibility=label_visibility)
        return None if choice == NOT_ANSWERED else spec.choices[choice]
    return st.text_input(label, key=key, label_visibility=label_visibility).strip() or None


def save_answers(store: TwinStore, pack: Pack, user_id: str, values: dict) -> None:
    """Answers given with a click or typed in the app are explicit, so they are stored as confirmed facts."""
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
        flash("Saved. Your plan is updated.", "success")
    st.rerun()


# -------------------------------------------------------- view: documents --
def view_documents(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView) -> None:
    """Upload, then each document of the procedure with what was read from it."""
    upload_form(store, ctx, pack)
    documents = [document for document in store.list_documents(ctx.user_id) if document.status != "deleted"]
    for spec in [s for s in pack.procedure.documents if not s.optional] + [s for s in pack.procedure.documents if s.optional]:
        document_row(store, ctx, pack, view, spec, [d for d in documents if d.doc_type == spec.doc_type])
    unknown = [d for d in documents if d.doc_type in (None, "unknown") or pack.document(d.doc_type) is None]
    for document in unknown:
        with st.container(border=True):
            st.markdown(f"❔ **{Path(document.file_path).name}** · I could not tell which document this is.")
            document_tools(store, ctx, pack, document)


def upload_form(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """Real -> twin: a document is read, and its values become facts of the Twin."""
    with st.form("upload", clear_on_submit=True, border=True):
        uploaded = st.file_uploader("Add a document (PDF, TXT or a photo). One PDF may hold several documents; "
                                    "I find out which is which.", type=UPLOAD_TYPES)
        submitted = st.form_submit_button("Upload and read", type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Choose a file first.")
        else:
            read_upload(store, ctx, pack, uploaded.name, uploaded.getvalue(), AUTO_TYPE)


def document_row(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, spec: DocumentSpec,
                 documents: list[TwinDocument]) -> None:
    """One document of the procedure: its status, the values read, the values to check (confirm them here)."""
    status = document_status(pack, documents, view, spec.doc_type)
    label = "Only if needed" if spec.optional and status == "missing" else CHECKLIST_LABELS[status]
    with st.container(border=True):
        st.markdown(f"{CHECKLIST_ICONS[status]} **{spec.description}** · :gray[{label}]")
        for document in documents:
            document_body(store, ctx, pack, view, document)


def document_body(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, document: TwinDocument) -> None:
    """What one uploaded document gave: values, values to check (with Confirm), notes; tools folded away."""
    observation = document.observation or {}
    st.caption(" · ".join(part for part in (Path(document.file_path).name, pages_text(observation.get("pages", [])),
                                            FOUND_BY.get(observation.get("found_by", ""), "")) if part))
    if document.processing_error:
        st.error(friendly_error(document.processing_error))
    known = [f"{readable(field)}: **{shown_value(pack, field, fact.value)}**" for field, facts in sorted(view.sources.items())
             for fact in facts if fact.source_ref == document.doc_id]
    if known:
        st.markdown(" · ".join(known))
    for fact in [fact for fact in view.unconfirmed if fact.source_ref == document.doc_id]:
        text, button = st.columns([4, 1], vertical_alignment="center")
        text.markdown(f":orange[:material/rate_review:] {readable(fact.field)}: "
                      f"**{shown_value(pack, fact.field, fact.value)}** (please check)")
        if button.button("Confirm", key=f"confirm_doc_{fact.fact_id}"):
            store.confirm_fact(ctx.user_id, fact.fact_id)
            st.rerun()
    for note in observation.get("notes", [])[:2] + [friendly_issue(i) for i in observation.get("issues", [])[:2]]:
        st.caption(f":material/info: {note}")
    document_tools(store, ctx, pack, document)


def document_tools(store: TwinStore, ctx: ToolContext, pack: Pack, document: TwinDocument) -> None:
    """Wrong type? Read it again as another document. Or delete it."""
    with st.popover("Wrong document? / Delete", icon=":material/more_horiz:"):
        labels = {spec.doc_type: spec.description for spec in pack.procedure.documents}
        chosen = st.selectbox("Read it again as", list(labels), format_func=labels.get, key=f"again_{document.doc_id}")
        if st.button("Read again", key=f"again_button_{document.doc_id}"):
            pages = (document.observation or {}).get("pages") or None

            def work() -> None:
                flash(reading_summary(pack, Path(document.file_path).name,
                                      run_extraction(store, ctx, document.doc_id, chosen, pages)), "success")

            if run_locked(ctx.user_id, "Reading the document again...", work):
                st.rerun()
        if st.button("Delete this document", key=f"delete_{document.doc_id}"):
            service.delete_document(store, ctx.user_id, pack.id, document.doc_id)
            st.rerun()


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


# ---------------------------------------------------------- view: details --
def view_details(store: TwinStore, ctx: ToolContext, pack: Pack, view: FactView, results: list[RuleResult]) -> None:
    """For examiners and developers: requirements, sandbox, how it works, the last answer, history, Twin, sources."""
    tabs = st.tabs(["Requirements", "Sandbox", "How it works", "Last answer", "History", "Twin", "Knowledge base"])
    with tabs[0]:
        requirements_overview(pack, results)
    with tabs[1]:
        sandbox_box(store, pack, ctx.user_id, ctx.today)
        outcome_box(store, pack, ctx.user_id)
        done_list(store, ctx.user_id)
    with tabs[2]:
        st.markdown(HOW_IT_WORKS.format(chat_steps=CHAT_MAX_STEPS, chat_repairs=CHAT_MAX_REPAIRS,
                                        review="on" if review_enabled(is_cloud(provider()))
                                        else "off with a local model (switch on with SELF_REVIEW=1)",
                                        worker_steps=WORKER_STEPS, max_steps=MAX_STEPS, max_repairs=MAX_REPAIRS,
                                        embed=EMBED_MODEL, reader=document_model_label(provider(), documents_cloud())))
        st.caption(f"Models: assistant {agent_model_label('chat', provider())}; agent team "
                   f"{agent_model_label('team', provider())}; documents "
                   f"{document_model_label(provider(), documents_cloud())}.")
        usage_box()
    with tabs[3]:
        last_answer(st.session_state.get("results", {}).get(ctx.user_id))
    with tabs[4]:
        run_history(store, pack, ctx.user_id)
    with tabs[5]:
        st.dataframe(fact_rows(view), hide_index=True)
        with st.expander("All rule results (raw)"):
            st.json([result.model_dump() for result in results])
    with tabs[6]:
        knowledge_box(store, pack)


def requirements_overview(pack: Pack, results: list[RuleResult]) -> None:
    """Every requirement per section, with its status, what it waits for and the official sources."""
    mine = user_results(pack, results)
    counts = count_status(mine)
    st.caption(f"{counts['pass']} met, {counts['fail']} not met, {counts['unknown']} waiting for information. "
               "Computed by the rule engine; the assistant never decides them.")
    for section in user_sections(pack):
        section_results = in_section(results, section.id)
        met = count_status(section_results)["pass"]
        st.markdown(f"**{section_icon(section_results)} {section.title} · {met}/{len(section_results)}**"
                    + (" · :gray[later step]" if section.later else ""))
        for result in section_results:
            st.markdown(rule_line(result))
    st.markdown("**Official sources**")
    for source in pack.sources:
        st.markdown(f"- [{source.title}]({source.url}) :gray[· {source.organization or source.authority} · "
                    f"retrieved {source.retrieved_at or 'never'}]")


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
    with st.expander("Something happened outside, or a value is wrong? Tell the Twin", icon=":material/campaign:"):
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
    """Every approved or denied action."""
    done = [proposal for proposal in store.list_proposals(user_id) if proposal.status not in OPEN_PROPOSALS]
    if done:
        st.markdown("**Approved and denied actions**")
        for proposal in done:
            st.markdown(f"{TOOL_ICONS.get(proposal.tool, ':material/task:')} {proposal.title} · {proposal.status}")


def last_answer(last: dict | None) -> None:
    """The ReAct steps of the last answer in this session, its reflection rounds and Gate 1's findings."""
    if not last:
        st.caption("No agent answer in this session yet. Ask the assistant something, then come back here.")
        return
    result = last["result"]
    st.caption(f"Message: {last['goal']} · status: {result.get('status')} · model calls: {result.get('llm_calls', 0)}")
    st.markdown("**ReAct steps** (decide → act → observe)")
    st.dataframe(step_rows(result), hide_index=True)
    rounds = result.get("reflections") or []
    st.markdown("**Reflection**: " + (reflection_summary(rounds) if rounds else "no correction was needed"))
    for item in rounds:
        for issue in item.get("issues", []):
            st.markdown(f"- *{item['by']}*: {issue}")
    if result.get("gate_issues"):
        st.markdown("**Gate 1 issues of the final answer**")
        st.json(result["gate_issues"])


def usage_box() -> None:
    """Model requests since the app started: how many, tokens, model time and the estimated cloud cost."""
    usage = usage_summary(USAGE.since(0))
    if not usage["requests"]:
        st.caption("No model request yet since the app started.")
        return
    tokens = ("n/a" if usage["input_tokens"] is None
              else f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
    cost = "n/a" if usage["cost_usd"] is None else f"${usage['cost_usd']:.4f}"
    cap = f" of {CLOUD_MAX_REQUESTS}" if CLOUD_MAX_REQUESTS else ""
    st.markdown(f"**Model usage since the app started:** {usage['requests']} requests "
                f"({usage['cloud_requests']}{cap} in the cloud, {usage['failures']} failed) · tokens {tokens} · "
                f"model time {usage['model_seconds']:.0f} s · estimated cloud cost {cost}")
    st.caption(f"Cost from the price list in config.py (as of {CLOUD_PRICES_AS_OF}); n/a when a model has no price. "
               "CLOUD_MAX_REQUESTS caps the cloud requests; past it the agent answers with the plan written by code.")


def run_history(store: TwinStore, pack: Pack, user_id: str) -> None:
    """Every run of this profile (memory across sessions) and the lessons kept from validated repairs."""
    runs = [run for run in store.list_runs(user_id) if run.procedure_id == pack.id]
    st.markdown("**Run log** (memory across sessions)")
    st.dataframe([{"when": run.started_at.strftime("%Y-%m-%d %H:%M"), "status": run.status, "steps": run.steps,
                   "llm calls": run.llm_calls, "rewrites": run.repairs, "passed": run_passed(run)}
                  for run in runs], hide_index=True)
    st.markdown("**Lessons from validated repairs** (episodic memory: shown to the model next time)")
    st.dataframe([{"failure": e.failure_type, "rule": e.rule_or_source_id, "lesson": e.corrective_action}
                  for e in store.find_episodes(pack.id, limit=20)], hide_index=True)


def knowledge_box(store: TwinStore, pack: Pack) -> None:
    """The official sources as searchable passages, with a box to test what a question retrieves."""
    st.caption(f"{len(pack_chunks(pack))} passages from {len(pack.sources)} official sources of this procedure "
               f"(embeddings: {EMBED_MODEL}, local). Web corpus for procedures without a checklist: "
               f"{len(load_corpus())} pages. Used to quote sources and answer questions the rules do not cover; "
               "never to decide a requirement (measured: docs/eval_rag.md).")
    with st.form("rag_test", border=False):
        query = st.text_input("Test a question", placeholder="Which health insurance is accepted?")
        if st.form_submit_button("Search the sources") and query.strip():
            try:
                for hit in search(store, pack, local_backend(), query.strip(), 3):
                    st.markdown(f"**{hit.title}** · similarity {hit.score:.2f}  \n{hit.text}")
            except Exception as error:   # UI boundary: the embedding model may be off
                show_error(error)


def run_passed(run) -> str:
    """'passed/total' of the rule results saved with a run ('' for runs logged before results were kept)."""
    if not run.results:
        return ""
    return f"{sum(result.status == 'pass' for result in run.results)}/{len(run.results)}"


# ------------------------------------------------------------------ page --
def tool_context(store: TwinStore, packs: dict[str, Pack], user_id: str, today: date) -> ToolContext:
    """Tools read documents with `document_backend`: the local models, or the cloud model when the sidebar says so."""
    active = store.active_procedure(user_id)
    return ToolContext(store=store, packs=packs, user_id=user_id, today=today,
                       backend=document_backend(provider(), documents_cloud()),
                       procedure_id=active if active in packs else None)


def no_procedure_page(store: TwinStore, ctx: ToolContext) -> None:
    """Before a procedure is chosen: the chat finds the procedure, or answers from the web corpus."""
    st.title("What do you want to do?")
    chat_panel(store, ctx, None, WELCOME, FIRST_SUGGESTIONS)
    st.caption("Procedures with a full checklist: " + "; ".join(pack_label(p) for p in ctx.packs.values())
               + ". For anything else the assistant answers from official pages, without checking your documents.")


def procedure_page(store: TwinStore, ctx: ToolContext, pack: Pack) -> None:
    """The journey on top, then the chosen view."""
    view = ctx.view()
    results = evaluate_rules(pack, view.fact_map())
    briefing = build_briefing(pack, store.list_documents(ctx.user_id), store.list_proposals(ctx.user_id), view, results)
    st.markdown(f"## {pack.procedure.title}")
    if pack.procedure.status == "unverified":
        st.warning("The rules of this procedure are not verified yet. Check every result with the official office.")
    journey_bar(pack, results, briefing)
    chosen = st.segmented_control("View", list(VIEWS), format_func=VIEWS.get, key="view",
                                  default=None if "view" in st.session_state else "assistant",
                                  label_visibility="collapsed") or "assistant"
    if chosen == "assistant":
        view_assistant(store, ctx, pack, view, briefing)
    elif chosen == "documents":
        view_documents(store, ctx, pack, view)
    else:
        view_details(store, ctx, pack, view, results)


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
