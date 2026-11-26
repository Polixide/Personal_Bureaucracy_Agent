"""Per-profile Streamlit model settings and content-free run diagnostics."""
from collections import Counter
from dataclasses import asdict
import streamlit as st

import config
from config import OPENAI_DOCUMENT_MODEL, OPENAI_MODEL, PLANNER_MODEL
from src.core.model_provider import AIOptions, ERROR_MESSAGES, cloud_configuration_issue, configured_budget
from src.evaluation.metrics import summarize_calls


PROVIDER_LABELS = {"ollama": "Local AI (Ollama)", "openai": "OpenAI API"}
DOCUMENT_LABELS = {"local": "Local text and OCR", "cloud": "Offer OpenAI vision for each file"}
DOCUMENT_CONSENT = (
    "The OpenAI action authorizes sending selected text or PDF page images from this file. "
    "They may contain personal information."
)


def selected_provider(thread_key):
    return st.session_state.get(f"selected_ai_provider_{thread_key}", config.LLM_PROVIDER)


def selected_document_provider(thread_key):
    return st.session_state.get(f"selected_document_provider_{thread_key}", config.DOCUMENT_PROVIDER)


def ui_ai_options(thread_key):
    provider = selected_provider(thread_key)
    budget = st.session_state.get(f"ai_budget_{thread_key}")
    if budget is None and (provider == "openai" or selected_document_provider(thread_key) == "cloud"):
        budget = configured_budget()
        st.session_state[f"ai_budget_{thread_key}"] = budget
    return AIOptions(provider=provider, budget=budget,
                     rag_enabled=st.session_state.get(f"selected_rag_{thread_key}", True))


def cloud_documents_available(thread_key):
    return selected_document_provider(thread_key) == "cloud" and cloud_configuration_issue(OPENAI_DOCUMENT_MODEL) is None


def describe_provider(thread_key):
    provider = selected_provider(thread_key)
    model = OPENAI_MODEL if provider == "openai" else PLANNER_MODEL
    label = PROVIDER_LABELS.get(provider, "Invalid LLM_PROVIDER")
    effort = config.OPENAI_REASONING_EFFORT if provider == "openai" else ""
    return " · ".join(value for value in (label, model, f"reasoning {effort}" if effort else "") if value)


def render_model_settings(thread_key):
    provider_key = f"widget_ai_provider_{thread_key}"
    saved_provider = f"selected_ai_provider_{thread_key}"
    doc_key = f"widget_document_provider_{thread_key}"
    saved_doc = f"selected_document_provider_{thread_key}"
    rag_key = f"widget_rag_{thread_key}"
    saved_rag = f"selected_rag_{thread_key}"

    def save_choice(widget, saved):
        st.session_state[saved] = st.session_state[widget]

    for label, choices, value, widget, saved in (
        ("AI model", PROVIDER_LABELS, selected_provider(thread_key), provider_key, saved_provider),
        ("Document processing", DOCUMENT_LABELS, selected_document_provider(thread_key), doc_key, saved_doc),
    ):
        st.selectbox(label, list(choices), format_func=choices.__getitem__,
                     index=list(choices).index(value) if value in choices else None, key=widget,
                     on_change=save_choice, args=(widget, saved))
    st.toggle("Use source retrieval in Copilot", value=st.session_state.get(saved_rag, True), key=rag_key,
              on_change=save_choice, args=(rag_key, saved_rag))
    st.caption(describe_provider(thread_key))
    if selected_provider(thread_key) == "openai":
        st.caption("OpenAI receives planning task categories and submitted guidance questions. Keep personal details out of questions.")
        if issue := cloud_configuration_issue(OPENAI_MODEL):
            st.warning(ERROR_MESSAGES[issue])
    if selected_document_provider(thread_key) == "cloud":
        st.caption("Cloud document model · " + (OPENAI_DOCUMENT_MODEL or "Set OPENAI_DOCUMENT_MODEL"))
        st.caption(DOCUMENT_CONSENT + " Choose the OpenAI action on the specific file to authorize it.")
        if issue := cloud_configuration_issue(OPENAI_DOCUMENT_MODEL):
            st.warning(ERROR_MESSAGES[issue])


def last_run_metrics(calls, elapsed, *, operation=None, retrieval_calls=0):
    rows = [asdict(call) for call in calls]
    return {"end_to_end_latency_seconds": elapsed, "summary": summarize_calls(rows), "calls": rows,
            "operation": operation, "retrieval_calls": retrieval_calls}


def remember_ai_run(thread_key, run):
    """Record once at execution boundaries. Renderers only read these summaries."""
    st.session_state[f"ai_run_metrics_{thread_key}"] = run
    session_key = f"ai_session_metrics_{thread_key}"
    previous = st.session_state.get(session_key, {})
    current = run["summary"]
    totals = {"interactions": previous.get("interactions", 0) + 1,
              "retrieval_calls": previous.get("retrieval_calls", 0) + run.get("retrieval_calls", 0),
              "provider_calls": dict(Counter(previous.get("provider_calls", {})) + Counter(current["provider_calls"]))}
    for key in ("llm_calls", "openai_requests", "prompt_tokens", "completion_tokens", "total_tokens",
                "transport_failures", "llm_latency_seconds"):
        before, after = previous.get(key, 0), current[key]
        totals[key] = None if before is None or after is None else before + after
    if not previous.get("openai_requests"):
        totals["estimated_cost_usd"] = current["estimated_cost_usd"]
    elif not current["openai_requests"]:
        totals["estimated_cost_usd"] = previous["estimated_cost_usd"]
    else:
        a, b = previous["estimated_cost_usd"], current["estimated_cost_usd"]
        totals["estimated_cost_usd"] = a + b if a is not None and b is not None else None
    st.session_state[session_key] = totals
