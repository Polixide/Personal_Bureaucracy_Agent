"""Daniele's public Knowledge base tester in the target's Your details workspace."""
from time import perf_counter

from ollama import ResponseError
import httpx
import streamlit as st

from src.agents.assistant import _guidance_answer, select_guidance_sources
from src.core.llm_diagnostics import collect_model_calls, last_model_call
from src.core.model_provider import OpenAIRequestError, use_ai_options
from src.rag.retriever import retrieval_status, search_official_sources
from src.ui.model_settings import describe_provider, last_run_metrics, remember_ai_run, ui_ai_options


SCOPES = {"case": "First residence permit", "public": "All public reference sources"}


def render_knowledge(state, thread_key, *, busy=False):
    st.subheader("Knowledge base")
    st.caption("Search public source snapshots. Guidance here does not change your case or its requirements.")
    scope = st.selectbox("Search scope", list(SCOPES), format_func=SCOPES.__getitem__, key=f"rag_scope_{thread_key}")
    try:
        status = retrieval_status(scope)
    except (OSError, ValueError, TypeError, KeyError):
        st.error("The public index could not be loaded. Check the source manifests and snapshots.")
        return
    st.write(f"{status['indexed_chunks']} searchable chunks · "
             f"{status['public_sources_loaded']}/{status['public_sources_listed']} source snapshots verified")
    if status["embedding_model"]:
        st.caption(f"Local vectors · {status['embedding_model']} · {status['vector_status']}"
                   f" · {status['indexed_vectors'] if status['indexed_vectors'] is not None else 0} vectors")
    else:
        st.caption("Lexical index ready")
    if status["issues"]:
        with st.expander("Source files requiring review"):
            for issue in status["issues"]:
                st.write(issue)

    options = ui_ai_options(thread_key)
    result_key = f"knowledge_result_{thread_key}"
    st.caption(describe_provider(thread_key))
    if options.provider == "openai":
        st.caption("Generate answer sends your question and the selected public passages to OpenAI. Use a public question.")
    with st.form(f"rag_test_{thread_key}"):
        question = st.text_input("Test a question", max_chars=1000,
                                 placeholder="What address proof do I need if I live in a dormitory?")
        retrieve = st.form_submit_button("Retrieve evidence", disabled=busy)
    if retrieve and question.strip():
        data = {"scope": scope, "question": question.strip(), "fingerprint": status["fingerprint"], "hits": []}
        details = {}
        start = perf_counter()
        with use_ai_options(options), collect_model_calls() as calls:
            try:
                hits = search_official_sources(data["question"], scope=scope, details=details)
                data["hits"] = [hit.model_dump(mode="json") for hit in hits]
            except (OSError, ValueError, TypeError, KeyError):
                data["error"] = "Source search could not finish. Check the corpus status and try again."
        data["retrieval"] = details
        st.session_state[result_key] = data
        remember_ai_run(thread_key, last_run_metrics(calls, perf_counter() - start, operation="source retrieval",
                                                     retrieval_calls=1))
    result = st.session_state.get(result_key)
    if result and result["scope"] == scope:
        if result["fingerprint"] != status["fingerprint"]:
            st.info("The public sources changed. Retrieve evidence again for this question.")
        else:
            _render_result(result, thread_key, options, busy)
    with st.expander("Evaluation and metrics"):
        from src.ui.technical import render_ai_performance

        render_ai_performance(state, thread_key)


def _render_result(result, thread_key, options, busy):
    if result.get("error"):
        st.error(result["error"])
        return
    hits = result["hits"]
    details = result["retrieval"]
    st.caption(f"Sources retrieved for this question · {len(hits)} "
               f"· {details.get('mode', 'lexical')} · {details.get('latency_seconds', 0):.3f} s")
    if details.get("embedding_message"):
        st.warning(details["embedding_message"])
    if not hits:
        st.info("No relevant public passages were found. Try a more specific question.")
        return
    if st.button("Generate answer", key=f"rag_answer_{thread_key}", disabled=busy):
        _generate(result, thread_key, options)
    if result.get("generation_error"):
        st.warning(result["generation_error"])
    if "answer" in result:
        st.markdown("#### RAG answer")
        st.markdown(result["answer"])
        if result.get("answer_model"):
            st.caption("Answer model · " + result["answer_model"])
    st.markdown("#### Retrieved sources")
    for hit in hits:
        with st.expander(hit["title"], expanded=False):
            st.write(hit["content"])
            st.caption(f"{hit['procedure'].replace('_', ' ')} · {hit['authority'].replace('_', ' ')}")
            if hit.get("url"):
                st.markdown(f"[Open source]({hit['url']})")
            if hit.get("file_ref"):
                st.caption(hit["file_ref"])
    with st.expander("Advanced retrieval details"):
        st.json({"retrieval": details, "sources": [
            {key: hit[key] for key in ("source_id", "chunk_id", "retrieved_at", "source_version", "score",
                                      "lexical_score", "cosine_score", "file_ref")}
            for hit in hits]})
        st.caption("Scores rank passages within this query. Labeled accuracy is available in benchmark reports.")


def _generate(result, thread_key, options):
    result.pop("generation_error", None)
    result.pop("answer_model", None)
    result.pop("answer", None)
    start = perf_counter()
    task = "details_rag_generation"
    with use_ai_options(options), collect_model_calls() as calls, st.spinner("Selecting evidence for the answer..."):
        try:
            selected = select_guidance_sources(result["question"], result["hits"], provider=options.provider, task=task)
        except OpenAIRequestError as error:
            result["generation_error"] = str(error)
        except (ResponseError, httpx.HTTPError, TimeoutError, ConnectionError, ValueError, TypeError, AttributeError):
            result["generation_error"] = "The selected model could not generate a valid source-grounded answer."
        else:
            result["answer"] = _guidance_answer(selected)
            if call := last_model_call(task):
                result["answer_model"] = f"{call.provider} · {call.response_model or call.model}"
    remember_ai_run(thread_key, last_run_metrics(calls, perf_counter() - start, operation="RAG answer generation"))
