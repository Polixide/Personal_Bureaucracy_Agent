"""On-demand diagnostics and testing tools for the saved case."""
import json

from src.ui.components import *
from src.ui.model_settings import describe_provider


def open_advanced(thread_key, panel):
    st.session_state[f"advanced_panel_{thread_key}"] = panel
    request_view(thread_key, "technical")

def view_technical(state, snapshot, graph, config, thread_key):
    panels = (
        "Overview", "Rules and sources", "Plan and repairs", "Saved state and document evidence",
        "Tool activity", "Saved run history", "Compare a source snapshot", "Experimental assistant",
        "Legal-stay calculator", "Individual sandbox tools", "AI performance", "Document diagnostics",
    )
    panel = st.selectbox("Advanced", panels, key=f"advanced_panel_{thread_key}")
    if panel == "Overview":
        st.caption("Choose a topic to inspect your saved case or open a testing tool.")
    elif panel == "Rules and sources":
        render_rule_diagnostics(state)
        if st.button("Recheck saved information", disabled=bool(snapshot.next), key=f"recheck_{thread_key}"):
            run_event(graph, config, UserEvent(kind="reevaluate"))
    elif panel == "Plan and repairs":
        from src.agents.planning_policy import workflow_stage

        st.caption(f"Workflow stage: {workflow_stage(state).replace('_', ' ').title()}"
                   f" · Repair attempts: {state.get('retry_count', 0)} / {MAX_REFLECTION_ATTEMPTS}")
        st.json(jsonable({
            "summary": state.get("plan_summary", ""), "candidate_plan": plan_steps(state),
            "plan_origin": state.get("plan_origin") or state.get("approved_plan_origin"),
            "validation_errors": display_records(state.get("validation_errors")),
            "feedback": state.get("repair_feedback", []), "history": state.get("repair_history", []),
            "planner_diagnostics": state.get("planner_diagnostics", {}),
            "planner_candidate_set": state.get("planning_context_snapshot", {}).get("canonical_candidates", []),
            "action_progress": state.get("action_progress", {}),
        }))
    elif panel == "Saved state and document evidence":
        st.json(jsonable(state))
    elif panel == "Tool activity":
        from src.evaluation.metrics import state_metrics
        from src.tools.registry import get_tool_specs

        error = st.session_state.get(f"_last_ui_error_{thread_key}")
        if error:
            st.code(error)
        st.json(jsonable({
            "metrics": state_metrics(state), "context": state.get("assistant_diagnostics", {}),
            "tool_trace": state.get("tool_outcomes", [])[-30:], "sandbox": state.get("sandbox_form", {}),
        }))
        st.json([spec.model_dump(mode="json") for spec in get_tool_specs()])
    elif panel == "Saved run history":
        from src.ui import components

        try:
            with TwinStore(components.TWIN_RECORD_PATH) as store:
                st.json([run.model_dump(mode="json") for run in store.list_runs(state.get("user_id", ""))][-20:])
                st.json([lesson.model_dump(mode="json") for lesson in store.find_lessons(
                    state.get("user_id", ""), state.get("active_procedure", "ikamet")
                )])
        except Exception as error:
            st.caption(str(error))
    elif panel == "Compare a source snapshot":
        render_source_monitor(snapshot, graph, config, thread_key)
    elif panel == "Experimental assistant":
        from src.ui.home import render_chat

        st.warning("Use the guided workflow for preparation and decisions.")
        render_chat(snapshot, graph, config, thread_key)
    elif panel == "Legal-stay calculator":
        from src.ui.documents import render_legal_stay_technical

        render_legal_stay_technical(snapshot, graph, config, thread_key)
    elif panel == "Individual sandbox tools":
        render_sandbox_technical(snapshot, graph, config, thread_key)
    elif panel == "AI performance":
        render_ai_performance(state, thread_key)
    elif panel == "Document diagnostics":
        render_document_diagnostics(state, thread_key)


def render_document_diagnostics(state, thread_key):
    documents = [
        document for document in state.get("documents", ())
        if isinstance(document, Document) and (document.info is not None or document.analysis_diagnostics)
    ]
    if not documents:
        st.caption("Analyze a document to see its request diagnostics.")
        return
    index = st.selectbox(
        "Saved document analysis", range(len(documents)), key=f"document_diagnostics_{thread_key}",
        format_func=lambda i: f"{document_label(documents[i].doc_type)} · {i+1} of {len(documents)}",
    )
    document = documents[index]
    analysis = document.analysis_diagnostics
    if not analysis:
        st.caption("This saved result predates per-request diagnostics. Request details are N/A.")
        return

    def available(value):
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            return {key: available(item) for key, item in value.items()}
        if isinstance(value, list):
            return [available(item) for item in value]
        return value

    from src.document_processing.analysis_diagnostics import document_analysis_findings

    classification, findings = document_analysis_findings(document.info, analysis)
    for finding in findings:
        st.write(finding)
    metadata = {**analysis, "classification": classification}
    summary = {name.replace("_", " ").capitalize(): value
               for name, value in metadata.items() if name != "requests"}
    st.json(available(summary))
    requests = analysis.get("requests", [])
    if not requests:
        st.caption("This attempt made no model requests. Request usage and latency are N/A.")
    for request in requests:
        with st.expander(request["task"].replace("_", " ").capitalize()):
            st.json(available({name.replace("_", " ").capitalize(): value for name, value in request.items()}))
    st.caption("Image counts describe requests attempted by the app. PDF pages outside the image selection can still contribute local text.")


def _metric_value(value, unit=""):
    if value is None:
        return "Unavailable"
    digits = 6 if unit == "USD" else 4
    formatted = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return f"{formatted} {unit}".strip()


def render_ai_performance(state, thread_key):
    st.caption(describe_provider(thread_key))
    st.caption("Change the provider in Case settings. Reports use saved measurements.")
    planner = state.get("planner_diagnostics") or {}
    if planner:
        st.write("Saved plan")
        st.write({"Origin": state.get("plan_origin") or state.get("approved_plan_origin"),
                  "Reflection attempts": state.get("retry_count", 0),
                  "Validation issues": len(state.get("validation_errors") or [])})
    run = st.session_state.get(f"ai_run_metrics_{thread_key}")
    if run:
        st.write("Current interaction")
        if run.get("operation"):
            st.caption(run["operation"].replace("_", " ").capitalize())
        summary = run["summary"]
        a, b, c = st.columns(3)
        a.metric("Model requests", _metric_value(summary["llm_calls"]))
        b.metric("Input tokens", _metric_value(summary["prompt_tokens"]))
        c.metric("Output tokens", _metric_value(summary["completion_tokens"]))
        a, b, c = st.columns(3)
        a.metric("Model time", _metric_value(summary["llm_latency_seconds"], "s"))
        b.metric("End-to-end time", _metric_value(run["end_to_end_latency_seconds"], "s"))
        c.metric("Estimated OpenAI cost", _metric_value(summary["estimated_cost_usd"], "USD"))
        if run["calls"]:
            with st.expander("Request and usage details"):
                st.json({"summary": summary, "requests": run["calls"]})
    session = st.session_state.get(f"ai_session_metrics_{thread_key}")
    if session:
        with st.expander("Whole-session usage"):
            st.write({"Interactions": session["interactions"], "Model requests": session["llm_calls"],
                      "Requests by provider": session["provider_calls"], "Source searches": session["retrieval_calls"],
                      "Input tokens": _metric_value(session["prompt_tokens"]),
                      "Output tokens": _metric_value(session["completion_tokens"]),
                      "Model time": _metric_value(session["llm_latency_seconds"], "s"),
                      "Estimated OpenAI cost": _metric_value(session["estimated_cost_usd"], "USD")})
    rag = (state.get("assistant_diagnostics") or {}).get("rag")
    if rag:
        st.write("Latest Copilot source retrieval")
        searches = rag.get("queries", [])
        st.write({"Retrieval enabled": rag["enabled"], "Searches": len(searches),
                  "Chunks retrieved": sum(row.get("result_count") or 0 for row in searches),
                  "Retrieval time (s)": sum(row.get("latency_seconds") or 0 for row in searches),
                  "Unavailable searches": sum(bool(row.get("failed")) for row in searches),
                  "Methods": sorted({row["mode"] for row in searches if row.get("mode")})})
    render_benchmark_performance(thread_key)


def render_benchmark_performance(thread_key):
    from scripts.evaluate_agent import ARCHITECTURE, RESULTS_DIR

    st.write("Synthetic benchmark reports")
    paths = {}
    try:
        folder = RESULTS_DIR.resolve()
        for path in RESULTS_DIR.glob("*.json"):
            if not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(folder):
                paths[path.stem] = path
    except OSError:
        st.caption("The benchmark results folder could not be read.")
        return
    if not paths:
        st.caption("Run python -m scripts.evaluate_agent to create a local report.")
        return
    selected = st.selectbox("Saved benchmark", sorted(paths, reverse=True), index=None, key=f"benchmark_report_{thread_key}")
    if not selected:
        return
    try:
        path = paths[selected]
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Report too large")
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata, summary = data["metadata"], data["summary"]
        if metadata.get("architecture") != ARCHITECTURE or not isinstance(summary, dict):
            raise ValueError("Unrecognized benchmark export")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        st.warning("This benchmark report could not be read.")
        return
    mode = metadata.get("execution_mode", "unrecorded mode")
    st.caption(f"{mode} · {metadata.get('provider') or 'Provider unspecified'} · {metadata.get('model') or 'Model unspecified'}")
    fields = {"task_success_rate": "Task success rate", "plan_validation_rate": "Plan validation rate",
              "reflection_recovery_rate": "Reflection recovery rate", "mean_llm_calls": "Mean model requests",
              "mean_end_to_end_latency_seconds": "Mean end-to-end time (s)", "estimated_cost_usd": "Estimated OpenAI cost (USD)",
              "rag_queries": "Source searches", "retrieval_precision_at_k": "Precision@k",
              "retrieval_recall_at_k": "Recall@k", "retrieval_hit_rate_at_k": "Hit rate@k",
              "retrieval_mrr_at_k": "Reciprocal rank", "citation_accuracy": "Citation accuracy"}
    rows = ["| Metric | Value |", "| --- | --- |"]
    for key, label in fields.items():
        value = summary.get(key)
        unit = "USD" if key == "estimated_cost_usd" else ""
        rows.append(f"| {label} | {_metric_value(value if type(value) in (int, float, bool) else None, unit)} |")
    st.markdown("\n".join(rows))


def render_rule_diagnostics(state):
    from src.agents.planner import actionable_needs
    from src.core.procedure import STAGE_RULE_IDS
    from src.rag.ingestion import source_catalogue

    requirements = display_records(state.get("relevant_requirements"))
    counts = requirement_summary(requirements)
    st.write(f"Active rules: {len(load_rules())} · Passed: {counts['PASS']} · Failed: {counts['FAIL']}"
             f" · Missing information: {counts['missing_information']}")
    stage = field(state.get("procedure"), "current_stage")
    ids = STAGE_RULE_IDS.get(stage)
    current = [row for row in requirements if ids is None or row.get("rule_id") in ids]
    source_ids = sorted({sid for row in current for sid in row.get("source_ids", [])})
    if source_ids:
        catalogue = source_catalogue()
        for source_id in source_ids:
            source = catalogue.get(source_id, {})
            if source.get("url"):
                st.markdown(f"[{source['title']}]({source['url']})")
            elif source.get("title"):
                st.caption(source["title"])
    needs = actionable_needs(state)
    st.json(jsonable({
        "requirements": requirements, "unknown_by_scope": unknown_requirement_groups(state, needs=needs),
        "priorities": needs,
    }))


def render_source_monitor(snapshot, graph, config, thread_key):
    from src.rag.ingestion import source_catalogue

    catalogue = {key: row for key, row in source_catalogue().items()
                 if row.get("authority_level") in {"government_primary", "sabanci_operational"}}
    st.caption("Compare an official-source snapshot for human review. The active rules stay in effect.")
    with st.form(f"source_review_{thread_key}"):
        source_id = st.selectbox("Allow-listed source", list(catalogue), format_func=lambda sid: catalogue[sid]["title"])
        content = st.text_area("New source snapshot text", max_chars=20000)
        submit = st.form_submit_button("Compare snapshot", disabled=bool(snapshot.next) or not catalogue)
    if submit and content.strip():
        if dispatch_tool(graph, config, "compare_official_snapshot",
                         {"source_id": source_id, "source_url": catalogue[source_id]["url"], "content": content}):
            st.rerun()
    for item in snapshot.values.get("tool_outcomes", [])[-10:]:
        if item.get("name") == "compare_official_snapshot":
            st.json(item.get("data", {}))
    st.caption("After human review, update the reviewed knowledge and its version, then run python -m pytest -q.")


def render_sandbox_technical(snapshot,graph,config,thread_key):
    labels={item.field_id:item.label for item in EIKAMET_FIELD_CATALOGUE}
    selected=st.selectbox('Sandbox field ID',list(labels),format_func=lambda key:key+' - '+labels[key],key=f'debug_field_{thread_key}',disabled=bool(snapshot.next))
    for name,label,args in (('simulate_fill_eikamet_field','Fill selected field',{'field_id':selected}),('read_eikamet_sandbox','Read sandbox state',{}),('simulate_application_validation','Validate simulated application',{}),('simulate_submission','Simulate submission',{})):
        if st.button(label,key=f'debug_{thread_key}_{name}',disabled=bool(snapshot.next)):
            if dispatch_tool(graph,config,name,args):st.rerun()
