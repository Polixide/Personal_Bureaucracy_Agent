from src.ui.components import *
def view_technical(state, snapshot):
    """Everything the earlier layout showed inline, in one place."""
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import workflow_stage
    requirements = display_records(state.get("relevant_requirements"))
    st.markdown('##### Active rule diagnostics')
    counts=requirement_summary(requirements)
    st.write(f"Active rules: {len(load_rules())} | PASS: {counts['PASS']} | FAIL: {counts['FAIL']} | UNKNOWN: {counts['missing_information']}")
    with st.expander('Full canonical state and document provenance',expanded=False):
        st.json(jsonable(state))
    if st.session_state.get('_last_ui_error'):
        with st.expander('Last processing error',expanded=False):st.code(st.session_state['_last_ui_error'])
    unknown_groups = unknown_requirement_groups(state)
    origin = state.get("plan_origin") or (state.get("approved_plan_origin") if state.get("approved_plan") else None)
    st.caption(f"Workflow stage: {workflow_stage(state).replace('_', ' ').title()}"
               f" | Plan source: {({'deterministic_fallback': 'Deterministic fallback', 'milestone_completion': 'Deterministic milestone assessment'}.get(origin, 'Local Ollama model') if origin else 'none')}"
               f" | Repair attempts: {state.get('retry_count', 0)} / {MAX_REFLECTION_ATTEMPTS}")
    st.caption("Not-yet-known checks by scope: " + " | ".join(
        f"{label}: {len(unknown_groups[key])}" for key, label in
        (("immediate", "needed now"), ("future", "later stage"), ("outside_scope", "outside Milestone 1"),
         ("guardrail_only", "safety guardrails"), ("waiting", "waiting on prerequisites"), ("advisory", "advisory"))))
    with st.expander("Active rule results"):
        st.json({"unknown_by_scope": unknown_groups, "requirements": requirements})
    with st.expander("Workflow priorities and dependencies"):
        st.json(jsonable(actionable_needs(state)))
    with st.expander("Structured plan"):
        st.json(jsonable({"summary": state.get("plan_summary", ""), "candidate_plan": plan_steps(state)}))
    with st.expander("Gate 1 and reflection"):
        st.json(jsonable({"validation_errors": display_records(state.get("validation_errors")),
                          "feedback": state.get("repair_feedback", []), "history": state.get("repair_history", []),
                          "planner_diagnostics": state.get("planner_diagnostics", {}),
                          "planner_candidate_set": state.get("planning_context_snapshot", {}).get("canonical_candidates", []),
                          "copilot_diagnostics": state.get("assistant_diagnostics", {}),
                          "action_progress": state.get("action_progress", {})}))
    with st.expander("Digital Twin"):
        st.json(jsonable({"extracted_facts": state.get("extracted_facts", {}),
                          "residence_permit_context": state.get("residence_permit_context", {}),
                          "verification_records": state.get("verification_records", {}),
                          "verification_conflicts": state.get("verification_conflicts", [])}))


    with st.expander("V2 tools, sandbox and working-context metrics"):
        from src.evaluation.metrics import state_metrics
        from src.tools.registry import get_tool_specs
        st.json(jsonable({"metrics":state_metrics(state),"context":state.get("assistant_diagnostics",{}),"tool_trace":state.get("tool_outcomes",[])[-30:],"sandbox":state.get("sandbox_form",{})}))
        st.json([spec.model_dump(mode="json") for spec in get_tool_specs()])
    with st.expander("Neda persistent episodes and accepted repairs"):
        try:
            with TwinStore(__import__("src.ui.components",fromlist=["TWIN_RECORD_PATH"]).TWIN_RECORD_PATH) as store:
                st.json([run.model_dump(mode="json") for run in store.list_runs(state.get("user_id",""))][-20:])
                st.json([lesson.model_dump(mode="json") for lesson in store.find_lessons(state.get("user_id",""),state.get("active_procedure", "ikamet"))])
        except Exception as error:st.caption(str(error))


def render_technical(snapshot,graph,config,thread_key):
    view_technical(snapshot.values,snapshot)


def render_source_monitor(snapshot,graph,config,thread_key):
    from src.rag.ingestion import source_catalogue
    catalogue={key:row for key,row in source_catalogue().items() if row.get("authority_level") in {"government_primary","sabanci_operational"}}
    with st.expander("Official source-change review (local snapshot)"):
        st.caption("Compare an operator-supplied official-source snapshot. Differences become review candidates only; active rules are never overwritten.")
        with st.form(f"source_review_{thread_key}"):
            source_id=st.selectbox("Allow-listed source",list(catalogue),format_func=lambda sid:catalogue[sid]["title"])
            content=st.text_area("New source snapshot text",max_chars=20000)
            submit=st.form_submit_button("Compare snapshot",disabled=bool(snapshot.next))
        if submit and content.strip():
            if dispatch_tool(graph,config,"compare_official_snapshot",{"source_id":source_id,"source_url":catalogue[source_id]["url"],"content":content}):st.rerun()
        for item in snapshot.values.get("tool_outcomes",[])[-10:]:
            if item.get("name")=="compare_official_snapshot":st.json(item.get("data",{}))
        st.caption("After human review: explicitly edit the reviewed knowledge, increment its version and run python -m pytest -q before using that new version. Retrieval cannot perform this promotion.")



def render_sandbox_technical(snapshot,graph,config,thread_key):
    labels={item.field_id:item.label for item in EIKAMET_FIELD_CATALOGUE}
    selected=st.selectbox('Sandbox field ID',list(labels),format_func=lambda key:key+' - '+labels[key],key=f'debug_field_{thread_key}',disabled=bool(snapshot.next))
    for name,label,args in (('simulate_fill_eikamet_field','Fill selected field',{'field_id':selected}),('read_eikamet_sandbox','Read sandbox state',{}),('simulate_application_validation','Validate simulated application',{}),('simulate_submission','Simulate submission',{})):
        if st.button(label,key=f'debug_{thread_key}_{name}',disabled=bool(snapshot.next)):
            if dispatch_tool(graph,config,name,args):st.rerun()
