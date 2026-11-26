"""Guided Streamlit workspace over the existing procedure graph and checkpointer."""
from src.ui.components import *
from src.ui.documents import *
from src.ui.actions import *
from src.ui.home import *
from src.ui.application import *
from src.ui.technical import *
from src.ui.model_settings import render_model_settings
from src.ui.layout import render_case_header, render_stage_rail, render_student_navigation, render_theme
from src.memory.checkpointer import initialize_procedure_thread, reset_thread, sqlite_checkpointer
from src.core.graph import build_graph

NAVIGATION = ("next", "copilot", "documents", "details", "checks", "application", "technical")


def render_state(snapshot, graph, config, thread_key, thread_id=None, *, planning=False):
    state = {**snapshot.values, "planning_scope": MILESTONE_SCOPE}
    summary = workflow_summary(state)
    view_key = f"view_{thread_key}"
    requested = st.session_state.pop(f"requested_view_{thread_key}", None)
    current_step_view = "documents" if summary["number"] == 1 else "application"
    view = st.session_state.get(view_key, "next")
    if requested == "current_step":
        view = current_step_view
    elif requested in NAVIGATION:
        view = requested
    if view not in NAVIGATION:
        view = "next"
    if view == "application" and current_step_view == "documents":
        existing_form_path = (summary["stage"] == "DOCUMENT_COLLECTION"
                              and field(state.get("procedure_entry_intent"), "mode") == "EIKAMET_COMPLETED")
        if not existing_form_path:
            view = "documents"
    st.session_state[view_key] = view

    render_stage_rail(state, summary)
    render_student_navigation(view, thread_key)
    render_case_header(summary, compact=view != "next")
    if planning:
        wait_for_run(config["configurable"]["thread_id"])
        return
    render_decision(state, snapshot, graph, config, thread_key, technical=view == "technical")
    if state.get("run_status") == "INVALID_INPUT":
        issue = next(iter(state.get("validation_errors") or []), None)
        message = str(field(issue, "message", ""))
        if message.startswith("Outside Sabanc"):
            st.warning("This workflow supports adult Sabancı exchange students preparing a first short-term permit.")
        else:
            st.warning("The last change needs review. Check your answers or file and try again.")

    if view == "copilot":
        render_copilot(snapshot, graph, config, thread_key)
    elif view == "documents":
        view_documents(state, snapshot, graph, config, thread_key, summary=summary)
    elif view == "details":
        render_details_workspace(state, snapshot, graph, config, thread_key, summary=summary)
    elif view == "checks":
        view_checks(state, snapshot, graph, config, thread_key, summary=summary)
    elif view == "application":
        render_application(snapshot, graph, config, thread_key, summary=summary)
    elif view == "technical":
        view_technical(state, snapshot, graph, config, thread_key)
    else:
        view_next_steps(state, snapshot, graph, config, thread_key, summary=summary)
        if not needs_onboarding(state) or state.get("email_drafts"):
            render_optional_tools(state, snapshot, graph, config, thread_key)


def leave_profile():
    st.session_state.pop("workspace_profile_id", None)
    st.session_state.pop("procedure_help", None)


def render_case_settings(thread_id, thread_key):
    if not st.sidebar.toggle("Case settings", key=f"case_settings_{thread_key}"):
        return
    with st.sidebar:
        render_model_settings(thread_key)
        st.button("Advanced diagnostics", key=f"advanced_{thread_key}", on_click=request_view,
                  args=(thread_key, "technical"))
        with st.expander("Reset this profile", icon=":material/delete:"):
            st.caption("Deletes this profile's saved progress and record. Uploaded files and other profiles are kept.")
            typed = st.text_input("Type the profile name to confirm", key=f"reset_typed_{thread_key}")
            if st.button("Reset", key=f"reset_{thread_key}",
                         disabled=typed != thread_id or profile_lock(thread_id).locked()):
                lock = profile_lock(thread_id)
                if not lock.acquire(blocking=False):
                    st.info("Wait for processing to finish before resetting.")
                    return
                try:
                    reset_thread(thread_id, typed)
                finally:
                    lock.release()
                for key in list(st.session_state):
                    if thread_key in key and not key.startswith("reset_"):
                        del st.session_state[key]
                st.rerun()


def main():
    st.set_page_config(page_title="Residence Permit Assistant", page_icon=":material/badge:", layout="wide")
    render_theme()
    thread_id = st.session_state.get("workspace_profile_id")
    if not thread_id:
        try:
            render_landing()
        except Exception as error:
            show_error(error)
        return

    thread_key = profile_key(thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    st.sidebar.markdown(":material/badge: **Residence Permit Assistant**")
    st.sidebar.text(thread_id)
    st.sidebar.button("Switch profile", key=f"switch_{thread_key}", on_click=leave_profile)
    try:
        # Keep the connection scoped to this execution. The checkpoint remains the source of case state.
        with sqlite_checkpointer() as saver:
            graph = build_graph(saver)
            lock = profile_lock(thread_id)
            was_running = lock.locked()
            snapshot = initialize_procedure_thread(graph, config)
            if lock.locked():
                if planning_message(thread_id):
                    render_state(snapshot, graph, config, thread_key, planning=True)
                else:
                    render_case_header(workflow_summary({**snapshot.values, "planning_scope": MILESTONE_SCOPE}))
                    wait_for_run(thread_id)
                return
            if was_running:
                snapshot = graph.get_state(config)
            render_plan_notice(thread_id, snapshot)
            render_state(snapshot, graph, config, thread_key)
            render_case_settings(thread_id, thread_key)
    except Exception as error:
        show_error(error, thread_id=thread_id)
        st.caption("Your progress is stored locally. Open the case again to retry.")


if __name__ == "__main__":
    main()
