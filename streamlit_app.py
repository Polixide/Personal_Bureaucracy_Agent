"""Neda UI shell over the current V2 graph, source-grounded Twin and tools."""
from src.ui.components import *
from src.ui.documents import *
from src.ui.actions import *
from src.ui.home import *
from src.ui.application import *
from src.ui.technical import *
from src.memory.checkpointer import sqlite_checkpointer,reset_thread,initialize_procedure_thread
from src.core.graph import build_graph

VIEWS={"next":":material/flag: Next steps","copilot":":material/smart_toy: AI Copilot","documents":":material/description: Documents", "details":":material/person: Your details","checks":":material/fact_check: Checks","application":":material/edit_document: e-Ikamet","technical":":material/code: Technical"}
NAVIGATION=tuple(VIEWS)


def render_state(snapshot,graph,config,thread_key,thread_id=None):
    state={**snapshot.values,"planning_scope":MILESTONE_SCOPE}
    render_status_bar(state,snapshot)
    render_decision(state,snapshot,graph,config,thread_key,technical=st.session_state.get("view_"+thread_key)=="technical")
    requested=st.session_state.pop('requested_view_'+thread_key,None)
    if requested in VIEWS:st.session_state['view_'+thread_key]=requested
    from src.core.procedure import user_step
    label=lambda key:(':material/edit_document: '+('e-Ikamet' if field(state.get('procedure'),'current_stage') in {'DOCUMENT_COLLECTION','EIKAMET_PREPARATION','EIKAMET_READY'} else user_step(state)['label'])) if key=='application' else VIEWS[key]
    view_key=f"view_{thread_key}"
    initial=None if view_key in st.session_state else "next"
    view=st.segmented_control("View",list(VIEWS),format_func=label,default=initial,key=view_key,label_visibility="collapsed") or "next"
    if view=="next":
        view_next_steps(state,snapshot,graph,config,thread_key)
        if not needs_onboarding(state):
            render_recommendations(snapshot,graph,config,thread_key)
            with st.expander('Prepare a reminder or email'):
                render_actions(snapshot,graph,config,thread_key,show_approval=False,show_results=False)
    elif view=="copilot":render_copilot(snapshot,graph,config,thread_key)
    elif view=="documents":view_documents(state,snapshot,graph,config,thread_key,thread_id or config["configurable"]["thread_id"])
    elif view=="details":
        view_details(state,snapshot,graph,config,thread_key)
        render_procedure_confirmations(snapshot,graph,config,thread_key)
        render_legal_stay(snapshot,graph,config,thread_key)
        render_results(state,thread_key)
    elif view=="checks":view_checks(state,snapshot,graph,config,thread_key)
    elif view=="application":
        render_application(snapshot,graph,config,thread_key)
    else:
        view_technical(state,snapshot)
        render_source_monitor(snapshot,graph,config,thread_key)
        with st.expander('Experimental assistant',expanded=False):
            st.warning('Experimental: free-form answers may be unreliable. Use the guided workflow for preparation and decisions.')
            render_chat(snapshot,graph,config,thread_key)
        with st.expander('Experimental legal-stay calculator',expanded=False):
            render_legal_stay_technical(snapshot,graph,config,thread_key)
        with st.expander('Individual sandbox tools',expanded=False):
            render_sandbox_technical(snapshot,graph,config,thread_key)

def main():
    st.set_page_config(page_title="Residence Permit Assistant", page_icon=":material/badge:", layout="wide")
    active=st.session_state.get('workspace_profile_id')
    if not active:
        try:
            with sqlite_checkpointer() as saver:render_landing(build_graph(saver))
        except Exception as error:show_error(error)
        return
    st.sidebar.markdown("### :material/badge: Residence Permit Assistant")
    st.sidebar.caption("Sabancı incoming exchange · first short-term residence permit")
    thread_id = st.sidebar.text_input("Your profile name", value=active,
                                      help="Use the same name next time to continue where you left off.").strip()
    if not thread_id:
        st.info("Enter a profile name to begin.")
        return
    if st.sidebar.button('Choose another profile / procedure'):
        st.session_state.pop('workspace_profile_id',None);st.rerun()
    thread_key = sha256(thread_id.encode()).hexdigest()[:16]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    try:
        # A fresh connection per rerun avoids caching a closed/shared connection.
        # No canonical state is stored in Streamlit session_state.
        with sqlite_checkpointer() as saver:
            graph = build_graph(saver)
            snapshot = initialize_procedure_thread(graph,config)
            st.sidebar.caption(":green[:material/cloud_done:] Saved progress restored" if snapshot.values
                               else "New profile")
            if st.sidebar.button("Refresh", icon=":material/refresh:"):
                st.rerun()
            with st.sidebar.expander("Reset this profile", icon=":material/delete:"):
                st.caption("Deletes this profile's saved progress and record. Uploaded files and other profiles are kept.")
                typed = st.text_input("Type the profile name to confirm", key=f"reset_typed_{thread_key}")
                if st.button("Reset", type="primary", key=f"reset_{thread_key}",
                             disabled=typed != thread_id or profile_lock(thread_id).locked()):
                    lock=profile_lock(thread_id)
                    if not lock.acquire(blocking=False):
                        st.info("Still processing your last action. Wait before resetting.");return
                    try:reset_thread(thread_id, typed)
                    finally:lock.release()
                    for key in list(st.session_state):
                        if thread_key in key and not key.startswith("reset_"): del st.session_state[key]
                    st.rerun()
            st.title("Your residence permit")
            if profile_lock(thread_id).locked():
                # A run is in progress elsewhere: show where things stand, offer no actions.
                render_status_bar({**snapshot.values, "planning_scope": MILESTONE_SCOPE}, snapshot)
                wait_for_run(thread_id)
                return
            render_state(snapshot, graph, config, thread_key, thread_id)
    except Exception as error:
        show_error(error)
        st.caption("Saved progress stays on this computer. Refresh to try again.")



if __name__=="__main__":main()
