"""Focused Streamlit presentation; all decisions use canonical backend functions."""
from src.ui.components import *


def render_onboarding(snapshot,graph,config,thread_key):
    with st.container(border=True):
        st.subheader('Where are you in the process?')
        st.caption('First short-term residence-permit preparation for Sabanci incoming exchange students. Your choice changes the starting view; evidence determines actual progress.')
        choices=(('START_FROM_SCRATCH',"I'm starting from the beginning",'documents'),('HAVE_DOCUMENTS','I already have some documents','documents'),('EIKAMET_COMPLETED','I already completed the online e-Ikamet application','application'))
        for mode,label,view in choices:
            if st.button(label,type='primary' if mode=='START_FROM_SCRATCH' else 'secondary',key=f'welcome_{mode}_{thread_key}'):
                if save_entry_intent(graph,config,mode,recorded_at=datetime.now(timezone.utc)):
                    if mode!='START_FROM_SCRATCH':request_view(thread_key,view)
                    st.rerun()


def _source_links(message):
    sources=field(message,"source_refs",[]) or []
    if not sources:return
    with st.expander("Sources used",expanded=False):
        for source in sources:
            title=field(source,"title",field(source,"source_id","Official guidance"))
            url=field(source,"url",field(source,"source_url"))
            if url and str(url).startswith(("https://","http://")):
                st.markdown(f"[{title}]({url})")
            else:st.write(title)
            kind=field(source,"content_kind")
            if kind=="project_rule_excerpt":st.caption("Project rule excerpt linked to its official reference; this is not a live legal update.")
            elif kind=="official_guidance":st.caption("Official guidance snapshot")
            elif kind=="project_operational_policy":st.caption("Team-supplied direct IRO instruction for this project; not a universal national-law source.")
            known_date=field(source,"verified_at") or field(source,"retrieved_at")
            if known_date:st.caption("Source snapshot date: "+str(known_date))
            excerpt=field(source,"excerpt",field(source,"text"))
            if excerpt:st.caption(str(excerpt)[:700])




def render_chat(snapshot,graph,config,thread_key,*,copilot=False):
    state=snapshot.values
    st.subheader("AI Copilot" if copilot else "What can I help you with?")
    st.caption("Ask about your case, missing documents, official guidance, reminders or an IRO email. Answers use your saved state and cited sources.")
    messages=state.get("assistant_messages",[]) or []
    if not messages:
        with st.chat_message("assistant"):
            st.write("I can explain what is missing, help review your preparation, and propose safe actions. For example: What do I need to start?")
    for message in messages[-16:]:
        role=field(message,"role","assistant")
        if role not in {"user","assistant"}:continue
        with st.chat_message(role):
            st.write(user_text(field(message,"content","")))
            _source_links(message)
    if snapshot.next:
        st.info("A decision or unfinished task needs attention. Use Next Actions before starting another conversation turn.")
    text=st.chat_input("Ask about your residence-permit preparation",disabled=bool(snapshot.next),key=f"chat_{thread_key}")
    if text and invoke(graph,ui_run(UserEvent(kind="assistant_message",text=text),config["configurable"]["thread_id"]),config):
        st.rerun()



def render_landing(graph):
    """Choose the existing thread/procedure; unsupported entries create no state."""
    from src.agents.assistant import PROCEDURE_CATALOG,recommend_procedure
    from src.core.procedure import user_step
    st.title('Residence Permit Assistant')
    profile=st.text_input('Profile ID',key='landing_profile_id',help='Use the same profile ID to restore your saved case.').strip()
    config={'configurable':{'thread_id':profile}}
    saved=graph.get_state(config) if profile else None
    if saved is not None and saved.values:
        st.success('Saved case found.')
        st.write('Procedure: First residence permit / Ikamet')
        step=user_step(saved.values);st.write('Current step: '+step['label'])
        if st.button('Resume case',type='primary',disabled=not profile):
            st.session_state['workspace_profile_id']=profile;st.rerun()
    st.subheader('What do you need help with?')
    for key,item in PROCEDURE_CATALOG.items():
        with st.container(border=True):
            st.caption('AVAILABLE' if item['implemented'] else 'NOT YET IMPLEMENTED')
            if st.button(item['label'],disabled=not item['implemented'] or not profile,key='catalog_'+key):
                st.session_state['workspace_profile_id']=profile;st.rerun()
    with st.container(border=True):
        st.markdown('##### Not sure which procedure you need?')
        with st.form('procedure_recommendation'):
            description=st.text_area('Describe your situation',max_chars=2000)
            ask=st.form_submit_button('Ask the assistant',disabled=not profile)
        if ask:
            if not description.strip():st.info('Describe your situation first.')
            else:
                with st.spinner('Checking the supported procedure catalogue...'):
                    st.session_state['landing_recommendation']=recommend_procedure(description)
        result=st.session_state.get('landing_recommendation')
        if result:
            st.info(result['message'])
            if result['implemented'] and st.button('Start this procedure',disabled=not profile):
                st.session_state['workspace_profile_id']=profile;st.rerun()


def render_copilot(snapshot,graph,config,thread_key):
    st.caption('Uses your saved case, accepted document information and attributed guidance. It cannot approve a permit or execute an external action without your approval.')
    quick=st.columns(3)
    for column,text in zip(quick,('What should I do now?','Which documents am I missing?','Explain my blockers')):
        if column.button(text,disabled=bool(snapshot.next),key=f'copilot_{thread_key}_{text}'):
            run_event(graph,config,UserEvent(kind='assistant_message',text=text))
    render_chat(snapshot,graph,config,thread_key,copilot=True)
    if snapshot.values.get('assistant_diagnostics',{}).get('route')=='plan_request':
        if st.button('Generate a stage plan',type='primary',disabled=bool(snapshot.next),key=f'copilot_plan_{thread_key}'):
            run_event(graph,config,UserEvent(kind='generate_plan'))
    from src.ui.actions import render_results
    render_results(snapshot.values,thread_key)
