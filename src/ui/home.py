"""Focused Streamlit presentation; all decisions use canonical backend functions."""
from src.ui.components import *
from src.ui.model_settings import describe_provider, selected_provider, ui_ai_options


def render_onboarding(snapshot, graph, config, thread_key):
    st.subheader("Where are you in the process?")
    choices = (
        ("START_FROM_SCRATCH", "I'm starting from the beginning", "Find out what to prepare first.", "next"),
        ("HAVE_DOCUMENTS", "I already have some documents", "Upload the files you have and see what is still needed.", "documents"),
        ("EIKAMET_COMPLETED", "I already completed the online e-Ikamet application", "Upload the real form to review it.", "application"),
    )
    for mode, label, description, view in choices:
        with st.container(border=True):
            st.caption(description)
            if st.button(label, type="primary" if mode == "START_FROM_SCRATCH" else "secondary",
                         key=f"welcome_{mode}_{thread_key}"):
                if save_entry_intent(graph, config, mode, recorded_at=utc_now()):
                    request_view(thread_key, view)
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
            if known_date:st.caption("Source snapshot date " + str(known_date))
            location = [str(field(source, key)) for key in ("section", "page") if field(source, key)]
            if location:
                st.caption(" · ".join(location))
            excerpt=field(source,"excerpt",field(source,"text"))
            if excerpt:st.caption(str(excerpt)[:700])




def render_chat(snapshot, graph, config, thread_key, *, copilot=False, questions=()):
    messages = snapshot.values.get("assistant_messages") or []
    st.subheader("Ask about this step" if copilot else "What can I help you with?")
    st.caption("Ask why evidence is needed, what a warning means, or how to resolve a blocker.")
    if selected_provider(thread_key) == "openai":
        st.caption(describe_provider(thread_key) + " · Your question and relevant public passages may be sent to OpenAI. Leave out personal details.")
    diagnostics = snapshot.values.get("assistant_diagnostics") or {}
    if provider_error := diagnostics.get("provider_error"):
        st.warning(ERROR_MESSAGES.get(provider_error, "OpenAI could not complete the last request."))
    busy = bool(snapshot.next)
    if questions and not messages:
        columns = st.columns(len(questions))
        for index, (column, question) in enumerate(zip(columns, questions)):
            if column.button(question, disabled=busy, key=f"copilot_question_{thread_key}_{index}"):
                run_event(graph, config, UserEvent(kind="assistant_message", text=question))
    for message in messages[-16:]:
        role = field(message, "role", "assistant")
        if role in {"user", "assistant"}:
            with st.chat_message(role):
                st.markdown(user_text(field(message, "content", ""), keep_lines=True))
                _source_links(message)
    if busy:
        st.info("Finish the action above before asking another question.")
    key = f"chat_{'copilot' if copilot else 'technical'}_{thread_key}"
    text = st.chat_input("Ask about your permit preparation", disabled=busy, key=key)
    if text and text.strip():
        run_event(graph, config, UserEvent(kind="assistant_message", text=text.strip()))


def render_landing():
    from src.agents.assistant import PROCEDURE_CATALOG, recommend_procedure

    with st.container(key="welcome_hero"):
        st.caption("RESIDENCE PERMIT ASSISTANT")
        st.title("Prepare your first residence permit")
        st.write("For Sabancı incoming exchange students.")
    case, help_panel = st.columns([1.25, 1], gap="large")
    with case:
        st.subheader("Open your preparation case")
        st.caption("Choose a profile name. Use the same name to return to the progress saved on this computer.")
        with st.form("open_case"):
            profile = st.text_input("Profile name", placeholder="For example, Alex").strip()
            open_case = st.form_submit_button("Open my case", type="primary")
        if open_case:
            if profile:
                st.session_state["workspace_profile_id"] = profile
                st.rerun()
            else:
                st.warning("Enter a profile name to open your case.")
    with help_panel:
        st.subheader("What you can do here")
        st.markdown("- Check which documents you need\n- Upload evidence and fix missing details\n- Prepare a package for university review")
        if st.toggle("Help choosing a procedure"):
            options = ui_ai_options("landing")
            if options.provider == "openai":
                st.caption("OpenAI receives the description you submit. Leave out identifying details.")
            with st.form("procedure_recommendation"):
                description = st.text_area("Describe your situation", max_chars=2000)
                ask = st.form_submit_button("Ask the assistant")
            if ask and description.strip():
                with use_ai_options(options), st.spinner("Checking the supported procedures..."):
                    recommendation = recommend_procedure(description)
                st.session_state["procedure_help"] = recommendation["message"]
                st.session_state["procedure_help_error"] = recommendation.get("provider_error")
            elif ask:
                st.info("Describe your situation first.")
            if st.session_state.get("procedure_help"):
                st.info(st.session_state["procedure_help"])
                if problem := st.session_state.get("procedure_help_error"):
                    st.warning(ERROR_MESSAGES[problem])
            unavailable = [item["label"] for item in PROCEDURE_CATALOG.values() if not item["implemented"]]
            st.caption("Other procedures currently unavailable: " + ", ".join(unavailable))


def open_detailed_plan(thread_key):
    st.session_state[f"show_plan_{thread_key}"] = True
    request_view(thread_key, "next")


def render_copilot(snapshot, graph, config, thread_key):
    from src.ui.actions import open_optional_tools

    questions = ("What should I do now?", "Which documents have I uploaded?", "Is my insurance ready?")
    render_chat(snapshot, graph, config, thread_key, copilot=True, questions=questions)
    state = snapshot.values
    if state.get("assistant_diagnostics", {}).get("route") == "plan_request":
        st.button("Open detailed action plan", type="primary", disabled=bool(snapshot.next),
                  key=f"copilot_plan_{thread_key}", on_click=open_detailed_plan, args=(thread_key,))

    latest = state.get("last_tool_outcome") or {}
    if field(latest, "status") != "SUCCEEDED":
        return
    name = field(latest, "name")
    if name == "create_calendar_reminder":
        st.button("Open saved reminders", key=f"copilot_reminders_{thread_key}", on_click=request_view,
                  args=(thread_key, "details"), kwargs={"detail_section": "reminders"})
    elif name == "prepare_iro_email":
        package = field(field(latest, "data", {}), "purpose") == "package_review"
        if package and field(state.get("procedure"), "current_stage") == "IRO_REVIEW":
            st.button("Open IRO package draft", key=f"copilot_iro_{thread_key}", on_click=request_view,
                      args=(thread_key, "application"))
        else:
            option = "Earlier IRO package email" if package else "Ask IRO a question"
            st.button("Open saved email draft", key=f"copilot_draft_{thread_key}", on_click=open_optional_tools,
                      args=(thread_key, option))
    elif name == "search_official_sources" and field(state.get("event"), "kind") == "run_tool":
        _source_links({"source_refs": field(latest, "data", {}).get("results", [])[:3]})
