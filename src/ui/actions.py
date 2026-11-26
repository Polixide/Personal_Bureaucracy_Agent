"""Neda cards and approvals, guided actions and visible user-controlled artifacts."""
from src.ui.components import *


def propose_reminder(graph,config,row):
    from datetime import time
    from zoneinfo import ZoneInfo
    from config import LOCAL_TIMEZONE
    day=date.fromisoformat(row['date'])
    return dispatch_tool(graph,config,'create_calendar_reminder',{'title':row['title'],'start_at':datetime.combine(day,time(9),tzinfo=ZoneInfo(LOCAL_TIMEZONE)).isoformat(),'description':row['kind'],'source_candidate_id':row.get('candidate_id'),'delivery':'calendar_app'})


def _draft_card(draft):
    st.write('**To:**',field(draft,'recipient') or 'Confirm a recipient')
    st.write('**Subject:**',field(draft,'subject',''))
    st.text(str(field(draft,'body','')))
    names=[document_label(field(item,'document_type')) for item in field(draft,'attachments',[]) or []]
    if names:st.caption('Selected attachments: '+', '.join(names))


def render_tool_approval(snapshot,graph,config,thread_key,technical=False):
    if not pending_tool(snapshot):return
    preview=tool_preview(snapshot.values);name,data,args=preview['name'],preview['data'],preview['arguments']
    st.subheader('Your approval is required')
    with st.container(border=True):
        if name=='create_calendar_reminder':
            st.write('**Prepare this calendar reminder**')
            st.write('Title:',data.get('title',args.get('title','')))
            st.write('Date:',data.get('start_at',args.get('start_at','')))
            st.write('Note:',data.get('description',args.get('description','')))
            st.caption('After approval: save a calendar link and calendar file in your case. Confirm the event in your calendar.')
        elif name in {'prepare_iro_email','send_iro_email'}:
            st.write('**Prepare this email draft**' if name=='prepare_iro_email' else '**Review the previously proposed send action**')
            draft=next((item for item in record_items(snapshot.values.get('email_drafts')) if field(item,'draft_id')==args.get('draft_id')),data)
            _draft_card(draft)
            st.caption('After approval: save the draft, then open it in your email app or download it. You remain responsible for pressing Send.' if name=='prepare_iro_email' else 'This is a previous advanced send proposal. Reject it to return to the email-app workflow, or inspect it in Advanced.')
        elif name=='generate_simulated_application':
            st.write('**Generate a simulated application PDF**')
            st.warning('Educational sandbox artifact. NOT AN OFFICIAL GOVERNMENT FORM. NOT SUBMITTED.')
            for label,value in data.get('prepared_values',{}).items():st.write(label+': '+user_value(label,value))
            st.caption('After approval: save this educational PDF. Real online completion is still required externally.')
        elif name=='build_checklist':
            st.write('**Prepare your current-step checklist**');st.markdown(data.get('content',''))
            st.caption('After approval: create the checklist download. This does not submit any documents.')
        else:st.write('Review the proposed action; the full preview is available in Advanced.')
    approve,reject=st.columns(2)
    if approve.button('Approve this action',type='primary',key=f"tool_approve_{thread_key}_{preview['proposal_id']}",disabled=name=='send_iro_email' and not technical):
        if invoke(graph,Command(resume=True),config):st.rerun()
    if reject.button('Reject this action',key=f"tool_reject_{thread_key}_{preview['proposal_id']}"):
        if invoke(graph,Command(resume=False),config):st.rerun()


def render_calendar_result(item,thread_key):
    from src.tools.calendar import google_calendar_link
    try:link=google_calendar_link(display_record(item))
    except (ValueError,TypeError,KeyError):st.info('This saved reminder needs a fresh preview before export.');return
    with st.container(border=True):
        st.success('Reminder ready');st.write('**Title:**',field(item,'title',''));st.write('**Date:**',field(item,'start_at',''))
        left,right=st.columns(2);left.link_button('Open in Google Calendar',link)
        if field(item,'ics'):right.download_button('Download .ics',data=field(item,'ics'),file_name='residence-permit-reminder.ics',mime='text/calendar',key=f"ics_{thread_key}_{field(item,'reminder_id')}",on_click='ignore')
        st.caption('An earlier configured-calendar action was recorded; this link opens an additional event draft.' if field(item,'external_id') else 'The event has not been added until you confirm it in your calendar.')
        st.caption('Saved in your case: this reminder and its approved details.')


def render_email_result(draft,state,thread_key,*,package=None):
    if field(draft, "purpose") == "package_review":
        from src.tools.email import package_attachments_current
        if package is None:
            from src.core.procedure import iro_package_readiness
            package = iro_package_readiness(state)
        selected = (field(item, "document_id") for item in field(draft, "attachments", []))
        if not package_attachments_current(package, selected):
            st.warning("The saved package-review draft needs a fresh review because the package or its address evidence changed. Its history is in Advanced.")
            return False
    from src.tools.email import email_app_link,email_file,email_text,gmail_compose_link
    with st.container(border=True):
        st.success('Email draft ready');_draft_card(draft)
        gmail,left,middle,right=st.columns(4)
        try:
            gmail.link_button('Open Gmail compose',gmail_compose_link(display_record(draft)))
            left.link_button('Open default email app',email_app_link(display_record(draft)))
        except (ValueError,TypeError,KeyError):left.caption('Confirm a recipient before opening your email app.')
        try:middle.download_button('Download .eml',data=email_file(display_record(draft)),file_name='iro-review-draft.eml',mime='message/rfc822',key=f"eml_{thread_key}_{field(draft,'draft_id')}",on_click='ignore')
        except (ValueError,OSError,TypeError,KeyError):st.warning('A selected attachment changed or is unavailable. Prepare a fresh draft to export it.')
        with right.popover('Copy text'):
            st.caption('Use the copy control on this text box.');st.code(email_text(display_record(draft)),language=None,wrap_lines=True)
        st.caption('Nothing is sent by opening this draft. Review it and press Send in your own email app.')
        if field(draft,'attachments'):st.caption('Gmail and default-email-app links do not attach files automatically. Attach the files manually, or open the downloaded draft with its checked attachments.')
        st.caption('Saved in your case: the email draft and selected attachment references.')
    return True


def latest_email_draft(state, purpose):
    drafts = record_items(state.get("email_drafts"))
    return next((draft for draft in reversed(drafts)
                 if (field(draft, "purpose") or "support") == purpose), None)


def render_email_form(state, snapshot, graph, config, thread_key, *, purpose="support", package=None):
    disabled = bool(snapshot.next)
    available = [doc for doc in state.get("documents") or [] if field(doc, "status") == "analyzed"
                 and (package is None or field(doc, "doc_id") in package["attachment_ids"])]
    labels = {}
    for doc in available:
        label = document_label(field(doc, "doc_type"))
        name = field(doc, "original_filename")
        labels[field(doc, "doc_id")] = f"{label} · {name}" if name else label
    with st.form(f"email_prepare_{purpose}_{thread_key}"):
        st.caption("Choose the attachments for your draft. You’ll review the email before approving it.")
        recipient = st.text_input("IRO recipient address (optional if configured)", disabled=disabled)
        subject = st.text_input("Subject", value="Residence permit preparation review request", disabled=disabled)
        attachments = st.multiselect("Documents to attach", list(labels), default=list(labels) if package else [],
                                     format_func=labels.get, disabled=disabled)
        prepared = st.form_submit_button("Prepare email draft", disabled=disabled)
    if prepared:
        args = {"subject": subject, "attachment_ids": attachments, "purpose": purpose}
        if recipient.strip():
            args["recipient"] = recipient.strip()
        if dispatch_tool(graph, config, "prepare_iro_email", args):
            st.rerun()


def render_reminders(snapshot, graph, config, thread_key):
    from datetime import time
    from zoneinfo import ZoneInfo
    from config import LOCAL_TIMEZONE
    from src.tools.calendar import declared_reminder_candidates

    st.subheader("Reminders")
    for row in declared_reminder_candidates(snapshot.values):
        day = date.fromisoformat(row["date"]).strftime("%d %B %Y")
        candidate = row["candidate_id"]
        label = {
            "declared_legal_stay": "Add reminder",
            "declared_preparation_target": "Add recommended preparation reminder",
            "declared_insurance_expiry": "Add insurance-expiry reminder",
        }[candidate]
        if candidate == "declared_preparation_target":
            st.caption(f"Suggested preparation target · {day}. This date is a planning suggestion.")
        elif candidate == "declared_insurance_expiry":
            st.caption(f"Insurance expiry · {day}")
        if st.button(label, key=f"reminder_{thread_key}_{candidate}", disabled=bool(snapshot.next)):
            if propose_reminder(graph, config, row):
                st.rerun()

    for reminder in record_items(snapshot.values.get("calendar_reminders"))[-2:]:
        render_calendar_result(reminder, thread_key)
    if not st.toggle("Prepare a personal reminder", key=f"custom_reminder_{thread_key}"):
        return
    disabled = bool(snapshot.next)
    with st.form(f"calendar_proposal_{thread_key}"):
        title = st.text_input("Reminder title", value="Complete residence-permit preparation", disabled=disabled)
        day = st.date_input("Reminder date", value=None, disabled=disabled)
        at = st.time_input("Reminder time", value=time(9), disabled=disabled)
        description = st.text_input("Reminder note", value="Personal preparation reminder. This date is for my own planning.",
                                    disabled=disabled)
        proposed = st.form_submit_button("Preview reminder", disabled=disabled)
    if proposed and day is None:
        st.warning("Choose a date before preparing the reminder.")
    elif proposed:
        args = {"title": title, "start_at": datetime.combine(day, at, tzinfo=ZoneInfo(LOCAL_TIMEZONE)).isoformat(),
                "duration_minutes": 30, "description": description, "delivery": "calendar_app"}
        if dispatch_tool(graph, config, "create_calendar_reminder", args):
            st.rerun()


def open_optional_tools(thread_key, option):
    st.session_state[f"optional_tool_{thread_key}"] = option
    request_view(thread_key, "next")


def render_optional_tools(state, snapshot, graph, config, thread_key):
    options = ["Choose an option", "Ask IRO a question", "Export a checklist"]
    package_draft = None
    if field(state.get("procedure"), "current_stage") != "IRO_REVIEW":
        package_draft = latest_email_draft(state, "package_review")
        if package_draft:
            options.append("Earlier IRO package email")
    choice = st.selectbox("Optional tools", options, key=f"optional_tool_{thread_key}")
    if choice == "Ask IRO a question":
        draft = latest_email_draft(state, "support")
        if draft:
            render_email_result(draft, state, thread_key)
        if not draft or st.toggle("Prepare a new support draft", key=f"new_support_draft_{thread_key}"):
            render_email_form(state, snapshot, graph, config, thread_key)
    elif choice == "Earlier IRO package email":
        render_email_result(package_draft, state, thread_key)
    elif choice == "Export a checklist":
        if st.button("Build checklist", key=f"checklist_{thread_key}", disabled=bool(snapshot.next)):
            if dispatch_tool(graph, config, "build_checklist", {}):
                st.rerun()
        seen = set()
        for item in reversed(state.get("tool_outcomes", [])[-8:]):
            call_id = field(item, "call_id")
            if field(item, "name") == "build_checklist" and field(item, "status") == "SUCCEEDED" and call_id not in seen:
                seen.add(call_id)
                st.download_button("Download checklist", data=field(item, "data", {}).get("content", ""),
                                   file_name="preparation-checklist.md", key=f"download_checklist_{thread_key}_{call_id}",
                                   on_click="ignore")


def perform_primary_action(action, graph, config, thread_key):
    from src.core.procedure import start_guided_preparation, continue_after_application_review, advance_current_stage

    advance = {
        "start_application": start_guided_preparation,
        "review_application": continue_after_application_review,
        "advance": advance_current_stage,
    }[action["kind"]]

    def transition(latest):
        updates = advance(latest, recorded_at=utc_now())
        current = field(latest.get("procedure"), "current_stage")
        destination = field(updates.get("procedure"), "current_stage")
        return updates if destination and destination != current else {}

    if persist_procedure_choice(graph, config, transition):
        request_view(thread_key, "application")
    st.rerun()


def view_next_steps(state, snapshot, graph, config, thread_key, *, summary=None):
    requested = st.session_state.get(f"show_plan_{thread_key}", False)
    just_requested = st.session_state.pop(f"open_plan_result_{thread_key}", False)
    if needs_onboarding(state):
        from src.ui.home import render_onboarding

        render_onboarding(snapshot, graph, config, thread_key)
        if not snapshot.next and (requested or just_requested):
            render_plan_cards(state, snapshot, graph, config, thread_key)
        return
    if summary is None:
        summary = workflow_summary(state)
    conflicts = conflict_messages(state)
    action = primary_action(state, checklist=summary["items"], conflicts=conflicts, collection=summary["collection"])
    if action:
        with st.container(border=True, key="next_task"):
            st.subheader("Your next step")
            if conflicts:
                st.warning(action["message"])
            else:
                st.write(action["message"])
            if action["kind"] == "navigate":
                st.button(action["label"], type="primary", disabled=bool(snapshot.next), key=f"primary_{thread_key}",
                          on_click=request_view, args=(thread_key, action["view"], action.get("document_id")),
                          kwargs={"document_type": action.get("document_type"), "detail_section": action.get("detail_section")})
            elif st.button(action["label"], type="primary", disabled=bool(snapshot.next), key=f"primary_{thread_key}"):
                perform_primary_action(action, graph, config, thread_key)

    from src.core.legal_stay import insurance_explanation

    warning = insurance_explanation(state)
    if warning:
        st.warning(warning)
    stage = field(state.get("procedure"), "current_stage")
    plan_stage = state.get("workflow_stage")
    current = not (stage and plan_stage and stage != plan_stage)
    saved_plan = current and (state.get("approved_plan") or field(state.get("event"), "kind") == "generate_plan")
    plan_relevant = stage != "COMPLETE" and (not action or action["kind"] == "navigate")
    if not interaction_state(snapshot)["approval_pending"] and (
        requested or just_requested or saved_plan or plan_relevant
    ):
        render_plan_cards(state, snapshot, graph, config, thread_key, current=current)


def render_plan_cards(state, snapshot, graph, config, thread_key, *, current=True):
    st.subheader("Detailed action plan")
    status = state.get("run_status", "NOT_STARTED") if current else "NOT_STARTED"
    busy = bool(snapshot.next)
    approved = state.get("approved_plan") if current else None
    if status == "PLANNING_UNAVAILABLE":
        st.warning("The planning assistant is unavailable. Your information is saved. You can retry or follow the next step above.")
    elif status in {"HUMAN_REVIEW_REQUIRED", "PLAN_INVALID"}:
        st.warning("The assistant could not prepare a plan that passed the checks. Review your saved information before retrying.")

    steps = field(approved, "actions", []) if approved else None
    origin = state.get("approved_plan_origin") or state.get("plan_origin")
    empty_result = (
        current and origin in {"stage_complete", "milestone_completion"}
        and state.get("candidate_plan") == [] and not state.get("validation_errors")
    )
    if empty_result or (approved and origin == "milestone_completion"):
        st.info(user_text(field(approved, "summary") or state.get("plan_summary") or "There are no additional actions to plan."))
    elif steps is None and status not in {"PLANNING_UNAVAILABLE", "HUMAN_REVIEW_REQUIRED", "PLAN_INVALID", "APPROVAL_DENIED"}:
        st.caption("Generate a plan for the current step using your saved documents and answers.")
    for step in display_records(steps):
        with st.container(border=True):
            description, progress = st.columns([5, 2])
            description.markdown(f"**{user_text(user_action(step))}**")
            if step.get("reason"):
                description.caption(user_text(step["reason"]))
            if step.get("due_date"):
                description.caption(f":material/event: Due {step['due_date']}")
            call = step.get("tool_call")
            if call:
                done = any(row.get("call_id") == call["call_id"] and row.get("status") == "SUCCEEDED"
                           for row in state.get("tool_outcomes", []))
                if done:
                    progress.caption("Outcome recorded")
                elif progress.button("Run approved action", disabled=busy,
                                     key=f"run_plan_{thread_key}_{call['call_id']}"):
                    if dispatch_tool_call(graph, config, call):
                        st.rerun()
            else:
                reported = any(row.get("action_id") == step.get("step_id") and row.get("action") == step.get("action")
                               for row in state.get("action_progress", {}).values())
                if reported:
                    progress.caption("Reported done")
                elif progress.button("Mark done", disabled=busy, key=f"done_{thread_key}_{step.get('step_id')}",
                                     help="Records your progress. Document and rule checks still use the available evidence."):
                    run_event(graph, config, UserEvent(kind="record_progress", action_id=step.get("step_id")))
                if not reported:
                    progress.caption("To do")
                    kind = step.get("request_kind")
                    if kind in {"REQUEST_DOCUMENT", "REVIEW_EXTRACTED_EVIDENCE", "RESOLVE_CONFLICT"}:
                        from src.workers.verification import PROFILE_TYPES
                        requested = step.get("requested_documents") or []
                        targets = {profile: doc_type for doc_type, profile in PROFILE_TYPES.items()}
                        target = targets.get(requested[0]) if len(requested) == 1 else None
                        description.button("Open Documents", key=f"plan_documents_{thread_key}_{step['step_id']}",
                                           on_click=request_view, args=(thread_key, "documents"),
                                           kwargs={"document_type": target})
    if steps is None and not empty_result:
        from src.ui.model_settings import describe_provider

        st.caption("Planning model · " + describe_provider(thread_key))
        if st.button("Generate plan", disabled=busy, key=f"plan_{thread_key}_{checkpoint_id(snapshot)}"):
            if start_detailed_plan(snapshot, config):
                st.session_state.pop(f"show_plan_{thread_key}", None)
                st.rerun()


def render_decision(state, snapshot, graph, config, thread_key, technical=False):
    """A pending approval or unfinished run comes first, before any view."""
    interaction = interaction_state(snapshot)
    if interaction["approval_pending"]:
        steps = state.get("candidate_plan")
        can_approve = not state.get("validation_errors") and (
            bool(steps) or (steps == [] and state.get("plan_origin") == "milestone_completion")
        )
        with st.container(border=True):
            st.markdown("#### :material/gavel: A plan is waiting for your decision")
            if steps:
                for index, step in enumerate(display_records(steps), 1):
                    st.markdown(f"**{index}.** {user_text(user_action(step))}")
                    if step.get("reason"):
                        st.caption(user_text(step["reason"]))
            elif can_approve:
                st.write(user_text(state.get("plan_summary", "")))
            else:
                st.warning("The saved proposal is incomplete. Deny it before generating another plan.")
            if state.get("plan_origin") == "deterministic_fallback":
                diagnostics = state.get("planner_diagnostics") or {}
                if diagnostics.get("provider") == "openai":
                    st.warning(ERROR_MESSAGES.get(diagnostics.get("provider_error"), "OpenAI could not prepare a plan."))
                    st.info("These proposed steps come from the saved information and checks.")
                else:
                    st.info("The local model was unavailable. These proposed steps come from the saved information and checks.")
            st.caption("Approving records these next steps only. Nothing is submitted, paid or "
                       "sent on your behalf. Deny the plan to edit your information.")
            approve, deny, _ = st.columns([1, 1, 3])
            decision = None
            if approve.button("Approve", type="primary", disabled=not can_approve,
                              key=f"approve_{thread_key}_{state.get('trace_id')}"):
                decision = True
            if deny.button("Deny", key=f"deny_{thread_key}_{state.get('trace_id')}"):
                decision = False
            if decision is not None and invoke(graph, Command(resume=decision), config):
                st.rerun()
    elif pending_tool(snapshot):
        render_tool_approval(snapshot,graph,config,thread_key,technical=technical)
    elif snapshot.next:
        with st.container(border=True):
            st.warning("Processing stopped before it finished. Retry to continue; editing is locked until then.")
            planning = field(state.get("event"), "kind") == "generate_plan"
            if planning:
                from src.ui.model_settings import describe_provider

                st.caption("The retry will use " + describe_provider(thread_key) + ".")
            if st.button("Retry unfinished processing", key=f"retry_{thread_key}"):
                started = start_detailed_plan(snapshot, config, resume=True) if planning else invoke(graph, None, config)
                if started:
                    st.rerun()
    else:
        outcome = state.get("last_tool_outcome") or {}
        if state.get("run_status") == "APPROVAL_DENIED":
            st.info("You declined the plan. You can update your information and prepare another plan.")
        elif field(outcome, "status") == "REJECTED":
            st.info("You declined the proposed action.")
        elif field(outcome, "status") == "FAILED":
            st.warning("The last action could not be prepared. Review the inputs and try again.")
