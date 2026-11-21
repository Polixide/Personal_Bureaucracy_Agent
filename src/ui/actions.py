"""Neda cards and approvals, guided actions and visible user-controlled artifacts."""
from src.ui.components import *
from src.ui.documents import render_checklist


def propose_reminder(graph,config,row):
    from datetime import time
    from zoneinfo import ZoneInfo
    from config import LOCAL_TIMEZONE
    day=date.fromisoformat(row['date'])
    return dispatch_tool(graph,config,'create_calendar_reminder',{'title':row['title'],'start_at':datetime.combine(day,time(9),tzinfo=ZoneInfo(LOCAL_TIMEZONE)).isoformat(),'description':row['kind'],'source_candidate_id':row.get('candidate_id'),'delivery':'calendar_app'})


def _draft_card(draft,state):
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
            _draft_card(draft,snapshot.values)
            st.caption('After approval: save the draft, then open it in your email app or download it. You remain responsible for pressing Send.' if name=='prepare_iro_email' else 'This is a previous advanced send proposal. Reject it to return to the email-app workflow, or inspect it in Technical.')
        elif name=='generate_simulated_application':
            st.write('**Generate a simulated application PDF**')
            st.warning('Educational sandbox artifact. NOT AN OFFICIAL GOVERNMENT FORM. NOT SUBMITTED.')
            for label,value in data.get('prepared_values',{}).items():st.write(label+': '+user_value(label,value))
            st.caption('After approval: save this educational PDF. Real online completion is still required externally.')
        elif name=='build_checklist':
            st.write('**Prepare your current-step checklist**');st.markdown(data.get('content',''))
            st.caption('After approval: create the checklist download. This does not submit any documents.')
        else:st.write('Review the proposed action; the exact state diff is retained in Technical.')
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
        if field(item,'ics'):right.download_button('Download .ics',data=field(item,'ics'),file_name='residence-permit-reminder.ics',mime='text/calendar',key=f"ics_{thread_key}_{field(item,'reminder_id')}")
        st.caption('An earlier configured-calendar action was recorded; this link opens an additional event draft.' if field(item,'external_id') else 'The event has not been added until you confirm it in your calendar.')
        st.caption('Saved in your case: this reminder and its approved details.')


def render_email_result(draft,state,thread_key):
    if field(draft,'purpose')=='package_review':
        from src.core.procedure import iro_package_readiness
        if not iro_package_readiness(state)['ready']:
            st.warning('The saved package-review draft needs a fresh review because your package is now incomplete. Its history remains in Technical.');return
    from src.tools.email import email_app_link,email_file,email_text,gmail_compose_link
    with st.container(border=True):
        st.success('Email draft ready');_draft_card(draft,state)
        gmail,left,middle,right=st.columns(4)
        try:
            gmail.link_button('Open Gmail compose',gmail_compose_link(display_record(draft)))
            left.link_button('Open default email app',email_app_link(display_record(draft)))
        except (ValueError,TypeError,KeyError):left.caption('Confirm a recipient before opening your email app.')
        try:middle.download_button('Download .eml',data=email_file(display_record(draft)),file_name='iro-review-draft.eml',mime='message/rfc822',key=f"eml_{thread_key}_{field(draft,'draft_id')}")
        except (ValueError,OSError,TypeError,KeyError):st.warning('A selected attachment changed or is unavailable. Prepare a fresh draft to export it.')
        with right.popover('Copy text'):
            st.caption('Use the copy control on this text box.');st.code(email_text(display_record(draft)),language=None,wrap_lines=True)
        with st.expander('Default email app did not open?'):
            st.info('Your system did not open a default mail application. Use Gmail compose, download .eml or copy the draft instead.')
        st.caption('Nothing is sent by opening this draft. Review it and press Send in your own email app.')
        if field(draft,'attachments'):st.caption('Gmail and default-email-app links do not attach files automatically. Attach the files manually, or open the downloaded draft with its checked attachments.')
        st.caption('Saved in your case: the email draft and selected attachment references.')


def render_results(state,thread_key):
    latest=state.get('last_tool_outcome') or {}
    if field(latest,'status')=='REJECTED':st.info('You rejected the proposal. No new action was performed.')
    elif field(latest,'status')=='FAILED':st.warning('This action could not be prepared. Check the inputs and retry; diagnostics are in Technical.')
    for item in record_items(state.get('calendar_reminders'))[-2:]:render_calendar_result(item,thread_key)
    for draft in record_items(state.get('email_drafts'))[-1:]:render_email_result(draft,state,thread_key)
    seen=set()
    for item in reversed(state.get('tool_outcomes',[])[-8:]):
        identifier=field(item,'call_id')
        if identifier in seen:continue
        seen.add(identifier)
        if field(item,'name')=='build_checklist' and field(item,'status')=='SUCCEEDED':
            st.success('Checklist ready');st.download_button('Download checklist',data=field(item,'data',{}).get('content',''),file_name='preparation-checklist.md',key=f"download_checklist_{thread_key}_{field(item,'call_id')}")
    if field(latest,'name')=='search_official_sources' and field(latest,'status')=='SUCCEEDED':
        st.success('Sources retrieved')
        for row in field(latest,'data',{}).get('results',[])[:3]:
            title=row.get('title','Guidance')
            if row.get('url'):st.markdown('['+title+']('+row['url']+')')
            else:st.write(title)
            st.caption(user_text(row.get('content','')))


def render_actions(snapshot,graph,config,thread_key,*,show_approval=True,show_results=True,email_only=False,email_purpose='support'):
    from datetime import time
    from zoneinfo import ZoneInfo
    from config import LOCAL_TIMEZONE
    state=snapshot.values;disabled=bool(snapshot.next)
    if show_approval:render_tool_approval(snapshot,graph,config,thread_key)
    if not email_only:
        with st.expander('Prepare a personal reminder',expanded=False):
            with st.form(f'calendar_proposal_{thread_key}'):
                title=st.text_input('Reminder title',value='Complete residence-permit preparation',disabled=disabled)
                day=st.date_input('Reminder date',value=None,disabled=disabled);at=st.time_input('Reminder time',value=time(9),disabled=disabled)
                description=st.text_input('Reminder note',value='Personal preparation reminder; not an official deadline.',disabled=disabled)
                proposed=st.form_submit_button('Preview reminder',disabled=disabled)
            if proposed:
                if day is None:st.warning('Choose the date before preparing the reminder.')
                elif dispatch_tool(graph,config,'create_calendar_reminder',{'title':title,'start_at':datetime.combine(day,at,tzinfo=ZoneInfo(LOCAL_TIMEZONE)).isoformat(),'duration_minutes':30,'description':description,'delivery':'calendar_app'}):st.rerun()
    from src.core.procedure import iro_package_readiness
    package=iro_package_readiness(state) if email_purpose=='package_review' else None
    email_disabled=disabled or bool(package and not package['ready'])
    label='Prepare IRO package-review email' if package is not None else 'Ask IRO a support question'
    with st.expander(label,expanded=email_only):
        if package is not None and not package['ready']:st.info('Complete the supported IRO package and required earlier steps before preparing its review email. You can ask IRO a separate support question from Next steps.')
        available=[d for d in state.get('documents',[]) if field(d,'status')=='analyzed' and (package is None or field(d,'doc_id') in package['attachment_ids'])];labels={field(d,'doc_id'):document_label(field(d,'doc_type')) for d in available}
        with st.form(f'email_prepare_{thread_key}'):
            st.caption('Select only documents you want to share. Review the exact draft before it is saved.')
            recipient=st.text_input('IRO recipient address (optional if configured)',disabled=disabled)
            subject=st.text_input('Subject',value='Residence permit preparation review request',disabled=disabled)
            attachments=st.multiselect('Documents to attach',list(labels),default=list(labels) if package is not None else [],format_func=labels.get,disabled=email_disabled)
            prepared=st.form_submit_button('Prepare email draft',disabled=email_disabled)
        if prepared:
            args={'subject':subject,'attachment_ids':attachments,'purpose':email_purpose}
            if recipient.strip():args['recipient']=recipient.strip()
            if dispatch_tool(graph,config,'prepare_iro_email',args):st.rerun()
    if show_results:render_results(state,thread_key)


def perform_primary_action(action,graph,config,thread_key):
    from src.core.procedure import start_guided_preparation,continue_after_application_review,advance_current_stage
    kind=action['kind']
    if kind=='navigate':request_view(thread_key,action['view'],action.get('document_id'));st.rerun()
    fn={'start_application':start_guided_preparation,'review_application':continue_after_application_review,'advance':advance_current_stage}.get(kind)
    if fn and persist_procedure_choice(graph,config,lambda state:fn(state,recorded_at=utc_now())):request_view(thread_key,'application');st.rerun()


def view_next_steps(state,snapshot,graph,config,thread_key):
    if needs_onboarding(state):
        from src.ui.home import render_onboarding
        render_onboarding(snapshot,graph,config,thread_key);return
    summary=workflow_summary(state)
    st.markdown('##### Current step');st.write('**'+summary['step_text']+'**');st.write(summary['progress_text'])
    st.markdown('##### Required for this step');render_checklist(state)
    action=primary_action(state)
    if conflict_messages(state):st.error('We found a document conflict that needs your attention.')
    if action:
        st.info(action['message'])
        if st.button(action['label'],type='primary',disabled=bool(snapshot.next),key=f'primary_{thread_key}'):perform_primary_action(action,graph,config,thread_key)
    ready_collection=field(state.get('procedure'),'current_stage')=='DOCUMENT_COLLECTION' and assess_document_collection(state).ready
    if not ready_collection:
        with st.expander('Detailed action plan (optional)',expanded=bool(state.get('approved_plan'))):render_plan_cards(state,snapshot,graph,config,thread_key)
    render_results(state,thread_key)


def render_recommendations(snapshot,graph,config,thread_key):
    from src.tools.calendar import declared_reminder_candidates
    from src.core.legal_stay import insurance_explanation
    state=snapshot.values;message=insurance_explanation(state)
    if message:st.info(message)
    for row in declared_reminder_candidates(state):
        if row['candidate_id']=='declared_legal_stay':
            st.write('Your declared legal-stay expiry is: '+date.fromisoformat(row['date']).strftime('%d %B %Y'));label='Add legal-stay reminder'
        elif row['candidate_id']=='declared_preparation_target':
            st.caption('Recommended preparation reminder: '+row['date']+'. Agent recommendation - not an official deadline.');label='Add preparation reminder'
        else:st.caption('Insurance expiry: '+row['date']);label='Add insurance-expiry reminder'
        if st.button(label,key=f"reminder_{thread_key}_{row['candidate_id']}",disabled=bool(snapshot.next)) and propose_reminder(graph,config,row):st.rerun()
    if st.button('Build checklist',key=f'checklist_{thread_key}',disabled=bool(snapshot.next)):
        if dispatch_tool(graph,config,'build_checklist',{}):st.rerun()


def render_plan_cards(state, snapshot, graph, config, thread_key):
    status = str(state.get("run_status", "NOT_STARTED"))
    busy = bool(snapshot.next)
    st.markdown("##### Your documents")
    render_checklist(state)

    st.markdown("##### Recommended next steps")
    if status == "PLANNING_UNAVAILABLE":
        st.warning("The local model could not write a plan. Your documents and answers are saved; try again when Ollama is running.")
    elif status == "HUMAN_REVIEW_REQUIRED":
        st.error("The plan could not be fixed automatically after "
                 f"{MAX_REFLECTION_ATTEMPTS} attempts. Review your information, then generate a new plan.")
    elif status == "APPROVED":
        st.success("You approved this plan. No official action was taken.")
    elif status == "APPROVAL_DENIED":
        st.info("You denied the last plan. Update your information and generate a new one.")
    approved = state.get("approved_plan")
    steps = plan_steps(state)
    origin = state.get("plan_origin") or (state.get("approved_plan_origin") if approved else None)
    if origin == "milestone_completion":
        st.info(user_text(state.get("plan_summary") or field(approved, "summary", "")))
    if steps is None:
        st.caption("Upload documents and answer the questions in **Your details**, then generate a plan.")
    for index, step in enumerate(display_records(steps), 1):
        with st.container(border=True):
            text, action = st.columns([5, 2])
            text.markdown(f"**{index}. {user_text(user_action(step))}**")
            if step.get("reason"):
                text.caption(user_text(step["reason"]))
            if step.get("due_date") and status in {"VALIDATED", "APPROVED"}:
                text.caption(f":material/event: Due {step['due_date']}")
            if approved and not busy and step.get("tool_call"):
                call=step["tool_call"]
                done=any(row.get("call_id")==call["call_id"] and row.get("status")=="SUCCEEDED" for row in state.get("tool_outcomes",[]))
                if done:action.caption("Outcome already recorded")
                elif action.button("Run proposed tool",key=f"run_plan_{thread_key}_{call['call_id']}"):
                    if dispatch_tool_call(graph,config,call):st.rerun()
            elif approved and not busy:
                marked = any(row.get("action_id") == step.get("step_id") and row.get("action") == step.get("action")
                             for row in state.get("action_progress", {}).values())
                if marked:
                    action.markdown(":green[:material/check: Marked done]")
                elif action.button("Mark done", key=f"done_{thread_key}_{step.get('step_id')}",
                                   help="Records your own declaration; requirements still depend on evidence."):
                    run_event(graph, config, UserEvent(kind="record_progress", action_id=step.get("step_id")))
    if state.get('validation_errors'):st.warning('The proposed next steps need review. Detailed diagnostics are in Technical.')
    if st.button("Generate plan" if steps is None else "Update plan", type="primary",
                 disabled=busy, key=f"plan_{thread_key}"):
        run_event(graph, config, UserEvent(kind="generate_plan"))

def render_decision(state, snapshot, graph, config, thread_key, technical=False):
    """A pending approval or unfinished run comes first, before any view."""
    interaction = interaction_state(snapshot)
    if interaction["approval_pending"]:
        steps = plan_steps(state)
        with st.container(border=True):
            st.markdown("#### :material/gavel: A plan is waiting for your decision")
            if steps:
                for index, step in enumerate(display_records(steps), 1):
                    st.markdown(f"**{index}.** {user_text(user_action(step))}")
            else:
                st.write(user_text(state.get("plan_summary", "")))
            st.caption("Approving records these next steps only. Nothing is submitted, paid or "
                       "sent on your behalf. Deny the plan to edit your information.")
            approve, deny, _ = st.columns([1, 1, 3])
            decision = None
            if approve.button("Approve", type="primary", key=f"approve_{thread_key}_{state.get('trace_id')}"):
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
            if st.button("Retry unfinished processing", key=f"retry_{thread_key}") and invoke(graph, None, config):
                st.rerun()
