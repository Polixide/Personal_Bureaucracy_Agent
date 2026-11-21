"""Offline presentation/action boundaries for the five focused V2 workspaces."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

from streamlit_app import NAVIGATION, user_case_view, procedure_view
from src.ui.components import (
    application_rows, document_sections, field_group, pending_tool, sandbox_rows,
    tool_preview, dispatch_tool,
)
from test_procedure import ready_bundle


def test_navigation_has_five_focused_workspaces():
    assert NAVIGATION == ("next", "copilot", "documents", "details", "checks", "application", "technical")


def test_documents_distinguish_current_and_later_without_changing_twin(ready_bundle):
    state=ready_bundle[2]
    before=deepcopy(state)
    sections=document_sections(state)
    assert len(sections["required_now"])==6
    assert len(sections["uploaded"])==4
    assert "Generated e-Ikamet form" in sections["later"]
    assert "IRO review and submission evidence" in sections["later"]
    assert state==before


def test_application_groups_show_values_sources_and_missing_fields(ready_bundle):
    state=ready_bundle[2]
    before=deepcopy(state)
    groups=application_rows(state)
    assert list(groups)==["Personal information","Passport","Contact / address","Residence request","Insurance"]
    passport=next(row for row in groups["Passport"] if row["Field"]=="Passport number")
    assert passport=={"Field":"Passport number","Value":"P1234567","Source":"Document","Status":"From your document"}
    entry=next(row for row in groups["Residence request"] if row["Field"]=="Latest Turkiye entry date")
    assert entry["Source"]=="Your confirmation" and entry["Status"]=="Confirmed by you"
    assert all(row["Value"]=="Not available" for row in groups["Contact / address"])
    assert state==before


def test_sandbox_display_reads_environment_without_updating_real_context(ready_bundle):
    state=deepcopy(ready_bundle[2])
    state["sandbox_form"]={"fields":{"passport_number":{"value":"P1234567","source_ref":"/verification_records/passport/passport_number","source_status":"VERIFIED"}},"revision":1,"status":"DRAFT"}
    before=deepcopy(state)
    rows=sandbox_rows(state)
    assert next(row for row in rows if row["Field"]=="Passport number")["Value"]=="P1234567"
    assert next(row for row in rows if row["Field"]=="Email")["Status"]=="MISSING"
    assert state==before


def test_field_groups_do_not_infer_unknown_values():
    assert field_group("first_name")=="Personal information"
    assert field_group("passport_expiry_date")=="Passport"
    assert field_group("turkiye_address")=="Contact / address"
    assert field_group("requested_permit_end")=="Residence request"
    assert field_group("insurance_category")=="Insurance"


def test_tool_approval_is_distinguished_from_plan_approval():
    assert pending_tool(SimpleNamespace(next=("tool_approval",)))
    assert not pending_tool(SimpleNamespace(next=("human_approval",)))
    assert not pending_tool(SimpleNamespace(next=()))


def test_tool_preview_is_detached_and_preserves_explicit_diff():
    state={"pending_tool_proposal":{"proposal_id":"call-1","tool_call":{"call_id":"call-1","name":"create_calendar_reminder","arguments":{"title":"Personal reminder"}},
        "preview":{"before":None,"after":{"title":"Personal reminder"}},
        "observation":{"data":{"title":"Personal reminder","mode":"LOCAL_ICS"}}}}
    before=deepcopy(state)
    preview=tool_preview(state)
    assert preview["name"]=="create_calendar_reminder" and preview["diff"]["before"] is None
    preview["diff"]["after"]["title"]="Display-only change"
    preview["arguments"]["title"]="Display-only change"
    assert state==before


def test_ui_tool_dispatch_uses_graph_event_without_direct_execution():
    graph=Mock()
    config={"configurable":{"thread_id":"ui-case"}}
    with patch("src.ui.components.st.spinner"):
        assert dispatch_tool(graph,config,"simulate_fill_eikamet_field",{"field_id":"passport_number"})
    payload,received_config=graph.invoke.call_args.args
    assert payload["event"].kind=="run_tool"
    assert payload["event"].tool_call["name"]=="simulate_fill_eikamet_field"
    assert payload["event"].tool_call["arguments"]=={"field_id":"passport_number"}
    assert payload["event"].tool_call["call_id"] and received_config==config


def test_legacy_saved_cases_are_not_given_inferred_progress():
    state={"user_id":"legacy","documents":[]}
    before=deepcopy(state)
    assert user_case_view(state)["legacy"] and procedure_view(state)["legacy"]
    assert state==before


def test_v2_document_card_readiness_does_not_reimpose_legacy_extra_attestations(ready_bundle):
    from src.ui.components import document_display_status
    state=ready_bundle[2]
    insurance=next(doc for doc in state["documents"] if doc.doc_type=="health_insurance")
    assert state["residence_permit_context"].insurance_valid_for_ikamet is None
    assert document_display_status(state,insurance)=="Ready for this step"
    assert insurance not in document_sections(state)["needs_review"]


def test_approved_plan_tool_dispatch_preserves_host_generated_call_id():
    from src.ui.components import dispatch_tool_call
    call={"call_id":"canonical-plan-call","name":"simulate_fill_eikamet_field","arguments":{"field_id":"passport_number"}}
    before=deepcopy(call)
    graph=Mock();config={"configurable":{"thread_id":"approved-plan-case"}}
    with patch("src.ui.components.st.spinner"):
        assert dispatch_tool_call(graph,config,call)
        assert dispatch_tool_call(graph,config,call)
    events=[invocation.args[0]["event"] for invocation in graph.invoke.call_args_list]
    assert all(event.kind=="run_tool" and event.tool_call==before for event in events)
    assert call==before


def test_neda_plan_button_dispatches_the_exact_approved_tool_call():
    from src.ui.actions import view_next_steps
    call={"call_id":"canonical-ui-plan-call","name":"simulate_fill_eikamet_field","arguments":{"field_id":"passport_number"}}
    step={"step_id":"prepare-passport","title":"Fill accepted passport number","tool_call":call}
    state={"run_status":"APPROVED","candidate_plan":[step],"approved_plan":{"actions":[step]}}
    snapshot=SimpleNamespace(next=())
    graph=Mock();config={"configurable":{"thread_id":"neda-plan-button"}}
    action_column=MagicMock();action_column.button.return_value=True
    with patch("src.ui.actions.st") as st,patch("src.ui.actions.render_checklist"),patch("src.ui.actions.dispatch_tool_call",return_value=False) as dispatch:
        st.columns.return_value=[MagicMock(),action_column]
        st.button.return_value=False
        view_next_steps(state,snapshot,graph,config,"neda-button")
    dispatch.assert_called_once_with(graph,config,call)


# Real Streamlit execution, isolated SQLite, and the existing graph/approval path.
import pytest
from hashlib import sha256
from datetime import datetime,timezone
from pathlib import Path
from streamlit.testing.v1 import AppTest


@pytest.fixture
def ui_case(tmp_path,monkeypatch):
    import src.memory.checkpointer as persistence
    import config as settings
    from src.core.graph import build_graph
    original=persistence.sqlite_checkpointer
    db=tmp_path/'ui.db'
    monkeypatch.setattr(persistence,'sqlite_checkpointer',lambda:original(db))
    monkeypatch.setattr(settings,'TWIN_RECORD_PATH',db)
    monkeypatch.setattr('src.ui.components.TWIN_RECORD_PATH',db)
    monkeypatch.setattr(settings,'PROJECT_ROOT',tmp_path)
    cfg={'configurable':{'thread_id':'demo-user'},'recursion_limit':60}
    key=sha256(b'demo-user').hexdigest()[:16]
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),default_timeout=25)
    app.session_state['workspace_profile_id']='demo-user'
    def seed(state):
        with original(db) as saver:build_graph(saver).update_state(cfg,deepcopy(state),as_node='respond')
    def read():
        with original(db) as saver:return build_graph(saver).get_state(cfg).values
    def nav(view):
        app.session_state['view_'+key]=view
        return app.run()
    def click(label):
        matches=[button for button in app.button if button.label==label]
        assert len(matches)==1,(label,[button.label for button in app.button])
        matches[0].click();app.run();healthy()
    def healthy():
        assert not app.exception,[error.value for error in app.exception]
        assert not any('Processing could not finish' in str(error.value) for error in app.error)
    def text():
        return '\n'.join(str(element.value) for kind in ('markdown','caption','info','warning','error','success','header','subheader','text','metric') for element in app.get(kind) if hasattr(element,'value'))
    return SimpleNamespace(app=app,seed=seed,read=read,nav=nav,click=click,healthy=healthy,text=text,config=cfg,key=key,tmp_path=tmp_path)


def entered(state,mode='START_FROM_SCRATCH'):
    from src.core.procedure import entry_intent_update
    state=deepcopy(state)
    state.update(entry_intent_update(state,mode,user_id='demo-user',trace_id=state.get('trace_id','test'),recorded_at=datetime(2026,11,20,tzinfo=timezone.utc)))
    return state


def choose(app,kind,label,value):
    element=next(item for item in app.get(kind) if item.label==label)
    element.set_value(value)


def test_acceptance_A_new_user_sees_five_steps_two_choices_and_no_chat(ui_case):
    ui_case.app.run();ui_case.healthy()
    assert 'Step 1 of 5 - Documents' in ui_case.text()
    assert not ui_case.app.chat_input and not ui_case.app.get('json')
    assert {"I'm starting from the beginning",'I already have some documents','I already completed the online e-Ikamet application'} <= {b.label for b in ui_case.app.button}
    assert not any(raw in ui_case.text() for raw in ('/ 24','/24','LOCAL_ICS','HUMAN_REVIEW_REQUIRED'))
    ui_case.click("I'm starting from the beginning")
    assert "I'm starting from the beginning" not in [b.label for b in ui_case.app.button]
    assert 'Upload passport identity page' in [b.label for b in ui_case.app.button]
    assert ui_case.read()['procedure'].current_stage=='DOCUMENT_COLLECTION'


def test_acceptance_B_unclear_nationality_is_readable_confirmation_not_json(ui_case,ready_bundle):
    from test_procedure import override
    state=deepcopy(ready_bundle[2])
    passport=next(d for d in state['documents'] if d.doc_type=='passport_copy')
    state=override(state,passport.doc_id,'nationality',None)
    passport=next(d for d in state['documents'] if d.doc_type=='passport_copy')
    passport.info.nationality=None;passport.info.missing_fields=['nationality']
    passport.info.source_evidence=[r for r in passport.info.source_evidence if r.field!='nationality']
    passport.info.source_pages.pop('nationality',None);passport.info.source_snippets.pop('nationality',None)
    ui_case.seed(entered(state));ui_case.nav('documents');ui_case.healthy()
    assert 'We could not read nationality confidently' in ui_case.text()
    assert any(item.label=='Nationality' for item in ui_case.app.text_input)
    assert not ui_case.app.get('json')
    assert '["nationality"]' not in ui_case.text()


def test_acceptance_C_conflict_explains_exact_document_and_field_on_home(ui_case,ready_bundle):
    from test_procedure import override
    state=ready_bundle[2];insurance=next(d for d in state['documents'] if d.doc_type=='health_insurance')
    state=override(state,insurance.doc_id,'holder_name','Different Person')
    ui_case.seed(entered(state));ui_case.app.run();ui_case.healthy()
    assert 'The holder name on Health insurance certificate does not match your passport identity.' in ui_case.text()
    ui_case.click('Review Health insurance certificate')
    assert ui_case.app.segmented_control[0].value=='documents'
    assert 'The holder names in your documents do not match' in ui_case.text()


def test_acceptance_D_ready_documents_continue_without_planner_or_reflection(ui_case,ready_bundle):
    state=entered(ready_bundle[2]);state['run_status']='HUMAN_REVIEW_REQUIRED';state['validation_errors']=[]
    ui_case.seed(state)
    with patch('src.agents.planner.create_action_plan',side_effect=AssertionError('No Planner for progression')):
        ui_case.app.run();ui_case.healthy()
        assert 'Your initial documents are ready.' in ui_case.text()
        assert not any(b.label in {'Generate plan','Update plan'} for b in ui_case.app.button)
        ui_case.click('Continue to e-Ikamet')
    assert ui_case.read()['procedure'].current_stage=='EIKAMET_PREPARATION'
    assert 'Step 2 of 5 - e-Ikamet' in ui_case.text()
    assert 'Have you already completed the online e-Ikamet application?' in ui_case.text()


def test_acceptance_E_existing_form_upload_review_and_explicit_payment_transition(ui_case,monkeypatch):
    from test_milestone_documents import SAMPLES,response
    ui_case.app.run();ui_case.click('I already completed the online e-Ikamet application')
    content=SAMPLES['ikamet_application_form'].encode()
    upload=SimpleNamespace(name='actual-application.txt',getvalue=lambda:content)
    path=ui_case.tmp_path/'actual-application.txt'
    def save(*args):path.write_bytes(content);return path
    monkeypatch.setattr('src.ui.application.save_upload',save)
    with patch('src.ui.application.st.file_uploader',return_value=upload),patch('src.document_processing.processor.Client') as client:
        client.return_value.chat.side_effect=[response({'document_type':'ikamet_application_form'}),response({})]
        ui_case.app.run();ui_case.click('Analyze application form')
    state=ui_case.read();assert state['procedure'].current_stage=='DOCUMENT_COLLECTION'
    assert len(state['documents'])==1 and state['documents'][0].doc_type=='ikamet_application_form'
    assert {'Matched information','Missing information'} <= set(e.value.strip('*') for e in ui_case.app.markdown)
    # Two real-world assertions use the existing document confirmation form.
    for box in ui_case.app.selectbox:
        if 'online application' in box.label.lower() or 'signed' in box.label.lower():box.set_value('Yes')
    choose(ui_case.app,'checkbox','I confirm the entered information is accurate',True)
    ui_case.click('Confirm information')
    ui_case.click('Continue to Payment & Fingerprint')
    state=ui_case.read()
    assert state['procedure'].current_stage=='PAYMENT' and not state.get('sandbox_form')
    assert 'Step 3 of 5 - Payment & Fingerprint' in ui_case.text()


def test_acceptance_F_assisted_prefill_one_review_approved_pdf_not_official_completion(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.app.run();ui_case.click('Continue to e-Ikamet')
    ui_case.click('No - Help me prepare it')
    filled=ui_case.read()['sandbox_form']['fields']
    assert len(filled)>1 and filled['passport_number']['value']=='P1234567'
    assert next(item for item in ui_case.app.text_input if item.label=='Passport number').disabled
    values={'First name':'Alex','Surname':'Example','Birth date':'2000-01-01','Passport issue date':'2020-01-01',
        'Passport issuing country':'Exampleland','Passport issuing authority':'Example Authority','Email':'student@example.test',
        'Turkish phone':'+90 5551234567','Turkiye address':'Example Street 12','Requested permit start':'2026-09-15', "Father's name":'Example Father',"Mother's name":'Example Mother',
        'Country of birth':'Exampleland','Province / city':'Istanbul','District':'Example District','Neighborhood / locality':'Example Neighborhood','Form date':'2026-11-21'}
    for label,value in values.items():choose(ui_case.app,'text_input',label,value)
    choose(ui_case.app,'selectbox','Gender','female');choose(ui_case.app,'selectbox','Passport type','ordinary')
    choose(ui_case.app,'selectbox','Preferred contact method','email');choose(ui_case.app,'selectbox','Marital status','single')
    choose(ui_case.app,'selectbox','Reason of stay','exchange_student_education')
    choose(ui_case.app,'checkbox','I have reviewed the guide declarations for this educational simulation',True)
    choose(ui_case.app,'checkbox','I confirm the information I entered',True)
    ui_case.click('Review application')
    assert ui_case.read()['sandbox_form']['status']=='VALIDATED'
    ui_case.click('Generate simulated application form')
    assert not ui_case.read()['sandbox_form'].get('simulated_application_ready')
    ui_case.click('Approve this action')
    state=ui_case.read();assert state['sandbox_form']['simulated_application_ready']
    assert state['procedure'].current_stage=='EIKAMET_PREPARATION'
    assert state['residence_permit_context'].online_application_completed is not True
    assert 'Simulation complete. You still need to complete the real e-Ikamet application externally.' in ui_case.text()
    assert any(item.proto.label=='Download simulated application PDF' for item in ui_case.app.get('download_button'))


def test_acceptance_G_declared_legal_stay_reminder_has_calendar_link_and_ics(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.nav('details');ui_case.healthy()
    assert not any('previous' in item.label.lower() or item.label=='Actual entry status' for item in ui_case.app.text_input)
    assert not ui_case.app.text_area
    assert 'https://www.mfa.gov.tr/visa-information-for-foreigners.en.mfa' in ui_case.text()
    choose(ui_case.app,'text_input','Legal stay expiry date (YYYY-MM-DD)','2026-12-16')
    choose(ui_case.app,'checkbox','I confirm this is my declared legal-stay expiry date',True)
    ui_case.click('Save legal-stay expiry');ui_case.click('Add reminder')
    assert not ui_case.read().get('calendar_reminders')
    ui_case.click('Approve this action');ui_case.healthy()
    assert 'Reminder ready' in ui_case.text()
    links=ui_case.app.get('link_button')
    assert any(item.proto.label=='Open in Google Calendar' and item.proto.url.startswith('https://calendar.google.com/calendar/render?') for item in links)
    assert any(item.proto.label=='Download .ics' for item in ui_case.app.get('download_button'))
    assert 'The event has not been added until you confirm it in your calendar.' in ui_case.text()
    assert 'LOCAL_ICS' not in ui_case.text()


def test_acceptance_H_iro_email_has_user_controlled_mailto_eml_and_copy(ui_case,ready_bundle):
    from src.evaluation.scenarios import copilot_benchmark_cases
    state=entered(copilot_benchmark_cases(ui_case.tmp_path/'email-fixtures')[-1]['state'])
    ui_case.seed(state);ui_case.nav('application');ui_case.healthy()
    ui_case.click('Prepare email draft')
    assert not ui_case.read().get('email_drafts')
    ui_case.click('Approve this action')
    assert 'Email draft ready' in ui_case.text()
    assert any(item.proto.label=='Open default email app' and item.proto.url.startswith('mailto:') for item in ui_case.app.get('link_button'))
    assert any(item.proto.label=='Download .eml' for item in ui_case.app.get('download_button'))
    assert any(item.proto.popover.label=='Copy text' for item in ui_case.app.get('popover'))
    assert ui_case.app.code and 'Nothing is sent by opening this draft.' in ui_case.text()
    assert not any('send email' in b.label.lower() for b in ui_case.app.button)


def test_acceptance_I_accepted_payment_receipt_avoids_duplicate_payment_question(ui_case,ready_bundle,tmp_path):
    from test_milestone_documents import analyze
    graph,cfg,_=ready_bundle
    state=analyze(graph,cfg,tmp_path,'card_fee_receipt')
    state=entered(state);state['procedure']=state['procedure'].model_copy(update={'current_stage':'PAYMENT'})
    ui_case.seed(state);ui_case.nav('application');ui_case.healthy()
    assert 'Card fee payment receipt received. Payment evidence ready.' in ui_case.text()
    assert not any(item.label=='Payment evidence' for item in ui_case.app.selectbox)
    assert 'Step 3 of 5 - Payment & Fingerprint' in ui_case.text()


@pytest.mark.parametrize('stage,step,heading',[('FINGERPRINT',3,'Fingerprint / GI'),('IRO_REVIEW',4,'Prepare your IRO package'),('TRACKING',5,'Submission / Tracking')])
def test_acceptance_J_later_steps_have_readable_progress_and_stage_specific_requirements(ui_case,ready_bundle,stage,step,heading):
    from src.core.procedure import stage_document_checklist
    state=entered(ready_bundle[2])
    if stage=='FINGERPRINT':
        from test_milestone_documents import analyze,confirm
        graph,cfg,_=ready_bundle
        state=analyze(graph,cfg,ui_case.tmp_path,'ikamet_application_form')
        doc=next(doc for doc in state['documents'] if doc.doc_type=='ikamet_application_form')
        state=entered(confirm(graph,cfg,doc.doc_id,{'online_application_completed':True,'application_form_signed':True}))
    state['procedure']=state['procedure'].model_copy(update={'current_stage':stage})
    ui_case.seed(state);ui_case.nav('application');ui_case.healthy()
    assert f'Step {step} of 5' in ui_case.text() and heading in ui_case.text()
    assert not any(raw in ui_case.text() for raw in ('/ 24','/24','LOCAL_ICS','LOCAL_SIMULATION','HUMAN_REVIEW_REQUIRED','TOOL_COMPLETED','sandbox revision','R022'))
    if stage=='FINGERPRINT':
        assert stage_document_checklist(state)==[{'label':'Fingerprint / GI confirmation','document_type':None,'status':'needs confirmation'}]
        assert 'No additional fingerprint document is required' in ui_case.text()


def test_user_value_translates_only_known_codes_without_mutating_canonical_values():
    from src.ui.components import user_value
    state={'permit_type':'short_term','insurance_evidence_type':'private_health_insurance','passport_number':'EXAMPLE_123'}
    before=deepcopy(state)
    assert user_value('permit_type',state['permit_type'])=='Short-term residence permit'
    assert user_value('insurance_evidence_type',state['insurance_evidence_type'])=='Private health insurance'
    assert user_value('passport_number',state['passport_number'])=='EXAMPLE_123'
    assert state==before



def test_landing_catalogue_recommendation_and_existing_profile_resume(ui_case,ready_bundle):
    del ui_case.app.session_state['workspace_profile_id']
    ui_case.app.run();ui_case.healthy()
    assert 'What do you need help with?' in ui_case.text() and not ui_case.app.segmented_control
    choose(ui_case.app,'text_input','Profile ID','new-guided-profile');ui_case.app.run()
    assert next(b for b in ui_case.app.button if b.label=='Residence permit renewal').disabled
    choose(ui_case.app,'text_area','Describe your situation','I am a Sabanci exchange student and just arrived in Turkey.')
    with patch('src.agents.assistant.recommend_procedure',return_value={'procedure_id':'first_ikamet','label':'First residence permit / Ikamet','implemented':True,'message':'First Ikamet is supported.'}):
        ui_case.click('Ask the assistant')
    assert 'Start this procedure' in [b.label for b in ui_case.app.button]
    ui_case.seed(entered(ready_bundle[2]))
    choose(ui_case.app,'text_input','Profile ID','demo-user');ui_case.app.run()
    assert 'Saved case found.' in ui_case.text() and 'Current step: Documents' in ui_case.text()
    ui_case.click('Resume case')
    assert 'Step 1 of 5 - Documents' in ui_case.text() and ui_case.app.segmented_control


def test_copilot_segment_answers_a_state_question_and_preserves_case(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.nav('copilot');ui_case.healthy()
    assert 'AI Copilot' in ui_case.text() and ui_case.app.chat_input
    before=deepcopy(ui_case.read()['residence_permit_context'])
    with patch('src.agents.planner.create_action_plan',side_effect=AssertionError('State questions do not plan')):
        ui_case.click('What should I do now?')
    assert 'Continue to e-Ikamet' in ui_case.text()
    assert ui_case.read()['residence_permit_context']==before


def test_incomplete_iro_package_email_is_disabled_with_specific_missing_items(ui_case,ready_bundle):
    from src.evaluation.scenarios import copilot_benchmark_cases
    state=entered(copilot_benchmark_cases(ui_case.tmp_path/'iro-incomplete')[5]['state'])
    ui_case.seed(state);ui_case.nav('application');ui_case.healthy()
    assert 'IRO package: 7 of 9 ready' in ui_case.text()
    assert 'Declaration' in ui_case.text() and 'receipt' in ui_case.text()
    assert next(button for button in ui_case.app.button if button.label=='Prepare email draft').disabled


def test_review_keeps_invalid_manual_drafts_visible_after_rerun_and_reload(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.app.run();ui_case.click('Continue to e-Ikamet');ui_case.click('No - Help me prepare it')
    choose(ui_case.app,'text_input',"Father's name",'Example Parent')
    choose(ui_case.app,'text_input','Birth date','not-a-date')
    choose(ui_case.app,'checkbox','I confirm the information I entered',True)
    ui_case.click('Review application')
    assert ui_case.read()['sandbox_form']['draft_answers']['father_name']=='Example Parent'
    assert ui_case.read()['residence_permit_context'].father_name is None
    ui_case.app.run();ui_case.healthy()
    assert next(item for item in ui_case.app.text_input if item.label=="Father's name").value=='Example Parent'
    # Recreate the app/session; restore the same SQLite thread explicitly.
    reloaded=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),default_timeout=25)
    reloaded.run();choose(reloaded,'text_input','Profile ID','demo-user');reloaded.run()
    next(b for b in reloaded.button if b.label=='Resume case').click();reloaded.run()
    reloaded.session_state['view_'+ui_case.key]='application';reloaded.run()
    assert not reloaded.exception
    assert next(item for item in reloaded.text_input if item.label=="Father's name").value=='Example Parent'
    assert next(item for item in reloaded.text_input if item.label=='Birth date').value=='not-a-date'
