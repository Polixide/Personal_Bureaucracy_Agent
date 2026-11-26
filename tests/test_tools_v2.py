"""Offline typed tools, explicit authorization, grounded sandbox and local MCP."""
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
from config import PROJECT_ROOT
from src.core.procedure import advance_procedure_if_ready
from src.tools.schemas import ToolCall, CalendarInput
from src.tools.registry import execute_tool, reject_tool, get_registry
from src.tools.calendar import reminder_preview, GoogleCalendarAdapter
from src.tools.email import GmailAdapter
from test_procedure import ready_bundle, override


@pytest.fixture(autouse=True)
def no_configured_external_adapters(monkeypatch):
    monkeypatch.setattr("src.tools.registry.configured_calendar_adapter",lambda:None)
    monkeypatch.setattr("src.tools.registry.configured_email_adapter",lambda:None)
    monkeypatch.delenv("GOOGLE_CALENDAR_ACCESS_TOKEN",raising=False)
    monkeypatch.delenv("GMAIL_ACCESS_TOKEN",raising=False)
    monkeypatch.delenv("IRO_EMAIL",raising=False)


def preparing(ready_bundle):
    state=deepcopy(ready_bundle[2])
    state.update(advance_procedure_if_ready(state,recorded_at=datetime(2026,11,19,tzinfo=timezone.utc)))
    return state


def observed(state, call, **kwargs):
    result=execute_tool(state,call,**kwargs)
    updated={**state,**result.state_updates,"tool_outcomes":[*state.get("tool_outcomes",[]),result.model_dump(mode="json",exclude={"state_updates"})]}
    return updated,result


def test_registry_has_typed_inputs_outputs_and_authorization_metadata():
    registry=get_registry()
    assert {"get_current_procedure_state","get_missing_documents","search_official_sources","read_eikamet_sandbox","simulate_fill_eikamet_field","simulate_application_validation","create_calendar_reminder","prepare_iro_email","send_iro_email"} <= registry.keys()
    assert registry["create_calendar_reminder"].risk_tier=="TIER_2"
    for spec in registry.values():
        assert spec.input_schema["additionalProperties"] is False
        assert spec.output_schema["type"]=="object"


def test_fill_is_a_real_grounded_state_diff_and_never_changes_verified_facts(ready_bundle):
    state=preparing(ready_bundle);before=deepcopy(state)
    result=execute_tool(state,ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"passport_number"}))
    assert result.status=="SUCCEEDED"
    assert result.state_updates["sandbox_form"]["fields"]["passport_number"]["value"]=="P1234567"
    assert result.diff["passport_number"]["before"] is None
    assert set(result.state_updates)=={"sandbox_form"}
    assert state==before


@pytest.mark.parametrize("arguments",[{"field_id":"passport_number","value":"FABRICATED"},{"field_id":"email"},{"field_id":"unknown"}])
def test_missing_fabricated_and_unsupported_values_cannot_fill(ready_bundle,arguments):
    result=execute_tool(preparing(ready_bundle),ToolCall(name="simulate_fill_eikamet_field",arguments=arguments))
    assert result.status=="FAILED" and not result.state_updates


def test_conflicts_and_collection_stage_block_automation(ready_bundle):
    state=ready_bundle[2]
    call=ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"passport_number"})
    assert execute_tool(state,call).status=="FAILED"
    state=preparing(ready_bundle);state["verification_conflicts"]=["passport_or_foreign_id"]
    assert execute_tool(state,call).status=="FAILED"


def test_stale_filled_field_is_detected_after_user_correction(ready_bundle):
    state=preparing(ready_bundle)
    state,result=observed(state,ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"passport_expiry_date"}))
    passport=next(doc for doc in state["documents"] if doc.doc_type=="passport_copy")
    state=override(state,passport.doc_id,"expiry_date","2031-01-01")
    result=execute_tool(state,ToolCall(name="validate_form_field",arguments={"field_id":"passport_expiry_date"}))
    assert result.status=="SUCCEEDED" and not result.data["valid"]
    assert result.data["stale_field_ids"]==["passport_expiry_date"]


def test_incomplete_simulated_submission_is_never_official_or_complete(ready_bundle):
    state=preparing(ready_bundle)
    result=execute_tool(state,ToolCall(name="simulate_submission"))
    assert result.status=="SUCCEEDED" and not result.data["valid"]
    assert result.data["form"]["status"]=="DRAFT"
    assert set(result.state_updates)=={"sandbox_form"}
    assert state["procedure"].current_stage=="EIKAMET_PREPARATION"


def test_model_cannot_approve_via_tool_arguments():
    result=execute_tool({},ToolCall(name="create_calendar_reminder",arguments={"title":"Reminder","start_at":"2027-01-01T12:00:00+03:00","approved":True}))
    assert result.status=="FAILED" and result.failure_category=="TOOL_SCHEMA_ERROR"


def test_calendar_preview_executes_only_after_approval_and_replays_once():
    adapter=Mock();adapter.create.return_value="external-event"
    call=ToolCall(call_id="calendar-one",name="create_calendar_reminder",arguments={"title":"Prepare documents","start_at":"2027-01-01T12:00:00+03:00"})
    preview=execute_tool({},call,adapters={"calendar":adapter})
    assert preview.status=="APPROVAL_REQUIRED" and preview.data["mode"]=="GOOGLE_CALENDAR"
    adapter.create.assert_not_called()
    state,result=observed({},call,approved=True,adapters={"calendar":adapter})
    assert result.status=="SUCCEEDED" and result.data["external_id"]=="external-event"
    assert execute_tool(state,call,approved=True,adapters={"calendar":adapter}).replayed
    adapter.create.assert_called_once()


def test_calendar_local_ics_is_usable_and_escapes_injection():
    call=ToolCall(name="create_calendar_reminder",arguments={"title":"Reminder\nBEGIN:VEVENT","start_at":"2027-01-01T12:00:00+03:00"})
    result=execute_tool({},call,approved=True,adapters={})
    assert result.status=="SUCCEEDED" and result.data["mode"]=="LOCAL_ICS"
    assert result.data["ics"].count("\r\nBEGIN:VEVENT\r\n")==1
    assert "DTSTART:20270101T090000Z" in result.data["ics"]
    assert len(result.state_updates["calendar_reminders"])==1


def test_rejection_is_terminal_for_that_proposal():
    call=ToolCall(name="create_calendar_reminder",arguments={"title":"Reminder","start_at":"2027-01-01T12:00:00+03:00"})
    rejected=reject_tool({},call)
    result=execute_tool({"tool_outcomes":[rejected.model_dump(exclude={"state_updates"})]},call,approved=True,adapters={})
    assert result.failure_category=="HUMAN_REJECTED" and not result.state_updates


def test_repeated_identical_failures_are_bounded():
    state={};call=ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"email"})
    state,_=observed(state,call);state,_=observed(state,call)
    assert execute_tool(state,call).failure_category=="TOOL_LOOP_LIMIT"


def test_email_draft_is_grounded_with_exact_selected_attachment_and_official_recipient(ready_bundle):
    state=preparing(ready_bundle);doc=state["documents"][0]
    result=execute_tool(state,ToolCall(name="prepare_iro_email",arguments={"attachment_ids":[doc.doc_id]}),approved=True)
    assert result.status=="SUCCEEDED"
    assert result.data["recipient"]=="irolegal@sabanciuniv.edu" and result.data["recipient_source_ref"]=="SAB_EXCHANGE_CURRENT"
    assert result.data["attachments"][0]["content_sha256"]==sha256(Path(doc.file_path).read_bytes()).hexdigest()
    assert "EIKAMET_PREPARATION" not in result.data["body"]
    assert "no official" not in result.data["body"].lower() or "does not claim official" in result.data["body"].lower()


def test_email_never_invents_attachment_or_accepts_header_injection(ready_bundle):
    for args in ({"attachment_ids":["absent"]},{"subject":"hello\r\nBcc:bad@example.test"}):
        result=execute_tool(ready_bundle[2],ToolCall(name="prepare_iro_email",arguments=args))
        assert result.status=="FAILED" and not result.state_updates


def test_email_send_preview_exact_draft_then_local_simulation(ready_bundle):
    state,draft=observed(preparing(ready_bundle),ToolCall(name="prepare_iro_email"),approved=True)
    call=ToolCall(name="send_iro_email",arguments={"draft_id":draft.data["draft_id"]})
    preview=execute_tool(state,call,adapters={})
    assert preview.status=="APPROVAL_REQUIRED" and preview.data["delivery_mode"]=="LOCAL_SIMULATION"
    state,result=observed(state,call,approved=True,adapters={})
    assert result.data["status"]=="SIMULATED_SEND"
    assert execute_tool(state,call,approved=True,adapters={}).replayed


def test_email_attachment_change_blocks_send_after_preview(ready_bundle):
    state=preparing(ready_bundle);doc=state["documents"][0]
    state,draft=observed(state,ToolCall(name="prepare_iro_email",arguments={"attachment_ids":[doc.doc_id]}),approved=True)
    Path(doc.file_path).write_text("changed synthetic content",encoding="utf-8")
    result=execute_tool(state,ToolCall(name="send_iro_email",arguments={"draft_id":draft.data["draft_id"]}),approved=True,adapters={})
    assert result.status=="FAILED" and not result.state_updates


def test_google_adapter_uses_deterministic_event_id_and_mock_transport_only():
    args=CalendarInput(title="Reminder",start_at="2027-01-01T12:00:00+03:00")
    reminder=reminder_preview(args,"stable-call")
    transport=Mock(return_value=BytesIO(json.dumps({"id":"event"}).encode()))
    adapter=GoogleCalendarAdapter("synthetic-token",transport=transport)
    assert adapter.create(reminder)=="event"
    request=transport.call_args.args[0]
    assert json.loads(request.data)["id"]==sha256(b"stable-call").hexdigest()
    assert request.get_method()=="POST"


def test_minimal_mcp_subprocess_lists_resource_and_readonly_tool():
    requests=[{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}},
        {"jsonrpc":"2.0","method":"notifications/initialized"},
        {"jsonrpc":"2.0","id":2,"method":"tools/list"},
        {"jsonrpc":"2.0","id":3,"method":"resources/list"},
        {"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"ikamet://official-guidance"}},
        {"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"search_official_sources","arguments":{"query":"passport validity"}}}]
    run=subprocess.run([sys.executable,"-m","src.integrations.mcp.server"],input="\n".join(map(json.dumps,requests))+"\n",
        cwd=PROJECT_ROOT,capture_output=True,text=True,check=True,timeout=30)
    replies=[json.loads(line) for line in run.stdout.splitlines()]
    assert len(replies)==5 and replies[0]["result"]["protocolVersion"]=="2025-11-25"
    assert replies[1]["result"]["tools"][0]["name"]=="search_official_sources"
    assert replies[2]["result"]["resources"][0]["uri"]=="ikamet://official-guidance"
    assert json.loads(replies[3]["result"]["contents"][0]["text"])
    assert not replies[4]["result"]["isError"]
    assert json.loads(replies[4]["result"]["content"][0]["text"])["results"]


def test_calendar_preview_accepts_canonical_nullable_pending_field():
    state={"pending_tool_proposal":None,"tool_outcomes":[]}
    result=execute_tool(state,ToolCall(name="create_calendar_reminder",arguments={"title":"Reminder","start_at":"2027-01-01T12:00:00+03:00"}),adapters={})
    assert result.status=="APPROVAL_REQUIRED"


def test_uncertain_gmail_delivery_is_not_retried_with_new_call_id(ready_bundle):
    state,draft=observed(preparing(ready_bundle),ToolCall(name="prepare_iro_email"),approved=True)
    adapter=Mock();adapter.send.side_effect=TimeoutError("uncertain delivery")
    call=ToolCall(name="send_iro_email",arguments={"draft_id":draft.data["draft_id"]})
    state,result=observed(state,call,approved=True,adapters={"email":adapter})
    assert result.failure_category=="EXTERNAL_OUTCOME_UNCERTAIN"
    retry=ToolCall(name="send_iro_email",arguments=call.arguments)
    result=execute_tool(state,retry,approved=True,adapters={"email":adapter})
    assert result.failure_category=="EXTERNAL_OUTCOME_UNCERTAIN"
    adapter.send.assert_called_once()


def test_calendar_rejects_implicit_numeric_timestamp():
    result=execute_tool({},ToolCall(name="create_calendar_reminder",arguments={"title":"Reminder","start_at":12345678}),adapters={})
    assert result.failure_category=="TOOL_SCHEMA_ERROR"


def test_proposal_validation_is_pure_and_shares_registry_schema():
    from src.tools.registry import validate_tool_call
    call=ToolCall(name="create_calendar_reminder",arguments={"title":"Reminder","start_at":"2027-01-01T12:00:00+03:00"})
    checked=validate_tool_call(call)
    assert checked.arguments["duration_minutes"]==30
    assert "duration_minutes" not in call.arguments
    with pytest.raises(ValueError):validate_tool_call(ToolCall(name="unknown"))
    with pytest.raises(ValueError):validate_tool_call(ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"passport_number","approved":True}))


def test_calendar_provider_conflict_reconciles_same_approved_event():
    from urllib.error import HTTPError
    args=CalendarInput(title="Reminder",start_at="2027-01-01T12:00:00+03:00")
    reminder=reminder_preview(args,"stable-event")
    event_id=sha256(b"stable-event").hexdigest()
    existing={"id":event_id,"summary":reminder.title,"description":reminder.description,
              "start":{"dateTime":"2027-01-01T09:00:00Z"},"end":{"dateTime":"2027-01-01T09:30:00Z"}}
    transport=Mock(side_effect=[HTTPError("https://example.test",409,"Exists",{},None),BytesIO(json.dumps(existing).encode())])
    assert GoogleCalendarAdapter("synthetic",transport=transport).create(reminder)==event_id
    assert transport.call_args.args[0].get_method()=="GET"


def test_gmail_adapter_serializes_exact_preview_with_mock_transport_only():
    import base64
    from email import policy
    from email.parser import BytesParser
    from src.tools.schemas import EmailResult
    draft=EmailResult(draft_id="synthetic-draft",recipient="iro@example.test",subject="Review request",body="Please review my saved preparation.",attachments=[])
    transport=Mock(return_value=BytesIO(b'{"id":"synthetic-message"}'))
    assert GmailAdapter("synthetic-token",transport=transport).send(draft)=="synthetic-message"
    request=transport.call_args.args[0]
    message=BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(json.loads(request.data)["raw"]))
    assert message["To"]==draft.recipient and message["Subject"]==draft.subject
    assert message["Message-ID"]=="<synthetic-draft@ikamet-local>"
    assert draft.body in message.get_content()


# Guided UX contracts: the same tools/approvals, without external side effects.
def test_calendar_app_delivery_never_uses_an_existing_api_adapter():
    from urllib.parse import urlparse, parse_qs
    from src.tools.calendar import google_calendar_link
    adapter=Mock()
    call=ToolCall(name="create_calendar_reminder",arguments={"title":"Prepare & review","start_at":"2027-01-01T12:00:00+03:00","delivery":"calendar_app"})
    assert execute_tool({},call,adapters={"calendar":adapter}).status=="APPROVAL_REQUIRED"
    result=execute_tool({},call,approved=True,adapters={"calendar":adapter})
    adapter.create.assert_not_called()
    assert result.status=="SUCCEEDED" and result.data["ics"].startswith("BEGIN:VCALENDAR")
    query=parse_qs(urlparse(google_calendar_link(result.data)).query)
    assert query["text"]==["Prepare & review"] and query["action"]==["TEMPLATE"]
    assert query["dates"]==["20270101T090000Z/20270101T093000Z"]


def test_primary_reminders_use_explicit_accepted_expiry_not_nationality_calculator(ready_bundle):
    from src.tools.calendar import declared_reminder_candidates
    state=deepcopy(ready_bundle[2])
    state["residence_permit_context"].legal_stay_expiry_date=date(2026,12,16)
    # A bare context value cannot create a declared reminder.
    assert not any(r['candidate_id']=='declared_legal_stay' for r in declared_reminder_candidates(state))
    state=override(state,'__declarations__','legal_stay_expiry_date','2026-12-16')
    before=deepcopy(state)
    with patch('src.core.legal_stay.assess_legal_stay',side_effect=AssertionError('No primary calculator')):
        rows={r['candidate_id']:r for r in declared_reminder_candidates(state)}
    assert rows['declared_legal_stay']['date']=='2026-12-16'
    assert rows['declared_preparation_target']['date']=='2026-11-16'
    assert 'not an official deadline' in rows['declared_preparation_target']['kind']
    assert state==before


def test_email_app_and_eml_exports_match_approved_draft_and_do_not_send(ready_bundle):
    from urllib.parse import urlparse,parse_qs
    from email import policy
    from email.parser import BytesParser
    from src.tools.email import email_app_link,email_file,email_text
    state,draft=observed(preparing(ready_bundle),ToolCall(name='prepare_iro_email',arguments={'attachment_ids':[ready_bundle[2]['documents'][0].doc_id]}),approved=True)
    before=deepcopy(state)
    query=parse_qs(urlparse(email_app_link(draft.data)).query)
    assert query['subject']==[draft.data['subject']] and query['body']==[draft.data['body']]
    message=BytesParser(policy=policy.default).parsebytes(email_file(draft.data))
    assert message['To']==draft.data['recipient'] and message['X-Unsent']=='1'
    assert len(list(message.iter_attachments()))==1
    assert draft.data['body'] in email_text(draft.data)
    assert state==before and not state.get('email_deliveries')


def test_batch_sandbox_prefill_resolves_every_accepted_value_without_fact_writes(ready_bundle):
    from src.tools.sandbox import prefill_fields,needs_prefill
    from src.core.procedure import build_eikamet_preparation,eikamet_field_value
    state=preparing(ready_bundle);before=deepcopy(state)
    result,diff=prefill_fields(state)
    fields=build_eikamet_preparation(state).fields
    expected={key:eikamet_field_value(state,item) for key,item in fields.items() if item.status in {'VERIFIED','USER_CONFIRMED'}}
    assert {key:item.value for key,item in result.form.fields.items()}==expected
    assert diff['before']['fields']=={} and len(expected)>1
    assert not needs_prefill({**state,'sandbox_form':result.form.model_dump(mode='json')})
    assert state==before


def test_simulated_pdf_requires_review_and_approval_and_never_establishes_completion(ready_bundle,tmp_path,monkeypatch):
    import pymupdf
    from src.core.procedure import checked_form_answers
    from src.tools.sandbox import read_simulated_pdf
    from src.document_processing.processor import process_document
    from src.document_processing.extractor import TextPage
    from unittest.mock import patch
    state=preparing(ready_bundle)
    values={'first_name':'Alex','last_name':'Example','birth_date':'2000-01-01','gender':'female','passport_type':'ordinary',
        'passport_issue_date':'2020-01-01','passport_issuing_country':'Exampleland','passport_issuing_authority':'Example Authority',
        'email':'student@example.test','turkish_phone':'+90 5551234567','residential_address':'Example Street 12',
        'requested_start_date':'2026-09-15','reason_of_stay':'Incoming exchange studies','father_name':'Example Father','mother_name':'Example Mother',
        'contact_method':'email','marital_status':'single','birth_country':'Exampleland','address_province':'Istanbul','address_district':'Example District',
        'address_locality':'Example Neighborhood','form_current_date':'2026-11-21','simulation_declaration_acknowledged':True}
    for key,value in checked_form_answers(state,values).items():state=override(state,'__declarations__',key,value)
    state,_=observed(state,ToolCall(name='prefill_eikamet_sandbox'))
    call=ToolCall(name='generate_simulated_application')
    assert execute_tool(state,call).status=='FAILED'  # One explicit Review is required.
    state,review=observed(state,ToolCall(name='simulate_application_validation'))
    assert review.data['valid'] and len(state['sandbox_form']['fields'])==30
    monkeypatch.setattr('config.PROJECT_ROOT',tmp_path)
    before=deepcopy(state)
    assert execute_tool(state,call).status=='APPROVAL_REQUIRED'
    assert not (tmp_path/'.local/artifacts').exists()
    state,result=observed(state,call,approved=True)
    assert result.status=='SUCCEEDED'
    data=read_simulated_pdf(state);assert data and data.startswith(b'%PDF')
    with pymupdf.open(stream=data,filetype='pdf') as pdf:text='\n'.join(page.get_text() for page in pdf)
    for label in ('SIMULATED E-IKAMET APPLICATION','Educational sandbox artifact','NOT AN OFFICIAL GOVERNMENT FORM','NOT SUBMITTED'):
        assert label in text
    assert state['documents']==before['documents'] and state['residence_permit_context']==before['residence_permit_context']
    assert state['procedure'].current_stage=='EIKAMET_PREPARATION'
    assert state['residence_permit_context'].online_application_completed is not True
    # Re-uploaded educational text cannot masquerade as official evidence, even with a hint.
    with patch('src.document_processing.processor.extract_pages',return_value=[TextPage(1,text)]),patch('src.document_processing.processor.Client') as client:
        info=process_document(tmp_path/'simulation.pdf',document_type_hint='ikamet_application_form')
    assert info.document_type=='unknown';client.assert_not_called()

    # Use an accepted correction for each case source after a PDF already exists.
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    new_passport = passport.model_copy(deep=True, update={"doc_id": "replacement-passport"})
    state["documents"] = [new_passport if doc.doc_id == passport.doc_id else doc for doc in state["documents"]]
    state = override(state, new_passport.doc_id, "passport_number", "P7654321")
    state = override(state, "__declarations__", "residential_address", "Updated Example Street 20")
    assert read_simulated_pdf(state) is None
    state, _ = observed(state, ToolCall(name="prefill_eikamet_sandbox"))
    form = state["sandbox_form"]
    assert form["fields"]["passport_number"] == {
        "value": "P7654321", "source_ref": "/verification_records/replacement-passport/passport_number",
        "source_status": "USER_CONFIRMED",
    }
    assert form["fields"]["turkiye_address"]["value"] == "Updated Example Street 20"
    assert form["validated_revision"] is None and not form["simulated_application_ready"]
    replacement = ToolCall(name="generate_simulated_application")
    assert execute_tool(state, replacement).status == "FAILED"
    state, refreshed = observed(state, ToolCall(name="simulate_application_validation"))
    assert refreshed.data["valid"]
    assert execute_tool(state, replacement).status == "APPROVAL_REQUIRED"
    state, _ = observed(state, replacement, approved=True)
    with pymupdf.open(stream=read_simulated_pdf(state), filetype="pdf") as pdf:
        updated = "\n".join(page.get_text() for page in pdf)
    assert "P7654321" in updated and "Updated Example Street 20" in updated
    assert "P1234567" not in updated and "Example Street 12" not in updated


def test_form_drafts_survive_failed_review_and_sqlite_recreation_without_legal_fact_writes(ready_bundle,tmp_path):
    from src.memory.checkpointer import sqlite_checkpointer
    from src.core.graph import build_graph
    state=preparing(ready_bundle);before=deepcopy(state)
    result=execute_tool(state,ToolCall(name='prefill_eikamet_sandbox',arguments={'answers':{'first_name':'Unsaved Name','birth_date':'not-a-date','father_name':'Example Parent'}}))
    assert result.status=='SUCCEEDED' and not result.data['valid']
    assert set(result.state_updates)=={'sandbox_form'}
    state.update(result.state_updates)
    assert state['residence_permit_context']==before['residence_permit_context'] and state['documents']==before['documents']
    cfg={'configurable':{'thread_id':'draft-restart'}};db=tmp_path/'draft.db'
    with sqlite_checkpointer(db) as saver:build_graph(saver).update_state(cfg,state,as_node='respond')
    with sqlite_checkpointer(db) as saver:loaded=build_graph(saver).get_state(cfg).values
    assert loaded['sandbox_form']['draft_answers']=={'first_name':'Unsaved Name','birth_date':'not-a-date','father_name':'Example Parent'}
    assert loaded['residence_permit_context'].father_name is None


def test_gmail_and_default_mail_urls_encode_exact_draft_without_sending():
    from src.tools.email import gmail_compose_link,email_app_link
    from urllib.parse import urlparse,parse_qs
    draft={'recipient':'iro@example.test','subject':'Review & questions?','body':'Hello\nFirst line & second line'}
    gmail=parse_qs(urlparse(gmail_compose_link(draft)).query);mailto=parse_qs(urlparse(email_app_link(draft)).query)
    assert gmail['view']==['cm'] and gmail['to']==[draft['recipient']]
    assert gmail['su']==mailto['subject']==[draft['subject']]
    assert gmail['body']==mailto['body']==[draft['body']]
