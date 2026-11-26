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


def test_navigation_keeps_student_workspaces_and_advanced():
    assert NAVIGATION == ("next", "copilot", "documents", "details", "checks", "application", "technical")


def test_documents_distinguish_current_and_later_without_changing_twin(ready_bundle):
    state=ready_bundle[2]
    before=deepcopy(state)
    sections=document_sections(state)
    assert len(sections["required_now"])==6
    assert {doc.doc_type for doc in sections["uploaded"]} == {
        "passport_copy", "turkiye_entry_stamp", "student_certificate", "health_insurance", "dorm_certificate",
    }
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
    with patch("src.ui.actions.st") as st,patch("src.ui.actions.dispatch_tool_call",return_value=False) as dispatch:
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
        return '\n'.join(str(element.value) for kind in ('title','markdown','caption','info','warning','error','success','header','subheader','text','metric') for element in app.get(kind) if hasattr(element,'value'))
    def view():
        return app.session_state["view_" + key] if "view_" + key in app.session_state else "next"
    def details(section):
        choose(app, "radio", "Details section", section)
        app.run()
        healthy()
    return SimpleNamespace(app=app, seed=seed, read=read, nav=nav, click=click, healthy=healthy, text=text,
                           view=view, details=details, config=cfg, key=key, tmp_path=tmp_path)


def entered(state,mode='START_FROM_SCRATCH'):
    from src.core.procedure import entry_intent_update
    state=deepcopy(state)
    state.update(entry_intent_update(state,mode,user_id='demo-user',trace_id=state.get('trace_id','test'),recorded_at=datetime(2026,11,20,tzinfo=timezone.utc)))
    return state


def choose(app,kind,label,value):
    element=next(item for item in app.get(kind) if item.label==label)
    element.set_value(value)


def test_open_current_step_recovers_legacy_documents_after_reopen_and_from_other_views(ui_case):
    from src.evaluation.scenarios import confirm, copilot_benchmark_cases

    state = copilot_benchmark_cases(ui_case.tmp_path / "navigation")[-1]["state"]
    state = confirm(state, "__declarations__", {"address_proof_type": None})
    del state["procedure"]
    state["documents"] = [doc for doc in state["documents"] if doc.doc_type in {
        "passport_copy", "turkiye_entry_stamp", "student_certificate", "health_insurance",
    }]
    ui_case.seed(state)
    saved = ui_case.read()

    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reopened.run()
    choose(reopened, "text_input", "Profile name", "demo-user")
    next(button for button in reopened.button if button.label == "Open my case").click()
    reopened.run()
    assert any(element.value == "4 of 7 items ready for this step" for element in reopened.markdown)
    next(button for button in reopened.sidebar.button if button.label == "Open current step").click()
    reopened.run()
    assert not reopened.exception
    assert any(button.label == "Upload and analyze" for button in reopened.button)
    assert reopened.session_state[f"view_{ui_case.key}"] == "documents"

    ui_case.app.run()
    choose(ui_case.app, "toggle", "Case settings", True)
    ui_case.app.run()
    for origin in ("Overview", "Documents", "Your details", "Checks and warnings",
                   "Ask about this step", "Advanced diagnostics"):
        ui_case.click(origin)
        ui_case.click("Open current step")
        assert ui_case.view() == "documents"
        assert "Upload or review a document" in ui_case.text()
        assert any(button.label == "Upload and analyze" for button in ui_case.app.button)

    ui_case.nav("application")  # A session left in the previously unsupported view.
    ui_case.healthy()
    assert ui_case.view() == "documents"
    assert "Upload or review a document" in ui_case.text()
    assert ui_case.read() == saved


@pytest.mark.parametrize(("stage", "expected_view", "control"), [
    ("DOCUMENT_COLLECTION", "documents", "Upload and analyze"),
    ("EIKAMET_PREPARATION", "application", "No - Help me prepare it"),
    ("EIKAMET_READY", "application", "Analyze application form"),
    ("PAYMENT", "application", "Upload payment receipt"),
    ("FINGERPRINT", "application", "Upload or review the real application form"),
    ("IRO_REVIEW", "application", "Save IRO progress and continue"),
    ("SUBMISSION", "application", "Save submission and continue"),
    ("TRACKING", "application", "Save tracking update"),
    ("COMPLETE", "application", None),
])
def test_open_current_step_renders_each_stage_and_survives_rerun(ui_case, stage, expected_view, control):
    from src.core.state import ProcedureState

    state = entered({"procedure": ProcedureState(), "user_id": "demo-user", "trace_id": "navigation"})
    state["procedure"] = ProcedureState(current_stage=stage,
                                       eikamet_path="UPLOADED_FORM" if stage == "EIKAMET_READY" else None)
    ui_case.seed(state)
    saved = ui_case.read()
    ui_case.nav("details")
    ui_case.click("Open current step")
    ui_case.app.run()
    ui_case.healthy()
    assert ui_case.view() == expected_view
    assert f"requested_view_{ui_case.key}" not in ui_case.app.session_state
    if control:
        assert any(button.label == control for button in ui_case.app.button)
    else:
        assert "Your reported official decision is recorded" in ui_case.text()
    assert ui_case.read() == saved


def test_open_current_step_follows_readiness_and_explicit_stage_transition(ui_case, ready_bundle):
    from test_procedure import override

    ui_case.seed(entered(override(ready_bundle[2], "__declarations__", "biometric_photo", False)))
    ui_case.app.run()
    ui_case.click("Open current step")
    assert "5 of 6 items ready for this step" in ui_case.text()
    assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)

    ui_case.click("Your details")
    choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "Yes")
    ui_case.click("Save and confirm preparation answers")
    ui_case.click("Open current step")
    assert ui_case.view() == "documents"
    assert "6 of 6 items ready for this step" in ui_case.text()
    ui_case.click("Continue to e-Ikamet")
    ui_case.click("Documents")
    ui_case.click("Open current step")
    assert ui_case.view() == "application"
    assert "Have you already completed the online e-Ikamet application?" in ui_case.text()
    assert ui_case.read()["procedure"].current_stage == "EIKAMET_PREPARATION"


def test_open_current_step_resolves_against_latest_checkpoint(ui_case):
    from src.core.state import ProcedureState

    state = entered({"procedure": ProcedureState(), "user_id": "demo-user", "trace_id": "navigation"})
    ui_case.seed(state)
    ui_case.app.run()
    # Another browser can update the case between rendering the shortcut and clicking it.
    ui_case.seed({**state, "procedure": ProcedureState(current_stage="PAYMENT")})
    ui_case.click("Open current step")
    assert ui_case.view() == "application"
    assert "Payment & Fingerprint" in [heading.value for heading in ui_case.app.header]


def test_unknown_view_recovers_to_overview_and_keeps_valid_navigation(ui_case, ready_bundle):
    ui_case.seed(entered(ready_bundle[2]))
    ui_case.app.session_state[f"requested_view_{ui_case.key}"] = "removed-route"
    ui_case.nav("old-view")
    ui_case.healthy()
    assert ui_case.view() == "next"
    assert "Get your documents ready" in [title.value for title in ui_case.app.title]
    ui_case.click("Documents")
    assert ui_case.view() == "documents"


def test_payments_overview_link_focuses_the_needed_document(ui_case, ready_bundle):
    state = entered(ready_bundle[2])
    state["procedure"] = state["procedure"].model_copy(update={"current_stage": "PAYMENT"})
    ui_case.seed(state)
    ui_case.nav("documents")
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    choose(ui_case.app, "selectbox", "Saved document", passport.doc_id)
    ui_case.app.run()
    ui_case.click("Overview")
    ui_case.click("Upload Card/document-fee payment receipt")
    assert ui_case.view() == "documents"
    assert next(box for box in ui_case.app.selectbox if box.label == "Saved document").value is None
    assert ui_case.app.session_state[f"document_target_{ui_case.key}"] == "card_fee_receipt"


def test_application_version_link_focuses_the_form_instead_of_a_previously_selected_passport(ui_case, ready_bundle):
    from test_document_integrity import add_evidence

    state = entered(ready_bundle[2])
    state["procedure"] = state["procedure"].model_copy(
        update={"current_stage": "EIKAMET_PREPARATION", "eikamet_path": "UPLOADED_FORM"})
    first = add_evidence(state, "ikamet_application_form", doc_id="first-synthetic-form")
    second = add_evidence(state, "ikamet_application_form", doc_id="second-synthetic-form")
    ui_case.seed(state)
    ui_case.nav("documents")
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    choose(ui_case.app, "selectbox", "Saved document", passport.doc_id)
    ui_case.app.run()
    ui_case.click("Open current step")
    ui_case.click("Choose application version in Documents")
    assert next(box for box in ui_case.app.selectbox if box.label == "Saved document").value in {first.doc_id, second.doc_id}
    assert any(button.label == "Use this version and replace other versions" for button in ui_case.app.button)


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
    ui_case.seed(entered(state))
    with patch("ollama.Client.chat", side_effect=AssertionError("Opening review is read-only")) as model:
        ui_case.nav("documents")
        ui_case.healthy()
        model.assert_not_called()
    assert "Missing information" in ui_case.text()
    assert "Key information" in ui_case.text()
    assert any("Passport number" in str(item.proto) for item in ui_case.app.get("html"))
    assert not any(item.label == "Review document" for item in ui_case.app.toggle)
    assert 'Nationality is missing' in ui_case.text()
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
    assert ui_case.view()=='documents'
    assert 'The holder names in your documents do not match' in ui_case.text()
    assert "Conflict detected" in ui_case.text()


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
    ui_case.click("Open current step")
    assert ui_case.view() == "documents"
    ui_case.click("Overview")
    ui_case.click("Review your e-Ikamet application form")
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
    assert not any("entered information is accurate" in item.label for item in ui_case.app.checkbox)
    ui_case.click('Save and confirm')
    ui_case.click('Continue to Payment & Fingerprint')
    state=ui_case.read()
    assert state['procedure'].current_stage=='PAYMENT' and not state.get('sandbox_form')
    assert 'Step 3 of 5 - Payment & Fingerprint' in ui_case.text()


def test_acceptance_F_assisted_prefill_one_review_approved_pdf_not_official_completion(ui_case, ready_bundle):
    import pymupdf
    from src.core.state import Document
    from src.evaluation.scenarios import extract_synthetic
    from src.tools.sandbox import read_simulated_pdf
    from test_procedure import override

    state = entered(ready_bundle[2])
    address = extract_synthetic("address_form", text=(
        "Address Form\nFirst name: Alex\nSurname: Example\nPassport number: P1234567\n"
        "Birth date: 2000-01-01\nDeclared address: Example Street 12\nSignature present: yes"
    ))
    state["documents"].append(Document(doc_id="signed-address", doc_type="address_form",
                                       file_path="synthetic-address.txt", status="analyzed", info=address))
    utility = extract_synthetic("utility_bill")
    state["documents"].append(Document(doc_id="synthetic-utility", doc_type="utility_bill",
                                       file_path="synthetic-utility.txt", status="analyzed", info=utility))
    state = override(state, "__declarations__", "address_proof_type", "utility_bill_and_address_form")
    ui_case.seed(state)
    ui_case.app.run()
    ui_case.click("Continue to e-Ikamet")
    state = ui_case.read()
    for key, value in {
        "email": "old@example.test", "passport_type": "ordinary", "passport_issue_date": "2020-01-01",
        "passport_issuing_country": "Exampleland", "passport_issuing_authority": "Example Authority",
        "address_province": "Istanbul", "address_district": "Example District", "address_locality": "Example Neighborhood",
    }.items():
        state = override(state, "__declarations__", key, value)
    ui_case.seed(state)
    with patch("src.ui.application.dispatch_tool", wraps=dispatch_tool) as dispatch:
        ui_case.click("No - Help me prepare it")
        ui_case.app.run()
        ui_case.healthy()
    assert [call.args[2] for call in dispatch.call_args_list] == ["prefill_eikamet_sandbox"]
    filled = ui_case.read()["sandbox_form"]["fields"]
    assert filled["passport_number"]["value"] == "P1234567"
    assert filled["turkiye_address"] == {
        "value": "Example Street 12", "source_ref": "/verification_records/signed-address/declared_address",
        "source_status": "VERIFIED",
    }
    assert filled["address_province"]["source_ref"] == "/verification_records/__declarations__/address_province"
    assert filled["email"]["value"] == "old@example.test"
    assert "From Passport identity page" in ui_case.text()
    assert "From Address Form" in ui_case.text()
    inputs = {item.label: item.value for item in ui_case.app.text_input if not item.disabled}
    assert {name: inputs[name] for name in (
        "Passport number", "Passport expiry date", "Nationality", "Turkiye address",
        "Province / city", "District", "Neighborhood / locality", "Email", "Passport issue date",
    )} == {
        "Passport number": "P1234567", "Passport expiry date": "2030-08-30", "Nationality": "Exampleland",
        "Turkiye address": "Example Street 12", "Province / city": "Istanbul", "District": "Example District",
        "Neighborhood / locality": "Example Neighborhood", "Email": "old@example.test",
        "Passport issue date": "2020-01-01",
    }
    assert inputs["Father's name"] == ""
    choose(ui_case.app, "text_input", "Email", "student@example.test")
    values = {
        "Turkish phone": "+90 5551234567", "Requested permit start": "2026-09-15",
        "Father's name": "Example Father", "Mother's name": "Example Mother",
        "Country of birth": "Exampleland", "Form date": "2026-11-21",
    }
    for label, value in values.items():
        choose(ui_case.app, "text_input", label, value)
    choose(ui_case.app, "selectbox", "Gender", "female")
    choose(ui_case.app,'selectbox','Preferred contact method','email');choose(ui_case.app,'selectbox','Marital status','single')
    choose(ui_case.app,'selectbox','Reason of stay','exchange_student_education')
    choose(ui_case.app,'checkbox','I have reviewed the guide declarations for this educational simulation',True)
    assert not any(item.label == 'I confirm the information I entered' for item in ui_case.app.checkbox)
    ui_case.click('Save and review simulated PDF')
    assert ui_case.read()['sandbox_form']['status'] == 'VALIDATED'
    assert not ui_case.read()['sandbox_form'].get('simulated_application_ready')
    assert 'Generate a simulated application PDF' in ui_case.text()
    ui_case.click('Reject this action')
    assert not ui_case.read()['sandbox_form'].get('simulated_application_ready')
    assert not list(ui_case.tmp_path.glob('.local/artifacts/**/*.pdf'))
    assert ui_case.read()["residence_permit_context"].email == "student@example.test"
    with patch('src.ui.application.dispatch_tool', wraps=dispatch_tool) as dispatch:
        ui_case.click('Save and review simulated PDF')
    assert [call.args[2] for call in dispatch.call_args_list] == ['generate_simulated_application']
    ui_case.click('Approve this action')
    state=ui_case.read();assert state['sandbox_form']['simulated_application_ready']
    assert state['procedure'].current_stage=='EIKAMET_PREPARATION'
    assert state['residence_permit_context'].online_application_completed is not True
    with pymupdf.open(stream=read_simulated_pdf(state), filetype="pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
    assert all(value in text for value in ("P1234567", "Example Street 12", "student@example.test", "Istanbul"))
    assert "old@example.test" not in text
    assert 'Simulation complete. Complete the real e-Ikamet application separately.' in ui_case.text()
    assert not any(button.label == 'Save and review simulated PDF' for button in ui_case.app.button)
    assert any(item.proto.label=='Download simulated application PDF' and item.proto.ignore_rerun for item in ui_case.app.get('download_button'))


def test_conflicting_passport_form_edit_stays_a_draft_and_can_be_restored(ui_case, ready_bundle):
    ui_case.seed(entered(ready_bundle[2]))
    ui_case.app.run()
    ui_case.click("Continue to e-Ikamet")
    ui_case.click("No - Help me prepare it")
    accepted = deepcopy(ui_case.read()["verification_records"])

    passport = next(item for item in ui_case.app.text_input if item.label == "Passport number")
    assert passport.value == "P1234567"
    passport.set_value("P7654321")
    ui_case.click("Save and review simulated PDF")
    assert "Your draft differs from accepted evidence for Passport number" in ui_case.text()
    assert not any(button.label == "Approve this action" for button in ui_case.app.button)
    state = ui_case.read()
    assert state["sandbox_form"]["draft_answers"]["passport_or_foreign_id"] == "P7654321"
    assert state["verification_records"] == accepted

    reloaded = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reloaded.run()
    choose(reloaded, "text_input", "Profile name", "demo-user")
    next(button for button in reloaded.button if button.label == "Open my case").click()
    reloaded.run()
    next(button for button in reloaded.button if button.label == "Open current step").click()
    reloaded.run()
    assert not reloaded.exception
    passport = next(item for item in reloaded.text_input if item.label == "Passport number")
    assert passport.value == "P7654321"
    passport.set_value("P1234567")
    next(button for button in reloaded.button if button.label == "Save and review simulated PDF").click()
    reloaded.run()
    assert not reloaded.exception
    assert "passport_or_foreign_id" not in ui_case.read()["sandbox_form"]["draft_answers"]
    assert ui_case.read()["verification_records"] == accepted


def test_assisted_draft_survives_navigation_and_reconciles_new_accepted_evidence(ui_case, ready_bundle):
    from test_procedure import override

    ui_case.seed(entered(ready_bundle[2]))
    ui_case.app.run()
    ui_case.click("Continue to e-Ikamet")
    ui_case.seed(override(ui_case.read(), "__declarations__", "residential_address", "Old Example Street 1"))
    ui_case.click("No - Help me prepare it")
    choose(ui_case.app, "text_input", "Passport number", "P7654321")
    choose(ui_case.app, "text_input", "Email", "draft@example.test")
    ui_case.click("Overview")
    assert "passport_or_foreign_id" not in ui_case.read()["sandbox_form"]["draft_answers"]
    assert ui_case.read()["residence_permit_context"].email is None

    state = ui_case.read()
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    state = override(state, passport.doc_id, "passport_number", "P9999999")
    state = override(state, "__declarations__", "residential_address", "New Example Street 2")
    ui_case.seed(state)
    ui_case.click("Open current step")
    inputs = {item.label: item.value for item in ui_case.app.text_input}
    assert inputs["Passport number"] == "P7654321"
    assert inputs["Turkiye address"] == "New Example Street 2"
    assert inputs["Email"] == "draft@example.test"
    assert inputs["District"] == ""
    assert "Accepted value · P9999999" in ui_case.text()
    ui_case.click("Save and review simulated PDF")
    assert "Your draft differs from accepted evidence for Passport number" in ui_case.text()
    assert not any(button.label == "Approve this action" for button in ui_case.app.button)
    state = ui_case.read()
    assert state["sandbox_form"]["fields"]["passport_number"]["value"] == "P9999999"
    assert state["sandbox_form"]["draft_answers"]["passport_or_foreign_id"] == "P7654321"
    assert state["residence_permit_context"].email is None

    ui_case.seed(override(state, "__declarations__", "passport_or_foreign_id", "P7777777"))
    ui_case.app.run()
    ui_case.healthy()
    passport_input = next(item for item in ui_case.app.text_input if item.label == "Passport number")
    assert passport_input.disabled and passport_input.value == "P7654321"
    assert "Compare the saved sources" in ui_case.text()
    assert "passport_number" not in ui_case.read()["sandbox_form"]["fields"]


def test_acceptance_G_declared_legal_stay_reminder_has_calendar_link_and_ics(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.nav('details');ui_case.healthy()
    ui_case.details("entry")
    assert not any('previous' in item.label.lower() or item.label=='Actual entry status' for item in ui_case.app.text_input)
    assert not ui_case.app.text_area
    assert 'https://www.mfa.gov.tr/visa-information-for-foreigners.en.mfa' in ui_case.text()
    choose(ui_case.app,'text_input','Legal stay expiry date (YYYY-MM-DD)','2026-12-16')
    assert not any('declared legal-stay expiry date' in item.label for item in ui_case.app.checkbox)
    ui_case.click('Confirm my legal-stay expiry date')
    ui_case.details('reminders')
    ui_case.click('Add reminder')
    assert not ui_case.read().get('calendar_reminders')
    ui_case.click('Approve this action');ui_case.healthy()
    assert 'Reminder ready' in ui_case.text()
    links=ui_case.app.get('link_button')
    assert any(item.proto.label=='Open in Google Calendar' and item.proto.url.startswith('https://calendar.google.com/calendar/render?') for item in links)
    assert any(item.proto.label=='Download .ics' and item.proto.ignore_rerun for item in ui_case.app.get('download_button'))
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
    assert any(item.proto.label=='Download .eml' and item.proto.ignore_rerun for item in ui_case.app.get('download_button'))
    assert any(item.proto.popover.label=='Copy text' for item in ui_case.app.get('popover'))
    assert ui_case.app.code and 'Nothing is sent by opening this draft.' in ui_case.text()
    assert not any('send email' in b.label.lower() for b in ui_case.app.button)

    with patch("src.tools.email.email_file", side_effect=AssertionError("Only the email view builds the export")) as export:
        for view in ("next", "details", "copilot"):
            ui_case.nav(view)
            ui_case.healthy()
            assert not any(item.proto.label == "Download .eml" for item in ui_case.app.get("download_button"))
        export.assert_not_called()
    ui_case.nav("application")
    ui_case.healthy()
    assert "Email draft ready" in ui_case.text()

    before = ui_case.read()
    assert before["residence_permit_context"].scans_sent_to_iro is not True
    assert not any("confirm these actual IRO outcomes" in box.label for box in ui_case.app.checkbox)
    ui_case.click("Save IRO progress and continue")
    assert ui_case.read()["trace_id"] == before["trace_id"]
    choose(ui_case.app, "selectbox", "Have you sent your scans to IRO?", "Yes")
    ui_case.click("Save IRO progress and continue")
    result = ui_case.read()
    assert result["event"].confirmed_fields == {"scans_sent_to_iro": True}
    assert result["procedure"].current_stage == "IRO_REVIEW"
    assert result["residence_permit_context"].iro_preverification_confirmed is not True
    assert "Waiting for IRO review" in ui_case.text()


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



def test_landing_recommends_a_procedure_and_reopens_the_same_case(ui_case, ready_bundle):
    from src.memory import checkpointer

    del ui_case.app.session_state["workspace_profile_id"]
    ui_case.app.run()
    ui_case.healthy()
    assert "Prepare your first residence permit" in ui_case.text()
    assert not any(button.label == "Overview" for button in ui_case.app.button)
    choose(ui_case.app, "toggle", "Help choosing a procedure", True)
    ui_case.app.run()
    assert "Residence permit renewal" in ui_case.text()
    assert not any(button.label == "Residence permit renewal" for button in ui_case.app.button)
    choose(ui_case.app, "text_area", "Describe your situation", "I am an incoming exchange student.")
    with patch("src.agents.assistant.recommend_procedure", return_value={"message": "First Ikamet is supported."}):
        ui_case.click("Ask the assistant")
    assert "First Ikamet is supported." in ui_case.text()
    assert [button.label for button in ui_case.app.button].count("Open my case") == 1
    state = entered(ready_bundle[2])
    ui_case.seed(state)
    with patch.object(checkpointer, "sqlite_checkpointer", wraps=checkpointer.sqlite_checkpointer) as saver:
        choose(ui_case.app, "text_input", "Profile name", "demo-user")
        ui_case.click("Open my case")
        saver.assert_called_once()
    assert "Step 1 of 5 - Documents" in ui_case.text()
    assert ui_case.read()["trace_id"] == state["trace_id"]


def test_copilot_segment_answers_a_state_question_and_preserves_case(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.nav('copilot');ui_case.healthy()
    assert 'Ask about this step' in ui_case.text() and ui_case.app.chat_input
    before=deepcopy(ui_case.read()['residence_permit_context'])
    with patch('src.agents.planner.create_action_plan',side_effect=AssertionError('State questions do not plan')):
        ui_case.click('What should I do now?')
    assert 'Continue to e-Ikamet' in ui_case.text()
    assert ui_case.read()['residence_permit_context']==before


def test_iro_review_hides_the_draft_from_a_previous_address_method(ui_case):
    from src.core.procedure import iro_package_readiness
    from src.evaluation.scenarios import confirm, copilot_benchmark_cases
    from src.tools.registry import execute_tool
    from src.tools.schemas import ToolCall
    from test_address_proof import SAMPLES, add_document

    state = copilot_benchmark_cases(ui_case.tmp_path / "iro-address")[-1]["state"]
    old_package = iro_package_readiness(state)
    request = ToolCall(name="prepare_iro_email", arguments={
        "purpose": "package_review", "attachment_ids": old_package["attachment_ids"],
    })
    draft = execute_tool(state, request, approved=True)
    assert draft.status == "SUCCEEDED"
    state.update(draft.state_updates)

    contract = add_document(state, "rental_agreement", path=ui_case.tmp_path / "rental.txt",
                            text=SAMPLES["rental_agreement"].replace("Alex Example", "Evaluation Student"))
    state = confirm(state, contract.doc_id, {"notary_certification": True})
    state = confirm(state, "__declarations__", {"address_proof_type": "rental_agreement"})
    current = iro_package_readiness(state)
    assert current["ready"] and contract.doc_id in current["attachment_ids"]
    assert current["address_attachment_ids"] == [contract.doc_id]
    assert "dorm_certificate" not in current["attachment_ids"]
    missing_address = ToolCall(name="prepare_iro_email", arguments={
        "purpose": "package_review",
        "attachment_ids": [key for key in current["attachment_ids"] if key != contract.doc_id],
    })
    assert execute_tool(state, missing_address).status == "FAILED"
    ui_case.seed(state)
    ui_case.nav("application")
    ui_case.healthy()
    assert "The saved package-review draft needs a fresh review" in ui_case.text()
    assert not any(item.proto.label == "Open Gmail compose" for item in ui_case.app.get("link_button"))
    assert any(button.label == "Prepare email draft" for button in ui_case.app.button)


def test_incomplete_iro_package_shows_missing_items_and_routes_support(ui_case,ready_bundle):
    from src.evaluation.scenarios import copilot_benchmark_cases
    state=entered(copilot_benchmark_cases(ui_case.tmp_path/'iro-incomplete')[5]['state'])
    ui_case.seed(state);ui_case.nav('application');ui_case.healthy()
    assert 'IRO package: 7 of 9 ready' in ui_case.text()
    assert 'Declaration' in ui_case.text() and 'receipt' in ui_case.text()
    assert not any(button.label == 'Prepare email draft' for button in ui_case.app.button)
    assert not any(widget.label == "Documents to attach" for widget in ui_case.app.multiselect)
    ui_case.click("Ask IRO a support question")
    assert ui_case.view() == "next"
    assert next(box for box in ui_case.app.selectbox if box.label == "Optional tools").value == "Ask IRO a question"
    assert next(widget for widget in ui_case.app.multiselect if widget.label == "Documents to attach").value == []
    assert ui_case.read()["procedure"].current_stage == "IRO_REVIEW"


def test_review_keeps_invalid_manual_drafts_visible_after_rerun_and_reload(ui_case,ready_bundle):
    ui_case.seed(entered(ready_bundle[2]));ui_case.app.run();ui_case.click('Continue to e-Ikamet');ui_case.click('No - Help me prepare it')
    choose(ui_case.app,'text_input',"Father's name",'Example Parent')
    choose(ui_case.app,'text_input','Birth date','not-a-date')
    ui_case.click('Save and review simulated PDF')
    assert not any(button.label == 'Approve this action' for button in ui_case.app.button)
    assert ui_case.read()['sandbox_form']['draft_answers']['father_name']=='Example Parent'
    assert ui_case.read()['residence_permit_context'].father_name is None
    ui_case.app.run();ui_case.healthy()
    assert next(item for item in ui_case.app.text_input if item.label=="Father's name").value=='Example Parent'
    # Recreate the app/session; restore the same SQLite thread explicitly.
    reloaded=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),default_timeout=25)
    reloaded.run();choose(reloaded,'text_input','Profile name','demo-user')
    next(b for b in reloaded.button if b.label=='Open my case').click();reloaded.run()
    reloaded.session_state['view_'+ui_case.key]='application';reloaded.run()
    assert not reloaded.exception
    assert next(item for item in reloaded.text_input if item.label=="Father's name").value=='Example Parent'
    assert next(item for item in reloaded.text_input if item.label=='Birth date').value=='not-a-date'

def test_document_save_explicitly_confirms_unresolved_entry_date_only(ui_case, ready_bundle):
    from test_procedure import override

    state = entered(ready_bundle[2])
    entry = next(doc for doc in state["documents"] if doc.doc_type == "turkiye_entry_stamp")
    state = override(state, entry.doc_id, "latest_entry_date", None)
    ui_case.seed(state)
    ui_case.nav("documents")
    assert "5 of 6 items ready for this step" in ui_case.text()
    assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)
    assert any(button.label == "Save and confirm" for button in ui_case.app.button)
    assert not any("entered information is accurate" in box.label for box in ui_case.app.checkbox)
    accepted_before = deepcopy(ui_case.read()["verification_records"])
    assert ui_case.read()["residence_permit_context"].latest_entry_date is None
    choose(ui_case.app, "text_input", "Date of entry into Türkiye (YYYY-MM-DD)", "2026-09-15")
    submit = next(button for button in ui_case.app.button
                  if button.label == "Save and confirm" and entry.doc_id in button.key)
    submit.click()
    ui_case.app.run()
    ui_case.healthy()
    result = ui_case.read()
    assert result["event"].confirmed_fields == {"latest_entry_date": "2026-09-15"}
    assert str(result["residence_permit_context"].latest_entry_date) == "2026-09-15"
    assert result["verification_records"][entry.doc_id]["latest_entry_date"].origin == "user_confirmed"
    assert next(doc for doc in result["documents"] if doc.doc_id == entry.doc_id).info == entry.info
    assert "Your changes were saved and confirmed" in ui_case.text()
    assert "Ready" in ui_case.text()
    assert result["procedure"].current_stage == "DOCUMENT_COLLECTION"
    assert any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)
    for doc_id, entries in accepted_before.items():
        if doc_id.startswith("__") or doc_id == entry.doc_id:
            continue
        after = result["verification_records"][doc_id]
        assert after.keys() == entries.keys()
        for name, before in entries.items():
            assert before.model_dump(exclude={"trace_id"}) == after[name].model_dump(exclude={"trace_id"})

    choose(ui_case.app, "toggle", "Document options", True)
    ui_case.app.run()
    ui_case.click("Use document reading for date of entry into türkiye")
    assert "5 of 6 items ready for this step" in ui_case.text()
    assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)
    choose(ui_case.app, "text_input", "Date of entry into Türkiye (YYYY-MM-DD)", "2026-09-15")
    ui_case.click("Save and confirm")

    # The action is case-level and remains available without a selected file.
    choose(ui_case.app, "selectbox", "Saved document", None)
    ui_case.app.run()
    ui_case.healthy()
    assert sum(button.label == "Continue to e-Ikamet" for button in ui_case.app.button) == 1
    ui_case.click("Continue to e-Ikamet")
    assert ui_case.view() == "application"
    assert ui_case.read()["procedure"].current_stage == "EIKAMET_PREPARATION"
    ui_case.click("Documents")
    ui_case.click("Overview")


def test_entry_stamp_draft_is_isolated_and_only_confirmation_survives_restart(ui_case, ready_bundle):
    from test_procedure import override

    state = entered(ready_bundle[2])
    stamp = next(doc for doc in state["documents"] if doc.doc_type == "turkiye_entry_stamp")
    passport = next(doc for doc in state["documents"] if doc.doc_type == "passport_copy")
    state = override(state, stamp.doc_id, "entry_stamp_present", None)
    state = override(state, stamp.doc_id, "latest_entry_date", None)
    ui_case.seed(state)
    ui_case.nav("documents")
    trace = ui_case.read()["trace_id"]
    date_label = "Date of entry into Türkiye (YYYY-MM-DD)"
    presence_label = "Entry stamp visible"
    choose(ui_case.app, "text_input", date_label, "2026-09-31")
    choose(ui_case.app, "selectbox", presence_label, "Yes")
    ui_case.app.run()
    assert next(item.value for item in ui_case.app.text_input if item.label == date_label) == "2026-09-31"
    assert ui_case.read()["trace_id"] == trace
    ui_case.click("Overview")
    ui_case.click("Documents")
    choose(ui_case.app, "selectbox", "Saved document", passport.doc_id)
    ui_case.app.run()
    assert not any(item.label == date_label for item in ui_case.app.text_input)
    choose(ui_case.app, "checkbox", "Edit accepted fields", True)
    ui_case.app.run()
    choose(ui_case.app, "text_input", "Passport number", "LOCAL-DRAFT")
    ui_case.app.run()
    ui_case.click("Discard draft")
    assert next(item.value for item in ui_case.app.text_input if item.label == "Passport number") == "P1234567"
    choose(ui_case.app, "selectbox", "Saved document", stamp.doc_id)
    ui_case.app.run()
    assert next(item.value for item in ui_case.app.text_input if item.label == date_label) == "2026-09-31"
    assert next(item.value for item in ui_case.app.selectbox if item.label == presence_label) == "Yes"

    ui_case.click("Save and confirm")
    assert ui_case.read()["validation_errors"]
    assert next(item.value for item in ui_case.app.text_input if item.label == date_label) == "2026-09-31"
    assert next(item.value for item in ui_case.app.selectbox if item.label == presence_label) == "Yes"
    choose(ui_case.app, "text_input", date_label, "2026-09-16")
    ui_case.click("Save and confirm")
    saved = ui_case.read()
    assert not saved["validation_errors"]
    assert str(saved["residence_permit_context"].latest_entry_date) == "2026-09-16"
    assert saved["residence_permit_context"].entry_stamp_present is True
    entries = saved["verification_records"][stamp.doc_id]
    assert entries["latest_entry_date"].value == "2026-09-16"
    assert entries["latest_entry_date"].origin == "user_confirmed"
    assert next(doc for doc in saved["documents"] if doc.doc_id == stamp.doc_id).info == stamp.info
    assert "Your changes were saved and confirmed." in ui_case.text()

    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reopened.run()
    choose(reopened, "text_input", "Profile name", "demo-user")
    next(button for button in reopened.button if button.label == "Open my case").click()
    reopened.run()
    next(button for button in reopened.button if button.label == "Documents").click()
    reopened.run()
    choose(reopened, "selectbox", "Saved document", stamp.doc_id)
    reopened.run()
    choose(reopened, "checkbox", "Edit accepted fields", True)
    reopened.run()
    assert not reopened.exception
    assert next(item.value for item in reopened.text_input if item.label == date_label) == "2026-09-16"
    assert next(item.value for item in reopened.selectbox if item.label == presence_label) == "Yes"


def test_unknown_pdf_type_drafts_survive_reclassification_navigation_and_reprocessing(ui_case, monkeypatch):
    import pymupdf
    from src.ui import components
    from test_milestone_documents import response

    directory = ui_case.tmp_path / "uploads"
    upload_path = components.upload_path
    save_upload = components.save_upload
    monkeypatch.setattr(components, "PROJECT_ROOT", ui_case.tmp_path)
    monkeypatch.setattr(components, "upload_path",
                        lambda name, data, thread, *args: upload_path(name, data, thread, directory))
    monkeypatch.setattr("src.ui.documents.save_upload",
                        lambda name, data, thread: save_upload(name, data, thread, directory))
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Synthetic document with fields requiring manual review")
        content = pdf.tobytes()
    upload = SimpleNamespace(name="unknown.pdf", getvalue=lambda: content)
    ui_case.app.run()
    ui_case.click("I'm starting from the beginning")
    ui_case.click("Documents")
    with patch("src.ui.documents.st.file_uploader", return_value=upload), patch(
        "src.document_processing.processor.Client"
    ) as client, patch("src.document_processing.processor.checked_classification", return_value="unknown"):
        client.return_value.chat.return_value = response({"document_type": "unknown"})
        ui_case.click("Upload and analyze")
    raw = ui_case.read()
    doc = raw["documents"][0]
    assert doc.doc_type == "unknown"
    assert not any(button.label == "Save and confirm" for button in ui_case.app.button)

    type_label = "Document type for reprocessing"
    choose(ui_case.app, "selectbox", type_label, "passport_copy")
    ui_case.app.run()
    choose(ui_case.app, "text_input", "Passport number", "SYNTH123")
    choose(ui_case.app, "text_input", "Passport expiry date (YYYY-MM-DD)", "2035-07-25")
    ui_case.app.run()
    choose(ui_case.app, "selectbox", type_label, "address_form")
    ui_case.app.run()
    assert "Draft entries kept from another document type" in ui_case.text()
    assert next(item.value for item in ui_case.app.text_input if item.label == "Passport number") == "SYNTH123"
    ui_case.click("Overview")
    ui_case.click("Documents")
    assert ui_case.read()["trace_id"] == raw["trace_id"]
    assert ui_case.view() == "documents"
    selected = next(item for item in ui_case.app.selectbox if item.label == "Saved document")
    assert selected.value == doc.doc_id, ui_case.text()
    assert next(item.value for item in ui_case.app.selectbox if item.label == type_label) == "address_form"
    assert next(item.value for item in ui_case.app.text_input if item.label == "Passport number") == "SYNTH123"

    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value = response({})
        ui_case.click("Reprocess document")
    assert ui_case.read()["documents"][0].doc_type == "address_form"
    assert next(item.value for item in ui_case.app.text_input if item.label == "Passport number") == "SYNTH123"
    choose(ui_case.app, "text_input", "Declared address", "Synthetic Street 23")
    ui_case.click("Save and confirm")
    accepted = ui_case.read()
    assert accepted["documents"][0].info.declared_address is None
    assert accepted["verification_records"][doc.doc_id]["passport_number"].value == "SYNTH123"
    correction = accepted["verification_records"][doc.doc_id]["declared_address"]
    assert correction.origin == "user_confirmed" and correction.value == "Synthetic Street 23"

    choose(ui_case.app, "toggle", "Document options", True)
    ui_case.app.run()
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value = response({})
        ui_case.click("Reprocess document")
    assert ui_case.read()["verification_records"][doc.doc_id]["declared_address"] == correction
    assert "Draft entries kept from another document type" in ui_case.text()

    choose(ui_case.app, "selectbox", type_label, "passport_copy")
    ui_case.app.run()
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value = response({})
        ui_case.click("Reprocess document")
    reclassified = ui_case.read()
    assert reclassified["documents"][0].doc_type == "passport_copy"
    assert all(entry.origin != "user_confirmed" for entry in reclassified["verification_records"][doc.doc_id].values())
    assert reclassified["residence_permit_context"].passport_or_foreign_id is None
    assert next(item.value for item in ui_case.app.text_input if item.label == "Passport number") == "SYNTH123"
    assert "Declared address" in ui_case.text()


def test_final_address_photo_confirmation_unlocks_and_revising_it_locks_again(ui_case, ready_bundle):
    from test_procedure import override

    state = entered(override(ready_bundle[2], "__declarations__", "biometric_photo", False))
    ui_case.seed(state)
    ui_case.nav("details")
    assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)

    choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "Yes")
    ui_case.click("Save and confirm preparation answers")
    assert sum(button.label == "Continue to e-Ikamet" for button in ui_case.app.button) == 1
    assert ui_case.view() == "details"
    assert ui_case.read()["procedure"].current_stage == "DOCUMENT_COLLECTION"

    reloaded = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reloaded.run()
    choose(reloaded, "text_input", "Profile name", "demo-user")
    next(button for button in reloaded.button if button.label == "Open my case").click()
    reloaded.run()
    next(button for button in reloaded.button if button.label == "Documents").click()
    reloaded.run()
    assert not reloaded.exception
    assert sum(button.label == "Continue to e-Ikamet" for button in reloaded.button) == 1

    choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "No")
    ui_case.click("Save and confirm preparation answers")
    ui_case.click("Documents")
    assert "5 of 6 items ready for this step" in ui_case.text()
    assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)


def test_identity_conflict_requires_a_deliberately_selected_value(ui_case, ready_bundle):
    from test_procedure import override

    state = entered(ready_bundle[2])
    insurance = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
    state = override(state, insurance.doc_id, "holder_name", "Other Synthetic Name")
    ui_case.seed(state)
    ui_case.nav("documents")
    selection = next(box for box in ui_case.app.selectbox if box.label == "Name to use for your case")
    assert selection.value is None
    before = ui_case.read()
    ui_case.click("Confirm chosen identity")
    assert "Choose the correct name" in ui_case.text()
    assert ui_case.read()["verification_conflicts"] == before["verification_conflicts"]
    assert ui_case.read()["trace_id"] == before["trace_id"]
    assert not any("explicitly resolve" in box.label for box in ui_case.app.checkbox)
    choose(ui_case.app, "selectbox", "Name to use for your case", 0)
    ui_case.click("Confirm chosen identity")
    result = ui_case.read()
    assert result["event"].kind == "resolve_conflict"
    assert "holder_name" not in result["verification_conflicts"]


def test_details_batches_changed_answers_and_ignores_unchanged_resubmission(ui_case):
    from src.core.state import ProcedureState

    ui_case.seed(entered({"procedure": ProcedureState(), "user_id": "demo-user", "trace_id": "details"}))
    ui_case.nav("details")
    ui_case.details("entry")
    choose(ui_case.app, "selectbox", "Are you currently in Türkiye?", "Yes")
    choose(ui_case.app, "selectbox", "Did you enter Türkiye using your passport?", "Yes")
    choose(ui_case.app, "text_input", "Planned application date (YYYY-MM-DD)", "2026-12-01")
    ui_case.click("Save and confirm answers")
    state = ui_case.read()
    assert state["event"].confirmed_fields == {
        "is_in_turkiye": True, "entered_with_passport": True, "application_date": "2026-12-01",
    }
    ui_case.click("Save and confirm answers")
    assert ui_case.read()["trace_id"] == state["trace_id"]
    assert "already up to date" in ui_case.text()
    ui_case.details("address")
    choose(ui_case.app, "selectbox", "Address-proof route", "dorm_certificate")
    choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "Yes")
    ui_case.click("Save and confirm preparation answers")
    assert ui_case.read()["event"].confirmed_fields == {
        "address_proof_type": "dorm_certificate", "biometric_photo": True,
    }


def test_new_upload_opens_its_review_and_is_not_analyzed_twice(ui_case, ready_bundle, monkeypatch):
    from src.ui.components import upload_path
    from test_milestone_documents import SAMPLES, response
    from test_procedure import drop_document

    initial = entered(drop_document(ready_bundle[2], "health_insurance"))
    passport = next(doc for doc in initial["documents"] if doc.doc_type == "passport_copy")
    ui_case.seed(initial)
    data = SAMPLES["health_insurance_certificate"].encode()
    upload = SimpleNamespace(name="synthetic-insurance.txt", getvalue=lambda: data)
    directory = ui_case.tmp_path / "uploads"
    monkeypatch.setattr("src.ui.components.upload_path",
                        lambda name, content, thread, *args: upload_path(name, content, thread, directory))
    with patch("src.ui.documents.st.file_uploader", return_value=upload), patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect = [response({"document_type": "health_insurance_certificate"}), response({})]
        ui_case.nav("documents")
        choose(ui_case.app, "selectbox", "Saved document", passport.doc_id)
        ui_case.app.run()
        assert not any(toggle.label == "Upload another document" for toggle in ui_case.app.toggle)
        ui_case.click("Upload and analyze")
        state = ui_case.read()
        added = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
        assert added.status == "analyzed"
        assert len(state["documents"]) == len(initial["documents"]) + 1
        assert next(item for item in ui_case.app.selectbox if item.label == "Saved document").value == added.doc_id
        assert "Key information" in ui_case.text()
        ui_case.click("Upload and analyze")
        assert "already in Documents" in ui_case.text()
        assert client.return_value.chat.call_count == 2
    assert ui_case.read()["trace_id"] == state["trace_id"]


def test_copilot_navigation_preserves_one_turn_without_model_calls(ui_case, ready_bundle):
    ui_case.seed(entered(ready_bundle[2]))
    ui_case.nav("copilot")
    with patch("ollama.Client.chat", side_effect=AssertionError("A saved-case question and UI rerender need no model")) as transport:
        ui_case.click("What should I do now?")
        messages = deepcopy(ui_case.read()["assistant_messages"])
        ui_case.app.run()
        ui_case.nav("checks")
        ui_case.nav("copilot")
        transport.assert_not_called()
    assert ui_case.read()["assistant_messages"] == messages
    assert len(messages) == 2
    assert len(ui_case.app.chat_input) == 1


def test_advanced_renders_only_the_selected_diagnostic(ui_case, ready_bundle):
    from src.ui import technical

    ui_case.seed(entered(ready_bundle[2]))
    with patch("src.ui.technical.TwinStore", wraps=technical.TwinStore) as store:
        ui_case.nav("technical")
        ui_case.healthy()
        store.assert_not_called()
        assert not ui_case.app.get("json")
        assert not ui_case.app.chat_input
        assert not any(button.label == "Compare snapshot" for button in ui_case.app.button)

        choose(ui_case.app, "selectbox", "Advanced", "Saved state and document evidence")
        ui_case.app.run()
        ui_case.healthy()
        store.assert_not_called()
        assert len(ui_case.app.get("json")) == 1

        choose(ui_case.app, "selectbox", "Advanced", "Saved run history")
        ui_case.app.run()
        ui_case.healthy()
        store.assert_called_once()
        assert len(ui_case.app.get("json")) == 2


def test_next_steps_shares_its_current_checklist_projection(ui_case):
    from src.core import procedure
    from src.core.state import ProcedureState

    ui_case.seed(entered({"procedure": ProcedureState(), "user_id": "demo-user", "trace_id": "status"}))
    with patch("src.core.procedure.stage_document_checklist", wraps=procedure.stage_document_checklist) as checklist:
        ui_case.nav("next")
        ui_case.healthy()
    assert checklist.call_count == 1
    assert "Upload passport identity page" in [button.label for button in ui_case.app.button]
    assert "Still needed for this step" not in ui_case.text()
    assert "Step checklist" not in [item.label for item in ui_case.app.get("expander")]
    assert not any(button.label in {"Build checklist", "Prepare email draft"} for button in ui_case.app.button)


def test_submission_and_tracking_record_only_explicit_updates(ui_case, ready_bundle):
    from src.core.state import ProcedureState

    state = entered(ready_bundle[2])
    state["procedure"] = ProcedureState(current_stage="SUBMISSION")
    ui_case.seed(state)
    ui_case.nav("application")
    original_trace = ui_case.read()["trace_id"]
    ui_case.click("Save submission and continue")
    assert ui_case.read()["trace_id"] == original_trace

    choose(ui_case.app, "selectbox", "Have you physically submitted the package to IRO?", "Not yet")
    ui_case.click("Save submission and continue")
    state = ui_case.read()
    assert state["procedure"].current_stage == "SUBMISSION"
    assert state["residence_permit_context"].submitted_to_iro is False

    choose(ui_case.app, "selectbox", "Have you physically submitted the package to IRO?", "Yes")
    ui_case.click("Save submission and continue")
    state = ui_case.read()
    assert state["procedure"].current_stage == "TRACKING"
    assert state["residence_permit_context"].submitted_to_iro is True

    for choice, received in (("waiting", False), ("received", True)):
        choose(ui_case.app, "selectbox", "What decision have you received?", choice)
        ui_case.click("Save tracking update")
        state = ui_case.read()
        assert state["procedure"].current_stage == "TRACKING"
        assert state["residence_permit_context"].official_decision_received is received
        assert state["residence_permit_context"].approval_status not in {"approved", "rejected"}

    choose(ui_case.app, "selectbox", "What decision have you received?", "rejected")
    ui_case.click("Save tracking update")
    state = ui_case.read()
    assert state["procedure"].current_stage == "COMPLETE"
    assert state["residence_permit_context"].approval_status == "rejected"


def test_save_and_continue_respects_conflicts_and_reuses_the_saved_answer(ui_case, ready_bundle):
    from src.core.state import ProcedureState
    from test_procedure import override

    state = entered(ready_bundle[2])
    insurance = next(doc for doc in state["documents"] if doc.doc_type == "health_insurance")
    state = override(state, insurance.doc_id, "holder_name", "Other Synthetic Name")
    state["procedure"] = ProcedureState(current_stage="SUBMISSION")
    ui_case.seed(state)
    ui_case.nav("application")
    choose(ui_case.app, "selectbox", "Have you physically submitted the package to IRO?", "Yes")
    ui_case.click("Save submission and continue")
    saved = ui_case.read()
    assert saved["residence_permit_context"].submitted_to_iro is True
    assert saved["procedure"].current_stage == "SUBMISSION"
    assert "holder_name" in saved["verification_conflicts"]

    ui_case.click("Documents")
    choose(ui_case.app, "selectbox", "Name to use for your case", 0)
    ui_case.click("Confirm chosen identity")
    assert "holder_name" not in ui_case.read()["verification_conflicts"]
    resolved_trace = ui_case.read()["trace_id"]
    ui_case.click("Open current step")
    ui_case.click("Save submission and continue")
    assert ui_case.read()["procedure"].current_stage == "TRACKING"
    assert ui_case.read()["trace_id"] == resolved_trace


def test_synthetic_student_prepares_the_iro_package_and_reopens_completed_case(ui_case, monkeypatch):
    import src.ui.components as components
    from src.core.address_proof import REVIEW_LABELS
    from src.core.procedure import iro_package_readiness
    from src.evaluation.scenarios import SYNTHETIC
    from test_milestone_documents import SAMPLES, response

    directory = ui_case.tmp_path / "uploads"
    upload_path = components.upload_path
    save_upload = components.save_upload
    monkeypatch.setattr(components, "PROJECT_ROOT", ui_case.tmp_path)
    monkeypatch.setattr(components, "upload_path",
                        lambda name, data, thread, *args: upload_path(name, data, thread, directory))
    for module in ("documents", "application"):
        monkeypatch.setattr(f"src.ui.{module}.save_upload",
                            lambda name, data, thread: save_upload(name, data, thread, directory))

    def upload(kind, button="Upload and analyze", *, content=None):
        data = (content or SAMPLES[kind]).encode()
        file = SimpleNamespace(name=f"{kind}.txt", getvalue=lambda: data)
        with patch("src.ui.components.st.file_uploader", return_value=file), patch(
            "src.document_processing.processor.Client"
        ) as client:
            client.return_value.chat.side_effect = [response({"document_type": kind}), response({})]
            ui_case.click(button)
        if kind == "dorm_certificate":
            client.return_value.chat.assert_not_called()
        else:
            assert 1 <= client.return_value.chat.call_count <= 2

    del ui_case.app.session_state["workspace_profile_id"]
    ui_case.app.run()
    choose(ui_case.app, "text_input", "Profile name", "demo-user")
    ui_case.click("Open my case")
    ui_case.click("I'm starting from the beginning")

    ui_case.click("Upload passport identity page")
    passport = SAMPLES["passport_identity_page"].replace("Nationality: Exampleland", "")
    upload("passport_identity_page", content=passport)
    choose(ui_case.app, "text_input", "Nationality", "Exampleland")
    ui_case.click("Save and confirm")
    assert ui_case.read()["residence_permit_context"].nationality == "Exampleland"

    upload("turkiye_entry_stamp")
    stamp = next(doc for doc in ui_case.read()["documents"] if doc.doc_type == "turkiye_entry_stamp")
    assert ui_case.read()["residence_permit_context"].latest_entry_date is None
    choose(ui_case.app, "text_input", "Date of entry into Türkiye (YYYY-MM-DD)", "2026-09-15")
    ui_case.click("Save and confirm")
    assert ui_case.read()["verification_records"][stamp.doc_id]["latest_entry_date"].origin == "user_confirmed"

    upload("student_enrollment_certificate")
    upload("health_insurance_certificate")
    choose(ui_case.app, "selectbox", "Insurance category", "private_health_insurance")
    ui_case.click("Save and confirm")

    ui_case.click("Your details")
    choose(ui_case.app, "selectbox", "Address-proof route", "dorm_certificate")
    choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "Yes")
    ui_case.click("Save and confirm preparation answers")
    ui_case.click("Upload Dormitory certificate")
    upload("dorm_certificate", content=SYNTHETIC["dorm_certificate"].replace("Evaluation Student", "Alex Example"))
    choose(ui_case.app, "selectbox", REVIEW_LABELS["dorm_certificate_format"], "Yes")
    ui_case.click("Save and confirm")
    ui_case.click("Overview")
    ui_case.click("Continue to e-Ikamet")
    ui_case.click("Yes - I have the application form")
    upload("ikamet_application_form", "Analyze application form")
    for box in ui_case.app.selectbox:
        if "online application" in box.label.lower() or "signed" in box.label.lower():
            box.set_value("Yes")
    ui_case.click("Save and confirm")
    ui_case.click("Continue to Payment & Fingerprint")

    ui_case.click("Upload payment receipt")
    upload("card_fee_receipt")
    assert ui_case.view() == "documents"
    assert sum(button.label == "Continue to Fingerprint / GI" for button in ui_case.app.button) == 1
    ui_case.click("Continue to Fingerprint / GI")
    choose(ui_case.app, "selectbox", "Have you completed fingerprinting?", "Yes")
    choose(ui_case.app, "selectbox", "Have you received a GI number?", "Yes")
    ui_case.click("Save fingerprint / GI and continue")
    assert ui_case.read()["procedure"].current_stage == "IRO_REVIEW"

    ui_case.click("Documents")
    upload("declaration_control_document")
    ui_case.click("Open current step")
    state = ui_case.read()
    package = iro_package_readiness(state)
    assert package["ready"], package["blockers"]
    assert len(package["attachment_ids"]) == 8
    certificate = next(doc for doc in state["documents"] if doc.doc_type == "dorm_certificate")
    assert certificate.doc_id in package["attachment_ids"]
    ui_case.click("Prepare email draft")
    ui_case.click("Reject this action")
    assert not ui_case.read().get("email_drafts")
    ui_case.click("Prepare email draft")
    ui_case.click("Approve this action")
    state = ui_case.read()
    assert state["email_drafts"]
    assert state["residence_permit_context"].scans_sent_to_iro is not True

    choose(ui_case.app, "selectbox", "Have you sent your scans to IRO?", "Yes")
    choose(ui_case.app, "selectbox", "Has IRO confirmed its review?", "Yes")
    ui_case.click("Save IRO progress and continue")
    assert ui_case.read()["procedure"].current_stage == "SUBMISSION"
    choose(ui_case.app, "selectbox", "Have you physically submitted the package to IRO?", "Yes")
    ui_case.click("Save submission and continue")
    choose(ui_case.app, "selectbox", "What decision have you received?", "approved")
    ui_case.click("Save tracking update")
    state = ui_case.read()
    assert state["procedure"].current_stage == "COMPLETE"
    assert state["residence_permit_context"].approval_status == "approved"
    assert all(Path(doc.file_path).resolve().is_relative_to(directory) for doc in state["documents"])

    ui_case.click("Switch profile")
    choose(ui_case.app, "text_input", "Profile name", "demo-user")
    ui_case.click("Open my case")
    ui_case.click("Open current step")
    assert "Your reported official decision is recorded" in ui_case.text()
    assert ui_case.read()["procedure"] == state["procedure"]


def test_landing_waits_for_a_profile_before_opening_checkpoints(ui_case):
    del ui_case.app.session_state["workspace_profile_id"]
    with patch("src.memory.checkpointer.sqlite_checkpointer", side_effect=AssertionError("The empty landing needs no database")) as saver:
        ui_case.app.run()
        ui_case.healthy()
        saver.assert_not_called()
    assert any(button.label == "Open my case" for button in ui_case.app.button)


def test_saved_address_conflict_shows_full_provenance_and_clears_in_one_save(ui_case, ready_bundle):
    from test_document_integrity import address_example, add_evidence, COMBINED_ROUTE, PASSPORT_FIELD
    from test_procedure import override

    state = entered(ready_bundle[2])
    address = address_example(state)
    add_evidence(state, "utility_bill")
    state = override(state, "__declarations__", "address_proof_type", COMBINED_ROUTE)
    state = override(state, address.doc_id, "phone", "05550000042")
    ui_case.seed(state)

    # Start a separate AppTest session and open the same SQLite checkpoint.
    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    with patch("ollama.Client.chat", side_effect=AssertionError("Rendering must not call Ollama")) as chat:
        reopened.run()
        choose(reopened, "text_input", "Profile name", "demo-user")
        next(button for button in reopened.button if button.label == "Open my case").click()
        reopened.run()
        reopened.session_state[f"view_{ui_case.key}"] = "documents"
        reopened.run()
        assert not reopened.exception
        message = "\n".join(row.value for row in reopened.warning)
        assert "Passport number conflict between Passport identity page and Address Form" in message
        sources = "\n".join(row.value for row in reopened.markdown)
        assert "Passport identity page \u00b7 P1234567" in sources
        assert "Address Form \u00b7 P7654321" in sources
        assert not any(button.label == "Continue to e-Ikamet" for button in reopened.button)

        next(button for button in reopened.button if button.label == "Review Address Form").click()
        reopened.run()
        assert not reopened.exception
        assert not next(box for box in reopened.checkbox if box.label == "Edit accepted fields").value
        assert not next(toggle for toggle in reopened.toggle if toggle.label == "View extraction evidence").value
        details = next(panel for panel in reopened.get("expander") if panel.label == "View all information")
        structured = "".join(row.proto.body for row in details.get("html"))
        for expected in ("05550000042", "05551234567", "Confirmed by you", "Original reading", "Birth date",
                         "Not recorded", "Exampleland", "Synthetic Street 10", "Form date", "Page 1", "P7654321"):
            assert expected in structured
        passport = next(box for box in reopened.text_input if box.label == "Passport number")
        assert passport.value == "P7654321"
        passport.set_value("P1234567")
        next(button for button in reopened.button if button.label == "Save and confirm").click()
        reopened.run()
        assert not reopened.exception
        assert not any("Passport number conflict" in row.value for row in reopened.warning)
        assert any(button.label == "Continue to e-Ikamet" for button in reopened.button)
        assert not chat.called

    saved = ui_case.read()
    assert PASSPORT_FIELD not in saved["verification_conflicts"]
    assert saved["verification_records"][address.doc_id]["passport_number"].value == "P1234567"
    assert saved["verification_records"][address.doc_id]["phone"].value == "05550000042"
    assert next(item for item in saved["documents"] if item.doc_id == address.doc_id).info.passport_number == "P7654321"


def test_current_address_requirement_shows_both_parts_without_hiding_other_saved_files(ui_case, ready_bundle):
    from test_document_integrity import address_example, COMBINED_ROUTE
    from test_procedure import override

    state = entered(ready_bundle[2])
    address = address_example(state, "P1234567")
    combined = override(state, "__declarations__", "address_proof_type", COMBINED_ROUTE)
    ui_case.seed(combined)
    ui_case.nav("documents")
    assert any(button.label == "Utility bill" for button in ui_case.app.button)
    ui_case.click("Address Form")
    assert "Required for this step" in ui_case.text()
    assert "Ready" in ui_case.text() and "Missing" in ui_case.text()
    assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)
    assert next(box for box in ui_case.app.selectbox if box.label == "Saved document").value == address.doc_id

    independent = override(combined, "__declarations__", "address_proof_type", "dorm_certificate")
    ui_case.seed(independent)
    ui_case.nav("documents")
    assert not any(button.label in {"Address Form", "Utility bill"} for button in ui_case.app.button)
    assert next(box for box in ui_case.app.selectbox if box.label == "Saved document").value == address.doc_id
    assert any(panel.label == "View all information" for panel in ui_case.app.get("expander"))


def test_document_conflict_shows_each_source_and_links_to_saved_answer(ui_case, ready_bundle):
    from test_procedure import override

    state = entered(ready_bundle[2])
    original = str(state["residence_permit_context"].passport_expiry_date)
    state = override(state, "__declarations__", "passport_expiry_date", "2031-02-01")
    ui_case.seed(state)
    ui_case.nav("documents")
    assert "Your details · 01 Feb 2031" in ui_case.text()
    assert "Passport identity page ·" in ui_case.text()
    assert ui_case.read()["residence_permit_context"].passport_expiry_date is None
    ui_case.click("Review your saved answers")
    assert ui_case.view() == "details"
    choose(ui_case.app, "text_input", "Passport expiry date (YYYY-MM-DD)", original)
    ui_case.click("Save and confirm answers")
    assert str(ui_case.read()["residence_permit_context"].passport_expiry_date) == original
    assert "passport_expiry_date" not in ui_case.read()["verification_conflicts"]


def test_document_review_focus_and_original_evidence_are_read_only(ui_case, ready_bundle):
    from src.ui import documents
    from test_procedure import override

    state = entered(ready_bundle[2])
    entry = next(doc for doc in state["documents"] if doc.doc_type == "turkiye_entry_stamp")
    state = override(state, entry.doc_id, "latest_entry_date", None)
    ui_case.seed(state)
    before = ui_case.read()
    ui_case.nav("next")
    with patch("src.ui.documents.document_card", wraps=documents.document_card) as card:
        ui_case.click("Review latest entry-stamp page")
        assert card.call_count == 1
        assert card.call_args.args[1].doc_id == entry.doc_id
    assert "Required for this step" in ui_case.text()
    assert "Needs review" in ui_case.text()
    assert "Ready" in ui_case.text()
    evidence = entry.info.source_evidence[0]
    assert evidence.snippet not in [item.value for item in ui_case.app.text]
    choose(ui_case.app, "toggle", "View extraction evidence", True)
    ui_case.app.run()
    ui_case.healthy()
    assert f"Page {evidence.page_number}" in ui_case.text()
    assert evidence.snippet in [item.value for item in ui_case.app.text]
    assert ui_case.read() == before


def test_checks_keep_coverage_advisory_and_load_rules_on_request(ui_case, ready_bundle):
    from src.rag import ingestion
    from src.core.legal_stay import INSURANCE_PREPARATION_ADVISORY

    state = entered(ready_bundle[2])
    context = state["residence_permit_context"]
    state["residence_permit_context"] = context.model_copy(update={"insurance_end_date": context.program_start_date})
    state["relevant_requirements"] = [
        rule.model_copy(update={"passed": False, "status": "failed"}) if rule.rule_id == "R015" else rule
        for rule in state["relevant_requirements"]
    ]
    ui_case.seed(state)
    with patch("src.rag.ingestion.source_catalogue", wraps=ingestion.source_catalogue) as catalogue:
        ui_case.nav("checks")
        ui_case.healthy()
        assert INSURANCE_PREPARATION_ADVISORY in ui_case.text()
        assert "R015" not in ui_case.text()
        assert not ui_case.app.get("json")
        catalogue.assert_not_called()
        ui_case.click("See rule details and sources")
        catalogue.assert_called_once()
        assert ui_case.view() == "technical"
        assert any("R015" in item.value for item in ui_case.app.get("json"))
    assert next(rule for rule in ui_case.read()["relevant_requirements"] if rule.rule_id == "R015").passed is False


def test_copilot_plan_shortcut_opens_one_plan_workspace_without_running_planner(ui_case, ready_bundle):
    state = entered(ready_bundle[2])
    state["assistant_diagnostics"] = {"route": "plan_request"}
    ui_case.seed(state)
    with patch("src.agents.planner.create_action_plan", side_effect=AssertionError("Navigation must not generate a plan")) as plan:
        ui_case.nav("copilot")
        ui_case.click("Open detailed action plan")
        assert ui_case.view() == "next"
        assert len([button for button in ui_case.app.button if button.label == "Generate plan"]) == 1
        plan.assert_not_called()


@pytest.mark.parametrize(("kinds", "next_label"), [
    ((), "Upload signed Address Form"),
    (("utility_bill",), "Upload signed Address Form"),
    (("address_form",), "Upload utility bill"),
    (("address_form", "utility_bill"), "Review photo readiness"),
])
def test_combined_address_option_saves_and_guides_required_evidence(ui_case, ready_bundle, kinds, next_label):
    from src.core.procedure import address_proof_status
    from test_document_integrity import address_example, add_evidence
    from test_procedure import override

    state = entered(ready_bundle[2])
    state = override(state, "__declarations__", "address_proof_type", None)
    state = override(state, "__declarations__", "biometric_photo", None)
    if "address_form" in kinds:
        address_example(state, state["residence_permit_context"].passport_or_foreign_id)
    if "utility_bill" in kinds:
        add_evidence(state, "utility_bill")
    ui_case.seed(state)
    ui_case.nav("details")
    choose(ui_case.app, "selectbox", "Address-proof route", "utility_bill_and_address_form")
    ui_case.click("Save and confirm preparation answers")
    saved = ui_case.read()
    assert saved["event"].confirmed_fields == {"address_proof_type": "utility_bill_and_address_form"}
    assert saved["residence_permit_context"].biometric_photo is None
    entry = saved["verification_records"]["__declarations__"]["address_proof_type"]
    assert entry.origin == "user_confirmed"
    assert entry.value == "utility_bill_and_address_form"
    assert next(box for box in ui_case.app.selectbox if box.label == "Address-proof route").value == entry.value
    assert "Saved address option" in ui_case.text()
    address = address_proof_status(saved)
    assert address["route"] == "utility_bill_and_address_form"
    assert address["ready"] is (len(kinds) == 2)
    rule = next(rule for rule in saved["relevant_requirements"] if rule.rule_id == "R017")
    assert (rule.passed is True) is address["ready"]
    for part in address["parts"]:
        label = "Ready" if part["document_type"] in kinds else "Missing"
        assert f"{part['label']} · {label}" in ui_case.text()

    ui_case.click("Save and confirm preparation answers")
    assert "already saved" in ui_case.text()
    assert ui_case.read()["trace_id"] == saved["trace_id"]
    ui_case.nav("next")
    if len(kinds) < 2:
        ui_case.click(next_label)
        focus = "Address Form" if "address_form" not in kinds else "Utility bill"
        assert f"Working on {focus}" in ui_case.text()
    else:
        assert not any(button.label == "Continue to e-Ikamet" for button in ui_case.app.button)
        ui_case.click(next_label)
        choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "Yes")
        ui_case.click("Save and confirm preparation answers")
        assert ui_case.read()["event"].confirmed_fields == {"biometric_photo": True}
        assert "Biometric photo ready" in ui_case.text()


@pytest.mark.parametrize("option", ["dorm_certificate", "rental_agreement", "notarized_commitment"])
def test_address_option_can_be_changed_after_saving(ui_case, option):
    from src.core.procedure import address_proof_status
    from src.core.state import ProcedureState

    ui_case.seed(entered({"procedure": ProcedureState(), "user_id": "demo-user", "trace_id": "form"}))
    ui_case.nav("details")
    choose(ui_case.app, "selectbox", "Address-proof route", "utility_bill_and_address_form")
    ui_case.click("Save and confirm preparation answers")
    assert not address_proof_status(ui_case.read())["ready"]
    choose(ui_case.app, "selectbox", "Address-proof route", option)
    ui_case.click("Save and confirm preparation answers")
    state = ui_case.read()
    assert state["residence_permit_context"].address_proof_type == option
    assert not address_proof_status(state)["ready"]
    assert f"{address_proof_status(state)['label']} · Missing" in ui_case.text()
    assert not any(button.label == "Upload signed Address Form" for button in ui_case.app.button)


def test_address_method_switch_updates_checklist_and_reuses_saved_certificate_after_reopen(ui_case, ready_bundle):
    from src.core.procedure import address_proof_status

    state = entered(ready_bundle[2])
    certificate = next(doc for doc in state["documents"] if doc.doc_type == "dorm_certificate")
    ui_case.seed(state)
    ui_case.nav("details")
    assert "Dormitory certificate \u00b7 Ready" in ui_case.text()
    with patch("ollama.Client.chat", side_effect=AssertionError("Changing the method must not call Ollama")) as model:
        choose(ui_case.app, "selectbox", "Address-proof route", "rental_agreement")
        ui_case.click("Save and confirm preparation answers")
        assert "Rental agreement \u00b7 Missing" in ui_case.text()
        assert "Biometric photo ready" in ui_case.text()
        ui_case.click("Upload Rental agreement")
        assert "5 of 6 items ready" in ui_case.text()
        assert any(button.label == "Rental agreement" for button in ui_case.app.button)
        assert not any(button.label == "Dormitory certificate" for button in ui_case.app.button)
        ui_case.click("Checks and warnings")
        assert "Rental agreement \u00b7 Missing" in ui_case.text()

        reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
        reopened.run()
        choose(reopened, "text_input", "Profile name", "demo-user")
        next(button for button in reopened.button if button.label == "Open my case").click()
        reopened.run()
        next(button for button in reopened.button if button.label == "Your details").click()
        reopened.run()
        assert next(box.value for box in reopened.selectbox if box.label == "Address-proof route") == "rental_agreement"
        choose(reopened, "selectbox", "Address-proof route", "dorm_certificate")
        next(button for button in reopened.button if button.label == "Save and confirm preparation answers").click()
        reopened.run()
        assert not reopened.exception
        assert any(button.label == "Continue to e-Ikamet" for button in reopened.button)
        model.assert_not_called()

    saved = ui_case.read()
    assert address_proof_status(saved)["ready"]
    assert address_proof_status(saved)["parts"][0]["document_ids"] == [certificate.doc_id]
    assert saved["verification_records"][certificate.doc_id]["dorm_certificate_format"].value is True
    assert saved["residence_permit_context"].biometric_photo is True


def test_photo_can_be_saved_without_claiming_address_readiness_and_survives_reload(ui_case):
    from src.core.procedure import address_proof_status
    from src.core.state import ProcedureState

    ui_case.seed(entered({"procedure": ProcedureState(), "user_id": "demo-user", "trace_id": "form"}))
    ui_case.nav("details")
    before = ui_case.read()["trace_id"]
    ui_case.click("Save and confirm preparation answers")
    assert "Choose an address option or record photo readiness" in ui_case.text()
    assert ui_case.read()["trace_id"] == before
    choose(ui_case.app, "selectbox", "Do you have a biometric photo ready?", "Yes")
    ui_case.click("Save and confirm preparation answers")
    state = ui_case.read()
    assert state["event"].confirmed_fields == {"biometric_photo": True}
    assert not address_proof_status(state)["ready"]
    choose(ui_case.app, "selectbox", "Address-proof route", "utility_bill_and_address_form")
    ui_case.click("Save and confirm preparation answers")
    state = ui_case.read()

    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    reopened.run()
    choose(reopened, "text_input", "Profile name", "demo-user")
    next(button for button in reopened.button if button.label == "Open my case").click()
    reopened.run()
    reopened.session_state[f"view_{ui_case.key}"] = "details"
    reopened.run()
    assert not reopened.exception
    assert next(box for box in reopened.selectbox if box.label == "Do you have a biometric photo ready?").value == "Yes"
    assert next(box for box in reopened.selectbox if box.label == "Address-proof route").value == "utility_bill_and_address_form"
    assert not address_proof_status(ui_case.read())["ready"]
    assert ui_case.read()["trace_id"] == state["trace_id"]


def test_new_student_can_follow_the_first_recommended_task(ui_case):
    del ui_case.app.session_state["workspace_profile_id"]
    ui_case.app.run()
    choose(ui_case.app, "text_input", "Profile name", "demo-user")
    ui_case.click("Open my case")
    assert "Get your documents ready" in ui_case.text()
    assert "Step 1 of 5 - Documents" in ui_case.text()
    assert not any(button.label == "Advanced diagnostics" for button in ui_case.app.button)
    ui_case.click("I'm starting from the beginning")
    ui_case.click("Upload passport identity page")
    assert ui_case.view() == "documents"
    assert "Required for this step" in ui_case.text()
    assert "Working on Passport identity page" in ui_case.text()
    assert any(button.label == "Upload and analyze" for button in ui_case.app.button)


def test_unsigned_address_form_stays_in_review(ui_case, ready_bundle):
    from src.core.procedure import address_proof_status
    from src.core.state import Document
    from src.evaluation.scenarios import extract_synthetic
    from test_procedure import override

    state = entered(ready_bundle[2])
    state["documents"].append(Document(doc_id="synthetic-signed-form", file_path="synthetic-address.txt",
                                       status="analyzed", doc_type="address_form", info=extract_synthetic("address_form")))
    state = override(state, "__declarations__", "address_proof_type", "utility_bill_and_address_form")
    state = override(state, "synthetic-signed-form", "signature_present", False)
    ui_case.seed(state)
    ui_case.nav("next")
    ui_case.click("Review Address Form")
    assert "Review the signature against the original" in ui_case.text()
    assert next(box for box in ui_case.app.selectbox if box.label == "Signed Address Form").value == "No"
    before = ui_case.read()["trace_id"]
    ui_case.click("Save and confirm")
    assert ui_case.read()["trace_id"] == before
    assert not address_proof_status(ui_case.read())["ready"]


def test_confirming_one_field_leaves_other_uncertain_readings_unconfirmed(ui_case, ready_bundle):
    from test_procedure import override

    state = entered(ready_bundle[2])
    passport_id = next(doc.doc_id for doc in state["documents"] if doc.doc_type == "passport_copy")
    for name in ("nationality", "expiry_date"):
        state = override(state, passport_id, name, None)
    ui_case.seed(state)
    ui_case.nav("documents")
    passport = next(doc for doc in ui_case.read()["documents"] if doc.doc_id == passport_id)
    original_expiry = passport.info.expiry_date
    assert original_expiry is not None
    expiry = next(item for item in ui_case.app.text_input if item.label == "Passport expiry date (YYYY-MM-DD)")
    assert expiry.value == ""
    choose(ui_case.app, "text_input", "Nationality", "Exampleland")
    ui_case.click("Save and confirm")
    state = ui_case.read()
    assert state["event"].confirmed_fields == {"nationality": "Exampleland"}
    assert state["verification_records"][passport_id]["expiry_date"].value is None
    assert next(doc for doc in state["documents"] if doc.doc_id == passport_id).info.expiry_date == original_expiry


@pytest.mark.parametrize(("kind", "status", "requirement", "conflicted", "expected"), [
    ("passport_copy", "pending", "complete", False, "Analyzing"),
    ("passport_copy", "rejected", "conflict", True, "Analysis failed"),
    ("unknown", "analyzed", None, False, "Unsupported"),
    ("health_insurance", "analyzed", "not_applicable", False, "Not applicable"),
    ("passport_copy", "analyzed", "complete", True, "Conflict detected"),
    ("student_certificate", "analyzed", "needs confirmation", False, "Needs review"),
    ("student_certificate", "analyzed", None, False, "Ready"),
])
def test_document_status_preserves_analysis_and_requirement_distinctions(kind, status, requirement, conflicted, expected):
    from src.ui.documents import document_status

    document = SimpleNamespace(doc_type=kind, status=status)
    assert document_status(document, {"can_confirm": True, "status": "complete"}, requirement, conflicted=conflicted) == expected
