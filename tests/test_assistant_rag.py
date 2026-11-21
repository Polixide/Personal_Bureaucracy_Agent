"""Offline home-agent, attributed retrieval and compact-memory regressions."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.agents.assistant import (assistant_turn, build_assistant_context,
    MAX_CONTEXT_CHARS, MAX_TOOL_CALLS, MAX_RETRIEVAL_CALLS)
from src.core.graph import build_graph
from src.core.state import ProcedureState, ResidencePermitContext, UserEvent, new_run
from src.core.procedure import advance_procedure_if_ready
from src.memory.checkpointer import sqlite_checkpointer
from src.rag.ingestion import build_corpus, ingest_guidance
from src.rag.retriever import search_official_sources
from src.tools.schemas import ToolObservation
from test_procedure import ready_bundle


NOW = datetime(2026, 11, 19, 10, tzinfo=timezone.utc)


def test_corpus_uses_attributed_authoritative_rule_excerpts():
    corpus = build_corpus()
    rules = {rid for chunk in corpus for rid in chunk.rule_ids}
    assert rules == {f"R{i:03}" for i in (i for i in range(1,30) if i not in (12,24,25,27,29))}
    assert all(chunk.authority in {"government_primary","sabanci_operational","sabanci_direct_instruction"} for chunk in corpus)
    assert all((chunk.url.startswith("https://") and chunk.retrieved_at) if chunk.authority!="sabanci_direct_instruction" else chunk.url is None for chunk in corpus)
    assert any(chunk.content_kind == "project_rule_excerpt" for chunk in corpus)


def test_passport_retrieval_finds_rule_and_preserves_source_metadata():
    results = search_official_sources("Why is passport validity required?")
    assert any("R006" in row.rule_ids for row in results)
    assert all(row.chunk_id and row.title and row.url for row in results)
    assert not search_official_sources("quantum banana xenon")


@pytest.mark.parametrize("limit", [0,6,True])
def test_retrieval_budget_is_typed_and_bounded(limit):
    with pytest.raises(ValueError): search_official_sources("passport",limit=limit)


def test_optional_official_cache_ingestion_is_validated_and_offline(tmp_path):
    cache = tmp_path / "guidance.json"
    assert ingest_guidance([{"source_id":"GOV_FAQ", "content":"Synthetic xenon guidance for ingestion only.", "verified_at":"2026-11-19"}], cache) == 1
    corpus = build_corpus(cache)
    result = search_official_sources("xenon", corpus=corpus)
    assert result[0].content_kind == "official_guidance"
    assert result[0].source_id == "GOV_FAQ" and result[0].verified_at == "2026-11-19"
    with pytest.raises(ValueError): ingest_guidance([{"source_id":"METU_RESIDENCE_2026", "content":"Cross-check only"}], cache)
    with pytest.raises(ValidationError): ingest_guidance([{"source_id":"GOV_FAQ", "content":"x", "url":"https://unapproved.invalid"}], cache)


def test_simple_state_question_avoids_llm_and_never_mutates_facts():
    state = {"procedure":ProcedureState(), "residence_permit_context":ResidencePermitContext()}
    original = deepcopy(state)
    with patch("src.agents.assistant.ollama_tool_proposal", side_effect=AssertionError("No model needed")):
        update = assistant_turn(state,"What documents am I missing?")
    assert "0 of 6" in update["assistant_messages"][-1]["content"]
    assert update["assistant_diagnostics"]["llm_calls"] == 0
    assert update["assistant_diagnostics"]["tool_calls"] == 1
    assert not {"documents","residence_permit_context","verification_records","procedure"}.intersection(update)
    assert state == original


def test_guidance_uses_retrieval_tool_and_attributed_evidence():
    update = assistant_turn({"procedure":ProcedureState()},"Why is passport validity required?",model=False)
    message = update["assistant_messages"][-1]
    assert "60 days" in message["content"] and "](https://" in message["content"]
    assert message["source_refs"] and update["assistant_diagnostics"]["retrieval_calls"] == 1
    assert update["assistant_diagnostics"]["llm_calls"] == 0


def test_working_context_prunes_episodes_and_excludes_raw_data(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state["assistant_messages"] = [{"role":"user","content":"PRIVATE OLD HISTORY "*1000} for _ in range(100)]
    state["extracted_facts"]["large"] = "RAW OCR PRIVATE "*20000
    context = build_assistant_context(state,"What should I do today?")
    encoded = json.dumps(context)
    assert len(encoded) <= MAX_CONTEXT_CHARS
    assert len(context["recent_messages"]) <= 6
    assert "RAW OCR PRIVATE" not in encoded and "source_evidence" not in encoded
    assert context["accepted_facts"]["passport_expiry_date"] == "2030-08-30"


def test_raw_legacy_flags_do_not_become_working_verified_facts():
    state={"procedure":ProcedureState(),"residence_permit_context":ResidencePermitContext(passport_expiry_date="2030-01-01",insurance_end_date="2030-01-01")}
    assert build_assistant_context(state,"missing")['accepted_facts'] == {}


def test_reminder_requires_exact_date_and_never_invents_a_deadline():
    update = assistant_turn({"procedure":ProcedureState()},"Remind me before my recommended target",model=False)
    assert "date, time and timezone" in update["assistant_messages"][-1]["content"]
    assert update["assistant_diagnostics"]["tool_calls"] == 0
    assert not update.get("calendar_reminders") and not update.get("tool_proposals")


def test_tier2_reminder_produces_one_exact_pending_preview_without_execution():
    state={"procedure":ProcedureState()}
    update = assistant_turn(state,"Remind me on 2027-02-03T10:30:00+03:00",model=False)
    assert len(update["tool_proposals"]) == 1
    proposal = update["tool_proposals"][0]
    assert proposal["status"] == "APPROVAL_REQUIRED" and proposal["tool_call"]["name"] == "create_calendar_reminder"
    assert "state_updates" not in proposal["observation"]
    assert not update.get("calendar_reminders")
    assert "explicitly approve or reject" in update["assistant_messages"][-1]["content"]


def test_claimed_completed_application_does_not_change_procedure():
    state={"procedure":ProcedureState()}
    update = assistant_turn(state,"I already completed e-Ikamet",model=False)
    assert "entry context, not verified completion" in update["assistant_messages"][-1]["content"]
    assert "procedure" not in update and "residence_permit_context" not in update


def test_repeated_successful_model_tool_call_stops_loop():
    def proposal(context,specs):
        return {"tool_calls":[{"name":"get_current_procedure_state","arguments":{}}]}
    update = assistant_turn({"procedure":ProcedureState()},"Tell me something contextual",model=proposal)
    assert update["assistant_diagnostics"]["tool_calls"] == 1
    assert "DUPLICATE_TOOL_CALL" in update["assistant_diagnostics"]["failure_categories"]


def test_identical_unknown_tool_failures_stop_after_two():
    def proposal(context,specs): return {"tool_calls":[{"name":"unknown_tool","arguments":{}}]}
    update = assistant_turn({"procedure":ProcedureState()},"A novel intent",model=proposal)
    assert update["assistant_diagnostics"]["tool_calls"] == 2
    assert "IDENTICAL_TOOL_FAILURE_LIMIT" in update["assistant_diagnostics"]["failure_categories"]
    assert "TOOL_SELECTION_ERROR" in update["assistant_diagnostics"]["failure_categories"]


def test_model_cannot_self_approve_inside_tool_arguments():
    def proposal(context,specs):
        return {"tool_calls":[{"name":"create_calendar_reminder","arguments":{"title":"Synthetic","start_at":"2027-02-03T10:00:00+03:00","approved":True}}]}
    update = assistant_turn({"procedure":ProcedureState()},"A novel intent",model=proposal)
    assert not update.get("calendar_reminders") and not update.get("tool_proposals")
    assert "TOOL_SCHEMA_ERROR" in update["assistant_diagnostics"]["failure_categories"]


def test_tool_observation_cannot_rewrite_canonical_verified_facts():
    def registry(state,call):
        return ToolObservation(call_id=call.call_id,name=call.name,status="SUCCEEDED",state_updates={"residence_permit_context":{"passport_expiry_date":"2099-01-01"}})
    update = assistant_turn({"procedure":ProcedureState()},"What documents am I missing?",registry=registry,model=False)
    assert "residence_permit_context" not in update
    assert "OBSERVATION_INTERPRETATION_ERROR" in update["assistant_diagnostics"]["failure_categories"]
    assert "state_updates" not in update["tool_outcomes"][0]


def test_retrieval_loops_are_bounded_independently_of_model_rounds():
    n = 0
    def model(context,specs):
        nonlocal n
        n += 1
        return {"tool_calls":[{"name":"search_official_sources","arguments":{"query":f"different query {n}"}}]}
    update = assistant_turn({"procedure":ProcedureState()},"A novel intent",model=model)
    assert update["assistant_diagnostics"]["retrieval_calls"] == MAX_RETRIEVAL_CALLS
    assert "RETRIEVAL_LOOP_LIMIT" in update["assistant_diagnostics"]["failure_categories"]


def test_model_transport_failure_degrades_without_changing_verified_context():
    def unavailable(context,specs): raise RuntimeError("PRIVATE transport details")
    update = assistant_turn({"procedure":ProcedureState()},"A novel intent",model=unavailable)
    assert update["assistant_diagnostics"]["failure_categories"]
    assert "PRIVATE" not in update["assistant_messages"][-1]["content"]


def test_bounded_sandbox_filling_uses_current_refs_and_progresses_on_next_turn(ready_bundle):
    state = deepcopy(ready_bundle[2])
    state.update(advance_procedure_if_ready(state,recorded_at=NOW))
    original_facts = deepcopy(state["verification_records"])
    first = assistant_turn(state,"Fill the verified form fields",model=False)
    assert 1 <= first["assistant_diagnostics"]["tool_calls"] <= MAX_TOOL_CALLS
    assert first["sandbox_form"]["fields"]
    state.update(first)
    first_count = len(state["sandbox_form"]["fields"])
    second = assistant_turn(state,"Fill the verified form fields",model=False)
    state.update(second)
    assert len(state["sandbox_form"]["fields"]) > first_count
    assert state["verification_records"] == original_facts
    assert state["procedure"].current_stage == "EIKAMET_PREPARATION"


def test_sandbox_automation_requires_explicit_preparation_transition(ready_bundle):
    state = ready_bundle[2]
    update = assistant_turn(state,"Fill the verified form fields",model=False)
    assert update["assistant_diagnostics"]["tool_calls"] == 0 and not update.get("sandbox_form")
    assert "explicitly" in update["assistant_messages"][-1]["content"]


def test_chat_episodes_survive_sqlite_recreation(tmp_path):
    path=tmp_path/"assistant.db"; config={"configurable":{"thread_id":"chat"}}
    with sqlite_checkpointer(path) as saver:
        graph=build_graph(saver)
        request=new_run(UserEvent(kind="assistant_message",text="What documents am I missing?"),"chat")
        graph.invoke(request,config)
        original=graph.get_state(config).values
    with sqlite_checkpointer(path) as saver:
        restored=build_graph(saver).get_state(config).values
    assert restored["assistant_messages"] == original["assistant_messages"]
    assert restored["tool_outcomes"] == original["tool_outcomes"]
    assert restored["assistant_diagnostics"]["tool_calls"] == 1


def test_live_proposal_transport_uses_central_model_timeout_and_safe_logs(capsys):
    from types import SimpleNamespace
    from config import OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT
    from src.agents.assistant import ollama_tool_proposal
    with patch("src.agents.assistant.Client") as client:
        client.return_value.chat.return_value=SimpleNamespace(message=SimpleNamespace(content='{"tool_calls": []}'))
        result=ollama_tool_proposal({"objective":"PRIVATE CURRENT OBJECTIVE"},[])
    assert result=={"tool_calls":[]}
    client.assert_called_once_with(host=OLLAMA_HOST,timeout=OLLAMA_TIMEOUT)
    request=client.return_value.chat.call_args.kwargs
    assert request["model"]==OLLAMA_MODEL and request["think"] is False and request["format"]["additionalProperties"] is False
    assert "PRIVATE" not in capsys.readouterr().err


def test_model_observes_only_bounded_retrieved_evidence_not_full_tool_payloads():
    calls=0
    observed=[]
    def model(context,specs):
        nonlocal calls
        calls+=1
        observed.append(deepcopy(context))
        assert all("output_schema" not in spec for spec in specs)
        return {"tool_calls":[{"name":"search_official_sources","arguments":{"query":"passport validity"}}]} if calls==1 else {"tool_calls":[]}
    update=assistant_turn({"procedure":ProcedureState()},"A contextual query",model=model)
    assert observed[1]["retrieved_evidence"]
    assert all(len(row["content"])<=500 for row in observed[1]["retrieved_evidence"])
    assert len(observed[1]["retrieved_evidence"])<=2
    assert update["assistant_diagnostics"]["llm_input_chars"]>0


def test_passport_validity_retrieval_excludes_unrelated_insurance_only_excerpts():
    from src.rag.retriever import search_official_sources,tokens
    rows=search_official_sources("Why is passport validity required?",limit=3)
    assert any("R006" in row.rule_ids for row in rows)
    assert all("passport" in tokens(row.content) for row in rows)
    assert not any("R016" in row.rule_ids for row in rows)


def test_mixed_passport_insurance_query_can_retrieve_both_topics():
    from src.rag.retriever import search_official_sources
    rows=search_official_sources("passport validity and insurance coverage",limit=5)
    assert any("R006" in row.rule_ids for row in rows)
    assert any("R015" in row.rule_ids for row in rows)



@pytest.fixture
def copilot_cases(tmp_path):
    from src.evaluation.scenarios import copilot_benchmark_cases
    return copilot_benchmark_cases(tmp_path/'copilot')


@pytest.mark.parametrize('index',[0,1,2,3,4,5,6,7])
def test_copilot_answers_current_case_without_planner_or_factual_writes(copilot_cases,index):
    case=copilot_cases[index];before=deepcopy(case['state'])
    with patch('src.agents.planner.create_action_plan',side_effect=AssertionError('State questions do not plan')):
        result=assistant_turn(case['state'],case['question'],model=False,copilot=True)
    assert case['expected_marker'].casefold() in result['assistant_messages'][-1]['content'].casefold()
    assert not {'documents','verification_records','residence_permit_context','procedure'} & result.keys()
    assert case['state']==before


def test_copilot_does_not_mark_missing_utility_bill_or_package_ready(copilot_cases):
    state=copilot_cases[2]['state']
    def fabricated(context,packet):return {'answer':'Your address proof is ready. Continue with the application now.','observation_ids':['case']}
    result=assistant_turn(state,'Why is my address proof incomplete?',model=False,copilot=True,synthesizer=fabricated)
    answer=result['assistant_messages'][-1]['content']
    assert 'Utility bill is still missing' in answer and 'incomplete' in answer
    assert result['assistant_diagnostics']['synthesis_valid'] is False


def test_structured_copilot_synthesis_uses_observed_case_and_bounded_memory(copilot_cases):
    state=deepcopy(copilot_cases[2]['state']);state['assistant_messages']=[{'role':'user','content':'Previous contextual question'}]*20
    seen={}
    def model(context,packet):
        seen.update(context=context,packet=packet)
        return {'answer':packet['grounded_answer'],'observation_ids':['case']}
    result=assistant_turn(state,'Why is my address proof incomplete?',copilot=True,model=False,synthesizer=model)
    assert result['assistant_diagnostics']['synthesis_valid']
    assert len(seen['context']['recent_messages'])<=6
    assert 'source_evidence' not in json.dumps(seen)
    assert 'Utility bill' in seen['packet']['grounded_answer']


@pytest.mark.parametrize('index,tool',[(8,'create_calendar_reminder'),(9,'prepare_iro_email')])
def test_copilot_action_requests_use_existing_exact_approval_preview(copilot_cases,index,tool):
    state=copilot_cases[index]['state'];before=deepcopy(state)
    result=assistant_turn(state,copilot_cases[index]['question'],model=False,copilot=True)
    proposal=result['tool_proposals'][-1]
    assert proposal['tool_call']['name']==tool and proposal['status']=='APPROVAL_REQUIRED'
    assert not result.get('calendar_reminders') and not result.get('email_drafts')
    assert state==before


def test_copilot_incomplete_package_cannot_create_submission_draft(copilot_cases):
    result=assistant_turn(copilot_cases[5]['state'],'Prepare my IRO email.',model=False,copilot=True)
    assert not result.get('tool_proposals') and not result.get('email_drafts')
    assert 'Declaration' in result['assistant_messages'][-1]['content'] and 'receipt' in result['assistant_messages'][-1]['content']


def test_model_cannot_invent_calendar_arguments_or_bypass_approval(copilot_cases):
    def proposal(context,specs):return {'tool_calls':[{'name':'create_calendar_reminder','arguments':{'title':'Invented','start_at':'2099-01-01T09:00:00+03:00'}}]}
    result=assistant_turn(copilot_cases[0]['state'],'What should I do now?',model=proposal,copilot=True,force_model=True)
    assert 'UNGROUNDED_ACTION_PROPOSAL' in result['assistant_diagnostics']['failure_categories']
    assert not result.get('tool_proposals') and not result.get('calendar_reminders')


@pytest.mark.parametrize('text,choice,supported',[
    ('I am a Sabanci exchange student and just arrived in Turkey.','first_ikamet',True),
    ('I need to renew my residence permit.','renewal',False),('I need a Kimlik procedure.','kimlik',False)])
def test_landing_recommendation_uses_only_fixed_catalogue(text,choice,supported):
    from src.agents.assistant import recommend_procedure
    result=recommend_procedure(text,model=lambda *args:{'procedure_id':choice})
    assert result['procedure_id']==choice and result['implemented'] is supported
    if not supported:assert result['message']=='This procedure is not implemented in the current prototype.'


def test_invented_landing_procedure_is_rejected_without_creating_state():
    from src.agents.assistant import recommend_procedure,PROCEDURE_CATALOG
    result=recommend_procedure('I need a residence permit renewal.',model=lambda *args:{'procedure_id':'invented_procedure'})
    assert result['procedure_id'] in PROCEDURE_CATALOG and not result['implemented']


def test_read_tools_share_canonical_case_without_mutation(copilot_cases):
    from src.tools.registry import execute_tool
    from src.tools.schemas import ToolCall
    state=copilot_cases[2]['state'];before=deepcopy(state)
    for name in ('get_case_summary','get_current_stage','get_stage_progress','get_missing_items','get_document_status','get_conflicts','get_next_actions','get_verified_facts'):
        observation=execute_tool(state,ToolCall(name=name))
        assert observation.status=='SUCCEEDED' and not observation.state_updates
    assert state==before



def test_copilot_does_not_restore_technical_visa_calculation(copilot_cases):
    def proposal(context,specs):return {'tool_calls':[{'name':'assess_legal_stay','arguments':{}}]}
    with patch('src.core.legal_stay.assess_legal_stay',side_effect=AssertionError('Technical calculator is not a Copilot tool')):
        result=assistant_turn(copilot_cases[0]['state'],'Explain my current situation',model=proposal,copilot=True,force_model=True)
    assert 'TOOL_SELECTION_ERROR' in result['assistant_diagnostics']['failure_categories']
    assert result['assistant_diagnostics']['tool_calls']==0



def test_live_selector_request_excludes_unneeded_factual_dump_and_large_schemas():
    from types import SimpleNamespace
    from src.agents.assistant import ollama_tool_proposal
    from src.tools.registry import get_tool_specs
    context={'objective':'Which documents am I missing?','procedure_stage':'DOCUMENT_COLLECTION','case':{'raw':'PRIVATE FULL TWIN '*10000},'recent_messages':[],'recent_tool_summaries':[]}
    specs=[spec.model_dump(mode='json') for spec in get_tool_specs()]
    with patch('src.agents.assistant.Client') as client:
        client.return_value.chat.return_value=SimpleNamespace(message=SimpleNamespace(content='{"tool_calls":[{"name":"get_missing_items","arguments":{}}]}'))
        result=ollama_tool_proposal(context,specs)
    request=client.return_value.chat.call_args.kwargs
    payload=request['messages'][1]['content']
    assert 'PRIVATE FULL TWIN' not in payload and len(payload)<5000
    assert result['tool_calls'][0]['name']=='get_missing_items'
    assert request['think'] is False


@pytest.mark.parametrize('name',['create_calendar_reminder','prepare_iro_email'])
def test_live_selector_constrains_known_grounded_action_request(name):
    from types import SimpleNamespace
    from src.agents.assistant import ollama_tool_proposal
    from src.tools.registry import get_tool_specs
    context={'objective':'Prepare the requested preview','canonical_action_requests':[{'name':name,'arguments':{}}]}
    specs=[spec.model_dump(mode='json') for spec in get_tool_specs()]
    with patch('src.agents.assistant.Client') as client:
        client.return_value.chat.return_value=SimpleNamespace(message=SimpleNamespace(content=json.dumps({'tool_calls':[{'name':name,'arguments':{}}]})))
        ollama_tool_proposal(context,specs)
    request=client.return_value.chat.call_args.kwargs
    assert request['format']['$defs']['AssistantToolIntent']['properties']['name']['enum']==[name]
    assert request['format']['properties']['tool_calls']['minItems']==1
    assert request['format']['properties']['tool_calls']['maxItems']==1
    assert request['format']['$defs']['AssistantToolIntent']['properties']['arguments']=={'type':'object','properties':{},'additionalProperties':False}
    assert [row['name'] for row in json.loads(request['messages'][1]['content'])['tools']]==[name]


def test_copilot_rejects_question_echo_and_keeps_attributed_guidance(copilot_cases):
    question='Why do I need fingerprinting? Please cite official guidance.'
    def echo(context,packet):return {'answer':question,'observation_ids':['case']}
    def redirected(context,specs):return {'tool_calls':[{'name':'search_official_sources','arguments':{'query':'generic passport rule'}}]}
    result=assistant_turn(copilot_cases[4]['state'],question,model=redirected,copilot=True,synthesizer=echo,force_model=True)
    answer=result['assistant_messages'][-1]['content']
    assert not answer.startswith(question) and 'fingerprint' in answer.casefold()
    lookup=next(row for row in result['assistant_events'] if row['name']=='search_official_sources')
    assert lookup['arguments']['query']=='Sabanci after online e-Ikamet fingerprint application form passport GI'
    assert 'SYNTHESIS_GROUNDING_FALLBACK' in result['assistant_diagnostics']['failure_categories']
    assert result['assistant_messages'][-1]['source_refs']
    assert any(row['source_id']=='SAB_EXCHANGE_CURRENT' for row in result['assistant_messages'][-1]['source_refs'])
