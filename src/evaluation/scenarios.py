"""Curated synthetic V2 outcomes using canonical processing, verification and tools.

Default runs use scripted transports only. Real private-document acceptance belongs
in the separate document benchmark; these scenarios never claim live OCR accuracy.
"""
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from time import perf_counter
from src.core.state import Document, ProcedureState, UserEvent, ValidationIssue, new_run
from src.core.procedure import assess_document_collection, advance_procedure_if_ready, document_collection_update
from src.document_processing.extractor import TextPage
from src.document_processing.processor import process_document
from src.workers.document_analyzer import TYPE_MAP
from src.workers.verification import verification_updates, MILESTONE_SCOPE
from src.workers.rules_engine import evaluate_requirements
from src.tools.registry import execute_tool, reject_tool
from src.tools.schemas import ToolCall
from src.evaluation.metrics import trajectory_metrics, reliability_metrics, LOOP_FAILURE_CATEGORIES

STAMP = datetime(2026,11,19,12,tzinfo=timezone.utc)
SYNTHETIC = {
    "utility_bill":"Utility Bill\nHolder name: Evaluation Landlord\nService address: Example Street 12\nBilling date: 2026-11-01\nProvider: Example Energy Ltd.",
    "address_form":"Address Form\nFirst name: Evaluation\nSurname: Student\nPassport number: X0000001\nDeclared address: Example Street 12\nSignature present: yes",
    "passport_identity_page":"Passport identity page\nHolder name: Evaluation Student\nPassport number: X0000001\nNationality: Exampleland\nExpiry date: 2030-08-30",
    "turkiye_entry_stamp":"TURKIYE ENTRY STAMP\nEntry date: 2026-09-15",
    "student_enrollment_certificate":"STUDENT CERTIFICATE\nHolder name: Evaluation Student\nStudent ID: EVAL0001\nInstitution: Sabanci University\nIssue date: 2026-10-14\nThe exchange student studies between September 15, 2026 - June 25, 2027 at Sabanci University",
    "health_insurance_certificate":"Health Insurance Certificate\nInsured: Evaluation Student\nInsurer: Example Insurance\nPolicy number: EVAL-POL-001\nCoverage start: 2026-09-01\nCoverage end: 2027-07-01",
    "ikamet_application_form":"Declaration for Short Term Residence Permit Registration Form\nApplicant: Evaluation Student\nPassport number: X0000001\nRequested start date: 2026-09-15\nRequested end date: 2027-06-25",
}

class ScriptedDocumentProvider:
    name, model, supports_images = "synthetic", "scripted-offline", False
    def __init__(self, kind): self.kind=kind
    def generate_json(self, *, stage, **kwargs):
        if stage=="classification":return {"document_type":self.kind}
        if self.kind=="ikamet_application_form":return {"applicant_name":"Evaluation Student"}
        if self.kind=="student_enrollment_certificate":return {"holder_name":"Evaluation Student","student_id":"EVAL0001","institution":"Sabanci University","issue_date":"2026-10-14"}
        return {}


def extract_synthetic(kind, *, text=None, scanned=False):
    page = TextPage(1,text or SYNTHETIC[kind],"rapidocr" if scanned else "native_text",0 if scanned else None,scanned,0.01 if scanned else 0,0.97 if scanned else None)
    return process_document("synthetic.txt",pages=[page],provider=ScriptedDocumentProvider(kind),mode="local-only")


def confirm(state, document_id, fields):
    updated={**deepcopy(state),**new_run(UserEvent(kind="confirm_information",document_id=document_id,confirmed_fields=fields),"evaluation-user",scope=MILESTONE_SCOPE)}
    updated.update(verification_updates(updated))
    updated["relevant_requirements"]=evaluate_requirements(updated)
    return updated


@lru_cache(maxsize=1)
def _ready_seed():
    state={**new_run(UserEvent(kind="reevaluate"),"evaluation-user",scope=MILESTONE_SCOPE),"procedure":ProcedureState()}
    state["documents"]=[Document(doc_id=kind,file_path="synthetic-"+kind+".txt",doc_type=TYPE_MAP[kind],status="analyzed",info=extract_synthetic(kind))
        for kind in ("passport_identity_page","turkiye_entry_stamp","student_enrollment_certificate","health_insurance_certificate")]
    state.update(verification_updates(state))
    state=confirm(state,"turkiye_entry_stamp",{"latest_entry_date":"2026-09-15"})
    state=confirm(state,"health_insurance_certificate",{"insurance_valid_for_ikamet":True})
    state=confirm(state,"__declarations__",{"address_proof_type":"dorm_certificate","biometric_photo":True,
        "insurance_evidence_type":"private_health_insurance","requested_permit_end_date":"2027-06-25"})
    state.update(document_collection_update(state))
    if not assess_document_collection(state).ready:raise ValueError("Synthetic readiness fixture is invalid")
    return state


def ready_state(): return deepcopy(_ready_seed())


def preparing_state():
    state=ready_state();state.update(advance_procedure_if_ready(state,recorded_at=STAMP));return state


def _apply(state, call, *, approved=False):
    obs=execute_tool(state,call,approved=approved,adapters={})
    state={**state,**obs.state_updates,"tool_outcomes":[*state.get("tool_outcomes",[]),obs.model_dump(mode="json",exclude={"state_updates"})]}
    return state,obs

GOLDEN_SCENARIOS = {
    "insurance_academic_target":"Coverage through academic end satisfies V2 despite a later requested permit end.",
    "utility_bill_plus_address_form":"Both independently accepted pieces satisfy the scoped operational R017 path.",
    "utility_bill_without_address_form":"One address piece remains incomplete and never passes R017.",
    "legal_stay_estimate":"Unknown previous history produces a labelled sourced estimate without changing legal facts.",
    "legal_stay_reminder":"A sourced stay-limit reminder requires explicit human approval.",
    "insurance_reminder":"Academic-period acceptance can coexist with an insurance-expiry reminder.",
    "rule_change_candidate":"Official snapshot changes are candidates only; no rule is overwritten.",
    "artifact_approval":"A checklist artifact remains a proposal until the host receives approval.",
    "all_prerequisites_missing":"Unknown facts remain missing; no readiness is invented.",
    "some_documents_present":"One passport does not satisfy the six collection prerequisites.",
    "alternate_insurance_layout":"Turkish start/end labels map to the existing insurance schema.",
    "insurance_insufficient":"Accepted coverage ending before both targets blocks collection.",
    "scanned_eikamet":"Synthetic OCR text follows the same profile/schema/grounding path.",
    "native_eikamet":"Native form text preserves requested dates and passport provenance.",
    "claimed_eikamet_without_evidence":"Entry intent does not complete a stage.",
    "all_documents_ready":"The explicit first transition preserves evidence and records one history entry.",
    "conflicting_passport":"A passport conflict blocks automated form filling.",
    "missing_entry_date":"Observed entry date is not accepted before explicit confirmation.",
    "rag_passport_validity":"Guidance retrieval returns attributed passport-validity evidence.",
    "rag_after_eikamet":"Post-application guidance is sourced rather than invented.",
    "calendar_reminder":"A reminder preview requires approval and becomes a usable local ICS.",
    "email_draft":"A grounded draft names only selected available attachments.",
    "tier2_approved":"Approved tool execution writes one outcome and replay executes no duplicate.",
    "tier2_rejected":"Rejected tool proposals cannot mutate the environment.",
    "reflection_success":"Gate errors produce structured repair context and a valid repaired plan.",
    "unchanged_invalid_circuit":"An unchanged-invalid signal takes the deterministic exhausted path.",
    "sandbox_field_diff":"Verified passport data fills the simulated environment, not legal facts.",
    "sandbox_missing_submission":"An incomplete form cannot simulate a successful submission.",
    "tool_schema_injection":"Model-supplied approval cannot bypass tool authorization.",
    "freeform_tool_proposal":"A bounded proposal loop selects a typed safe tool without modifying facts.",
}


EXPECTED_TOOLS = {
    "conflicting_passport":["simulate_fill_eikamet_field"],
    "rag_passport_validity":["search_official_sources"], "rag_after_eikamet":["search_official_sources"],
    "calendar_reminder":["create_calendar_reminder"], "tier2_approved":["create_calendar_reminder"],
    "email_draft":["prepare_iro_email"], "sandbox_field_diff":["simulate_fill_eikamet_field"],
    "sandbox_missing_submission":["simulate_submission"], "tool_schema_injection":["create_calendar_reminder"],
    "freeform_tool_proposal":["get_missing_documents"],
}
EXPECTED_ARGUMENTS = {
    "conflicting_passport":{"simulate_fill_eikamet_field":{"field_id":"passport_number"}},
    "sandbox_field_diff":{"simulate_fill_eikamet_field":{"field_id":"passport_number"}},
    "calendar_reminder":{"create_calendar_reminder":{"title":"Finish preparation","start_at":"2027-01-01T12:00:00+03:00","duration_minutes":30}},
    "tier2_approved":{"create_calendar_reminder":{"title":"Finish preparation","start_at":"2027-01-01T12:00:00+03:00","duration_minutes":30}},
    "email_draft":{"prepare_iro_email":{"attachment_ids":[]}},
    "freeform_tool_proposal":{"get_missing_documents":{}},
}


def run_scenario(name, *, assistant_model=None, rag_judge=None):
    if name not in GOLDEN_SCENARIOS:raise ValueError("Unknown golden scenario")
    started=perf_counter();trace=[];detail={}
    if name=="all_prerequisites_missing":
        result=assess_document_collection({"procedure":ProcedureState()});ok=not result.ready and len(result.prerequisites)==6
    elif name=="some_documents_present":
        state=ready_state();state["documents"]=state["documents"][:1]
        result=assess_document_collection(state);ok=not result.ready and "R014" in result.blockers
    elif name=="alternate_insurance_layout":
        text="Health Insurance Certificate\nInsured: Evaluation Student\nInsurer: Example Insurance\nPolicy number: EVAL-POL-002\nBaslangic Tarihi: 01.10.2026\nBitis Tarihi: 30.09.2027"
        info=extract_synthetic("health_insurance_certificate",text=text)
        ok=info.coverage_start_date.isoformat()=="2026-10-01" and info.coverage_end_date.isoformat()=="2027-09-30"
    elif name=="insurance_insufficient":
        state=confirm(ready_state(),"health_insurance_certificate",{"coverage_end_date":"2027-02-01"})
        result=assess_document_collection(state);ok=not result.ready and "R015" in result.blockers
    elif name in {"scanned_eikamet","native_eikamet"}:
        info=extract_synthetic("ikamet_application_form",scanned=name=="scanned_eikamet")
        ok=info.document_type=="ikamet_application_form" and info.requested_end_date.isoformat()=="2027-06-25" and info.source_pages["requested_end_date"]==1
        detail["route"]=info.extraction_diagnostics["extraction_route"]
    elif name=="claimed_eikamet_without_evidence":
        from src.core.procedure import entry_intent_update
        state={"procedure":ProcedureState()};state.update(entry_intent_update(state,"EIKAMET_COMPLETED",user_id="eval",trace_id="eval-intent",recorded_at=STAMP))
        ok=state["procedure"].current_stage=="DOCUMENT_COLLECTION" and not assess_document_collection(state).ready
    elif name=="all_documents_ready":
        state=ready_state();before=deepcopy(state)
        state.update(advance_procedure_if_ready(state,recorded_at=STAMP))
        ok=state["procedure"].current_stage=="EIKAMET_PREPARATION" and state["documents"]==before["documents"] and len(state["procedure"].stage_history)==1
    elif name=="conflicting_passport":
        state=preparing_state();state["verification_conflicts"]=["passport_or_foreign_id"]
        _,obs=_apply(state,ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"passport_number"}));trace.append(obs)
        ok=obs.status=="FAILED" and not obs.state_updates
    elif name=="missing_entry_date":
        state=ready_state();state["verification_records"]["turkiye_entry_stamp"].pop("latest_entry_date")
        result=assess_document_collection(state);ok=not result.ready and "confirmation:latest_entry_date" in result.blockers
    elif name in {"rag_passport_validity","rag_after_eikamet"}:
        from src.agents.assistant import assistant_turn
        query="Why is passport validity required?" if name=="rag_passport_validity" else "What should I do after e-Ikamet?"
        result=assistant_turn(ready_state(),query,model=False)
        answer=result["assistant_messages"][-1]
        trace=result.get("tool_outcomes",[])
        ok=bool(answer.get("source_refs")) and bool(answer.get("content"))
        detail["source_count"]=len(answer.get("source_refs",[]))
        from src.evaluation.rag_eval import evaluate_rag
        sources=[source for event in trace if event.get("name")=="search_official_sources" for source in event.get("data",{}).get("results",[])]
        detail["rag_evaluation"]=evaluate_rag(query,answer["content"],sources,judge=rag_judge)
    elif name in {"calendar_reminder","tier2_approved","tier2_rejected"}:
        state={};call=ToolCall(call_id="eval-reminder",name="create_calendar_reminder",arguments={"title":"Finish preparation","start_at":"2027-01-01T12:00:00+03:00"})
        preview=execute_tool(state,call,adapters={});trace.append(preview)
        if name=="tier2_rejected":
            obs=reject_tool(state,call);trace.append(obs);ok=obs.status=="REJECTED" and not obs.state_updates
        else:
            state,obs=_apply(state,call,approved=True);trace.append(obs)
            replay=execute_tool(state,call,approved=True,adapters={});trace.append(replay)
            ok=preview.status=="APPROVAL_REQUIRED" and obs.status=="SUCCEEDED" and "BEGIN:VCALENDAR" in obs.data["ics"] and replay.replayed and len(state["calendar_reminders"])==1
    elif name=="email_draft":
        _,obs=_apply(ready_state(),ToolCall(name="prepare_iro_email"),approved=True);trace.append(obs)
        ok=obs.status=="SUCCEEDED" and not obs.data["attachments"] and obs.data["status"]=="DRAFT" and obs.data["recipient_source_ref"]=="SAB_EXCHANGE_CURRENT"
    elif name=="reflection_success":
        from src.agents.planning_policy import fallback_plan
        from src.agents.planner import actionable_needs
        from src.guardrails.gate_1 import validate_plan
        from src.memory.reflection import reflection
        # Repair needs actual current work; a complete stage has no filler plan.
        state=confirm(ready_state(),"__declarations__",{"biometric_photo":None})
        plan=fallback_plan(state,actionable_needs(state));bad=plan.model_copy(deep=True)
        bad.actions[0].action=bad.actions[0].title="information"
        errors=validate_plan(state,bad)
        repair=reflection({**state,"candidate_plan":bad.actions,"validation_errors":errors})
        ok=bool(errors) and repair["retry_count"]==1 and not validate_plan(state,plan)
        trace=[{"kind":"reflection"}]
    elif name=="unchanged_invalid_circuit":
        from src.core.graph import route_gate_1,respond
        state={"run_status":"PLAN_INVALID","retry_count":1,"validation_errors":[ValidationIssue(code="UNCHANGED_INVALID_PLAN",message="same invalid plan",source="planner")]}
        ok=route_gate_1(state)=="exhausted" and respond(state)["run_status"]=="HUMAN_REVIEW_REQUIRED"
    elif name=="sandbox_field_diff":
        state=preparing_state();before=deepcopy(state)
        state,obs=_apply(state,ToolCall(name="simulate_fill_eikamet_field",arguments={"field_id":"passport_number"}));trace.append(obs)
        ok=state["sandbox_form"]["fields"]["passport_number"]["value"]=="X0000001" and state["verification_records"]==before["verification_records"] and state["procedure"]==before["procedure"]
    elif name=="sandbox_missing_submission":
        _,obs=_apply(preparing_state(),ToolCall(name="simulate_submission"));trace.append(obs)
        ok=obs.status=="SUCCEEDED" and not obs.data["valid"] and obs.data["form"]["status"]=="DRAFT"
    elif name=="tool_schema_injection":
        obs=execute_tool({},ToolCall(name="create_calendar_reminder",arguments={"title":"Unsafe","start_at":"2027-01-01T12:00:00+03:00","approved":True}),adapters={});trace.append(obs)
        ok=obs.status=="FAILED" and obs.failure_category=="TOOL_SCHEMA_ERROR"
    elif name=="insurance_academic_target":
        state=confirm(ready_state(),"__declarations__",{"requested_permit_end_date":"2027-09-30"})
        rows={r.rule_id:r for r in evaluate_requirements(state)}
        ok=rows["R015"].passed is True and assess_document_collection(state).ready
    elif name in {"utility_bill_plus_address_form","utility_bill_without_address_form"}:
        kinds=("utility_bill","address_form") if name=="utility_bill_plus_address_form" else ("utility_bill",)
        state={**new_run(UserEvent(kind="reevaluate"),"evaluation-user",scope=MILESTONE_SCOPE),"procedure":ProcedureState(),"documents":[Document(doc_id=k,file_path="synthetic.txt",doc_type=k,status="analyzed",info=extract_synthetic(k)) for k in kinds]}
        state.update(verification_updates(state))
        row=next(r for r in evaluate_requirements(state) if r.rule_id=="R017")
        ok=row.passed is True if len(kinds)==2 else row.status=="missing_information"
    elif name in {"legal_stay_estimate","legal_stay_reminder","insurance_reminder"}:
        from src.core.legal_stay import assess_legal_stay,reminder_candidates,insurance_explanation
        from src.core.state import ResidencePermitContext
        state={"residence_permit_context":ResidencePermitContext(nationality="ITA",passport_type="ordinary",visa_status="visa_exempt",latest_entry_date="2026-10-01",insurance_end_date="2027-07-20",program_end_date="2027-06-20",requested_permit_end_date="2027-09-20")}
        assessment=assess_legal_stay(state,as_of=STAMP.date())
        if name=="legal_stay_estimate":ok=assessment.confidence_type=="ESTIMATE" and state["residence_permit_context"].legal_stay_expiry_date is None
        else:
            key="legal_stay" if name=="legal_stay_reminder" else "insurance_expiry"
            row=next(r for r in reminder_candidates(state,as_of=STAMP.date()) if r["candidate_id"]==key)
            call=ToolCall(name="create_calendar_reminder",arguments={"title":row["title"],"start_at":row["date"]+"T09:00:00+03:00","source_candidate_id":key})
            preview=execute_tool(state,call,adapters={});trace.append(preview)
            state,obs=_apply(state,call,approved=True);trace.append(obs)
            ok=preview.status=="APPROVAL_REQUIRED" and obs.status=="SUCCEEDED" and bool(insurance_explanation(state))
    elif name=="rule_change_candidate":
        from src.rag.ingestion import build_corpus
        source=next(r for r in build_corpus() if r.source_id=="SAB_EXCHANGE_CURRENT" and r.content_kind=="official_guidance")
        _,obs=_apply({},ToolCall(name="compare_official_snapshot",arguments={"source_id":source.source_id,"source_url":source.url,"content":source.content+" Synthetic snapshot update for review."}));trace.append(obs)
        ok=obs.status=="SUCCEEDED" and obs.data["data"]["status"]=="RULE_CHANGE_CANDIDATE" and not obs.state_updates
    elif name=="artifact_approval":
        call=ToolCall(name="build_checklist")
        preview=execute_tool({},call,adapters={});trace.append(preview)
        # Reject the preview: deterministic no-artifact assertion, no unnecessary file creation.
        rejected=reject_tool({},call);trace.append(rejected)
        ok=preview.status=="APPROVAL_REQUIRED" and rejected.status=="REJECTED" and not rejected.state_updates
    else:
        from src.agents.assistant import assistant_turn
        calls=[0]
        def scripted(context,tools):
            calls[0]+=1
            return {"tool_calls":[{"name":"get_missing_documents","arguments":{}}]} if calls[0]==1 else {"tool_calls":[]}
        state=ready_state();result=assistant_turn(state,"Please organize my immediate next safe actions using the available environment.",model=assistant_model or scripted)
        trace=result.get("tool_outcomes",[])
        ok=any(item.get("status")=="SUCCEEDED" for item in trace) and not any(key in result for key in ("documents","residence_permit_context","procedure"))
        detail["assistant_diagnostics"]=result.get("assistant_diagnostics",{})
    metrics=trajectory_metrics(trace,expected_tools=EXPECTED_TOOLS.get(name),expected_arguments=EXPECTED_ARGUMENTS.get(name))
    if name in {"freeform_tool_proposal","rag_passport_validity","rag_after_eikamet"}:
        diagnostics=result.get("assistant_diagnostics",{})
        metrics.update(llm_calls=diagnostics.get("llm_calls",0),
            input_characters=diagnostics.get("llm_input_chars") or diagnostics.get("input_chars",0),
            output_characters=diagnostics.get("output_chars",0),
            loop_stopped=metrics["loop_stopped"] or bool(LOOP_FAILURE_CATEGORIES.intersection(diagnostics.get("failure_categories",[]))))
    return {"scenario":name,"purpose":GOLDEN_SCENARIOS[name],"passed":bool(ok),"elapsed_seconds":perf_counter()-started,
            "trajectory":metrics,"details":detail}


def evaluate_scenarios(*, repeats=3, names=None, assistant_model=None, rag_judge=None):
    if type(repeats) is not int or not 1<=repeats<=10:raise ValueError("Repeat count must be 1..10")
    reports=[]
    for name in names or GOLDEN_SCENARIOS:
        runs=[]
        for index in range(repeats):
            try: result=run_scenario(name,assistant_model=assistant_model,rag_judge=rag_judge)
            except Exception as error:
                result={"scenario":name,"purpose":GOLDEN_SCENARIOS[name],"passed":False,"error_type":type(error).__name__}
            result["repeat"]=index+1;runs.append(result)
        reports.append({"scenario":name,"runs":runs,"reliability":reliability_metrics([row["passed"] for row in runs],k=repeats)})
    return {"scenario_count":len(reports),"run_count":len(reports)*repeats,
        "passed_runs":sum(run["passed"] for row in reports for run in row["runs"]),"reports":reports,
        "live_model_requested":assistant_model is not None,"live_judge_requested":rag_judge is not None}



def copilot_benchmark_cases(directory):
    """Ten sanitized cases, no private PDFs or official-action execution."""
    from pathlib import Path
    from src.core.procedure import choose_eikamet_path
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def add(state,kind,text=None):
        info=extract_synthetic(kind,text=text)
        path=directory/(kind+'.txt');path.write_text(text or SYNTHETIC[kind],encoding='utf-8')
        state=deepcopy(state);state['documents'].append(Document(doc_id=kind,file_path=str(path),doc_type=TYPE_MAP[kind],status='analyzed',info=info))
        state['event']=UserEvent(kind='reevaluate');state.update(verification_updates(state));state['relevant_requirements']=evaluate_requirements(state)
        return state
    base=ready_state()
    for doc in base['documents']:
        path=directory/(doc.doc_id+'.txt');path.write_text(SYNTHETIC[doc.doc_id],encoding='utf-8');doc.file_path=str(path)
    address=confirm(base,'__declarations__',{'address_proof_type':None})
    address=add(address,'address_form')
    complete=add(base,'ikamet_application_form')
    complete=confirm(complete,'ikamet_application_form',{'online_application_completed':True,'application_form_signed':True})
    complete=confirm(complete,'__declarations__',{'fingerprint_step_completed':True,'gi_number_present':True})
    complete['procedure']=complete['procedure'].model_copy(update={'current_stage':'IRO_REVIEW'})
    incomplete=deepcopy(complete)
    complete=add(complete,'declaration_control_document','DECLARATION CONTROL DOCUMENT\nHolder name: Evaluation Student\nSigned: yes')
    complete=add(complete,'card_fee_receipt','RESIDENCE PERMIT CARD FEE RECEIPT\nPayment date: 2026-11-20\nReceipt number: EVAL-RECEIPT-1\nAmount: 1234 TRY')
    conflict=confirm(base,'health_insurance_certificate',{'holder_name':'Different Person'})
    expiry=confirm(base,'__declarations__',{'legal_stay_expiry_date':'2027-02-15'})
    rows=[
        ('What should I do now?',address,{'get_case_summary','get_next_actions','get_missing_items'},'Utility bill'),
        ('Which documents am I missing?',address,{'get_missing_items','get_case_summary','get_document_status'},'Utility bill'),
        ('Why is my address proof incomplete?',address,{'get_case_summary','get_missing_items'},'Utility bill'),
        ('Have I already completed e-Ikamet?',base,{'get_case_summary','get_current_stage','get_verified_facts'},'form'),
        ('What do I need before fingerprinting?',base,{'get_case_summary','get_stage_progress','get_missing_items','search_official_sources'},'form'),
        ('Can I send my documents to IRO now?',incomplete,{'get_case_summary','get_stage_progress','get_missing_items'},'receipt'),
        ("Why can't I proceed to IRO?",incomplete,{'get_case_summary','get_missing_items','get_stage_progress'},'declaration'),
        ('What is wrong with my current case?',conflict,{'get_conflicts','get_case_summary'},'name'),
        ('Remind me before my legal stay expires.',expiry,{'create_calendar_reminder'},'approve'),
        ('Prepare my IRO email.',complete,{'prepare_iro_email'},'approve'),
    ]
    return [{'question':question,'state':deepcopy(state),'allowed_tools':allowed,'expected_marker':marker} for question,state,allowed,marker in rows]


def benchmark_copilot(models,*,directory,question_numbers=None,include_planner=True):
    from src.agents.assistant import assistant_turn,ollama_tool_proposal,ollama_synthesize,safe_synthesis,CopilotAnswer
    from src.agents.planner import create_action_plan
    from src.guardrails.gate_1 import validate_plan
    from pathlib import Path
    import json
    cases=copilot_benchmark_cases(Path(directory)/'synthetic_cases');reports=[]
    cases.append({'question':'Why do I need fingerprinting? Please cite official guidance.','state':deepcopy(cases[4]['state']),'allowed_tools':{'search_official_sources'},'expected_marker':'fingerprint'})
    for model in models:
        rows=[]
        for number,case in enumerate(cases,1):
            if question_numbers is not None and number not in question_numbers:continue
            raw={};transport={'intent_calls':0,'synthesis_calls':0}
            def intent(context,specs):
                transport['intent_calls']+=1
                value=ollama_tool_proposal(context,specs,model=model);raw['proposal']=value;return value
            def synthesize(context,packet):
                transport['synthesis_calls']+=1
                raw['synthesis_schema_valid']=False
                value=ollama_synthesize(context,packet,model=model);CopilotAnswer.model_validate(value);raw['synthesis_schema_valid']=True;raw['synthesis_safe']=safe_synthesis(CopilotAnswer.model_validate(value),packet);return value
            started=perf_counter()
            try:
                update=assistant_turn(case['state'],case['question'],model=intent,copilot=True,synthesizer=synthesize,force_model=True)
                answer=update['assistant_messages'][-1]['content'];diag=update['assistant_diagnostics']
                selected=[call['name'] for call in raw.get('proposal',{}).get('tool_calls',[])]
                schema_valid=bool(raw.get('proposal') is not None) and raw.get('synthesis_schema_valid') is not False
                tool_correct=bool(selected) and set(selected)<=case['allowed_tools']
                grounded=case['expected_marker'].casefold() in answer.casefold() and answer.split('\n\n')[0].casefold().strip(' .?!')!=case['question'].casefold().strip(' .?!')
                unsupported=raw.get('synthesis_safe') is False or 'UNGROUNDED_ACTION_PROPOSAL' in diag['failure_categories']
                rows.append({'number':number,'question':case['question'],'answer':answer,'selected_tools':selected,'tool_selection_correct':tool_correct,
                    'schema_valid':schema_valid,'state_grounded_correct':grounded,'unsupported_model_claims':unsupported,
                    'fallback':diag.get('synthesis_valid') is False,'latency_seconds':round(perf_counter()-started,3),'transport':transport,
                    'failure_categories':diag['failure_categories'],'approval_required':bool(update.get('tool_proposals'))})
            except Exception as error:
                rows.append({'number':number,'question':case['question'],'schema_valid':False,'tool_selection_correct':False,'state_grounded_correct':False,
                    'unsupported_model_claims':False,'latency_seconds':round(perf_counter()-started,3),'failure':type(error).__name__})
            print(f"[COPILOT BENCHMARK] model={model} question={number} seconds={rows[-1]['latency_seconds']} tool_correct={rows[-1]['tool_selection_correct']} grounded={rows[-1]['state_grounded_correct']}",flush=True)
        reports.append({'model':model,'questions':rows,'schema_valid':sum(row['schema_valid'] for row in rows),'tool_selection_correct':sum(row['tool_selection_correct'] for row in rows),
            'state_grounded_correct':sum(row['state_grounded_correct'] for row in rows),'unsupported_model_claims':sum(row['unsupported_model_claims'] for row in rows),
            'diagnostic_failure_rate':sum(bool(row.get('failure') or row.get('failure_categories')) for row in rows)/len(rows),
            'failure_rate':sum(not (row['schema_valid'] and row['tool_selection_correct'] and row['state_grounded_correct']) or row['unsupported_model_claims'] or bool(row.get('failure')) for row in rows)/len(rows),
            'average_latency_seconds':sum(row['latency_seconds'] for row in rows)/len(rows)})
    # Planner quality is measured separately on two compact current-stage cases.
    import src.agents.planner as planner_module
    planner_rows=[];original=planner_module.OLLAMA_MODEL
    try:
        for model in models if include_planner else []:
            planner_module.OLLAMA_MODEL=model
            for name,state in [('missing_address_utility',cases[0]['state']),('incomplete_iro_package',cases[5]['state'])]:
                started=perf_counter()
                try:
                    plan=create_action_plan(state);errors=validate_plan(state,plan)
                    planner_rows.append({'model':model,'scenario':name,'gate_1_passed':not errors,'error_codes':[error.code for error in errors],
                        'actions':[action.model_dump(mode='json') for action in plan.actions],'seconds':round(perf_counter()-started,3)})
                except Exception as error:planner_rows.append({'model':model,'scenario':name,'gate_1_passed':False,'failure':type(error).__name__,'seconds':round(perf_counter()-started,3)})
    finally:planner_module.OLLAMA_MODEL=original
    return {'models':reports,'planner_cases':planner_rows,'live':True,'external_actions_executed':False,'document_benchmark_repeated_for_model_comparison':False}
