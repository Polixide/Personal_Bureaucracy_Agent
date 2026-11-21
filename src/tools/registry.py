"""Typed host dispatch and authorization; no model-supplied approval is trusted."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from time import perf_counter
from pydantic import ValidationError
from src.core.procedure import assess_document_collection
from src.tools.schemas import (SandboxAnswersInput,
    ToolCall, ToolSpec, ToolObservation, EmptyInput, SearchInput, FillFieldInput,
    ValidateFieldInput, CalendarInput, EmailDraftInput, SendEmailInput,
    ReadResult,SnapshotInput,ArtifactResult, SandboxResult, ProcedureResult, MissingDocumentsResult, SearchResult, CalendarResult, EmailResult,
)
from src.tools.sandbox import read_form, fill_field, validate_field, validate_form, apply_validation, prefill_fields, simulated_application_preview, generate_simulated_pdf
from src.tools.schemas import SimulatedApplicationResult
from src.tools.calendar import reminder_preview, configured_calendar_adapter
from src.tools.email import prepare_draft, configured_email_adapter

# One canonical implementation per tool; public aliases are deliberately absent.
_DEFINITIONS = {
    "get_case_summary": ("Read the accepted case, current stage, package readiness and next actions.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_current_stage": ("Read the saved stage and completed stages; infer no completion.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_stage_progress": ("Read current-stage prerequisites and explicit next-stage readiness.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_document_status": ("Read uploaded documents and unresolved observation fields.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_conflicts": ("Read unresolved accepted-information conflicts.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_next_actions": ("Read only currently available deterministic next actions.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_verified_facts": ("Read source-grounded and user-confirmed accepted facts, never raw observations.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "read_twin": ("Read compact accepted facts and current procedure, excluding raw evidence.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "get_missing_items": ("Read active missing requirements and document prerequisites.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "evaluate_rules": ("Evaluate active deterministic rules without mutating facts.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "assess_legal_stay": ("Assess declared stay conditions and estimate explicitly when history is unknown.",EmptyInput,ReadResult,"TIER_1","READ_ONLY"),
    "compare_official_snapshot": ("Compare an allow-listed local guidance snapshot and return a human-review candidate; never change rules.",SnapshotInput,ReadResult,"TIER_1","READ_ONLY"),
    "build_checklist": ("Preview a deterministic checklist artifact; create it only after human approval.",EmptyInput,ArtifactResult,"TIER_2","ARTIFACT_MUTATION"),
    "get_current_procedure_state": ("Read the saved procedure stage and history.", EmptyInput, ProcedureResult, "TIER_1", "READ_ONLY"),
    "get_missing_documents": ("Assess the six current document prerequisites deterministically.", EmptyInput, MissingDocumentsResult, "TIER_1", "READ_ONLY"),
    "search_official_sources": ("Retrieve bounded official-source guidance with citations.", SearchInput, SearchResult, "TIER_1", "READ_ONLY"),
    "read_eikamet_sandbox": ("Read the local form simulation without changing official facts.", EmptyInput, SandboxResult, "TIER_1", "READ_ONLY"),
    "prefill_eikamet_sandbox": ("Prefill all currently accepted fields into the local simulated form.",SandboxAnswersInput,SandboxResult,"TIER_1","SANDBOX_WRITE"),
    "generate_simulated_application": ("Preview and, after explicit approval, create a clearly marked educational PDF.",EmptyInput,SimulatedApplicationResult,"TIER_2","ARTIFACT_MUTATION"),
    "simulate_fill_eikamet_field": ("Fill one form field only from accepted canonical evidence.", FillFieldInput, SandboxResult, "TIER_1", "SANDBOX_WRITE"),
    "validate_form_field": ("Check a simulated field against current canonical evidence.", ValidateFieldInput, SandboxResult, "TIER_1", "READ_ONLY"),
    "simulate_application_validation": ("Validate all supported simulated form inputs.", EmptyInput, SandboxResult, "TIER_1", "SANDBOX_WRITE"),
    "simulate_submission": ("Simulate submission only if all catalogue inputs validate; never contact a government site.", EmptyInput, SandboxResult, "TIER_1", "SANDBOX_WRITE"),
    "create_calendar_reminder": ("Preview a personal reminder, require approval, then export ICS or use a configured calendar adapter.", CalendarInput, CalendarResult, "TIER_2", "EXTERNAL_MUTATION"),
    "prepare_iro_email": ("Prepare a grounded IRO draft with only selected existing attachments.", EmailDraftInput, EmailResult, "TIER_2", "ARTIFACT_MUTATION"),
    "send_iro_email": ("Preview the exact saved draft, require approval, then simulate send or use a configured Gmail adapter.", SendEmailInput, EmailResult, "TIER_2", "EXTERNAL_MUTATION"),
}


def get_registry():
    return {name: ToolSpec(name=name, description=description, input_schema=inputs.model_json_schema(),
        output_schema=outputs.model_json_schema(), risk_tier=risk, effect=effect)
        for name, (description, inputs, outputs, risk, effect) in _DEFINITIONS.items()}


def get_tool_specs():
    return list(get_registry().values())



def validate_tool_call(call):
    """Validate/normalize a proposal only; never execute or inspect adapters."""
    call = ToolCall.model_validate(call)
    definition = _DEFINITIONS.get(call.name)
    if definition is None:
        raise ValueError("Unknown tool")
    arguments = definition[1].model_validate(call.arguments).model_dump(mode="json")
    return call.model_copy(update={"arguments":arguments})


def _fingerprint(call):
    return sha256(json.dumps({"name":call.name,"arguments":call.arguments},sort_keys=True,default=str,separators=(",",":")).encode()).hexdigest()


def _saved_observation(raw):
    return raw.get("observation", raw) if isinstance(raw, dict) else {}


def _outcome(state, call, status, *, data=None, diff=None, failure=None, updates=None, started=0):
    return ToolObservation(call_id=call.call_id, name=call.name, status=status, data=data or {}, arguments=deepcopy(call.arguments), diff=diff or {},
        failure_category=failure, elapsed_seconds=perf_counter()-started if started else 0,
        state_updates=updates or {}, fingerprint=_fingerprint(call))


def reject_tool(state, call, reason="User rejected the proposed external action"):
    call = ToolCall.model_validate(call)
    definition = _DEFINITIONS.get(call.name)
    if definition:
        call = call.model_copy(update={"arguments":definition[1].model_validate(call.arguments).model_dump(mode="json")})
    return _outcome(state, call, "REJECTED", data={"message":reason}, failure="HUMAN_REJECTED")


def execute_tool(state, call, *, approved=False, adapters=None):
    """Return a canonical partial update; input state is never modified.

    approved is a host-only keyword supplied after the graph's approval interrupt.
    Credentials are loaded only after approval. No endpoint is contacted in preview.
    """
    started = perf_counter()
    try:
        call = ToolCall.model_validate(call)
    except ValidationError:
        fallback = ToolCall(name="invalid_tool_call")
        return _outcome(state, fallback, "FAILED", failure="TOOL_SCHEMA_ERROR", data={"message":"Malformed typed tool call"}, started=started)
    definition = _DEFINITIONS.get(call.name)
    if definition is None:
        return _outcome(state, call, "FAILED", failure="TOOL_SELECTION_ERROR", data={"message":"Unknown tool"}, started=started)
    try:
        call = validate_tool_call(call)
        args = definition[1].model_validate(call.arguments)
    except ValidationError as error:
        return _outcome(state, call, "FAILED", failure="TOOL_SCHEMA_ERROR", data={"message":"Tool arguments failed schema validation",
            "fields":[".".join(map(str,item["loc"])) for item in error.errors(include_input=False)]}, started=started)
    fingerprint = _fingerprint(call)
    previous = [_saved_observation(item) for item in state.get("tool_outcomes", [])]
    same_id = [item for item in previous if item.get("call_id")==call.call_id]
    if call.name=="send_iro_email" and any(item.get("fingerprint")==fingerprint and item.get("failure_category")=="EXTERNAL_OUTCOME_UNCERTAIN" for item in previous):
        return _outcome(state, call, "FAILED", failure="EXTERNAL_OUTCOME_UNCERTAIN", data={"message":"A previous delivery of this draft is uncertain. Reconcile Sent mail before preparing a new send; automatic retries are stopped."}, started=started)
    if any(item.get("fingerprint") not in (None, fingerprint) for item in same_id):
        return _outcome(state, call, "FAILED", failure="TOOL_SCHEMA_ERROR", data={"message":"A call ID cannot be reused with different arguments"}, started=started)
    if any(item.get("status")=="REJECTED" for item in same_id):
        return _outcome(state, call, "FAILED", failure="HUMAN_REJECTED", data={"message":"A rejected proposal requires a new proposal and approval"}, started=started)
    successful = next((item for item in reversed(same_id) if item.get("status")=="SUCCEEDED"), None)
    if successful:
        return ToolObservation.model_validate({key:value for key,value in successful.items() if key in ToolObservation.model_fields} | {"replayed":True,"state_updates":{}})
    failures = sum(item.get("fingerprint")==fingerprint and item.get("status")=="FAILED" for item in previous)
    if failures >= 2:
        return _outcome(state, call, "FAILED", failure="TOOL_LOOP_LIMIT", data={"message":"Repeated identical tool failures stopped"}, started=started)
    updates, diff = {}, {}
    try:
        if call.name in {"get_case_summary","get_current_stage","get_stage_progress","get_document_status","get_conflicts","get_next_actions","get_verified_facts"}:
            from src.core.procedure import case_summary
            summary=case_summary(state)
            keys={'get_current_stage':('stage','user_step','completed_stages'),'get_stage_progress':('stage_readiness','current_items','collection','iro_package'),
                'get_document_status':('documents',),'get_conflicts':('conflicts',),'get_next_actions':('next_actions',),'get_verified_facts':('verified_facts',)}
            data={'data':{key:summary[key] for key in keys.get(call.name,summary)}}
        elif call.name in {"read_twin","get_missing_items","evaluate_rules","assess_legal_stay","compare_official_snapshot"}:
            from src.workers.rules_engine import evaluate_requirements
            if call.name=="read_twin":
                from src.agents.assistant import build_assistant_context
                data={"data":build_assistant_context(state,"Read my saved case")}
            elif call.name=="get_missing_items":
                from src.core.procedure import case_summary
                summary=case_summary(state)
                data={'data':{key:summary[key] for key in ('stage','current_items','address_proof','iro_package','next_actions')}}
            elif call.name=="evaluate_rules":data={"data":{"requirements":[r.model_dump(mode="json") for r in evaluate_requirements(state)]}}
            elif call.name=="assess_legal_stay":
                from src.core.legal_stay import assess_legal_stay,reminder_candidates
                data={"data":{"assessment":assess_legal_stay(state).model_dump(mode="json"),"reminders":reminder_candidates(state)}}
            else:
                from src.rag.monitoring import compare_snapshot
                data={"data":compare_snapshot(args.source_id,args.source_url,args.content)}
        elif call.name=="build_checklist":
            from src.workers.rules_engine import evaluate_requirements
            from config import PROJECT_ROOT
            rows=evaluate_requirements(state)
            content="# Residence-permit preparation checklist\n\nPreparation only; not official approval or submission.\n\n"+"\n".join(f"- [{'x' if r.passed is True else '?' if r.passed is None else ' '}] {r.rule_id}: {r.description} ({r.status})" for r in rows)
            if state.get('procedure') is not None:
                from src.core.procedure import user_step,stage_document_checklist
                step=user_step(state)
                content='# Residence-permit preparation checklist\n\n'+f"Step {step['number']} of 5 - {step['label']}"+'\n\nPreparation only; not official approval or submission.\n\n'+'\n'.join(f"- [{'x' if row['status']=='complete' else ' '}] {row['label']} - {'Ready' if row['status']=='complete' else 'Needs review'}" for row in stage_document_checklist(state))
            data={"title":"Residence-permit checklist","content":content,"artifact_path":None,"status":"PROPOSED"}
            diff={"before":None,"after":data}
            if not approved:return _outcome(state,call,"APPROVAL_REQUIRED",data=data,diff=diff,started=started)
            directory=PROJECT_ROOT/".local"/"artifacts"/sha256(str(state.get("user_id","")).encode()).hexdigest()[:16]
            from pathlib import Path
            if os.name=="nt":directory=Path(chr(92)*2+"?"+chr(92)+str(directory.resolve()))
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/(sha256(call.call_id.encode()).hexdigest()+".md")
            path.write_text(content,encoding="utf-8")
            data={**data,"artifact_path":str(path),"status":"CREATED"}
        elif call.name=="get_current_procedure_state":
            value = state.get("procedure")
            data = ProcedureResult(procedure=value.model_dump(mode="json") if hasattr(value,"model_dump") else value).model_dump(mode="json")
        elif call.name=="get_missing_documents":
            assessment = assess_document_collection(state)
            data = MissingDocumentsResult(ready=assessment.ready, prerequisites=assessment.prerequisites, blockers=assessment.blockers).model_dump(mode="json")
        elif call.name=="search_official_sources":
            from src.rag.retriever import search_official_sources
            data = SearchResult(results=[item.model_dump(mode="json") if hasattr(item,"model_dump") else item
                for item in search_official_sources(args.query, limit=args.limit)]).model_dump(mode="json")
        elif call.name=="read_eikamet_sandbox":
            data = validate_form(state).model_dump(mode="json")
        elif call.name=="validate_form_field":
            data = validate_field(state, args.field_id).model_dump(mode="json")
        elif call.name=="prefill_eikamet_sandbox":
            result,diff=prefill_fields(state,args.answers)
            data=result.model_dump(mode='json');updates['sandbox_form']=result.form.model_dump(mode='json')
        elif call.name=="generate_simulated_application":
            preview=simulated_application_preview(state)
            data=preview.model_dump(mode='json');diff={'before':None,'after':data,'scope':'Educational PDF only; not an official application or submission'}
            if not approved:return _outcome(state,call,'APPROVAL_REQUIRED',data=data,diff=diff,started=started)
            artifact,form=generate_simulated_pdf(state,call.call_id)
            data=artifact.model_dump(mode='json');updates['sandbox_form']=form.model_dump(mode='json')
        elif call.name=="simulate_fill_eikamet_field":
            result, diff = fill_field(state, args.field_id, args.value)
            data = result.model_dump(mode="json")
            updates["sandbox_form"] = result.form.model_dump(mode="json")
        elif call.name in {"simulate_application_validation", "simulate_submission"}:
            result = apply_validation(state, submit=call.name=="simulate_submission")
            data = result.model_dump(mode="json")
            updates["sandbox_form"] = result.form.model_dump(mode="json")
            diff = {"before":read_form(state).model_dump(mode="json"), "after":updates["sandbox_form"]}
        elif call.name=="create_calendar_reminder":
            if args.source_candidate_id:
                if args.source_candidate_id.startswith('declared_'):
                    from src.tools.calendar import declared_reminder_candidates
                    candidates=declared_reminder_candidates(state)
                else:
                    from src.core.legal_stay import reminder_candidates
                    candidates=reminder_candidates(state)  # Existing Technical/evaluation capability.
                candidate=next((r for r in candidates if r["candidate_id"]==args.source_candidate_id),None)
                if candidate is None or candidate["date"]!=args.start_at.date().isoformat():raise ValueError("Reminder source changed; request a fresh proposal")
            reminder = reminder_preview(args, call.call_id)
            external = args.delivery!='calendar_app' and (adapters.get("calendar") is not None if adapters is not None else bool(os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN")))
            reminder.mode = "GOOGLE_CALENDAR" if external else "LOCAL_ICS"
            expected = (state.get("pending_tool_proposal") or {}).get("preview", {}).get("after", {}).get("mode")
            if approved and expected and expected != reminder.mode:
                raise ValueError("Calendar execution target changed; request a fresh preview")
            data = reminder.model_dump(mode="json")
            diff = {"before":None, "after":data, "scope":"Personal reminder; not an official deadline or appointment"}
            if not approved:
                return _outcome(state, call, "APPROVAL_REQUIRED", data=data, diff=diff, started=started)
            adapter = None if args.delivery=='calendar_app' else adapters.get("calendar") if adapters is not None else configured_calendar_adapter()
            if external and adapter is None: raise ValueError("Configured calendar adapter is unavailable; request a fresh preview")
            if adapter is not None:
                try:
                    reminder.external_id = adapter.create(reminder)
                except Exception:
                    return _outcome(state, call, "FAILED", failure="EXTERNAL_OUTCOME_UNCERTAIN", data={"message":"Calendar outcome could not be confirmed. Check the calendar before proposing another reminder."}, started=started)
                reminder.mode = "GOOGLE_CALENDAR"
                data = reminder.model_dump(mode="json")
            updates["calendar_reminders"] = [*deepcopy(state.get("calendar_reminders", [])), data]
        elif call.name=="prepare_iro_email":
            draft = prepare_draft(state, args, call.call_id)
            data = draft.model_dump(mode="json")
            diff = {"before":None, "after":data}
            if not approved:return _outcome(state,call,"APPROVAL_REQUIRED",data=data,diff=diff,started=started)
            updates["email_drafts"] = [*deepcopy(state.get("email_drafts", [])), data]
        elif call.name=="send_iro_email":
            raw = next((item for item in state.get("email_drafts", []) if item.get("draft_id")==args.draft_id), None)
            if raw is None:
                raise ValueError("Select an existing prepared email draft")
            draft = EmailResult.model_validate(raw)
            if draft.purpose=='package_review':
                from src.core.procedure import iro_package_readiness
                package=iro_package_readiness(state)
                if not package['ready'] or any(item['document_id'] not in package['attachment_ids'] for item in draft.attachments):raise ValueError('The IRO package changed; prepare a fresh package-review draft')
            if not draft.recipient:
                raise ValueError("Confirm an exact recipient before sending")
            if draft.status != "DRAFT":
                raise ValueError("This draft already has a send outcome")
            # Prevent stale/deleted attachments from being sent after a preview.
            from pathlib import Path
            active = {getattr(doc,"doc_id",None):doc for doc in state.get("documents",[]) if getattr(doc,"status",None)=="analyzed"}
            if any(item["document_id"] not in active or active[item["document_id"]].file_path!=item["file_path"] for item in draft.attachments):
                raise ValueError("Selected attachments changed; prepare a fresh draft")
            for attachment in draft.attachments:
                path = Path(attachment["file_path"]).resolve()
                from config import PROJECT_ROOT
                if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != attachment.get("content_sha256"):
                    raise ValueError("Selected attachment content changed; prepare a fresh draft")
            external = adapters.get("email") is not None if adapters is not None else bool(os.getenv("GMAIL_ACCESS_TOKEN"))
            draft.delivery_mode = "GMAIL" if external else "LOCAL_SIMULATION"
            expected = (state.get("pending_tool_proposal") or {}).get("preview", {}).get("after", {}).get("delivery_mode")
            if approved and expected and expected != draft.delivery_mode:
                raise ValueError("Email execution target changed; request a fresh preview")
            diff = {"before":{"status":"DRAFT"}, "after":draft.model_dump(mode="json"), "scope":"Exact selected recipient, body and attachments"}
            data = draft.model_dump(mode="json")
            if not approved:
                return _outcome(state, call, "APPROVAL_REQUIRED", data=data, diff=diff, started=started)
            adapter = adapters.get("email") if adapters is not None else configured_email_adapter()
            if external and adapter is None: raise ValueError("Configured email adapter is unavailable; request a fresh preview")
            if adapter is not None:
                try:
                    draft.external_id = adapter.send(draft)
                except Exception:
                    return _outcome(state, call, "FAILED", failure="EXTERNAL_OUTCOME_UNCERTAIN", data={"message":"Email delivery could not be confirmed. Check Sent mail before proposing another send."}, started=started)
                draft.status = "SENT"
            else:
                draft.status = "SIMULATED_SEND"
            data = draft.model_dump(mode="json")
            updates["email_drafts"] = [data if item.get("draft_id")==args.draft_id else deepcopy(item) for item in state.get("email_drafts", [])]
        else:
            raise ValueError("Tool handler unavailable")
        # Handler result is checked against the public output contract.
        data = definition[2].model_validate(data).model_dump(mode="json")
        return _outcome(state, call, "SUCCEEDED", data=data, diff=diff, updates=updates, started=started)
    except (ValueError, TypeError, KeyError) as error:
        return _outcome(state, call, "FAILED", failure="VALIDATION_POLICY_FAILURE", data={"message":str(error)}, started=started)
    except Exception as error:
        # Network exceptions must not leak keys, private payloads or provider bodies.
        return _outcome(state, call, "FAILED", failure="TOOL_EXECUTION_ERROR", data={"message":"Tool adapter failed; no successful outcome was recorded","error_type":type(error).__name__}, started=started)
