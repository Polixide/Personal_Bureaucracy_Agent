"""Offline persistence checks using real SQLite files and graph checkpoints."""
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from copy import deepcopy
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from langgraph.types import Command

from config import PROJECT_ROOT
from src.core.graph import build_graph
from src.core.state import (
    ActionPlan, AgentState, CHECKPOINT_TYPES, Document, PlanStep, ProcedureState,
    ResidencePermitContext, StageHistoryEntry, UserEvent, VerifiedInput, new_run,
)
from src.document_processing.schemas import DocumentInfo, StudentCertificate
from src.memory.checkpointer import sqlite_checkpointer


def thread(name="demo-user"):
    return {"configurable": {"thread_id": name}, "recursion_limit": 60}


def proposal():
    return ActionPlan(actions=[PlanStep(
        step_id="ask", action="Provide entry details", priority=1,
        reason="Information is missing", related_rule_id="R003",
    )], summary="Gather information")


def create_pending_graph(db):
    """Run real workers/gates, with one failed candidate then a valid repair."""
    info = StudentCertificate(
        document_type="student_enrollment_certificate", holder_name="Alex Example",
        student_id="TEST-0001", institution="Example University", issue_date="2026-10-01",
    )
    bad = proposal()
    bad.actions[0].reason = "R003 failed"
    event = UserEvent(kind="upload", file_path=str(
        PROJECT_ROOT / "data/synthetic_samples/student_enrollment.txt"))
    request = new_run(event, "demo-user")
    request["residence_permit_context"] = ResidencePermitContext(
        legal_stay_expiry_date=date(2026, 12, 1))
    request["extracted_facts"] = {"known_fact": "retained"}
    request["blockers"] = ["R003"]
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        with patch("src.workers.document_analyzer.process_document", return_value=info), \
             patch("src.agents.planner.create_action_plan", side_effect=[bad, proposal()]):
            result = graph.invoke(request, thread())
        assert result["__interrupt__"]
        assert result["retry_count"] == 1
        return graph.get_state(thread()).values


def test_full_state_survives_recreation_and_fresh_python_process(tmp_path):
    db = tmp_path / "twin.db"
    before = create_pending_graph(db)
    with sqlite_checkpointer(db) as saver:
        after = build_graph(saver).get_state(thread()).values
    assert after == before
    assert isinstance(after["documents"][0], Document)
    assert isinstance(after["documents"][0].info, DocumentInfo)
    assert after["residence_permit_context"].legal_stay_expiry_date == date(2026, 12, 1)
    assert len(after["relevant_requirements"]) == 24
    assert after["candidate_plan"][0].step_id == "ask"
    # The engine refreshes the deliberately stale seeded blocker list before
    # approval. All 26 blocking rules lack inputs in this fixture's context.
    assert after["blockers"] == [f"R{i:03}" for i in (i for i in range(1,27) if i not in (12,24,25))]
    assert after["repair_feedback"][0].code == "UNKNOWN_AS_FAILURE"
    assert len(after["repair_history"]) == 1

    # A separate interpreter has no graph, saver, mocks or Digital Twin in memory.
    code = """
import json, sys
from src.memory.checkpointer import sqlite_checkpointer
from src.core.graph import build_graph
with sqlite_checkpointer(sys.argv[1]) as saver:
    snapshot = build_graph(saver).get_state({'configurable': {'thread_id': 'demo-user'}})
    print(json.dumps({'documents': len(snapshot.values['documents']),
                      'fact': snapshot.values['extracted_facts']['known_fact'],
                      'pending': list(snapshot.next),
                      'approval': snapshot.values['approval_decision']}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(db)], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert json.loads(process.stdout) == {
        "documents": 1, "fact": "retained", "pending": ["human_approval"], "approval": None,
    }


def test_thread_ids_are_isolated(tmp_path):
    db = tmp_path / "twin.db"
    create_pending_graph(db)
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        assert graph.get_state(thread("other-user")).values == {}
        graph.update_state(thread("other-user"), {
            "user_id": "other-user", "extracted_facts": {"other": "private"},
        })
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        first = graph.get_state(thread()).values
        second = graph.get_state(thread("other-user")).values
        assert "other" not in first["extracted_facts"]
        assert second["extracted_facts"] == {"other": "private"}
        assert not second.get("documents")


def test_document_upserts_and_fact_merges_survive_multiple_reopens(tmp_path):
    db = tmp_path / "twin.db"
    first = Document(doc_id="a", file_path="a.txt")
    second = Document(doc_id="b", file_path="b.txt")
    with sqlite_checkpointer(db) as saver:
        build_graph(saver).update_state(thread(), {
            "documents": [first], "extracted_facts": {"old": 1, "replace": 1},
        })
    with sqlite_checkpointer(db) as saver:
        build_graph(saver).update_state(thread(), {
            "documents": [first.model_copy(update={"status": "rejected"}), second],
            "extracted_facts": {"replace": 2, "new": 3},
        })
    with sqlite_checkpointer(db) as saver:
        values = build_graph(saver).get_state(thread()).values
    assert {d.doc_id: d.status for d in values["documents"]} == {"a": "rejected", "b": "pending"}
    assert values["extracted_facts"] == {"old": 1, "replace": 2, "new": 3}


def test_pending_approval_requires_explicit_resume_after_restart(tmp_path):
    db = tmp_path / "twin.db"
    create_pending_graph(db)
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        snapshot = graph.get_state(thread())
        assert snapshot.next == ("human_approval",)
        assert snapshot.values["approval_decision"] is None
        assert "approved_plan" not in snapshot.values
        # Even invoking without an approval decision must re-interrupt.
        still_pending = graph.invoke(None, thread())
        assert still_pending["__interrupt__"]
        assert "approved_plan" not in still_pending
        approved = graph.invoke(Command(resume=True), thread())
        assert approved["run_status"] == "APPROVED"
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(saver)
        saved = graph.get_state(thread())
        assert saved.next == ()
        assert saved.values["approved_plan"].actions[0].step_id == "ask"
        assert saved.values["blockers"]
        with patch("src.agents.planner.create_action_plan", return_value=proposal()):
            new = graph.invoke(new_run(UserEvent(kind="message", text="Next plan"), "demo-user"), thread())
        assert new["__interrupt__"]
        assert new["approval_decision"] is None  # Previous consent cannot approve a new plan.


def test_denial_survives_restart(tmp_path):
    db = tmp_path / "twin.db"
    create_pending_graph(db)
    with sqlite_checkpointer(db) as saver:
        denied = build_graph(saver).invoke(Command(resume=False), thread())
        assert denied["run_status"] == "APPROVAL_DENIED"
    with sqlite_checkpointer(db) as saver:
        saved = build_graph(saver).get_state(thread())
    assert saved.next == ()
    assert saved.values["validation_errors"][0].code == "APPROVAL_DENIED"
    assert "approved_plan" not in saved.values


def test_database_paths_must_stay_in_project():
    with pytest.raises(ValueError, match="inside the project"):
        with sqlite_checkpointer(PROJECT_ROOT.parent / "outside.db"):
            pass


@pytest.mark.parametrize("kind,fields", [
    ("student_enrollment_certificate", {"holder_name":"Alex", "student_id":"S123", "institution":"Example", "issue_date":"2026-10-01"}),
    ("health_insurance_certificate", {"holder_name":"Alex", "policy_number":"P123", "insurer":"Example", "coverage_start_date":"2026-10-01", "coverage_end_date":"2027-10-01"}),
    ("ikamet_application_form", {"applicant_name":"Alex", "application_number":"APP123", "requested_start_date":"2026-10-01", "requested_end_date":"2027-10-01"}),
    ("unknown", {}),
])
def test_document_specific_schemas_survive_sqlite_restart(tmp_path,kind,fields):
    from src.document_processing.schemas import DOCUMENT_SCHEMAS
    info=DOCUMENT_SCHEMAS[kind](**fields)
    mapping={"student_enrollment_certificate":"student_certificate", "health_insurance_certificate":"health_insurance"}
    document=Document(doc_id="typed",file_path="typed.txt",status="analyzed",doc_type=mapping.get(kind,kind),info=info)
    db=tmp_path / "typed.db"
    with sqlite_checkpointer(db) as saver:
        build_graph(saver).update_state(thread(),{"documents":[document]})
    with sqlite_checkpointer(db) as saver:
        restored=build_graph(saver).get_state(thread()).values["documents"][0]
    assert type(restored.info) is type(info)
    assert restored.info == info
    assert restored == document



def procedure_snapshot():
    """Explicit synthetic lifecycle input; no stage inference is under test."""
    return ProcedureState(
        current_stage="IRO_REVIEW",stage_status="WAITING_CONFIRMATION",
        completed_stages=["DOCUMENT_COLLECTION","EIKAMET_PREPARATION","EIKAMET_READY","PAYMENT","FINGERPRINT"],
        stage_blockers=["R021","conflict:holder_name","confirmation:iro_preverification_confirmed"],
        stage_history=[
            StageHistoryEntry(stage="PAYMENT",status="COMPLETED",trace_id="record-payment",
                recorded_at=datetime(2026,11,18,9,30,tzinfo=timezone.utc),reason_refs=["R010"]),
            StageHistoryEntry(stage="IRO_REVIEW",status="WAITING_CONFIRMATION",trace_id="await-iro",
                recorded_at=datetime(2026,11,18,9,45,tzinfo=timezone.utc),reason_refs=["R021"]),
        ],
    )


def test_procedure_can_be_explicitly_initialized_without_reusing_plan_progress():
    state: AgentState={**new_run(UserEvent(kind="message",text="Prepare my application"),"new-user"),
                       "procedure":ProcedureState(),"completed_steps":["payment","fingerprint"]}
    procedure=state["procedure"]
    assert procedure.current_stage=="DOCUMENT_COLLECTION" and procedure.stage_status=="NOT_STARTED"
    assert procedure.completed_stages==procedure.stage_blockers==procedure.stage_history==[]
    from typing import get_args
    for kind in get_args(UserEvent.model_fields["kind"].annotation):
        assert "procedure" not in new_run(UserEvent(kind=kind),"new-user")
    assert set(ProcedureState.model_fields)=={"current_stage","stage_status","completed_stages","stage_blockers","stage_history","eikamet_preparation","eikamet_path","reviewed_application_document_id"}
    assert set(StageHistoryEntry.model_fields)=={"stage","status","trace_id","recorded_at","reason_refs"}
    assert (StageHistoryEntry.__module__,StageHistoryEntry.__name__) in CHECKPOINT_TYPES
    assert (ProcedureState.__module__,ProcedureState.__name__) in CHECKPOINT_TYPES


def test_procedure_literal_contract_and_independent_default_lists():
    schema=ProcedureState.model_json_schema()
    assert schema["properties"]["current_stage"]["enum"]==[
        "DOCUMENT_COLLECTION","EIKAMET_PREPARATION","EIKAMET_READY","PAYMENT","FINGERPRINT",
        "IRO_REVIEW","SUBMISSION","TRACKING","COMPLETE"]
    assert schema["properties"]["stage_status"]["enum"]==[
        "NOT_STARTED","IN_PROGRESS","BLOCKED","READY","WAITING_CONFIRMATION","COMPLETED","NOT_APPLICABLE"]
    first,second=ProcedureState(),ProcedureState()
    first.completed_stages.append("DOCUMENT_COLLECTION")
    first.stage_blockers.append("R006")
    first.stage_history.append(procedure_snapshot().stage_history[0])
    assert second.completed_stages==second.stage_blockers==second.stage_history==[]
    first_entry=StageHistoryEntry(stage="PAYMENT",status="READY",trace_id="one",recorded_at="2026-11-18T09:00:00Z")
    second_entry=StageHistoryEntry(stage="PAYMENT",status="READY",trace_id="two",recorded_at=first_entry.recorded_at)
    first_entry.reason_refs.append("R010")
    assert second_entry.reason_refs==[]


@pytest.mark.parametrize("payload",[
    {"current_stage":"UNSUPPORTED"},
    {"stage_status":"FAILED"},
    {"completed_stages":["UNSUPPORTED"]},
    {"stage_blockers":[{"rule_id":"R006","passed":False}]},
    {"stage_history":[{"stage":"UNSUPPORTED","status":"READY","trace_id":"test","recorded_at":"2026-11-18T09:00:00Z"}]},
    {"stage_history":[{"stage":"PAYMENT","status":"FAILED","trace_id":"test","recorded_at":"2026-11-18T09:00:00Z"}]},
    {"stage_history":[{"stage":"PAYMENT","status":"READY","trace_id":"test","recorded_at":"not-a-date"}]},
    {"stage_history":[{"stage":"PAYMENT","status":"READY","trace_id":"test"}]},
    {"stage_history":[{"stage":"PAYMENT","status":"READY","trace_id":"test","recorded_at":"2026-11-18T09:00:00Z","official_event_date":"2026-11-17"}]},
    {"next_stage_prerequisites":["R006"]},
    {"available_actions":["pay"]},
    {"requirements":[]},
])
def test_procedure_schema_rejects_invalid_values_and_duplicated_state(payload):
    with pytest.raises(ValidationError):ProcedureState.model_validate(payload)


def test_procedure_and_history_round_trip_through_existing_serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    original=procedure_snapshot()
    for value in (original,*original.stage_history):
        restored=serde.loads_typed(serde.dumps_typed(value))
        assert type(restored) is type(value) and restored==value
    assert isinstance(serde.loads_typed(serde.dumps_typed(original)).stage_history[0],StageHistoryEntry)


def test_procedure_survives_sqlite_recreation_and_separate_python_process(tmp_path):
    db=tmp_path/"procedure.db";original=procedure_snapshot();config=thread("procedure-user")
    with sqlite_checkpointer(db) as saver:
        build_graph(saver).update_state(config,{"user_id":"procedure-user","procedure":original})
    with sqlite_checkpointer(db) as saver:
        restored=build_graph(saver).get_state(config).values["procedure"]
    assert type(restored) is ProcedureState and restored==original
    assert type(restored.stage_history[0]) is StageHistoryEntry
    code="""
import json,sys
from src.core.graph import build_graph
from src.core.state import ProcedureState,StageHistoryEntry
from src.memory.checkpointer import sqlite_checkpointer
with sqlite_checkpointer(sys.argv[1]) as saver:
    p=build_graph(saver).get_state({'configurable':{'thread_id':'procedure-user'}}).values['procedure']
    print(json.dumps({'procedure':p.model_dump(mode='json'),'typed':isinstance(p,ProcedureState),
                      'history_typed':all(isinstance(e,StageHistoryEntry) for e in p.stage_history)}))
"""
    result=subprocess.run([sys.executable,"-c",code,str(db)],cwd=PROJECT_ROOT,capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(result.stdout)=={"procedure":original.model_dump(mode="json"),"typed":True,"history_typed":True}


def test_procedure_thread_isolation_survives_independent_updates_and_reopens(tmp_path):
    db=tmp_path/"isolated-procedures.db"
    first=procedure_snapshot();second=ProcedureState()
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        graph.update_state(thread("first"),{"procedure":first})
        graph.update_state(thread("second"),{"procedure":second})
    updated=second.model_copy(update={"stage_status":"BLOCKED","stage_blockers":["R006"]},deep=True)
    with sqlite_checkpointer(db) as saver:
        build_graph(saver).update_state(thread("second"),{"procedure":updated})
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        assert graph.get_state(thread("first")).values["procedure"]==first
        assert graph.get_state(thread("second")).values["procedure"]==updated
        assert graph.get_state(thread("new-third")).values=={}


@pytest.mark.parametrize("explicit_none",[False,True])
def test_legacy_checkpoint_without_procedure_does_not_infer_completion(tmp_path,explicit_none):
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    db=tmp_path/"legacy.db";config=thread("legacy")
    legacy={"user_id":"legacy","residence_permit_context":ResidencePermitContext(
        online_application_completed=True,fee_receipt_present=True,fingerprint_step_completed=True,
        gi_number_present=True,iro_preverification_confirmed=True,submitted_to_iro=True),
        "completed_steps":["payment","fingerprint","iro_submission"]}
    if explicit_none:legacy["procedure"]=None
    with sqlite_checkpointer(db) as saver:
        # Representative pre-Phase-1 checkpoint: no new Pydantic types present.
        saver.serde=JsonPlusSerializer(allowed_msgpack_modules=[t for t in CHECKPOINT_TYPES if t[1] not in {"ProcedureState","StageHistoryEntry"}])
        build_graph(saver).update_state(config,legacy)
        values=saver.get_tuple(config).checkpoint["channel_values"]
        assert ("procedure" in values)==explicit_none
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        assert graph.get_state(config).values.get("procedure") is None
        with patch("src.agents.planner.create_action_plan",return_value=proposal()):
            result=graph.invoke(new_run(UserEvent(kind="generate_plan"),"legacy"),config)
        assert result["__interrupt__"] and result.get("procedure") is None
        assert result["completed_steps"]==legacy["completed_steps"]


@pytest.mark.parametrize("kind",["message","generate_plan","reevaluate"])
def test_procedure_is_preserved_by_new_run_and_actual_graph_execution(tmp_path,kind):
    db=tmp_path/"runs.db";config=thread("runs");original=procedure_snapshot()
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        graph.update_state(config,{"procedure":original})
        updates=new_run(UserEvent(kind=kind,text="Review next steps"),"runs")
        assert "procedure" not in updates
        with patch("src.agents.planner.create_action_plan",return_value=proposal()) as planner:
            result=graph.invoke(updates,config)
        expected = original
        if kind=="reevaluate":
            from src.core.procedure import procedure_evidence_update
            expected = procedure_evidence_update(result).get("procedure", original)
            # A factual re-check refreshes blockers/status, never lifecycle/history.
            assert result["procedure"].current_stage==original.current_stage
            assert result["procedure"].completed_stages==original.completed_stages
            assert result["procedure"].stage_history==original.stage_history
            assert result["procedure"].stage_status=="BLOCKED"
            assert "evidence:complete_submission_file" in result["procedure"].stage_blockers
        assert result["procedure"]==expected
        if kind=="reevaluate":planner.assert_not_called()
        else:planner.assert_called_once()
    with sqlite_checkpointer(db) as saver:
        assert build_graph(saver).get_state(config).values["procedure"]==expected


@pytest.mark.parametrize("approve",[True,False])
def test_plan_approval_or_denial_never_advances_procedure(tmp_path,approve):
    db=tmp_path/"approval-procedure.db";config=thread("approval-procedure");original=ProcedureState()
    request=new_run(UserEvent(kind="message",text="Plan the next steps"),"approval-procedure")
    request.update(procedure=original,completed_steps=["historical-plan-step"])
    with sqlite_checkpointer(db) as saver:
        with patch("src.agents.planner.create_action_plan",return_value=proposal()):
            result=build_graph(saver).invoke(request,config)
        assert result["__interrupt__"] and result["procedure"]==original
    with sqlite_checkpointer(db) as saver:
        result=build_graph(saver).invoke(Command(resume=approve),config)
        assert result["run_status"]==("APPROVED" if approve else "APPROVAL_DENIED")
        assert result["procedure"]==original
        assert result["completed_steps"]==["historical-plan-step"]
        c=result["residence_permit_context"]
        assert c.fee_receipt_present is c.fingerprint_step_completed is c.submitted_to_iro is None
    with sqlite_checkpointer(db) as saver:
        assert build_graph(saver).get_state(config).values["procedure"]==original


def test_adding_procedure_preserves_documents_evidence_confirmations_and_requirements(tmp_path):
    from src.document_processing.schemas import SourceEvidence
    from src.workers.rules_engine import evaluate_requirements
    info=StudentCertificate(holder_name="Alex Example",student_id="TEST-0001",institution="Example University",issue_date="2026-10-01",
        source_pages={"student_id":1},source_snippets={"student_id":"Student ID: TEST-0001"},
        source_evidence=[SourceEvidence(field="student_id",original_value="TEST-0001",normalized_value="TEST-0001",snippet="Student ID: TEST-0001",line_number=1,page_number=1)])
    document=Document(doc_id="student",file_path="synthetic.txt",doc_type="student_certificate",status="analyzed",info=info)
    source=VerifiedInput(value="TEST-0001",origin="source_grounded",document_id="student",field_name="student_id",source_page=1,source_snippet="Student ID: TEST-0001",trace_id="source-trace")
    confirmed=VerifiedInput(value="Alex Reviewed",origin="user_confirmed",document_id="student",field_name="holder_name",confirmed_by="synthetic-user",trace_id="confirmation-trace")
    seed={"documents":[document],"extracted_facts":{"student":info.model_dump(mode="json")},
          "verification_records":{"student":{"student_id":source,"holder_name":confirmed}},
          "residence_permit_context":ResidencePermitContext(holder_name="Alex Reviewed",student_id="TEST-0001"),
          "verified_context_fields":["holder_name","student_id"],"verification_conflicts":[]}
    seed["relevant_requirements"]=evaluate_requirements(seed)
    seed["blockers"]=[r.rule_id for r in seed["relevant_requirements"] if r.severity=="blocking" and r.passed is not True]
    untouched=deepcopy(seed);db=tmp_path/"evidence-procedure.db";config=thread("evidence-procedure")
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver);graph.update_state(config,seed)
        before=deepcopy(graph.get_state(config).values)
        graph.update_state(config,{"procedure":procedure_snapshot()})
    with sqlite_checkpointer(db) as saver:
        after=build_graph(saver).get_state(config).values
    assert {key:value for key,value in after.items() if key!="procedure"}==before
    assert seed==untouched
    assert after["documents"][0].info.holder_name=="Alex Example"
    assert after["verification_records"]["student"]["holder_name"].value=="Alex Reviewed"
    assert after["verification_records"]["student"]["student_id"].origin=="source_grounded"
