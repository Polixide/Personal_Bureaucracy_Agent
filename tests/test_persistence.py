"""Offline persistence checks using real SQLite files and graph checkpoints."""
import json
import subprocess
import sys
from datetime import date
from unittest.mock import patch

import pytest
from langgraph.types import Command

from config import PROJECT_ROOT
from src.core.graph import build_graph
from src.core.state import (
    ActionPlan, Document, PlanStep, ResidencePermitContext, UserEvent, new_run,
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
    assert len(after["relevant_requirements"]) == 29
    assert after["candidate_plan"][0].step_id == "ask"
    # The engine refreshes the deliberately stale seeded blocker list before
    # approval. All 26 blocking rules lack inputs in this fixture's context.
    assert after["blockers"] == [f"R{i:03}" for i in range(1, 27)]
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
