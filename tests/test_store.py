"""Twin record (schema v2): mirror semantics, typed round trips, integrity, run log, lessons."""
import sqlite3
from datetime import date, datetime, timezone

import pytest

from src.core.state import (ActionPlan, Document, PlanStep, Requirement, ResidencePermitContext,
                            UserEvent, ValidationIssue, VerifiedInput, new_run)
from src.document_processing.schemas import StudentCertificate
from src.memory.codec import EncodingError
from src.memory.twin_store import Lesson, SchemaVersionError, TwinStore, twin_view
from src.memory.twin_sync import lessons_from_run, run_outcome

T0 = datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc)
CERT_INFO = StudentCertificate(holder_name="Alex Example", student_id="SYN-001",
                               institution="Sabancı University", issue_date="2026-11-01")
CERT = Document(doc_id="cert.txt", file_path="/u/cert.txt", status="analyzed",
                doc_type="student_certificate", issue_date=date(2026, 11, 1), info=CERT_INFO)
INSURANCE = Document(doc_id="ins.pdf", file_path="/u/ins.pdf", status="replacement_pending",
                     doc_type="health_insurance", issue_date=date(2026, 9, 1),
                     expiry_date=date(2027, 6, 30))
PLAN = ActionPlan(summary="Next steps", actions=[
    PlanStep(step_id="ask", action="Provide entry details", priority=1, reason="Unknown",
             related_rule_id="R003", action_type="information", due_date=date(2026, 11, 20))])


def full_twin(**extra):
    return {
        "user_id": "u1", "active_procedure": "ikamet", "approved_plan_origin": "model",
        "documents": [CERT, INSURANCE],
        "extracted_facts": {"cert.txt": CERT_INFO.model_dump(mode="json"), "n": 3, "ok": True,
                            "appointment": date(2026, 11, 20)},
        "deadlines": {"appointment": date(2026, 11, 20), "legal_stay": date(2026, 12, 1)},
        "verification_records": {"cert.txt": {
            "holder_name": VerifiedInput(value="Alex Example", origin="source_grounded",
                                         document_id="cert.txt", field_name="holder_name",
                                         source_page=1, source_snippet="Holder name: Alex Example",
                                         trace_id="t0"),
            "student_certificate_present": VerifiedInput(
                value=True, origin="derived", document_id="cert.txt",
                field_name="student_certificate_present", trace_id="t0",
                derived_from=["holder_name"], derivation_policy="present_if_analyzed")}},
        "completed_steps": [], "blockers": ["R003", "R015"],
        "residence_permit_context": ResidencePermitContext(
            legal_stay_expiry_date=date(2026, 12, 1), exchange_student=True, age=22),
        "approved_plan": PLAN,
        "action_progress": {"t1:ask": {"action_id": "ask", "status": "declared_completed"}},
        "identity_resolutions": {"holder_name": {"value": "Alex Example", "signature": "abc"}},
        "verified_context_fields": ["exchange_student"], "verification_conflicts": [],
        **extra}


# ------------------------------------------------------------- round trip --
def test_unknown_user_loads_empty(store):
    assert store.load_twin("nobody") == {}


def test_full_twin_round_trip_keeps_types(store):
    state = full_twin()
    store.save_twin(state)
    loaded = store.load_twin("u1")
    assert twin_view(loaded) == twin_view(state)
    assert isinstance(loaded["documents"][0].info, StudentCertificate)
    assert loaded["deadlines"]["appointment"] == date(2026, 11, 20)
    assert isinstance(loaded["residence_permit_context"].legal_stay_expiry_date, date)
    assert loaded["approved_plan"].actions[0].due_date == date(2026, 11, 20)
    assert loaded["extracted_facts"]["appointment"] == date(2026, 11, 20)


def test_record_survives_reopening(db_path):
    with TwinStore(db_path) as store:
        store.save_twin(full_twin())
    with TwinStore(db_path) as store:
        assert twin_view(store.load_twin("u1")) == twin_view(full_twin())


def test_mirror_removes_deleted_documents_facts_and_deadlines(store):
    store.save_twin(full_twin())
    store.save_twin({"user_id": "u1", "documents": [CERT], "extracted_facts": {"n": 4},
                     "deadlines": {}})
    loaded = store.load_twin("u1")
    assert [d.doc_id for d in loaded["documents"]] == ["cert.txt"]
    assert loaded["extracted_facts"] == {"n": 4}
    assert loaded["deadlines"] == {}
    assert loaded["blockers"] == ["R003", "R015"]          # absent keys are untouched
    assert loaded["approved_plan"] == PLAN


def test_document_order_follows_the_state(store):
    store.save_twin({"user_id": "u1", "documents": [CERT, INSURANCE]})
    store.save_twin({"user_id": "u1", "documents": [INSURANCE, CERT]})
    assert [d.doc_id for d in store.load_twin("u1")["documents"]] == ["ins.pdf", "cert.txt"]


@pytest.mark.parametrize("status", ["pending", "analyzed", "rejected", "replacement_pending",
                                    "deleted"])
def test_every_document_status_is_accepted(store, status):
    doc = CERT.model_copy(update={"status": status})
    store.save_twin({"user_id": "u1", "documents": [doc]})
    assert store.load_twin("u1")["documents"][0].status == status


def test_completed_at_is_kept_when_a_step_is_saved_again(store):
    store.save_twin({"user_id": "u1", "completed_steps": ["s1"]})
    first = store.completed_at("u1")["s1"]
    store.save_twin({"user_id": "u1", "completed_steps": ["s1", "s2"]})
    times = store.completed_at("u1")
    assert times["s1"] == first and times["s2"] > first


def test_students_are_isolated(store):
    store.save_twin(full_twin())
    store.save_twin({"user_id": "u2", "blockers": ["R9"]})
    assert store.load_twin("u2")["documents"] == []
    assert store.load_twin("u1")["blockers"] == ["R003", "R015"]
    assert store.user_ids() == ["u1", "u2"]


# ------------------------------------------------------- dates & integrity --
def test_date_hidden_in_an_untyped_field_is_refused_and_rolled_back(store):
    """identity_resolutions is dict[str, Any]: a date there would come back as a string."""
    store.save_twin({"user_id": "u1", "blockers": ["R1"]})
    with pytest.raises(EncodingError):
        store.save_twin({"user_id": "u1", "blockers": ["R2"],
                         "identity_resolutions": {"seen_on": date(2026, 11, 1)}})
    assert store.load_twin("u1")["blockers"] == ["R1"]


def test_datetime_in_a_date_field_is_refused(store):
    bad = CERT.model_construct(**{**dict(CERT), "expiry_date": datetime(2027, 1, 1, 12, 0)})
    with pytest.raises(EncodingError):
        store.save_twin({"user_id": "u1", "documents": [bad]})


@pytest.mark.parametrize("bad", ["2026-02-30", "2026/11/20", "20261120", "tomorrow"])
def test_database_itself_rejects_bad_dates(store, bad):
    store.save_twin({"user_id": "u1"})
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("INSERT INTO deadlines (user_id, key, due_date) VALUES ('u1','x',?)",
                            (bad,))
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO documents (user_id, doc_id, position, file_path, status, expiry_date,"
            " updated_at) VALUES ('u1','x',0,'x','pending',?,'2026-11-18T09:00:00+00:00')", (bad,))


def test_database_rejects_unknown_status_and_orphans(store):
    store.save_twin({"user_id": "u1"})
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO documents (user_id, doc_id, position, file_path, status, updated_at)"
            " VALUES ('u1','x',0,'x','lost','2026-11-18T09:00:00+00:00')")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("INSERT INTO blockers (user_id, rule_id, position) "
                            "VALUES ('ghost', 'R1', 0)")


def test_old_schema_version_is_refused_with_advice(db_path):
    sqlite3.connect(db_path).execute("PRAGMA user_version = 1")
    with pytest.raises(SchemaVersionError, match="rebuilt"):
        TwinStore(db_path)


# ------------------------------------------------------------- run history --
def run_state(**extra):
    state = new_run(UserEvent(kind="upload", file_path="/u/cert.txt"), user_id="u1")
    return {**state, "documents": [CERT], "final_response": "done", "run_status": "VALIDATED",
            "relevant_requirements": [
                Requirement(rule_id="R003", description="d", passed=None,
                            status="missing_information", message="m",
                            missing_fields=["is_in_turkiye"]),
                Requirement(rule_id="R014", description="d", passed=True, status="passed",
                            message="m")], **extra}


def test_paused_run_is_updated_in_place_when_resumed(store):
    state = run_state()
    store.commit_snapshot(state, "awaiting_approval", started_at=T0)
    store.commit_snapshot({**state, "run_status": "APPROVED", "approved_plan": PLAN},
                          "pass", started_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    [run] = store.list_runs("u1")
    assert (run.outcome, run.run_status) == ("pass", "APPROVED")
    assert run.started_at == T0                                       # original start kept
    assert run.requirements == {"R003": "missing_information", "R014": "passed"}
    assert store.load_twin("u1")["approved_plan"] == PLAN


def test_run_issues_are_logged(store):
    issue = ValidationIssue(code="PREMATURE_SUBMISSION", message="too early", source="gate_1",
                            item="submit", rule_id="R023")
    store.commit_snapshot(run_state(validation_errors=[issue], run_status="PLAN_INVALID"),
                          "fail", T0)
    [run] = store.list_runs("u1")
    assert run.issues == [issue] and run.outcome == "fail"


def test_unknown_outcome_is_refused(store):
    with pytest.raises(ValueError):
        store.commit_snapshot(run_state(), "maybe", T0)


def test_commit_is_atomic(store):
    """If the twin part fails, the run is not logged either."""
    bad = run_state(identity_resolutions={"when": date(2026, 1, 1)})
    with pytest.raises(EncodingError):
        store.commit_snapshot(bad, "pass", T0)
    assert store.list_runs("u1") == [] and store.load_twin("u1") == {}


def test_delete_user_forgets_everything(store):
    state = run_state()
    store.commit_snapshot(state, "pass", T0)
    store.add_lesson(make_lesson(trace_id=state["trace_id"]))
    store.delete_user("u1")
    assert store.load_twin("u1") == {}
    assert store.list_runs("u1") == [] and store.find_lessons("u1", "ikamet") == []
    assert store._conn.execute("SELECT COUNT(*) FROM run_requirements").fetchone()[0] == 0


# ------------------------------------------------------- reflection memory --
def make_lesson(**overrides):
    fields = dict(user_id="u1", procedure="ikamet", failure_type="PREMATURE_SUBMISSION",
                  affected_item="submit_to_iro", validator_error="prerequisites unverified",
                  rule_or_source_id="R023", rule_version="1.0",
                  corrective_action="Revised plan: Obtain an accepted address document",
                  outcome="Revised plan passed Gate 1 after 1 repair(s)",
                  correction_validated=True)
    return Lesson(**{**fields, **overrides})


def test_unvalidated_lesson_is_refused_by_code_and_database(store):
    store.save_twin({"user_id": "u1"})
    with pytest.raises(ValueError):
        store.add_lesson(make_lesson(correction_validated=False))
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO lessons (user_id, procedure, failure_type, validator_error,"
            " corrective_action, outcome, correction_validated, created_at)"
            " VALUES ('u1','ikamet','X','e','a','o',0,'2026-11-18T09:00:00+00:00')")


def test_lessons_are_written_once_per_run(store):
    state = run_state()
    store.commit_snapshot(state, "pass", T0)
    assert store.add_lessons_once(state["trace_id"], [make_lesson(trace_id=state["trace_id"])]) == 1
    assert store.add_lessons_once(state["trace_id"], [make_lesson(trace_id=state["trace_id"])]) == 0
    assert len(store.find_lessons("u1", "ikamet")) == 1


def test_lessons_are_filtered_newest_first_and_isolated(store):
    store.save_twin({"user_id": "u1"})
    store.save_twin({"user_id": "u2"})
    old = store.add_lesson(make_lesson())
    new = store.add_lesson(make_lesson(corrective_action="another fix"))
    store.add_lesson(make_lesson(failure_type="DATE_CONFLICT"))
    store.add_lesson(make_lesson(user_id="u2"))
    found = store.find_lessons("u1", "ikamet", failure_type="PREMATURE_SUBMISSION")
    assert [lesson.lesson_id for lesson in found] == [new, old]
    assert store.find_lessons("u1", "other_procedure") == []


def test_lessons_are_invalidated_when_their_rule_changes(store):
    store.save_twin({"user_id": "u1"})
    store.add_lesson(make_lesson())
    store.add_lesson(make_lesson(rule_or_source_id="R017"))
    assert store.invalidate_lessons("R023") == 1
    assert [l.rule_or_source_id for l in store.find_lessons("u1", "ikamet")] == ["R017"]


# ------------------------------------------------------------ sync rules --
@pytest.mark.parametrize("values, next_nodes, expected", [
    ({"run_status": "VALIDATED"}, ("human_approval",), "awaiting_approval"),
    ({"run_status": "RUNNING"}, ("planner",), "incomplete"),
    ({"run_status": "APPROVED"}, (), "pass"),
    ({"run_status": "DOCUMENT_ANALYZED"}, (), "pass"),
    ({"run_status": "HUMAN_REVIEW_REQUIRED"}, (), "fail"),
    ({"run_status": "APPROVAL_DENIED"}, (), "fail"),
    ({"run_status": "APPROVED", "validation_errors": [object()]}, (), "fail"),
])
def test_run_outcome(values, next_nodes, expected):
    assert run_outcome(values, next_nodes) == expected


def repaired_run(status):
    error = {"code": "PREMATURE_SUBMISSION", "message": "prerequisites unverified",
             "source": "gate_1", "item": "submit_to_iro", "rule_id": "R023"}
    return {**run_state(run_status=status, retry_count=1, candidate_plan=PLAN.actions),
            "repair_history": [{"attempt": 1, "errors": [error, error]}]}


def test_lessons_come_only_from_repairs_that_passed_gate_1():
    [lesson] = lessons_from_run(repaired_run("VALIDATED"))          # duplicate error -> one
    assert (lesson.failure_type, lesson.rule_or_source_id, lesson.rule_version) == \
        ("PREMATURE_SUBMISSION", "R023", "1.0")
    assert lesson.corrective_action == "Fix only the structured Gate 1 errors; preserve valid actions and use canonical source-grounded candidates."
    assert lessons_from_run(repaired_run("HUMAN_REVIEW_REQUIRED")) == []
    assert lessons_from_run(run_state()) == []                       # no repair, no lesson


def test_record_survives_a_new_python_process(db_path):
    """Acceptance: state written by one process is read back, typed, by another."""
    import json
    import subprocess
    import sys

    from config import PROJECT_ROOT

    write = ("import sys; from datetime import date; from src.memory.twin_store import TwinStore;"
             "TwinStore(sys.argv[1]).save_twin({'user_id': 'u1', 'blockers': ['R015'],"
             " 'deadlines': {'appointment': date(2026, 11, 20)}})")
    read = ("import sys, json; from src.memory.twin_store import TwinStore;"
            "t = TwinStore(sys.argv[1]).load_twin('u1');"
            "print(json.dumps([t['blockers'], type(t['deadlines']['appointment']).__name__,"
            " str(t['deadlines']['appointment'])]))")
    for code in (write, read):
        done = subprocess.run([sys.executable, "-c", code, str(db_path)], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [["R015"], "date", "2026-11-20"]
