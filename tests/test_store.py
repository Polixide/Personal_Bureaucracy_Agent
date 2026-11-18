"""TwinStore: tables, write semantics, integrity constraints, reflection memory."""
import sqlite3
from datetime import date, datetime, timezone

import pytest

from src.core.state import Document, UserEvent, ValidationIssue, new_run
from src.memory.checkpointer import Lesson, SchemaVersionError, TwinStore
from src.memory.codec import EncodingError

INSURANCE = Document(doc_id="ins.pdf", file_path="/u/ins.pdf", status="analyzed",
                     doc_type="health_insurance", issue_date=date(2026, 11, 1),
                     expiry_date=date(2027, 11, 1))
CERT = Document(doc_id="cert.pdf", file_path="/u/cert.pdf")          # pending, no dates


def twin(**extra):
    return {"user_id": "u1", "active_procedure": "ikamet", **extra}


# ------------------------------------------------------------- round trip --
def test_unknown_user_loads_empty(store):
    assert store.load_twin("nobody") == {}


def test_full_twin_round_trip(store):
    state = twin(documents=[INSURANCE, CERT],
                 extracted_facts={"appointment_date": date(2026, 11, 20), "dormitory": True,
                                  "planned_stay_months": 5, "city": "Istanbul"},
                 completed_steps=["register_address"], blockers=["R015", "R017"])
    store.save_twin(state)
    loaded = store.load_twin("u1")
    assert loaded == {**state, "documents": [INSURANCE, CERT]}
    assert isinstance(loaded["extracted_facts"]["appointment_date"], date)


def test_documents_are_upserted_never_dropped(store):
    store.save_twin(twin(documents=[INSURANCE, CERT]))
    analyzed = CERT.model_copy(update={"status": "analyzed", "doc_type": "student_certificate"})
    store.save_twin(twin(documents=[analyzed]))                       # only the changed one
    docs = {d.doc_id: d for d in store.load_twin("u1")["documents"]}
    assert docs == {"ins.pdf": INSURANCE, "cert.pdf": analyzed}


def test_facts_are_merged(store):
    store.save_twin(twin(extracted_facts={"a": 1, "b": 2}))
    store.save_twin(twin(extracted_facts={"b": "two", "c": date(2026, 1, 1)}))
    assert store.load_twin("u1")["extracted_facts"] == {"a": 1, "b": "two",
                                                        "c": date(2026, 1, 1)}


def test_blockers_and_steps_are_replaced_and_ordered(store):
    store.save_twin(twin(blockers=["R3", "R1", "R2"], completed_steps=["s1", "s2"]))
    store.save_twin(twin(blockers=["R2", "R9"], completed_steps=["s2", "s1", "s3"]))
    loaded = store.load_twin("u1")
    assert loaded["blockers"] == ["R2", "R9"]
    assert loaded["completed_steps"] == ["s2", "s1", "s3"]


def test_missing_keys_leave_tables_untouched(store):
    store.save_twin(twin(blockers=["R1"], completed_steps=["s1"]))
    store.save_twin({"user_id": "u1"})
    loaded = store.load_twin("u1")
    assert loaded["blockers"] == ["R1"] and loaded["completed_steps"] == ["s1"]
    assert loaded["active_procedure"] == "ikamet"


def test_completed_at_is_kept_when_a_step_is_saved_again(store):
    store.save_twin(twin(completed_steps=["s1"]))
    first = store.completed_at("u1")["s1"]
    store.save_twin(twin(completed_steps=["s1", "s2"]))
    times = store.completed_at("u1")
    assert times["s1"] == first and times["s2"] > first


def test_users_are_isolated(store):
    store.save_twin(twin(documents=[INSURANCE], blockers=["R1"]))
    store.save_twin({"user_id": "u2", "blockers": ["R9"]})
    assert store.load_twin("u2")["documents"] == []
    assert store.load_twin("u2")["blockers"] == ["R9"]
    assert store.load_twin("u1")["blockers"] == ["R1"]


# ------------------------------------------------------- dates & integrity --
def test_datetime_in_a_date_field_is_refused_and_nothing_is_written(store):
    store.save_twin(twin(blockers=["R1"]))
    bad = INSURANCE.model_construct(**{**INSURANCE.model_dump(),
                                       "expiry_date": datetime(2027, 11, 1, 12, 0)})
    with pytest.raises(EncodingError):
        store.save_twin(twin(documents=[bad], blockers=["R2"]))
    loaded = store.load_twin("u1")                                    # transaction rolled back
    assert loaded["blockers"] == ["R1"] and loaded["documents"] == []


@pytest.mark.parametrize("bad", ["2026-02-30", "2026/11/20", "20261120", "tomorrow"])
def test_database_itself_rejects_bad_dates(store, bad):
    """Even a writer that bypasses TwinStore cannot store an invalid date."""
    store.save_twin(twin())
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO documents (user_id, doc_id, file_path, status, expiry_date, updated_at)"
            " VALUES ('u1', 'x', 'x', 'pending', ?, '2026-11-18T09:00:00+00:00')", (bad,))


def test_database_rejects_unknown_status(store):
    store.save_twin(twin())
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO documents (user_id, doc_id, file_path, status, updated_at)"
            " VALUES ('u1', 'x', 'x', 'lost', '2026-11-18T09:00:00+00:00')")


def test_foreign_keys_are_enforced(store):
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("INSERT INTO blockers (user_id, rule_id, position) "
                            "VALUES ('ghost', 'R1', 0)")


def test_newer_schema_version_is_refused(db_path):
    TwinStore(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(SchemaVersionError):
        TwinStore(db_path)


# ------------------------------------------------------------- run history --
def finished_run(user_id="u1", issues=()):
    state = new_run(UserEvent(kind="upload", file_path="ins.pdf"), user_id=user_id)
    return {**state, "documents": [INSURANCE], "validation_errors": list(issues),
            "final_response": "done"}


def test_commit_run_saves_twin_and_logs_the_run(store):
    issue = ValidationIssue(code="DATE_CONFLICT", message="expires early", source="gate_1",
                            item="attend_appointment", rule_id="R015")
    state = finished_run(issues=[issue])
    store.commit_run(state, started_at=datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc))
    [run] = store.list_runs("u1")
    assert run.trace_id == state["trace_id"] and run.outcome == "fail"
    assert run.issues == [issue] and run.event_file == "ins.pdf"
    assert run.finished_at >= run.started_at
    assert store.load_twin("u1")["documents"] == [INSURANCE]


def test_commit_run_is_atomic(store):
    """If logging the run fails, the twin is not saved either."""
    started = datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc)
    state = finished_run()
    store.commit_run(state, started)
    second = {**finished_run(), "trace_id": state["trace_id"],          # duplicate key
              "documents": [CERT]}
    with pytest.raises(sqlite3.IntegrityError):
        store.commit_run(second, started)
    assert store.load_twin("u1")["documents"] == [INSURANCE]
    assert len(store.list_runs("u1")) == 1


def test_delete_user_forgets_everything(store):
    store.commit_run(finished_run(), datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc))
    store.add_lesson(make_lesson())
    store.delete_user("u1")
    assert store.load_twin("u1") == {}
    assert store.list_runs("u1") == [] and store.find_lessons("u1", "ikamet") == []


# ------------------------------------------------------- reflection memory --
def make_lesson(**overrides):
    fields = dict(user_id="u1", procedure="ikamet", failure_type="DATE_CONFLICT",
                  affected_item="health_insurance",
                  validator_error="insurance expires 2026-11-10 before appointment 2026-11-20",
                  rule_or_source_id="R015", rule_version="1.0",
                  corrective_action="insert 'renew insurance' before the appointment",
                  outcome="revised plan passed Gate 1", correction_validated=True)
    return Lesson(**{**fields, **overrides})


def test_unvalidated_lesson_is_refused(store):
    store.save_twin(twin())
    with pytest.raises(ValueError):
        store.add_lesson(make_lesson(correction_validated=False))
    with pytest.raises(sqlite3.IntegrityError):                       # and by the DB itself
        store._conn.execute(
            "INSERT INTO lessons (user_id, procedure, failure_type, validator_error,"
            " corrective_action, outcome, correction_validated, created_at)"
            " VALUES ('u1','ikamet','X','e','a','o',0,'2026-11-18T09:00:00+00:00')")


def test_lessons_are_retrieved_by_filter_newest_first(store):
    store.save_twin(twin())
    store.save_twin({"user_id": "u2"})
    old = store.add_lesson(make_lesson())
    new = store.add_lesson(make_lesson(corrective_action="book a later appointment"))
    store.add_lesson(make_lesson(failure_type="MISSING_PREREQUISITE", affected_item="R021"))
    store.add_lesson(make_lesson(user_id="u2"))                       # other user
    found = store.find_lessons("u1", "ikamet", failure_type="DATE_CONFLICT")
    assert [l.lesson_id for l in found] == [new, old]
    assert all(l.user_id == "u1" for l in store.find_lessons("u1", "ikamet"))
    assert len(store.find_lessons("u1", "ikamet", limit=1)) == 1
    assert store.find_lessons("u1", "other_procedure") == []


def test_lessons_are_invalidated_when_their_rule_changes(store):
    store.save_twin(twin())
    store.add_lesson(make_lesson())
    store.add_lesson(make_lesson(rule_or_source_id="R017"))
    assert store.invalidate_lessons("R015") == 1
    assert [l.rule_or_source_id for l in store.find_lessons("u1", "ikamet")] == ["R017"]
