"""src/twin/store.py: schema, integrity, history and every table. Ports Neda's v1 store tests."""
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timezone

import pytest

from src.models import Issue
from src.twin.models import Episode, Event, Fact, ProposedAction, RunRecord, RunRuleResult, Simulation, TwinDocument
from src.twin.store import SCHEMA_VERSION, SchemaVersionError, TwinStore, schema_statements

USER = "synthetic_student_01"


def fact(field="age", value=22, **extra):
    data = dict(user_id=USER, procedure_id="ikamet_first", field=field, value=value, provenance="user_stated",
                source_ref="app", confirmed=True)
    data.update(extra)
    return Fact(**data)


def test_new_database_has_current_schema_and_no_users(store):
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert not store.user_exists("nobody") and store.current_facts("nobody", "ikamet_first") == []


@pytest.mark.parametrize("version", [1, 99])
def test_other_schema_versions_are_refused(db_path, version):
    connection = sqlite3.connect(db_path)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.close()
    with pytest.raises(SchemaVersionError):
        TwinStore(db_path)


def test_existing_langgraph_tables_do_not_block_the_twin(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE checkpoints (thread_id TEXT)")
    connection.close()
    with TwinStore(db_path) as twin:
        twin.add_fact(fact())
        assert len(twin.current_facts(USER, "ikamet_first")) == 1


def test_new_fact_from_same_source_closes_the_previous_one(store):
    first = store.add_fact(fact(value=22))
    second = store.add_fact(fact(value=23))
    current = store.current_facts(USER, "ikamet_first")
    history = store.fact_history(USER, "ikamet_first", "age")
    assert [f.value for f in current] == [23]
    assert [f.value for f in history] == [22, 23]
    assert history[0].valid_to == second.valid_from and first.valid_to is None


def test_facts_from_different_sources_coexist(store):
    store.add_fact(fact(source_ref="doc1", provenance="document"))
    store.add_fact(fact(source_ref="doc2", provenance="document"))
    assert len(store.current_facts(USER, "ikamet_first")) == 2


def test_confirming_keeps_history(store):
    stated = store.add_fact(fact(confirmed=False, source_ref="chat"))
    confirmed = store.confirm_fact(USER, stated.fact_id)
    assert confirmed.confirmed and store.get_fact(stated.fact_id).valid_to is not None
    with pytest.raises(ValueError):
        store.confirm_fact(USER, stated.fact_id)
    with pytest.raises(ValueError):
        store.confirm_fact("someone_else", confirmed.fact_id)


def test_close_source_ends_all_its_facts(store):
    store.add_fact(fact(field="age", source_ref="doc1", provenance="document"))
    store.add_fact(fact(field="holder_name", value="Alex", source_ref="doc1"))
    store.add_fact(fact(field="nationality", value="IT", source_ref="doc2"))
    assert store.close_source(USER, "ikamet_first", "doc1") == 2
    assert [f.field for f in store.current_facts(USER, "ikamet_first")] == ["nationality"]


@pytest.mark.parametrize("value", [date(2027, 1, 31), True, "text", 7, None])
def test_fact_values_keep_their_type(store, value):
    assert store.add_fact(fact(value=value)).value == value


def test_database_rejects_bad_dates_even_without_the_store(store):
    store.ensure_user(USER)
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("""INSERT INTO facts (user_id, procedure_id, field, value_type, value, provenance,
                               source_ref, confirmed, valid_from) VALUES (?, 'p', 'd', 'date', '2026-02-30',
                               'document', 'x', 1, '2026-11-18T09:00:00+00:00')""", (USER,))


def test_foreign_keys_are_enforced(store):
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("""INSERT INTO events (user_id, kind, direction, payload, created_at)
                               VALUES ('ghost', 'k', 'internal', '{}', '2026-11-18T09:00:00+00:00')""")


def test_failed_transaction_rolls_back_everything(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.add_fact(fact())
            store.add_event(Event(user_id=USER, kind="x", direction="internal"))
            raise RuntimeError("boom")
    assert store.current_facts(USER, "ikamet_first") == [] and store.list_events(USER) == []


def test_delete_user_forgets_everything(store):
    store.add_fact(fact())
    store.save_document(USER, TwinDocument(doc_id="d1", file_path="x.pdf", status="pending"))
    store.add_event(Event(user_id=USER, kind="x", direction="internal"))
    store.delete_user(USER)
    assert not store.user_exists(USER)
    assert store.fact_history(USER, "ikamet_first") == [] and store.list_documents(USER, include_deleted=True) == []


def test_documents_roundtrip_with_observation(store):
    store.save_document(USER, TwinDocument(doc_id="d1", file_path="a.pdf", status="pending"))
    saved = store.save_document(USER, TwinDocument(doc_id="d1", file_path="a.pdf", status="analyzed",
                                                   doc_type="passport_identity_page", observation={"grounded": {"x": 1}}))
    assert saved.observation == {"grounded": {"x": 1}} and saved.status == "analyzed"
    store.save_document(USER, saved.model_copy(update={"status": "deleted"}))
    assert store.list_documents(USER) == [] and len(store.list_documents(USER, include_deleted=True)) == 1


def test_events_keep_direction_and_order(store):
    store.add_event(Event(user_id=USER, kind="document_uploaded", direction="real_to_twin", payload={"doc_id": "d1"}))
    store.add_event(Event(user_id=USER, kind="artifact_executed", direction="twin_to_real"))
    events = store.list_events(USER)
    assert [e.direction for e in events] == ["real_to_twin", "twin_to_real"] and events[0].payload == {"doc_id": "d1"}
    assert len(store.list_events(USER, kind="artifact_executed")) == 1


def test_proposal_lifecycle(store):
    store.add_proposal(ProposedAction(action_id="a1", user_id=USER, procedure_id="ikamet_first", tool="create_ics",
                                      title="Reminder", payload={"ics": "BEGIN:VCALENDAR"}, refs=["R004"]))
    assert store.get_proposal("a1").status == "proposed" and len(store.list_proposals(USER, "proposed")) == 1
    decided = store.decide_proposal("a1", "executed", artifact_path="data/outbox/x.ics")
    assert decided.status == "executed" and decided.decided_at is not None and decided.refs == ["R004"]


def test_latest_simulation(store):
    for accepted in (False, True):
        store.add_simulation(Simulation(user_id=USER, procedure_id="ikamet_first", stage_id="iro_submission",
                                        accepted=accepted, facts_hash="h", result={"failing": []}))
    assert store.latest_simulation(USER, "ikamet_first", "iro_submission").accepted is True
    assert store.latest_simulation(USER, "ikamet_first", "fingerprint") is None


def episode(**extra):
    data = dict(user_id=USER, procedure_id="ikamet_first", failure_type="CLAIM_MISMATCH",
                validator_error="claim differs from evaluate_rules", corrective_action="claims copied from evaluation",
                outcome="accepted", correction_validated=True, rule_or_source_id="R015")
    data.update(extra)
    return Episode(**data)


def test_unvalidated_episodes_are_refused(store):
    with pytest.raises(ValueError):
        store.add_episode(episode(correction_validated=False))


def test_episodes_are_found_by_procedure_and_failure_type_newest_first(store):
    store.add_episode(episode(corrective_action="first"))
    store.add_episode(episode(corrective_action="second"))
    store.add_episode(episode(failure_type="UNSUPPORTED_DATE"))
    found = store.find_episodes("ikamet_first", "CLAIM_MISMATCH")
    assert [e.corrective_action for e in found] == ["second", "first"]
    assert len(store.find_episodes("ikamet_first")) == 3 and store.find_episodes("other") == []


def test_episodes_are_retired_when_their_rule_changes(store):
    store.add_episode(episode())
    assert store.invalidate_episodes("R015") == 1
    assert store.find_episodes("ikamet_first") == []


def test_runs_are_logged_with_issues(store):
    start = datetime(2026, 11, 18, 9, tzinfo=timezone.utc)
    store.log_run(RunRecord(trace_id="t1", user_id=USER, procedure_id="ikamet_first", goal="apply", status="answered",
                            llm_calls=4, started_at=start, finished_at=start,
                            issues=[Issue(code="CLAIM_MISMATCH", message="m", item="R015")]))
    run = store.list_runs(USER)[0]
    assert run.llm_calls == 4 and run.issues[0].code == "CLAIM_MISMATCH"


def test_runs_keep_the_rule_verdicts_at_their_end(store):
    start = datetime(2026, 11, 18, 9, tzinfo=timezone.utc)
    verdicts = [RunRuleResult(rule_id="R006", status="unknown", severity="blocking", missing_fields=["passport_expiry_date"]),
                RunRuleResult(rule_id="R027", status="pass", severity="guardrail")]
    store.log_run(RunRecord(trace_id="t2", user_id=USER, procedure_id="ikamet_first", goal="apply", status="answered",
                            started_at=start, finished_at=start, results=verdicts))
    assert store.list_runs(USER)[0].results == verdicts
    store.delete_user(USER)
    assert store._conn.execute("SELECT COUNT(*) FROM run_results").fetchone()[0] == 0


def old_twin(db_path, version: int, missing: tuple[str, ...]) -> None:
    """A Twin file as an older version wrote it: without the tables added later, with one run."""
    connection = sqlite3.connect(db_path)
    for statement in schema_statements():
        if not any(statement.startswith(f"CREATE TABLE {table} ") for table in missing):
            connection.execute(statement)
    connection.execute("INSERT INTO users (user_id, created_at, updated_at) VALUES ('old', '2026-11-18T09:00:00+00:00', "
                       "'2026-11-18T09:00:00+00:00')")
    connection.execute("INSERT INTO runs (trace_id, user_id, goal, status, started_at, finished_at) VALUES "
                       "('t0', 'old', 'g', 'answered', '2026-11-18T09:00:00+00:00', '2026-11-18T09:00:01+00:00')")
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()


@pytest.mark.parametrize("version,missing", [(2, ("run_results", "chat_messages")), (3, ("chat_messages",))])
def test_an_older_twin_is_upgraded_without_losing_data(db_path, version, missing):
    """v2 files (before run_results) and v3 files (before chat_messages) gain the new tables; their runs stay."""
    old_twin(db_path, version, missing)
    with TwinStore(db_path) as store:
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
        [run] = store.list_runs("old")
        assert run.trace_id == "t0" and run.results == []
        assert store.list_chat("old") == []


def test_chat_messages_are_kept_in_order_and_cleared_per_user(store):
    from src.twin.models import ChatMessage
    store.add_chat_message(ChatMessage(user_id="u1", role="user", content="What is missing?"))
    saved = store.add_chat_message(ChatMessage(user_id="u1", role="assistant", content="Your insurance.",
                                               details={"sources": ["GOV_FAQ"]}))
    store.add_chat_message(ChatMessage(user_id="u2", role="user", content="Hello"))
    assert saved.message_id and saved.created_at is not None
    assert [(m.role, m.content) for m in store.list_chat("u1")] == [("user", "What is missing?"), ("assistant", "Your insurance.")]
    assert store.list_chat("u1")[1].details == {"sources": ["GOV_FAQ"]}
    assert store.clear_chat("u1") == 2 and store.list_chat("u1") == [] and len(store.list_chat("u2")) == 1
    store.delete_user("u2")
    assert store.list_chat("u2") == []


def test_vector_cache_detects_changed_text(store):
    store.put_vector("ikamet_first", "SRC#0", "bge-m3", "SRC", "hash1", [0.5, 1.0])
    assert store.get_vector("ikamet_first", "SRC#0", "bge-m3", "hash1") == [0.5, 1.0]
    assert store.get_vector("ikamet_first", "SRC#0", "bge-m3", "hash2") is None
    assert store.get_vector("ikamet_first", "SRC#0", "other-model", "hash1") is None


def test_state_survives_a_fresh_python_process(db_path):
    """Neda's restart guarantee: only the SQLite file carries state between processes."""
    script = ("import sys; from src.twin.store import TwinStore; from src.twin.models import Fact\n"
              "with TwinStore(sys.argv[1]) as s: s.add_fact(Fact(user_id='u', procedure_id='p', field='age', value=30,"
              " provenance='user_stated', source_ref='app', confirmed=True))")
    subprocess.run([sys.executable, "-c", script, str(db_path)], check=True)
    with TwinStore(db_path) as reopened:
        assert [f.value for f in reopened.current_facts("u", "p")] == [30]
