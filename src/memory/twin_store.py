"""Twin record: permanent, queryable copy of each student's state (schema.sql v2).

Division of labour
  * src/memory/checkpointer.py (LangGraph SqliteSaver) is the live source of
    truth. The approval interrupt needs it to pause and resume a run.
  * TwinStore keeps independent tables that mirror the checkpointed Digital
    Twin after every graph invocation (see twin_sync.py), plus the run log and
    validated reflection lessons. Evaluation, inspection and per-student
    deletion work on these tables.

Because the tables mirror the checkpoint, writes REPLACE: a document or fact
that the agent deleted (Marco's reducers drop status="deleted" documents and
facts set to None) is removed here too. A twin key absent from the state
leaves its table untouched.

Every value is encoded explicitly (codec.py) or by Pydantic against its
AgentState type, and each nested object must round-trip to an equal value, or
the write is refused and rolled back. Nothing is stored with a changed meaning.
"""
import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, get_type_hints

from pydantic import BaseModel, TypeAdapter

from src.core.state import AgentState, Document, Requirement, ValidationIssue, VerifiedInput
from src.memory.codec import (EncodingError, decode_date, decode_fact, decode_timestamp,
                              encode_date, encode_fact, encode_timestamp, utc_now)

SCHEMA_VERSION = 2
SCHEMA_FILE = Path(__file__).with_name("schema.sql")

# Twin fields stored as validated JSON in `twin_objects` (nested or evolving shapes).
OBJECT_KEYS = ("residence_permit_context", "approved_plan", "action_progress",
               "identity_resolutions", "verified_context_fields", "verification_conflicts",
               "procedure", "procedure_entry_intent", "knowledge_snapshot", "requirements_knowledge_version", "assistant_messages", "assistant_events",
               "assistant_diagnostics", "tool_proposals", "tool_outcomes", "pending_tool_proposal",
               "sandbox_form", "calendar_reminders", "email_drafts", "last_tool_outcome")
TWIN_KEYS = ("user_id", "active_procedure", "approved_plan_origin", "documents",
             "extracted_facts", "deadlines", "verification_records", "completed_steps",
             "blockers", *OBJECT_KEYS)
RUN_OUTCOMES = ("pass", "fail", "awaiting_approval", "incomplete")


_EMPTY = {"documents": list, "extracted_facts": dict, "deadlines": dict,
          "verification_records": dict, "completed_steps": list, "blockers": list}


def twin_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """The twin fields of `state`, with absent table-backed fields as empty containers.

    The record cannot tell "never set" from "set to empty"; comparing
    twin_view(live_state) with twin_view(store.load_twin(...)) ignores only that.
    """
    view = {key: state[key] for key in TWIN_KEYS if key in state}
    for key, empty in _EMPTY.items():
        if view.get(key) is None:
            view[key] = empty()
    return view


class SchemaVersionError(RuntimeError):
    """The database was written by a different schema version."""


class Lesson(BaseModel):
    """One validated reflection lesson (plan section 3.3). No chain-of-thought."""
    user_id: str
    procedure: str
    failure_type: str                    # usually a ValidationIssue code
    validator_error: str
    corrective_action: str
    outcome: str
    correction_validated: bool
    affected_item: str | None = None
    rule_or_source_id: str | None = None
    rule_version: str | None = None
    trace_id: str | None = None
    lesson_id: int | None = None
    created_at: datetime | None = None


class RunRecord(BaseModel):
    trace_id: str
    user_id: str
    procedure: str | None
    event_kind: str
    event_text: str
    event_file: str | None
    event_document: str | None
    run_status: str | None
    outcome: str
    retry_count: int
    final_response: str
    started_at: datetime
    finished_at: datetime
    issues: list[ValidationIssue]
    requirements: dict[str, str]         # rule_id -> passed | failed | missing_information


# ----------------------------------------------------- typed JSON helpers --
@cache
def _adapter(key: str) -> TypeAdapter:
    return TypeAdapter(get_type_hints(AgentState)[key])


_INFO_ADAPTER = TypeAdapter(Document.model_fields["info"].annotation)


def _dump_exact(adapter: TypeAdapter, value: Any, what: str) -> str:
    """JSON that Pydantic reads back to an EQUAL value, or EncodingError."""
    text = adapter.dump_json(value).decode()
    if adapter.validate_json(text) != value:
        raise EncodingError(f"{what} does not round-trip through JSON unchanged "
                            f"(a date or custom object inside an untyped field?)")
    return text


def _decode_document_info(text: str | None) -> dict:
    if text is None:
        return {"info": None}
    data = json.loads(text)
    if "observation" not in data:
        return {"info": _INFO_ADAPTER.validate_python(data)}
    return {
        "info": _INFO_ADAPTER.validate_python(data["observation"]),
        "analysis_diagnostics": data["analysis_diagnostics"],
        "prior_observations": data["prior_observations"],
    }


def _encode_document_info(document: Document) -> str | None:
    """Keep legacy observation JSON readable and use the same column for analysis metadata."""
    info = (_dump_exact(_INFO_ADAPTER, document.info, f"document {document.doc_id} info")
            if document.info is not None else None)
    if not (document.analysis_diagnostics or document.prior_observations):
        return info
    return json.dumps({
        "observation": json.loads(info) if info is not None else None,
        "analysis_diagnostics": document.analysis_diagnostics,
        "prior_observations": document.prior_observations,
    }, allow_nan=False)


def _encode_verified_value(value: Any) -> tuple[str, str | None]:
    value_type, text = encode_fact(value)
    if value_type not in ("null", "bool", "int", "str"):
        raise EncodingError(f"verified input value must be str/bool/int/None: {value!r}")
    return value_type, text


class TwinStore:
    """Read/write the twin record. One instance per process; close() when done.

    `clock` returns the current time as an aware datetime; tests pass a fixed one.
    """

    def __init__(self, path: str | Path, clock: Callable[[], datetime] = utc_now):
        self.path = str(path)
        self._clock = clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)   # explicit transactions
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    # ------------------------------------------------------------ plumbing --
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TwinStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing: an error anywhere rolls back every write in the block."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _now(self) -> str:
        return encode_timestamp(self._clock())

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            self._v2_columns()
            return
        if version != 0:
            raise SchemaVersionError(
                f"{self.path} has twin-record schema v{version}, this code expects "
                f"v{SCHEMA_VERSION}. The record mirrors the LangGraph checkpoint, so it "
                f"can be deleted and rebuilt by the next run of each student.")
        with self._transaction() as conn:
            for statement in SCHEMA_FILE.read_text().split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._v2_columns()

    def _v2_columns(self):
        columns={row[1] for row in self._conn.execute("PRAGMA table_info(documents)")}
        if "original_filename" not in columns:
            self._conn.execute("ALTER TABLE documents ADD COLUMN original_filename TEXT")

    # --------------------------------------------------------- Digital Twin --
    def load_twin(self, user_id: str) -> dict[str, Any]:
        """The stored twin fields for `user_id`, typed as in AgentState; {} if unknown."""
        conn = self._conn
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if user is None:
            return {}
        twin: dict[str, Any] = {"user_id": user_id}
        for key in ("active_procedure", "approved_plan_origin"):
            if user[key] is not None:
                twin[key] = user[key]

        twin["documents"] = [
            Document(doc_id=r["doc_id"], file_path=r["file_path"], status=r["status"], original_filename=r["original_filename"],
                     doc_type=r["doc_type"], issue_date=decode_date(r["issue_date"]),
                     expiry_date=decode_date(r["expiry_date"]),
                     processing_error=r["processing_error"], **_decode_document_info(r["info_json"]))
            for r in conn.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY position",
                                  (user_id,))]
        twin["extracted_facts"] = {
            r["key"]: decode_fact(r["value_type"], r["value"])
            for r in conn.execute("SELECT * FROM facts WHERE user_id = ? ORDER BY key", (user_id,))}
        twin["deadlines"] = {
            r["key"]: decode_date(r["due_date"])
            for r in conn.execute("SELECT * FROM deadlines WHERE user_id = ? ORDER BY key",
                                  (user_id,))}
        records: dict[str, dict[str, VerifiedInput]] = {}
        for r in conn.execute("SELECT * FROM verification_records WHERE user_id = ? "
                              "ORDER BY record_key, field_key", (user_id,)):
            records.setdefault(r["record_key"], {})[r["field_key"]] = VerifiedInput(
                value=decode_fact(r["value_type"], r["value"]), origin=r["origin"],
                document_id=r["document_id"], field_name=r["field_name"],
                source_page=r["source_page"], source_snippet=r["source_snippet"],
                confirmed_by=r["confirmed_by"], trace_id=r["trace_id"],
                derived_from=json.loads(r["derived_from"]),
                derivation_policy=r["derivation_policy"])
        twin["verification_records"] = records
        twin["completed_steps"] = [r["step_id"] for r in conn.execute(
            "SELECT step_id FROM completed_steps WHERE user_id = ? ORDER BY position", (user_id,))]
        twin["blockers"] = [r["rule_id"] for r in conn.execute(
            "SELECT rule_id FROM blockers WHERE user_id = ? ORDER BY position", (user_id,))]
        for r in conn.execute("SELECT key, value_json FROM twin_objects WHERE user_id = ?",
                              (user_id,)):
            twin[r["key"]] = _adapter(r["key"]).validate_json(r["value_json"])
        return twin

    def save_twin(self, state: Mapping[str, Any]) -> None:
        """Mirror the twin fields of `state` in one transaction."""
        with self._transaction() as conn:
            self._write_twin(conn, state)

    def _write_twin(self, conn: sqlite3.Connection, state: Mapping[str, Any]) -> None:
        user_id, now = state["user_id"], self._now()
        conn.execute(
            """INSERT INTO users (user_id, active_procedure, approved_plan_origin,
                                  created_at, updated_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (user_id) DO UPDATE SET
                 active_procedure = COALESCE(excluded.active_procedure, users.active_procedure),
                 approved_plan_origin = COALESCE(excluded.approved_plan_origin,
                                                 users.approved_plan_origin),
                 updated_at = excluded.updated_at""",
            (user_id, state.get("active_procedure"), state.get("approved_plan_origin"), now, now))

        if "documents" in state:
            docs = list(state["documents"] or [])
            self._keep_only(conn, "documents", "doc_id", user_id, [d.doc_id for d in docs])
            for position, doc in enumerate(docs):
                info = _encode_document_info(doc)
                conn.execute(
                    """INSERT INTO documents (user_id, doc_id, position, file_path, status,
                           doc_type, issue_date, expiry_date, processing_error, info_json,
                           updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (user_id, doc_id) DO UPDATE SET
                         position = excluded.position, file_path = excluded.file_path,
                         status = excluded.status, doc_type = excluded.doc_type,
                         issue_date = excluded.issue_date, expiry_date = excluded.expiry_date,
                         processing_error = excluded.processing_error,
                         info_json = excluded.info_json, updated_at = excluded.updated_at""",
                    (user_id, doc.doc_id, position, doc.file_path, doc.status, doc.doc_type,
                     encode_date(doc.issue_date), encode_date(doc.expiry_date),
                     doc.processing_error, info, now))

                conn.execute("UPDATE documents SET original_filename=? WHERE user_id=? AND doc_id=?",(doc.original_filename,user_id,doc.doc_id))

        if "extracted_facts" in state:
            facts = dict(state["extracted_facts"] or {})
            self._keep_only(conn, "facts", "key", user_id, list(facts))
            for key, value in facts.items():
                value_type, text = encode_fact(value)
                conn.execute(
                    """INSERT INTO facts (user_id, key, value_type, value, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (user_id, key) DO UPDATE SET value_type = excluded.value_type,
                         value = excluded.value, updated_at = excluded.updated_at""",
                    (user_id, key, value_type, text, now))

        if "deadlines" in state:
            conn.execute("DELETE FROM deadlines WHERE user_id = ?", (user_id,))
            conn.executemany("INSERT INTO deadlines (user_id, key, due_date) VALUES (?, ?, ?)",
                             [(user_id, key, encode_date(day))
                              for key, day in (state["deadlines"] or {}).items()])

        if "verification_records" in state:
            conn.execute("DELETE FROM verification_records WHERE user_id = ?", (user_id,))
            for record_key, fields in (state["verification_records"] or {}).items():
                for field_key, item in fields.items():
                    item = VerifiedInput.model_validate(item)       # dicts from old checkpoints
                    value_type, text = _encode_verified_value(item.value)
                    conn.execute(
                        """INSERT INTO verification_records (user_id, record_key, field_key,
                               document_id, field_name, value_type, value, origin, source_page,
                               source_snippet, confirmed_by, trace_id, derived_from,
                               derivation_policy)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (user_id, record_key, field_key, item.document_id, item.field_name,
                         value_type, text, item.origin, item.source_page, item.source_snippet,
                         item.confirmed_by, item.trace_id, json.dumps(item.derived_from),
                         item.derivation_policy))

        if "completed_steps" in state:
            steps = list(dict.fromkeys(state["completed_steps"] or []))   # dedupe, keep order
            self._keep_only(conn, "completed_steps", "step_id", user_id, steps)
            for position, step_id in enumerate(steps):
                conn.execute(                    # keep completed_at of steps already recorded
                    """INSERT INTO completed_steps (user_id, step_id, position, completed_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT (user_id, step_id) DO UPDATE SET position = excluded.position""",
                    (user_id, step_id, position, now))

        if "blockers" in state:
            conn.execute("DELETE FROM blockers WHERE user_id = ?", (user_id,))
            conn.executemany("INSERT INTO blockers (user_id, rule_id, position) VALUES (?, ?, ?)",
                             [(user_id, rule_id, i) for i, rule_id
                              in enumerate(dict.fromkeys(state["blockers"] or []))])

        for key in OBJECT_KEYS:
            if key in state:
                conn.execute(
                    """INSERT INTO twin_objects (user_id, key, value_json, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT (user_id, key) DO UPDATE SET
                         value_json = excluded.value_json, updated_at = excluded.updated_at""",
                    (user_id, key, _dump_exact(_adapter(key), state[key], key), now))

    @staticmethod
    def _keep_only(conn: sqlite3.Connection, table: str, column: str, user_id: str,
                   keep: list[str]) -> None:
        placeholders = ",".join("?" * len(keep))
        conn.execute(f"DELETE FROM {table} WHERE user_id = ? AND {column} NOT IN ({placeholders})",
                     (user_id, *keep))

    def completed_at(self, user_id: str) -> dict[str, datetime]:
        """When each completed step was first recorded."""
        return {r["step_id"]: decode_timestamp(r["completed_at"]) for r in self._conn.execute(
            "SELECT step_id, completed_at FROM completed_steps WHERE user_id = ?", (user_id,))}

    def user_ids(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT user_id FROM users ORDER BY user_id")]

    def delete_user(self, user_id: str) -> None:
        """Forget a student entirely: twin, run history and lessons (cascade)."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------- episodes --
    def commit_snapshot(self, state: Mapping[str, Any], outcome: str,
                        started_at: datetime) -> None:
        """Mirror the twin AND log the run atomically: both are stored or neither.

        Called again for the same trace_id (a run resumed after approval), the
        run row is updated in place and keeps its original started_at.
        """
        if outcome not in RUN_OUTCOMES:
            raise ValueError(f"unknown run outcome {outcome!r}")
        with self._transaction() as conn:
            self._write_twin(conn, state)
            if state.get("trace_id") and state.get("event") is not None:
                self._write_run(conn, state, outcome, started_at)

    def _write_run(self, conn: sqlite3.Connection, state: Mapping[str, Any], outcome: str,
                   started_at: datetime) -> None:
        event, trace_id = state["event"], state["trace_id"]
        # A resumed run keeps its original start. Read it first: SQLite checks
        # CHECK constraints on the proposed row before ON CONFLICT applies.
        existing = conn.execute("SELECT started_at FROM runs WHERE trace_id = ?",
                                (trace_id,)).fetchone()
        started = existing["started_at"] if existing else encode_timestamp(started_at)
        conn.execute(
            """INSERT INTO runs (trace_id, user_id, procedure, event_kind, event_text, event_file,
                   event_document, run_status, outcome, retry_count, final_response,
                   started_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (trace_id) DO UPDATE SET
                 run_status = excluded.run_status, outcome = excluded.outcome,
                 retry_count = excluded.retry_count, final_response = excluded.final_response,
                 finished_at = excluded.finished_at""",
            (trace_id, state["user_id"], state.get("active_procedure"), event.kind, event.text,
             event.file_path, getattr(event, "document_id", None), state.get("run_status"),
             outcome, state.get("retry_count", 0), state.get("final_response", ""),
             started, self._now()))
        conn.execute("DELETE FROM run_issues WHERE trace_id = ?", (trace_id,))
        conn.executemany(
            """INSERT INTO run_issues (trace_id, position, code, message, source, item, rule_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(trace_id, i, e.code, e.message, e.source, e.item, e.rule_id)
             for i, e in enumerate(state.get("validation_errors") or [])])
        conn.execute("DELETE FROM run_requirements WHERE trace_id = ?", (trace_id,))
        requirements: list[Requirement] = state.get("relevant_requirements") or []
        conn.executemany(
            """INSERT INTO run_requirements (trace_id, rule_id, status, severity, error_code,
                   missing_fields) VALUES (?, ?, ?, ?, ?, ?)""",
            [(trace_id, r.rule_id, r.status, r.severity, r.error_code,
              json.dumps(r.missing_fields)) for r in requirements])

    def list_runs(self, user_id: str) -> list[RunRecord]:
        """Run history for one student, oldest first."""
        records = []
        for r in self._conn.execute(
                "SELECT * FROM runs WHERE user_id = ? ORDER BY started_at, rowid", (user_id,)):
            issues = [ValidationIssue(code=i["code"], message=i["message"], source=i["source"],
                                      item=i["item"], rule_id=i["rule_id"])
                      for i in self._conn.execute(
                          "SELECT * FROM run_issues WHERE trace_id = ? ORDER BY position",
                          (r["trace_id"],))]
            requirements = {q["rule_id"]: q["status"] for q in self._conn.execute(
                "SELECT rule_id, status FROM run_requirements WHERE trace_id = ? ORDER BY rule_id",
                (r["trace_id"],))}
            records.append(RunRecord(
                **{k: r[k] for k in ("trace_id", "user_id", "procedure", "event_kind",
                                     "event_text", "event_file", "event_document", "run_status",
                                     "outcome", "retry_count", "final_response")},
                started_at=decode_timestamp(r["started_at"]),
                finished_at=decode_timestamp(r["finished_at"]),
                issues=issues, requirements=requirements))
        return records

    # ---------------------------------------------------- reflection memory --
    def add_lesson(self, lesson: Lesson) -> int:
        """Store a validated lesson. Unvalidated corrections are refused (plan 3.4)."""
        if not lesson.correction_validated:
            raise ValueError("only corrections that passed Gate 1 or a human check are stored")
        with self._transaction() as conn:
            return self._insert_lesson(conn, lesson)

    def add_lessons_once(self, trace_id: str, lessons: list[Lesson]) -> int:
        """Store a run's lessons unless that run already has lessons (idempotent sync)."""
        if any(not lesson.correction_validated for lesson in lessons):
            raise ValueError("only corrections that passed Gate 1 or a human check are stored")
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM lessons WHERE trace_id = ?", (trace_id,)).fetchone():
                return 0
            for lesson in lessons:
                self._insert_lesson(conn, lesson)
            return len(lessons)

    def _insert_lesson(self, conn: sqlite3.Connection, lesson: Lesson) -> int:
        return conn.execute(
            """INSERT INTO lessons (user_id, procedure, failure_type, affected_item,
                   validator_error, rule_or_source_id, rule_version, corrective_action,
                   outcome, correction_validated, trace_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (lesson.user_id, lesson.procedure, lesson.failure_type, lesson.affected_item,
             lesson.validator_error, lesson.rule_or_source_id, lesson.rule_version,
             lesson.corrective_action, lesson.outcome, lesson.trace_id, self._now())).lastrowid

    def find_lessons(self, user_id: str, procedure: str, failure_type: str | None = None,
                     affected_item: str | None = None, limit: int = 5) -> list[Lesson]:
        """Targeted retrieval (plan 3.4): same student and procedure, newest first,
        optionally narrowed by failure type and item. Invalidated lessons are skipped."""
        sql = ["SELECT * FROM lessons WHERE user_id = ? AND procedure = ?",
               "AND invalidated_at IS NULL"]
        args: list[Any] = [user_id, procedure]
        if failure_type is not None:
            sql.append("AND failure_type = ?")
            args.append(failure_type)
        if affected_item is not None:
            sql.append("AND affected_item = ?")
            args.append(affected_item)
        sql.append("ORDER BY created_at DESC, lesson_id DESC LIMIT ?")
        args.append(limit)
        return [Lesson(**{k: r[k] for k in r.keys()
                          if k not in ("created_at", "invalidated_at", "correction_validated")},
                       correction_validated=bool(r["correction_validated"]),
                       created_at=decode_timestamp(r["created_at"]))
                for r in self._conn.execute(" ".join(sql), args)]

    def invalidate_lessons(self, rule_or_source_id: str) -> int:
        """Retire every lesson tied to a rule/source that changed. Returns the count."""
        with self._transaction() as conn:
            return conn.execute(
                "UPDATE lessons SET invalidated_at = ? "
                "WHERE rule_or_source_id = ? AND invalidated_at IS NULL",
                (self._now(), rule_or_source_id)).rowcount
