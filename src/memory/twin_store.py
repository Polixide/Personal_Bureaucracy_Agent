"""SQLite persistence layer for the Digital Twin (Workstream: memory & evaluation).

`TwinStore` maps the persistent part of `AgentState` onto independent tables
(see schema.sql) instead of a LangGraph checkpoint blob, so each fact can be
queried, tested and deleted on its own.

What is stored
  * Digital Twin : users, documents, facts (extracted_facts), completed_steps, blockers
  * Episodes     : runs + run_issues, one row per graph run (PASS or FAIL)
  * Reflection   : lessons, validated corrections only (plan section 3.3-3.4)

Write semantics follow the reducers in src/core/state.py:
  * documents, facts : upsert (merged by id/key, nothing is dropped)
  * completed_steps, blockers : replaced by the list in the state
  * a key missing from the state leaves its table untouched
"""
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.core.state import AgentState, Document, ValidationIssue
from src.memory.codec import (decode_date, decode_fact, decode_timestamp, encode_date,
                              encode_fact, encode_timestamp, utc_now)

SCHEMA_VERSION = 1
SCHEMA_FILE = Path(__file__).with_name("schema.sql")
TWIN_KEYS = ("user_id", "active_procedure", "documents", "extracted_facts",
             "completed_steps", "blockers")


class SchemaVersionError(RuntimeError):
    """The database was written by a different schema version."""


class Lesson(BaseModel):
    """One validated reflection lesson (plan section 3.3). No chain-of-thought."""
    user_id: str
    procedure: str
    failure_type: str                    # usually a ValidationIssue code, e.g. "DATE_CONFLICT"
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
    outcome: str                         # "pass" | "fail"
    retry_count: int
    final_response: str
    started_at: datetime
    finished_at: datetime
    issues: list[ValidationIssue]


class TwinStore:
    """Read/write the Digital Twin. One instance per process; close() when done.

    `clock` returns the current time as an aware datetime; tests pass a fixed one.
    """

    def __init__(self, path: str | Path, clock: Callable[[], datetime] = utc_now):
        self.path = str(path)
        self._clock = clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # autocommit mode: transactions are opened explicitly in _transaction()
        self._conn = sqlite3.connect(self.path, isolation_level=None)
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
            return
        if version != 0:
            raise SchemaVersionError(
                f"{self.path} has schema v{version}, this code expects v{SCHEMA_VERSION}")
        with self._transaction() as conn:
            for statement in SCHEMA_FILE.read_text().split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # --------------------------------------------------------- Digital Twin --
    def load_twin(self, user_id: str) -> dict[str, Any]:
        """The persistent AgentState fields for `user_id`; {} for an unknown user."""
        conn = self._conn
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if user is None:
            return {}
        documents = [
            Document(doc_id=r["doc_id"], file_path=r["file_path"], status=r["status"],
                     doc_type=r["doc_type"], issue_date=decode_date(r["issue_date"]),
                     expiry_date=decode_date(r["expiry_date"]))
            for r in conn.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY rowid",
                                  (user_id,))]
        facts = {r["key"]: decode_fact(r["value_type"], r["value"])
                 for r in conn.execute("SELECT * FROM facts WHERE user_id = ? ORDER BY key",
                                       (user_id,))}
        steps = [r["step_id"] for r in conn.execute(
            "SELECT step_id FROM completed_steps WHERE user_id = ? ORDER BY position", (user_id,))]
        blockers = [r["rule_id"] for r in conn.execute(
            "SELECT rule_id FROM blockers WHERE user_id = ? ORDER BY position", (user_id,))]
        twin = {"user_id": user_id, "documents": documents, "extracted_facts": facts,
                "completed_steps": steps, "blockers": blockers}
        if user["active_procedure"] is not None:
            twin["active_procedure"] = user["active_procedure"]
        return twin

    def save_twin(self, state: AgentState) -> None:
        """Persist the Digital Twin fields of `state` in one transaction."""
        with self._transaction() as conn:
            self._write_twin(conn, state)

    def _write_twin(self, conn: sqlite3.Connection, state: AgentState) -> None:
        user_id = state["user_id"]
        now = self._now()
        conn.execute(
            """INSERT INTO users (user_id, active_procedure, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (user_id) DO UPDATE SET
                 active_procedure = COALESCE(excluded.active_procedure, users.active_procedure),
                 updated_at = excluded.updated_at""",
            (user_id, state.get("active_procedure"), now, now))

        for doc in state.get("documents") or []:
            conn.execute(
                """INSERT INTO documents (user_id, doc_id, file_path, status, doc_type,
                                          issue_date, expiry_date, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user_id, doc_id) DO UPDATE SET
                     file_path = excluded.file_path, status = excluded.status,
                     doc_type = excluded.doc_type, issue_date = excluded.issue_date,
                     expiry_date = excluded.expiry_date, updated_at = excluded.updated_at""",
                (user_id, doc.doc_id, doc.file_path, doc.status, doc.doc_type,
                 encode_date(doc.issue_date), encode_date(doc.expiry_date), now))

        for key, value in (state.get("extracted_facts") or {}).items():
            value_type, text = encode_fact(value)
            conn.execute(
                """INSERT INTO facts (user_id, key, value_type, value, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (user_id, key) DO UPDATE SET
                     value_type = excluded.value_type, value = excluded.value,
                     updated_at = excluded.updated_at""",
                (user_id, key, value_type, text, now))

        if "completed_steps" in state:
            steps = list(dict.fromkeys(state["completed_steps"] or []))   # dedupe, keep order
            placeholders = ",".join("?" * len(steps))
            conn.execute(f"DELETE FROM completed_steps WHERE user_id = ? "
                         f"AND step_id NOT IN ({placeholders})", (user_id, *steps))
            for position, step_id in enumerate(steps):
                # keep the original completed_at of steps that were already done
                conn.execute(
                    """INSERT INTO completed_steps (user_id, step_id, position, completed_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT (user_id, step_id) DO UPDATE SET position = excluded.position""",
                    (user_id, step_id, position, now))

        if "blockers" in state:
            conn.execute("DELETE FROM blockers WHERE user_id = ?", (user_id,))
            conn.executemany(
                "INSERT INTO blockers (user_id, rule_id, position) VALUES (?, ?, ?)",
                [(user_id, rule_id, i)
                 for i, rule_id in enumerate(dict.fromkeys(state["blockers"] or []))])

    def completed_at(self, user_id: str) -> dict[str, datetime]:
        """When each completed step was first recorded."""
        return {r["step_id"]: decode_timestamp(r["completed_at"]) for r in self._conn.execute(
            "SELECT step_id, completed_at FROM completed_steps WHERE user_id = ?", (user_id,))}

    def delete_user(self, user_id: str) -> None:
        """Forget a user entirely: twin, run history and lessons (cascade)."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------- episodes --
    def commit_run(self, final_state: AgentState, started_at: datetime) -> None:
        """Save the twin AND log the run atomically: either both are stored or neither."""
        with self._transaction() as conn:
            self._write_twin(conn, final_state)
            self._write_run(conn, final_state, started_at)

    def _write_run(self, conn: sqlite3.Connection, state: AgentState,
                   started_at: datetime) -> None:
        event = state["event"]
        issues = state.get("validation_errors") or []
        conn.execute(
            """INSERT INTO runs (trace_id, user_id, procedure, event_kind, event_text, event_file,
                                 outcome, retry_count, final_response, started_at, finished_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (state["trace_id"], state["user_id"], state.get("active_procedure"), event.kind,
             event.text, event.file_path, "fail" if issues else "pass",
             state.get("retry_count", 0), state.get("final_response", ""),
             encode_timestamp(started_at), self._now()))
        conn.executemany(
            """INSERT INTO run_issues (trace_id, position, code, message, source, item, rule_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(state["trace_id"], i, issue.code, issue.message, issue.source, issue.item,
              issue.rule_id) for i, issue in enumerate(issues)])

    def list_runs(self, user_id: str) -> list[RunRecord]:
        """Run history for one user, oldest first."""
        records = []
        for r in self._conn.execute(
                "SELECT * FROM runs WHERE user_id = ? ORDER BY started_at, rowid", (user_id,)):
            issues = [ValidationIssue(code=i["code"], message=i["message"], source=i["source"],
                                      item=i["item"], rule_id=i["rule_id"])
                      for i in self._conn.execute(
                          "SELECT * FROM run_issues WHERE trace_id = ? ORDER BY position",
                          (r["trace_id"],))]
            records.append(RunRecord(
                **{k: r[k] for k in ("trace_id", "user_id", "procedure", "event_kind",
                                     "event_text", "event_file", "outcome", "retry_count",
                                     "final_response")},
                started_at=decode_timestamp(r["started_at"]),
                finished_at=decode_timestamp(r["finished_at"]), issues=issues))
        return records

    # ---------------------------------------------------- reflection memory --
    def add_lesson(self, lesson: Lesson) -> int:
        """Store a validated lesson. Unvalidated corrections are refused (plan 3.4)."""
        if not lesson.correction_validated:
            raise ValueError("only corrections that passed Gate 1 or a human check are stored")
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO lessons (user_id, procedure, failure_type, affected_item,
                       validator_error, rule_or_source_id, rule_version, corrective_action,
                       outcome, correction_validated, trace_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (lesson.user_id, lesson.procedure, lesson.failure_type, lesson.affected_item,
                 lesson.validator_error, lesson.rule_or_source_id, lesson.rule_version,
                 lesson.corrective_action, lesson.outcome, lesson.trace_id, self._now()))
            return cursor.lastrowid

    def find_lessons(self, user_id: str, procedure: str, failure_type: str | None = None,
                     affected_item: str | None = None, limit: int = 5) -> list[Lesson]:
        """Targeted retrieval (plan 3.4): same user and procedure, newest first,
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
