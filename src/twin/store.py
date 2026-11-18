"""SQLite storage of the Digital Twin (schema v2 in schema.sql). Adapted from Neda's TwinStore.

Write rules:
  * every write runs in a transaction (all-or-nothing); transactions nest safely;
  * a new fact from the same (field, provenance, source) closes the previous one (valid_to),
    so the Twin keeps the full history of what was observed or stated, and when;
  * instants come from an injectable clock so tests are deterministic.
"""
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import Issue
from src.twin.codec import decode_fact, decode_timestamp, encode_fact, encode_timestamp, utc_now
from src.twin.models import Episode, Event, Fact, ProposedAction, RunRecord, Simulation, TwinDocument

SCHEMA_VERSION = 2
SCHEMA_FILE = Path(__file__).with_name("schema.sql")


class SchemaVersionError(RuntimeError):
    """The database was written by a different schema version."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _instant(text: str | None) -> datetime | None:
    return decode_timestamp(text) if text else None


def schema_statements() -> list[str]:
    """SQL statements of schema.sql, with `--` comments removed before splitting on ';'."""
    lines = [line.split("--", 1)[0] for line in SCHEMA_FILE.read_text(encoding="utf-8").splitlines()]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


class TwinStore:
    """Read/write the Twin. One instance per process or app run; close() when done."""

    def __init__(self, path: str | Path, clock: Callable[[], datetime] = utc_now):
        self.path = str(path)
        self._clock = clock
        self._depth = 0
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
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
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing block. Nested blocks join the outer transaction."""
        if self._depth:
            self._depth += 1
            try:
                yield self._conn
            finally:
                self._depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._depth = 0

    def now(self) -> str:
        """Current instant as stored text."""
        return encode_timestamp(self._clock())

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise SchemaVersionError(f"{self.path} has schema v{version}, this code expects v{SCHEMA_VERSION}")
        with self.transaction() as conn:
            for statement in schema_statements():
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # --------------------------------------------------------------- users --
    def ensure_user(self, user_id: str) -> None:
        """Create the user row if needed (every user-owned write calls this)."""
        now = self.now()
        with self.transaction() as conn:
            conn.execute("""INSERT INTO users (user_id, created_at, updated_at) VALUES (?, ?, ?)
                            ON CONFLICT (user_id) DO UPDATE SET updated_at = excluded.updated_at""",
                         (user_id, now, now))

    def set_active_procedure(self, user_id: str, procedure_id: str | None) -> None:
        with self.transaction() as conn:
            self.ensure_user(user_id)
            conn.execute("UPDATE users SET active_procedure = ? WHERE user_id = ?", (procedure_id, user_id))

    def active_procedure(self, user_id: str) -> str | None:
        row = self._conn.execute("SELECT active_procedure FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row["active_procedure"] if row else None

    def user_exists(self, user_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is not None

    def delete_user(self, user_id: str) -> None:
        """Forget a user entirely (documents, facts, events, proposals, runs, episodes)."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    # ----------------------------------------------------------- documents --
    def save_document(self, user_id: str, document: TwinDocument) -> TwinDocument:
        """Insert or replace one document row."""
        now = self.now()
        observation = _json(document.observation) if document.observation is not None else None
        with self.transaction() as conn:
            self.ensure_user(user_id)
            conn.execute(
                """INSERT INTO documents (user_id, doc_id, file_path, status, doc_type, observation,
                                          processing_error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user_id, doc_id) DO UPDATE SET
                     file_path = excluded.file_path, status = excluded.status, doc_type = excluded.doc_type,
                     observation = excluded.observation, processing_error = excluded.processing_error,
                     updated_at = excluded.updated_at""",
                (user_id, document.doc_id, document.file_path, document.status, document.doc_type,
                 observation, document.processing_error, now, now))
        return self.get_document(user_id, document.doc_id)

    def get_document(self, user_id: str, doc_id: str) -> TwinDocument | None:
        row = self._conn.execute("SELECT * FROM documents WHERE user_id = ? AND doc_id = ?",
                                 (user_id, doc_id)).fetchone()
        return self._document(row) if row else None

    def list_documents(self, user_id: str, include_deleted: bool = False) -> list[TwinDocument]:
        rows = self._conn.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY created_at, doc_id",
                                  (user_id,)).fetchall()
        documents = [self._document(row) for row in rows]
        return [d for d in documents if include_deleted or d.status != "deleted"]

    @staticmethod
    def _document(row: sqlite3.Row) -> TwinDocument:
        return TwinDocument(doc_id=row["doc_id"], file_path=row["file_path"], status=row["status"],
                            doc_type=row["doc_type"],
                            observation=json.loads(row["observation"]) if row["observation"] else None,
                            processing_error=row["processing_error"], created_at=_instant(row["created_at"]),
                            updated_at=_instant(row["updated_at"]))

    # --------------------------------------------------------------- facts --
    def add_fact(self, fact: Fact) -> Fact:
        """Store a value; the current value from the same field/provenance/source is closed."""
        now = self.now()
        value_type, text = encode_fact(fact.value)
        with self.transaction() as conn:
            self.ensure_user(fact.user_id)
            conn.execute(
                """UPDATE facts SET valid_to = ? WHERE user_id = ? AND procedure_id = ? AND field = ?
                   AND provenance = ? AND source_ref = ? AND valid_to IS NULL""",
                (now, fact.user_id, fact.procedure_id, fact.field, fact.provenance, fact.source_ref))
            cursor = conn.execute(
                """INSERT INTO facts (user_id, procedure_id, field, value_type, value, provenance, source_ref,
                                      page, snippet, confirmed, note, valid_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fact.user_id, fact.procedure_id, fact.field, value_type, text, fact.provenance,
                 fact.source_ref, fact.page, fact.snippet, int(fact.confirmed), fact.note, now))
        return self.get_fact(cursor.lastrowid)

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self._conn.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        return self._fact(row) if row else None

    def confirm_fact(self, user_id: str, fact_id: int) -> Fact:
        """Mark a current fact as confirmed by the user (a new row; the old one is closed)."""
        fact = self.get_fact(fact_id)
        if fact is None or fact.user_id != user_id or fact.valid_to is not None:
            raise ValueError("Only a current fact of this user can be confirmed")
        return self.add_fact(fact.model_copy(update={"fact_id": None, "confirmed": True}))

    def close_fact(self, user_id: str, fact_id: int) -> None:
        """End the validity of one current fact (history is kept)."""
        with self.transaction() as conn:
            conn.execute("UPDATE facts SET valid_to = ? WHERE fact_id = ? AND user_id = ? AND valid_to IS NULL",
                         (self.now(), fact_id, user_id))

    def close_source(self, user_id: str, procedure_id: str, source_ref: str, provenance: str | None = None) -> int:
        """End every current fact from one source (for example a deleted document). Returns the count."""
        sql = "UPDATE facts SET valid_to = ? WHERE user_id = ? AND procedure_id = ? AND source_ref = ? AND valid_to IS NULL"
        args: list[Any] = [self.now(), user_id, procedure_id, source_ref]
        if provenance is not None:
            sql += " AND provenance = ?"
            args.append(provenance)
        with self.transaction() as conn:
            return conn.execute(sql, args).rowcount

    def current_facts(self, user_id: str, procedure_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? AND procedure_id = ? AND valid_to IS NULL ORDER BY fact_id",
            (user_id, procedure_id)).fetchall()
        return [self._fact(row) for row in rows]

    def fact_history(self, user_id: str, procedure_id: str, field: str | None = None) -> list[Fact]:
        sql = "SELECT * FROM facts WHERE user_id = ? AND procedure_id = ?"
        args: list[Any] = [user_id, procedure_id]
        if field is not None:
            sql += " AND field = ?"
            args.append(field)
        return [self._fact(row) for row in self._conn.execute(sql + " ORDER BY fact_id", args)]

    @staticmethod
    def _fact(row: sqlite3.Row) -> Fact:
        return Fact(fact_id=row["fact_id"], user_id=row["user_id"], procedure_id=row["procedure_id"],
                    field=row["field"], value=decode_fact(row["value_type"], row["value"]),
                    provenance=row["provenance"], source_ref=row["source_ref"], page=row["page"],
                    snippet=row["snippet"], confirmed=bool(row["confirmed"]), note=row["note"],
                    valid_from=_instant(row["valid_from"]), valid_to=_instant(row["valid_to"]))

    # -------------------------------------------------------------- events --
    def add_event(self, event: Event) -> Event:
        with self.transaction() as conn:
            self.ensure_user(event.user_id)
            cursor = conn.execute(
                "INSERT INTO events (user_id, procedure_id, kind, direction, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (event.user_id, event.procedure_id, event.kind, event.direction, _json(event.payload), self.now()))
        row = self._conn.execute("SELECT * FROM events WHERE event_id = ?", (cursor.lastrowid,)).fetchone()
        return self._event(row)

    def list_events(self, user_id: str, kind: str | None = None, limit: int = 200) -> list[Event]:
        sql = "SELECT * FROM events WHERE user_id = ?"
        args: list[Any] = [user_id]
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        rows = self._conn.execute(sql + " ORDER BY event_id DESC LIMIT ?", [*args, limit]).fetchall()
        return [self._event(row) for row in reversed(rows)]

    @staticmethod
    def _event(row: sqlite3.Row) -> Event:
        return Event(event_id=row["event_id"], user_id=row["user_id"], procedure_id=row["procedure_id"],
                     kind=row["kind"], direction=row["direction"], payload=json.loads(row["payload"]),
                     created_at=_instant(row["created_at"]))

    # ----------------------------------------------------------- proposals --
    def add_proposal(self, action: ProposedAction) -> ProposedAction:
        with self.transaction() as conn:
            self.ensure_user(action.user_id)
            conn.execute(
                """INSERT INTO proposed_actions (action_id, user_id, procedure_id, tool, title, payload, refs,
                                                 requires_stage, status, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)""",
                (action.action_id, action.user_id, action.procedure_id, action.tool, action.title,
                 _json(action.payload), _json(action.refs), action.requires_stage, action.note, self.now()))
        return self.get_proposal(action.action_id)

    def get_proposal(self, action_id: str) -> ProposedAction | None:
        row = self._conn.execute("SELECT * FROM proposed_actions WHERE action_id = ?", (action_id,)).fetchone()
        return self._proposal(row) if row else None

    def list_proposals(self, user_id: str, status: str | None = None) -> list[ProposedAction]:
        sql = "SELECT * FROM proposed_actions WHERE user_id = ?"
        args: list[Any] = [user_id]
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        return [self._proposal(row) for row in self._conn.execute(sql + " ORDER BY created_at, action_id", args)]

    def decide_proposal(self, action_id: str, status: str, artifact_path: str | None = None, note: str = "") -> ProposedAction:
        """Record the user's decision (or a blocked approval) on a proposal."""
        with self.transaction() as conn:
            conn.execute("UPDATE proposed_actions SET status = ?, artifact_path = ?, note = ?, decided_at = ? WHERE action_id = ?",
                         (status, artifact_path, note, self.now(), action_id))
        return self.get_proposal(action_id)

    @staticmethod
    def _proposal(row: sqlite3.Row) -> ProposedAction:
        return ProposedAction(action_id=row["action_id"], user_id=row["user_id"], procedure_id=row["procedure_id"],
                              tool=row["tool"], title=row["title"], payload=json.loads(row["payload"]),
                              refs=json.loads(row["refs"]), requires_stage=row["requires_stage"],
                              status=row["status"], artifact_path=row["artifact_path"], note=row["note"],
                              created_at=_instant(row["created_at"]), decided_at=_instant(row["decided_at"]))

    # --------------------------------------------------------- simulations --
    def add_simulation(self, simulation: Simulation) -> Simulation:
        with self.transaction() as conn:
            self.ensure_user(simulation.user_id)
            cursor = conn.execute(
                """INSERT INTO simulations (user_id, procedure_id, stage_id, accepted, facts_hash, result, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (simulation.user_id, simulation.procedure_id, simulation.stage_id, int(simulation.accepted),
                 simulation.facts_hash, _json(simulation.result), self.now()))
        return simulation.model_copy(update={"simulation_id": cursor.lastrowid})

    def list_simulations(self, user_id: str, procedure_id: str, stage_id: str | None = None) -> list[Simulation]:
        sql = "SELECT * FROM simulations WHERE user_id = ? AND procedure_id = ?"
        args: list[Any] = [user_id, procedure_id]
        if stage_id is not None:
            sql += " AND stage_id = ?"
            args.append(stage_id)
        rows = self._conn.execute(sql + " ORDER BY simulation_id", args).fetchall()
        return [Simulation(simulation_id=r["simulation_id"], user_id=r["user_id"], procedure_id=r["procedure_id"],
                           stage_id=r["stage_id"], accepted=bool(r["accepted"]), facts_hash=r["facts_hash"],
                           result=json.loads(r["result"]), created_at=_instant(r["created_at"])) for r in rows]

    def latest_simulation(self, user_id: str, procedure_id: str, stage_id: str) -> Simulation | None:
        simulations = self.list_simulations(user_id, procedure_id, stage_id)
        return simulations[-1] if simulations else None

    # ------------------------------------------------------------ episodes --
    def add_episode(self, episode: Episode) -> int:
        """Store a validated repair lesson. Unvalidated corrections are refused (also by the DB)."""
        if not episode.correction_validated:
            raise ValueError("only corrections that passed Gate 1 are stored")
        with self.transaction() as conn:
            self.ensure_user(episode.user_id)
            cursor = conn.execute(
                """INSERT INTO episodes (user_id, procedure_id, failure_type, affected_item, validator_error,
                       corrective_action, outcome, correction_validated, rule_or_source_id, rule_version,
                       trace_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                (episode.user_id, episode.procedure_id, episode.failure_type, episode.affected_item,
                 episode.validator_error, episode.corrective_action, episode.outcome,
                 episode.rule_or_source_id, episode.rule_version, episode.trace_id, self.now()))
            return cursor.lastrowid

    def find_episodes(self, procedure_id: str, failure_type: str | None = None, limit: int = 5) -> list[Episode]:
        """Valid lessons for a procedure (optionally one failure type), newest first."""
        sql = "SELECT * FROM episodes WHERE procedure_id = ? AND invalidated_at IS NULL"
        args: list[Any] = [procedure_id]
        if failure_type is not None:
            sql += " AND failure_type = ?"
            args.append(failure_type)
        sql += " ORDER BY created_at DESC, episode_id DESC LIMIT ?"
        rows = self._conn.execute(sql, [*args, limit]).fetchall()
        fields = [name for name in Episode.model_fields if name not in ("created_at", "correction_validated")]
        return [Episode(**{name: row[name] for name in fields}, correction_validated=bool(row["correction_validated"]),
                        created_at=_instant(row["created_at"])) for row in rows]

    def invalidate_episodes(self, rule_or_source_id: str) -> int:
        """Retire lessons tied to a rule or source that changed. Returns the count."""
        with self.transaction() as conn:
            return conn.execute("UPDATE episodes SET invalidated_at = ? WHERE rule_or_source_id = ? AND invalidated_at IS NULL",
                                (self.now(), rule_or_source_id)).rowcount

    # ---------------------------------------------------------------- runs --
    def log_run(self, run: RunRecord) -> None:
        """Store one agent run and its Gate 1 issues atomically."""
        with self.transaction() as conn:
            self.ensure_user(run.user_id)
            conn.execute(
                """INSERT INTO runs (trace_id, user_id, procedure_id, goal, status, llm_calls, steps, repairs,
                       fallback_used, needs_human_review, summary, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run.trace_id, run.user_id, run.procedure_id, run.goal, run.status, run.llm_calls, run.steps,
                 run.repairs, int(run.fallback_used), int(run.needs_human_review), run.summary,
                 encode_timestamp(run.started_at), encode_timestamp(run.finished_at)))
            conn.executemany("INSERT INTO run_issues (trace_id, position, code, message, item) VALUES (?, ?, ?, ?, ?)",
                             [(run.trace_id, i, issue.code, issue.message, issue.item) for i, issue in enumerate(run.issues)])

    def list_runs(self, user_id: str) -> list[RunRecord]:
        records = []
        for row in self._conn.execute("SELECT * FROM runs WHERE user_id = ? ORDER BY started_at, rowid", (user_id,)):
            issues = [Issue(code=i["code"], message=i["message"], item=i["item"]) for i in self._conn.execute(
                "SELECT * FROM run_issues WHERE trace_id = ? ORDER BY position", (row["trace_id"],))]
            records.append(RunRecord(
                trace_id=row["trace_id"], user_id=row["user_id"], procedure_id=row["procedure_id"], goal=row["goal"],
                status=row["status"], llm_calls=row["llm_calls"], steps=row["steps"], repairs=row["repairs"],
                fallback_used=bool(row["fallback_used"]), needs_human_review=bool(row["needs_human_review"]),
                summary=row["summary"], started_at=_instant(row["started_at"]),
                finished_at=_instant(row["finished_at"]), issues=issues))
        return records

    # ------------------------------------------------------------- vectors --
    def get_vector(self, procedure_id: str, chunk_id: str, model: str, text_hash: str) -> list[float] | None:
        """Cached embedding, or None when missing or computed from different text."""
        row = self._conn.execute("SELECT text_hash, vector FROM rag_vectors WHERE procedure_id = ? AND chunk_id = ? AND model = ?",
                                 (procedure_id, chunk_id, model)).fetchone()
        if row is None or row["text_hash"] != text_hash:
            return None
        return json.loads(row["vector"])

    def put_vector(self, procedure_id: str, chunk_id: str, model: str, source_id: str, text_hash: str,
                   vector: list[float]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO rag_vectors (procedure_id, chunk_id, model, source_id, text_hash, vector, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (procedure_id, chunk_id, model) DO UPDATE SET
                     source_id = excluded.source_id, text_hash = excluded.text_hash, vector = excluded.vector,
                     created_at = excluded.created_at""",
                (procedure_id, chunk_id, model, source_id, text_hash, json.dumps(vector), self.now()))
