-- Digital Twin schema, version 2 (v1 was Neda's milestone-1 TwinStore; see docs/modules/twin.md).
--
-- Conventions kept from v1:
--   * calendar dates: TEXT 'YYYY-MM-DD', checked with `date(x) IS x` (IS, not =: date() returns NULL
--     for garbage and a NULL CHECK would pass);
--   * instants (*_at, valid_from, valid_to): TEXT ISO-8601 UTC, written by the store only;
--   * every user-owned row cascades on user deletion (right to be forgotten).
-- New in v2: facts carry provenance and a validity interval; events, proposed actions, sandbox
-- simulations, episodes (validated repairs), agent run logs and a RAG vector cache.

CREATE TABLE users (
    user_id          TEXT PRIMARY KEY CHECK (length(user_id) > 0),
    active_procedure TEXT,
    created_at       TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    updated_at       TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL)
);

-- Uploaded documents. `observation` is the JSON DocumentObservation (what the document showed).
CREATE TABLE documents (
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id           TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('pending', 'analyzed', 'rejected', 'deleted')),
    doc_type         TEXT,
    observation      TEXT,
    processing_error TEXT,
    created_at       TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    updated_at       TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL),
    PRIMARY KEY (user_id, doc_id)
);

-- One row per observed/stated value. A newer value from the same source closes the older one
-- (valid_to), so history is never lost. Only confirmed, currently valid rows reach the rules.
CREATE TABLE facts (
    fact_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id TEXT NOT NULL,
    field        TEXT NOT NULL,
    value_type   TEXT NOT NULL CHECK (value_type IN ('null','bool','int','float','str','date','datetime','json')),
    value        TEXT,
    provenance   TEXT NOT NULL CHECK (provenance IN ('document', 'user_stated', 'tool', 'derived')),
    source_ref   TEXT NOT NULL,
    page         INTEGER,
    snippet      TEXT,
    confirmed    INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
    note         TEXT NOT NULL DEFAULT '',
    valid_from   TEXT NOT NULL CHECK (julianday(valid_from) IS NOT NULL),
    valid_to     TEXT CHECK (valid_to IS NULL OR julianday(valid_to) IS NOT NULL),
    CHECK ((value_type = 'null') = (value IS NULL)),
    CHECK (value_type <> 'date' OR (length(value) = 10 AND date(value) IS value)),
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
CREATE INDEX facts_current ON facts(user_id, procedure_id, valid_to, field);

-- Everything that crossed the boundary: real -> twin (documents, outcomes, rule updates),
-- twin -> real (approved artifacts), and internal agent events.
CREATE TABLE events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id TEXT,
    kind         TEXT NOT NULL CHECK (length(kind) > 0),
    direction    TEXT NOT NULL CHECK (direction IN ('real_to_twin', 'twin_to_real', 'internal')),
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL)
);
CREATE INDEX events_by_user ON events(user_id, event_id);

-- Artifact tools only PROPOSE. Approval runs the promotion gate, then writes the artifact.
CREATE TABLE proposed_actions (
    action_id      TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id   TEXT NOT NULL,
    tool           TEXT NOT NULL,
    title          TEXT NOT NULL,
    payload        TEXT NOT NULL,
    refs           TEXT NOT NULL,
    requires_stage TEXT,
    status         TEXT NOT NULL CHECK (status IN ('proposed', 'executed', 'denied', 'blocked')),
    artifact_path  TEXT,
    note           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    decided_at     TEXT CHECK (decided_at IS NULL OR julianday(decided_at) IS NOT NULL)
);

-- Mock-world attempts. facts_hash ties a simulation to the exact facts it saw.
CREATE TABLE simulations (
    simulation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id  TEXT NOT NULL,
    stage_id      TEXT NOT NULL,
    accepted      INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    facts_hash    TEXT NOT NULL,
    result        TEXT NOT NULL,
    created_at    TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL)
);

-- Episodic memory: only repairs that passed Gate 1 (the CHECK makes anything else impossible).
CREATE TABLE episodes (
    episode_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id         TEXT NOT NULL,
    failure_type         TEXT NOT NULL,
    affected_item        TEXT,
    validator_error      TEXT NOT NULL,
    corrective_action    TEXT NOT NULL,
    outcome              TEXT NOT NULL,
    correction_validated INTEGER NOT NULL CHECK (correction_validated = 1),
    rule_or_source_id    TEXT,
    rule_version         TEXT,
    trace_id             TEXT,
    created_at           TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    invalidated_at       TEXT CHECK (invalidated_at IS NULL OR julianday(invalidated_at) IS NOT NULL)
);
CREATE INDEX episodes_lookup ON episodes(procedure_id, failure_type);

-- One row per agent run (evaluation metrics come from here).
CREATE TABLE runs (
    trace_id           TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id       TEXT,
    goal               TEXT NOT NULL,
    status             TEXT NOT NULL,
    llm_calls          INTEGER NOT NULL DEFAULT 0 CHECK (llm_calls >= 0),
    steps              INTEGER NOT NULL DEFAULT 0 CHECK (steps >= 0),
    repairs            INTEGER NOT NULL DEFAULT 0 CHECK (repairs >= 0),
    fallback_used      INTEGER NOT NULL DEFAULT 0 CHECK (fallback_used IN (0, 1)),
    needs_human_review INTEGER NOT NULL DEFAULT 0 CHECK (needs_human_review IN (0, 1)),
    summary            TEXT NOT NULL DEFAULT '',
    started_at         TEXT NOT NULL CHECK (julianday(started_at)  IS NOT NULL),
    finished_at        TEXT NOT NULL CHECK (julianday(finished_at) IS NOT NULL),
    CHECK (finished_at >= started_at)
);
CREATE TABLE run_issues (
    trace_id TEXT NOT NULL REFERENCES runs(trace_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    code     TEXT NOT NULL,
    message  TEXT NOT NULL,
    item     TEXT,
    PRIMARY KEY (trace_id, position)
);
-- The rule verdicts at the end of each run (Neda's run_requirements): progress across sessions. Added in v3.
CREATE TABLE run_results (
    trace_id       TEXT NOT NULL REFERENCES runs(trace_id) ON DELETE CASCADE,
    rule_id        TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('pass', 'fail', 'unknown')),
    severity       TEXT NOT NULL,
    missing_fields TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(missing_fields)),
    PRIMARY KEY (trace_id, rule_id)
);

-- The conversation with the assistant (chat view), per user. `details` is JSON: sources cited, facts
-- recorded, the agent run's status. Added in v4.
CREATE TABLE chat_messages (
    message_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure_id TEXT,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    trace_id     TEXT,
    details      TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(details)),
    created_at   TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL)
);

-- RAG cache: one vector per (pack, chunk, embedding model); text_hash detects edited sources.
CREATE TABLE rag_vectors (
    procedure_id TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    model        TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    text_hash    TEXT NOT NULL,
    vector       TEXT NOT NULL,
    created_at   TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    PRIMARY KEY (procedure_id, chunk_id, model)
);
