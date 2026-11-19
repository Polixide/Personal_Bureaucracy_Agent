-- Twin record schema, version 2.
--
-- Role: the PERMANENT RECORD of each student's state, plus a run log and
-- validated reflection lessons. The live agent's source of truth is LangGraph's
-- own checkpoint (src/memory/checkpointer.py), which the approval interrupt
-- needs. After every graph invocation, src/memory/twin_sync.py mirrors the
-- checkpointed state into these tables, so they can be queried, inspected,
-- evaluated and deleted per student without LangGraph.
--
-- Conventions
--   * calendar dates: TEXT 'YYYY-MM-DD'. The CHECK `date(x) IS x` rejects
--     wrong formats ('2026/11/20') and impossible days ('2026-02-30', which
--     SQLite would otherwise normalise to '2026-03-02'). It must be `IS`, not
--     `=`: date() returns NULL for garbage, and a CHECK that evaluates to NULL
--     passes, so `date(x) = x` would let '2026/11/20' through.
--   * instants (*_at): TEXT ISO-8601 in UTC, written by the store only.
--   * nested Pydantic objects: JSON produced and re-validated by Pydantic
--     against the AgentState type, so dates inside them round-trip exactly.
--   * every student-owned row cascades on user deletion.

-- ---------------------------------------------------------- Digital Twin --
CREATE TABLE users (
    user_id              TEXT PRIMARY KEY CHECK (length(user_id) > 0),
    active_procedure     TEXT,
    approved_plan_origin TEXT,
    created_at           TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    updated_at           TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL)
);

CREATE TABLE documents (
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id           TEXT NOT NULL,
    position         INTEGER NOT NULL,
    file_path        TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('pending', 'analyzed', 'rejected',
                                                     'replacement_pending', 'deleted')),
    doc_type         TEXT,
    issue_date       TEXT CHECK (issue_date  IS NULL OR (length(issue_date)  = 10 AND date(issue_date)  IS issue_date)),
    expiry_date      TEXT CHECK (expiry_date IS NULL OR (length(expiry_date) = 10 AND date(expiry_date) IS expiry_date)),
    processing_error TEXT,
    info_json        TEXT CHECK (info_json IS NULL OR json_valid(info_json)),  -- extracted observation
    updated_at       TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL),
    PRIMARY KEY (user_id, doc_id)
);

-- extracted_facts: one row per key, with an explicit type tag (see codec.py).
CREATE TABLE facts (
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (value_type IN ('null','bool','int','float','str','date','datetime','json')),
    value      TEXT,
    updated_at TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL),
    PRIMARY KEY (user_id, key),
    CHECK ((value_type = 'null') = (value IS NULL)),
    CHECK (value_type <> 'date' OR (length(value) = 10 AND date(value) IS value)),
    CHECK (value_type <> 'datetime' OR julianday(value) IS NOT NULL),
    CHECK (value_type <> 'json' OR json_valid(value))
);

CREATE TABLE deadlines (
    user_id  TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key      TEXT NOT NULL,
    due_date TEXT NOT NULL CHECK (length(due_date) = 10 AND date(due_date) IS due_date),
    PRIMARY KEY (user_id, key)
);

-- verification_records[record_key][field_key] -> VerifiedInput
CREATE TABLE verification_records (
    user_id           TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    record_key        TEXT NOT NULL,
    field_key         TEXT NOT NULL,
    document_id       TEXT NOT NULL,
    field_name        TEXT NOT NULL,
    value_type        TEXT NOT NULL CHECK (value_type IN ('null', 'bool', 'int', 'str')),
    value             TEXT,
    origin            TEXT NOT NULL CHECK (origin IN ('source_grounded', 'user_confirmed', 'derived')),
    source_page       INTEGER,
    source_snippet    TEXT,
    confirmed_by      TEXT,
    trace_id          TEXT NOT NULL,
    derived_from      TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(derived_from)),
    derivation_policy TEXT,
    PRIMARY KEY (user_id, record_key, field_key),
    CHECK ((value_type = 'null') = (value IS NULL))
);

CREATE TABLE completed_steps (
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    step_id      TEXT NOT NULL,
    position     INTEGER NOT NULL,
    completed_at TEXT NOT NULL CHECK (julianday(completed_at) IS NOT NULL),
    PRIMARY KEY (user_id, step_id)
);

CREATE TABLE blockers (
    user_id  TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    rule_id  TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (user_id, rule_id)
);

-- Remaining twin fields whose shape is a nested model or still evolving:
-- residence_permit_context, approved_plan, action_progress,
-- identity_resolutions, verified_context_fields, verification_conflicts.
CREATE TABLE twin_objects (
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    updated_at TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL),
    PRIMARY KEY (user_id, key)
);

-- ------------------------------------------------- episodes (run history) --
-- One row per run (trace_id). A run paused at human approval is stored as
-- 'awaiting_approval' and updated in place when it is resumed.
CREATE TABLE runs (
    trace_id       TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure      TEXT,
    event_kind     TEXT NOT NULL,
    event_text     TEXT NOT NULL,
    event_file     TEXT,
    event_document TEXT,
    run_status     TEXT,
    outcome        TEXT NOT NULL CHECK (outcome IN ('pass', 'fail', 'awaiting_approval', 'incomplete')),
    retry_count    INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    final_response TEXT NOT NULL,
    started_at     TEXT NOT NULL CHECK (julianday(started_at)  IS NOT NULL),
    finished_at    TEXT NOT NULL CHECK (julianday(finished_at) IS NOT NULL),
    CHECK (finished_at >= started_at)
);
CREATE INDEX runs_by_user ON runs(user_id, started_at);

CREATE TABLE run_issues (
    trace_id TEXT NOT NULL REFERENCES runs(trace_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    code     TEXT NOT NULL,
    message  TEXT NOT NULL,
    source   TEXT NOT NULL,
    item     TEXT,
    rule_id  TEXT,
    PRIMARY KEY (trace_id, position)
);

-- Rule results of each run: "how often does R015 fail?" is one query.
CREATE TABLE run_requirements (
    trace_id       TEXT NOT NULL REFERENCES runs(trace_id) ON DELETE CASCADE,
    rule_id        TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('passed', 'failed', 'missing_information')),
    severity       TEXT NOT NULL,
    error_code     TEXT,
    missing_fields TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(missing_fields)),
    PRIMARY KEY (trace_id, rule_id)
);

-- ------------------------------------------------------ reflection memory --
-- Validated lessons only (plan section 3.4): the CHECK makes it impossible to
-- store a correction that did not pass Gate 1 or human confirmation.
CREATE TABLE lessons (
    lesson_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure            TEXT NOT NULL,
    failure_type         TEXT NOT NULL,
    affected_item        TEXT,
    validator_error      TEXT NOT NULL,
    rule_or_source_id    TEXT,
    rule_version         TEXT,
    corrective_action    TEXT NOT NULL,
    outcome              TEXT NOT NULL,
    correction_validated INTEGER NOT NULL CHECK (correction_validated = 1),
    trace_id             TEXT REFERENCES runs(trace_id) ON DELETE SET NULL,
    created_at           TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    invalidated_at       TEXT CHECK (invalidated_at IS NULL OR julianday(invalidated_at) IS NOT NULL)
);
CREATE INDEX lessons_lookup ON lessons(user_id, procedure, failure_type);
