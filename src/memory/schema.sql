-- Digital Twin persistence schema, version 1.
--
-- One table per kind of fact, keyed by user_id, so the data can be queried,
-- inspected and tested without LangGraph. Conventions:
--   * calendar dates: TEXT 'YYYY-MM-DD'. The CHECK `date(x) IS x` rejects
--     wrong formats ('2026/11/20') and impossible days ('2026-02-30', which
--     SQLite would otherwise normalise to '2026-03-02'). It must be `IS`, not
--     `=`: date() returns NULL for garbage, and a CHECK that evaluates to NULL
--     passes, so `date(x) = x` would let '2026/11/20' through.
--   * instants (*_at): TEXT ISO-8601 in UTC, written by the store only.
--   * every user-owned row cascades on user deletion (right to be forgotten).

-- ---------------------------------------------------------- Digital Twin --
CREATE TABLE users (
    user_id          TEXT PRIMARY KEY CHECK (length(user_id) > 0),
    active_procedure TEXT,
    created_at       TEXT NOT NULL CHECK (julianday(created_at) IS NOT NULL),
    updated_at       TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL)
);

CREATE TABLE documents (
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'analyzed', 'rejected')),
    doc_type    TEXT,
    issue_date  TEXT CHECK (issue_date  IS NULL OR (length(issue_date)  = 10 AND date(issue_date) IS issue_date)),
    expiry_date TEXT CHECK (expiry_date IS NULL OR (length(expiry_date) = 10 AND date(expiry_date) IS expiry_date)),
    updated_at  TEXT NOT NULL CHECK (julianday(updated_at) IS NOT NULL),
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
    CHECK (value_type <> 'datetime' OR julianday(value) IS NOT NULL)
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

-- ------------------------------------------------- episodes (run history) --
-- One row per graph run, PASS or FAIL: the "past failures" the plan needs,
-- and the raw material for evaluation metrics.
CREATE TABLE runs (
    trace_id       TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    procedure      TEXT,
    event_kind     TEXT NOT NULL CHECK (event_kind IN ('message', 'upload')),
    event_text     TEXT NOT NULL,
    event_file     TEXT,
    outcome        TEXT NOT NULL CHECK (outcome IN ('pass', 'fail')),
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
    code     TEXT NOT NULL CHECK (code IN ('INVALID_INPUT', 'MISSING_PREREQUISITE',
                                           'DATE_CONFLICT', 'SCHEMA_ERROR', 'APPROVAL_DENIED')),
    message  TEXT NOT NULL,
    source   TEXT NOT NULL,
    item     TEXT,
    rule_id  TEXT,
    PRIMARY KEY (trace_id, position)
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
