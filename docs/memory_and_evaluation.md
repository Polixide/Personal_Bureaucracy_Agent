# Memory & Evaluation

Workstream deliverable: SQLite tables and persistence functions for the Digital
Twin, synthetic test cases, and test harnesses.

**Acceptance criteria**
- *User state survives application restart.* `tests/test_restart.py` runs each
  session in a separate Python process, so only the SQLite file carries state.
- *Initial unit tests run.* `pytest` passes 104 tests in about 2 seconds.

```bash
pip install -r requirements.txt
pytest                      # unit + restart + scenario + knowledge tests
python -m scripts.run_eval  # scenario report (exit code 1 if a scenario fails)
```

## Files

| File | What it does |
|---|---|
| `src/memory/schema.sql` | Table definitions with integrity checks (schema v1) |
| `src/memory/codec.py` | Strict date / timestamp / fact encoding |
| `src/memory/checkpointer.py` | `TwinStore`: load/save twin, run log, reflection lessons |
| `src/memory/session.py` | `run_event()`: load twin → run graph → persist, atomically |
| `src/evaluation/harness.py` | Scenario runner with simulated restarts + metrics report |
| `data/synthetic_samples/scenarios.json` | 5 synthetic students, 10 sessions |
| `scripts/run_eval.py` | Prints the evaluation report |

## Why independent tables (not a LangGraph checkpoint)

LangGraph's `SqliteSaver` stores the whole state as one serialized blob per
step. Separate tables are easier to query ("which users have insurance
expiring before their appointment?"), inspect in any SQLite viewer, test
column by column, and delete per user. The graph code is unchanged: the
wrapper `run_event()` loads the twin into the input state and writes the final
state back.

Later, `human_approval` will use `interrupt()`, which needs a LangGraph
checkpointer to pause and resume a run. The two can coexist. The LangGraph
checkpointer would hold only the paused run, and these tables stay the
Digital Twin.

## Tables

```
users            user_id PK, active_procedure, created_at, updated_at
documents        (user_id, doc_id) PK, file_path, status, doc_type, issue_date, expiry_date
facts            (user_id, key) PK, value_type, value          ← extracted_facts
completed_steps  (user_id, step_id) PK, position, completed_at
blockers         (user_id, rule_id) PK, position
runs             trace_id PK, user_id, event_*, outcome pass|fail, retry_count, started/finished_at
run_issues       (trace_id, position) PK, code, message, source, item, rule_id
lessons          lesson_id PK, user_id, procedure, failure_type, affected_item, validator_error,
                 rule_or_source_id, rule_version, corrective_action, outcome,
                 correction_validated (must be 1), trace_id, created_at, invalidated_at
```

All user rows cascade on `delete_user()`. Foreign keys are enforced.

**Write semantics** follow the reducers in `src/core/state.py`:
- `documents` and `facts` are upserted; nothing is dropped.
- `completed_steps` and `blockers` are replaced by the list in the state.
- A key missing from the state leaves its table untouched.

**PASS vs FAIL runs.** Every run's final state is saved. Only `state_update`
writes blockers and completed steps, and it runs on PASS only. So a FAIL run
records what was observed (for example, an insurance that expires too early,
so it is not asked for again) without changing the plan status. Every run is
logged in `runs` / `run_issues`. Twin and log are saved in one transaction.

## Dates

| Kind | Python | Stored as | Guard |
|---|---|---|---|
| Calendar day (expiry, appointment) | `date` | `'YYYY-MM-DD'`, no time zone | `datetime` refused (it would lose its time); DB `CHECK (date(x) IS x)` |
| Instant (`*_at`) | aware `datetime` | UTC ISO-8601 | naive datetimes refused |
| Fact value | any of null/bool/int/float/str/date/datetime/list/dict | value + type tag | `"2026-11-20"` the string stays a string; dates inside list/dict facts refused |

The database checks reject `'2026-02-30'`, `'2026/11/20'` and `'20261120'`
even when the writer bypasses `TwinStore`. They use `IS`, not `=`: `date()`
returns NULL for garbage, and a NULL check passes.

Scenario files mark dates explicitly as `{"$date": "2026-11-20"}`.
`load_rule_cases()` parses every `*_date` input in `knowledge/test_cases.json`
into a `date` for the Rules Engine.

## Reflection memory (plan section 3.3–3.4)

- `add_lesson()` refuses unvalidated corrections, and so does the database.
- `find_lessons()` filters by user, procedure, failure type and item, newest
  first, with a limit.
- `invalidate_lessons(rule_id)` retires lessons when a rule or source changes.
- Lessons are stored per user.

## Evaluation harness

Each scenario seeds a twin, then runs each session with a new connection and
a newly compiled graph (a simulated restart). It checks the outcome, error
codes and nodes visited, then reopens the database and compares the stored
documents, facts, blockers and run counts. The report gives:

- scenarios passed
- checks passed
- persisted-state checks
- failures caught by the gates
- false alarms

| Scenario | Covers |
|---|---|
| S01 normal path, two sessions | documents accumulate, blockers shrink across restarts |
| S02 question only | no documents → both blockers, remembered |
| S03 insurance expires before appointment | Gate 1 DATE_CONFLICT; FAIL keeps the document, leaves blockers |
| S04 unreadable document | rejected status remembered, not re-analysed |
| S05 unsupported file / blank message | Gate 0 INVALID_INPUT; nothing added, run logged |

**Not covered yet.** These need the real Rules Engine, Planner and Reflection:

- deadline pressure
- housing branches
- stale or conflicting rules
- bounded replanning (`retry_count` ≤ 3)
- writing a lesson after a successful replan

Add them as scenarios when those nodes replace the stubs. The harness needs
no changes.
