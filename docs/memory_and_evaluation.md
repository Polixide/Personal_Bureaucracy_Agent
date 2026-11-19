# Memory & Evaluation (integrated with the Milestone-1 prototype)

This branch builds on Marco's Milestone-1 prototype (`feature/marco-milestone1`).
It adds a permanent, queryable record of each student's state, a run log,
validated reflection lessons, and a scored evaluation harness.

```bash
pip install -r requirements-dev.txt
python -m pytest            # 603 tests, offline (model calls mocked)
python -m scripts.run_eval  # scored report, exit code 1 if any scenario fails
```

## Two stores, two roles (team option B)

| Store | File | Role |
|---|---|---|
| LangGraph checkpoint (`src/memory/checkpointer.py`) | `data/digital_twin.db` | **Live source of truth.** The approval `interrupt()` needs it to pause and resume a run. |
| Twin record (`src/memory/twin_store.py`) | `data/twin_record.db` | **Permanent record.** Independent tables mirroring each student's state after every graph invocation, plus the run log and lessons. Used for evaluation, inspection and per-student deletion. |

After each `graph.invoke(...)` the sync step (`src/memory/twin_sync.py`)
copies the checkpointed state into the twin record and logs the run. The sync
is hooked into the Streamlit UI (`invoke()`) and `scripts/run_demo.py`.

- **Sync failures never block the UI.** The checkpoint is already saved, and the next successful sync re-mirrors the whole twin.
- **Reset deletes both.** `reset_thread()` removes the student's checkpoint thread and their record: state, run log and lessons.
- **Moving to option C later.** In option C the tables become the source of truth and the agent loads from them. That changes only how state is loaded; the tables already exist.

## Twin record tables (schema v2)

| Table | Holds |
|---|---|
| `users` | student id, procedure, approved-plan origin |
| `documents` | id, status (5 values incl. `replacement_pending`, `deleted`), type, issue/expiry dates, processing error, extracted info (JSON) |
| `facts` | `extracted_facts`, one row per key with a type tag |
| `deadlines` | `deadlines`, real date columns |
| `verification_records` | every `VerifiedInput`: value, origin, page, snippet, who confirmed, trace |
| `completed_steps`, `blockers` | ordered lists |
| `twin_objects` | `residence_permit_context`, `approved_plan`, `action_progress`, `identity_resolutions`, `verified_context_fields`, `verification_conflicts` as validated JSON |
| `runs` | one row per run: event, `run_status`, outcome (`pass` / `fail` / `awaiting_approval` / `incomplete`), repairs, response, times |
| `run_issues` | the run's validation errors |
| `run_requirements` | the run's 29 rule results (status, severity, missing fields) |
| `lessons` | validated reflection lessons (see below) |

**Mirror semantics.** The tables copy the checkpoint exactly. A document the
agent deleted, or a fact set to `None`, disappears from the record too. A
paused run is stored as `awaiting_approval`, and its row is updated in place
when the user approves or denies, keeping its original start time.

## Dates

- **Calendar days** (expiry, deadlines, appointment) are stored as `YYYY-MM-DD` with no time zone. A `datetime` in a date field is refused.
- **Database checks** reject impossible days (`2026-02-30`) and wrong formats (`2026/11/20`) even when the writer bypasses the store. The check is `date(x) IS x`; `=` would let NULL-producing garbage through.
- **Timestamps** must be timezone-aware and are stored in UTC.
- **Nested objects** are dumped by Pydantic against their `AgentState` type. They must read back to an equal value or the write is refused. This catches a date hidden in an untyped field such as `identity_resolutions`, which would otherwise come back as a string.

## Reflection memory (plan 3.3–3.4)

- **When lessons are written:** when a run needed repairs and its final plan then passed Gate 1. The sync writes one lesson per distinct Gate 1 error: failure type, item, error message, rule id plus its `rules.json` version, corrective action and outcome. This happens once per run.
- **What is refused:** unvalidated lessons, by both the code and a database CHECK.
- **`find_lessons()`** filters by student, procedure, failure type and item, newest first.
- **`invalidate_lessons(rule_id)`** retires lessons when a rule changes.
- **Not yet done:** the planner does not read lessons yet. The next step is to add `find_lessons(...)` results to the planner's repair context.

## Evaluation harness (`src/evaluation/`)

`scenarios.py` holds Marco's 10 curated scenarios, unchanged. `harness.py` runs
them, plus 3 multi-session scenarios. Each run uses the real graph, rules
engine, Gate 1, repair loop, approval interrupt and both databases; only model
transport is mocked. Compared with the previous single-session test, it:

- **restarts the app** (closes both databases, recompiles the graph) between the approval pause and the user's decision
- **compares the record with the checkpoint** after reopening
- **checks the run log**, including each run's 29 rule results
- **checks lessons** are written exactly when a repair passed Gate 1
- **collects every expectation** as a check and aggregates metrics instead of stopping at the first failed assert

| Multi-session scenario | Covers |
|---|---|
| M1 three sessions | upload → restart → approve → restart → re-evaluate. Documents persist. Re-evaluation clears the old approval (Marco's `new_run` rule), and the record follows. |
| M2 denied after restart | the paused run is logged as `awaiting_approval`, then updated to `fail`/`APPROVAL_DENIED`; no approved plan in checkpoint or record |
| M3 reset one student | reset removes one student's checkpoint, record and run log; another student is untouched |

Current report (offline):

```
Scenarios passed           : 13/13 (100%)
Checks passed              : 169/169 (100%)
Final status correct       : 10/10 (100%)
Rule statuses correct      : 290/290 (100%)
Invalid plans caught (G1)  : 5/5 (100%)
False alarms (G1)          : 0
Repaired, then passed G1   : 1 (expected 1)
Stopped at 3-repair limit  : 1 (expected 1)
Record == checkpoint       : 12/12 (100%)  (after restart; reset scenario excluded)
Validated lessons written  : 2
```

**What these numbers do not measure.** Model calls are mocked, so the report
says nothing about extraction or planning quality with a live Qwen model. It
measures the deterministic parts (rules, gates, repair limit, approval, memory)
and the plumbing around the model. Live-model evaluation needs a running
Ollama server and a labelled set of real-format synthetic documents.
