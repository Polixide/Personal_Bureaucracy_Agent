# `src/twin/` — the Digital Twin (store, facts, service)

## What it does
The Twin is the agent's environment: a SQLite database holding what is known about one user for one procedure.
It stores documents, facts (with provenance and validity dates), events, proposed actions, sandbox simulations,
episodes and run logs, including the rule verdicts at the end of each run (`run_results`, Neda's
`run_requirements`), so progress across sessions can be read back. `facts.py` turns the stored facts into the values the rule engine may use.

## Where it sits in the flow
- **`service.py` is the real → twin path:**
  - `register_upload` and `store_extraction` (documents),
  - `record_statement` (chat, unconfirmed) and `record_app_value` (form, confirmed),
  - `correct_document_field`, `resolve_conflict` and `delete_document`,
  - `record_outcome` (results the user reports from the real world),
  - `record_rule_update` (an official rule changed; its lessons are retired).
- **Readers:** tools, Gate 1 and the sandbox call `twin_view(store, pack, user, today)`. That returns a
  `FactView`, and `view.fact_map()` goes to `src/rules.py`.
- **Twin → real:** this happens only through approved proposals (`src/sandbox/approval.py`).

## Inputs and outputs
A stored fact:
```json
{"field": "insurance_end_date", "value": "2027-06-30", "provenance": "document", "source_ref": "doc_3f2a",
 "page": 1, "snippet": "Coverage end: 30.06.2027", "confirmed": true,
 "valid_from": "2026-11-18T09:00:01+00:00", "valid_to": null}
```
- **Fact view:** `FactView{values, sources, conflicts, unconfirmed, derived}`.
- **Provenance values:** `document`, `user_stated`, `tool` and `derived`.

## Why it is built this way
- **History is never overwritten.** A newer value from the same source closes the older one (`valid_to`). You
  can always see what a document said and what the user changed.
- **Only confirmed facts reach the rules.** Chat statements (`record_fact`) and values the pack marks
  `requires_confirmation` (for example the entry date) are listed as *unconfirmed* until the user confirms them.
  This is the approved decision from the migration plan.
- **Conflicts are reported, not guessed.** If two documents disagree, the value becomes unknown until the user
  resolves it. A correction of a document by the user beats that document's own observation.
- **Derived values are computed on read** (months of stay, "certificate complete", today). They cannot go stale
  in the database.
- **Separate tables, not a LangGraph checkpoint blob.** This is Neda's reason: every fact can be queried,
  tested and deleted on its own. `delete_user` cascades everywhere.

## How to test it
```
python -m pytest -q tests/test_twin_store.py tests/test_twin_facts.py tests/test_codec.py
```

## Known limits
- **Restating a value.** A new chat statement for a field replaces an earlier confirmed chat statement for that
  field; the user must confirm again. This is deliberate.
- **Schema versions.** v3 (2026-11-19) added `run_results`; v4 (2026-11-20) added `chat_messages` (the
  assistant's conversation: role, text, procedure, trace id, JSON details such as cited sources and recorded facts;
  deleted with the user, or with "Clear the conversation"). A v2 or v3 file gains the missing tables on open, with
  its data kept (`MIGRATIONS` in `store.py`). Schema v1 files (Neda's milestone-1 store) are refused; no such data exists.
- **Concurrency.** One process at a time per database file. This is a local prototype.
