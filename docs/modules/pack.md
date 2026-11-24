# `src/models.py` and `src/pack.py` — Procedure Packs

## What it does
A **Procedure Pack** is a folder that fully describes one bureaucratic procedure as data:
`procedures/<id>/procedure.json`, `rules.json`, `sources.json`, `sources_text/` and `eval/scenarios.json`.
`src/models.py` defines the Pydantic contract of those files. `src/pack.py` loads them, validates them,
and turns user or document input into typed facts.

## Where it sits in the flow
- `load_pack(id)` / `load_all_packs()` run when a tool context is built (agent, app, eval).
- `coerce_fact(pack, field, raw)` is used by `record_fact`, the app's confirmation forms, document
  extraction and scenario seeding.
- `validate_pack_dir(id)` backs the CLI `python -m src.pack validate <id>` and a test that checks every
  committed pack.

## Inputs and outputs
A part of `procedure.json`:
```json
{"id": "ikamet_first", "status": "verified",
 "sections": [{"id": "passport_photo", "title": "Passport, entry pages and photo", "order": 2}],
 "fields": [{"name": "passport_expiry_date", "type": "date", "question": "Provide the passport expiration date..."}],
 "documents": [{"doc_type": "passport_identity_page", "extracts": [{"field": "passport_expiry_date", "from_field": "expiry_date"}]}],
 "derived": [{"field": "planned_stay_months", "op": "months_between", "args": ["program_start_date", "program_end_date"]}],
 "stages": [{"id": "iro_submission", "requires": ["R023", "R024", "R025", "R026"]}]}
```
- **Validation output** is a list of `Issue{code, message, item}`. The codes include `UNKNOWN_FIELD`,
  `UNKNOWN_SOURCE`, `DEPENDENCY_CYCLE`, `EMPTY_SECTION`, `TYPE_MISMATCH`, `NO_SOURCE`, `FEW_SCENARIOS` and
  `SCHEMA_ERROR`.
- **Example of typing:** `coerce_fact(pack, "institution", " sabanci UNIVERSITY")` returns
  `"Sabancı University"` (through a field alias).

## Why it is built this way
- **The İkamet pack was converted, not written by hand.** A one-off script (`scripts/convert_knowledge.py`,
  commit fdc5ebd, deleted afterwards) read `knowledge/` and the milestone-1 code (field types, document
  mappings, scope test, stage prerequisites). Each condition
  mirrors the original `deterministic_condition` text, which is kept as `legacy_condition` for audit.
- **Strict models.** `extra="forbid"` means a typo such as `"sevrity"` is an error instead of a silently
  ignored key.
- **Unverified rules.** A rule with no source must say `"status": "unverified"`, and answers that use it
  must label it.
- **Aliases vs qualifiers (2026-11-24).** `search_procedures` scores alias words, and adds qualifier words only
  when an alias matched. Before this split, generic words in the ID-card packs' aliases ("lost", "first",
  "expired", "renew") sent "I lost my passport" or "My visa expired" to an ID-card pack and tied a first İkamet
  request with the first ID card. `python -m eval.routing_eval` (38 goals): dev 16/26 → 26/26, held-out
  3/12 → 9/12 with 0 wrong or out-of-scope selections (the 3 left are ties, where the agent asks).

## Rule volatility and updates
Each rule has a `volatility` (from the milestone-1 change-monitoring policy). Re-check its sources: `very_high`
before an active submission, `high` weekly, `medium` monthly, `low` each semester or when a source changes. A
changed source never edits a rule automatically: a person reviews it, bumps the rule `version`, runs the tests,
and reports it with `twin.service.record_rule_update`, which retires the lessons tied to the old rule.

## How to test it
```
python -m pytest -q tests/test_pack.py
python -m src.pack validate ikamet_first
```

## Known limits
- **`sources_text/` placeholders.** The files are compiled summaries of the rule data, not copies of the
  official pages. Each file says so in its header; replace them with the saved page text.
- **Truncated URL.** `GOV_RESIDENCE_INFO` keeps its truncated URL and is flagged in `notes`.
- **Field types.** Only `bool`, `int`, `str` and `date` are supported; there are no lists or money amounts.
