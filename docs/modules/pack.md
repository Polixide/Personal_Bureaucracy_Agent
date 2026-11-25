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
- **When and how a field is asked.** `ask`: `"now"` (default), `"later"` (a step after the application: shown under
  later steps) or `"never"` (computed, read from documents only, or published by an office). `choices`: for a text
  field, the answers shown (label → stored value), e.g. "Rental agreement in my name" → `rental_agreement`; a printed
  title that contains an alias ("KİRA SÖZLEŞMESİ / RENTAL AGREEMENT") maps to its value too.
- **Sections and documents.** `Section.later`: rules about steps after the application (never "problems to fix
  today"). `DocumentSpec.optional`: needed only in some cases (not counted; suggested only when a rule waits for
  nothing else). `ExtractSpec.lines`: a value written over several lines (an address: 3).
- **Derived document counts.** `missing_documents` / `documents_complete` over `doc_types`, unknown until the first
  document is read.

## Why it is built this way
- **The İkamet pack was converted, not written by hand.** A one-off script (`scripts/convert_knowledge.py`,
  commit fdc5ebd, deleted afterwards) read `knowledge/` and the milestone-1 code (field types, document
  mappings, scope test, stage prerequisites). Each condition
  mirrors the original `deterministic_condition` text, which is kept as `legacy_condition` for audit.
- **Strict models.** `extra="forbid"` means a typo such as `"sevrity"` is an error instead of a silently
  ignored key.
- **Unverified rules.** A rule with no source must say `"status": "unverified"`, and answers that use it
  must label it.

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
