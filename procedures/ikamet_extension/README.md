# Procedure Pack: `ikamet_extension` (UNVERIFIED draft)

Short-term residence permit extension for Sabancı University incoming exchange students. It was added in phase 10
to show that a second procedure needs **no change in `src/`** (checked by `tests/test_generalization.py`).

## Status
- **Draft rules.** All 8 rules and the procedure are `"status": "unverified"`. The agent must say so in every
  answer, and Gate 1 enforces it (`UNVERIFIED_NOT_LABELLED`).
- **Sources.** The 4 sources are copied unchanged from `procedures/ikamet_first/sources.json` (originally the
  milestone-1 `knowledge/sources.json`). No URL or date was invented.
- **Source texts.** `sources_text/` holds placeholders that list the draft rules, not the official text.

## Review list for a person (before switching any rule to `verified`)
| Rule | Draft | Check against |
|---|---|---|
| EXT001 | apply while the current permit is still valid | extension part of SAB_EXCHANGE_CURRENT: is there an earliest or latest window? |
| EXT002 | a copy of the current permit card is needed | SAB_EXCHANGE_CURRENT document list |
| EXT003 | still an exchange student for the extension period | SAB_EXCHANGE_CURRENT, GOV_RESIDENCE_TYPES |
| EXT004 | passport valid 60 days beyond the requested end | GOV_FAQ (verified for first applications as ikamet_first R006) |
| EXT005 | insurance covers the requested extension period | GOV_RESIDENCE_TYPES, SAB_EXCHANGE_CURRENT |
| EXT006 | application through e-İkamet | GOV_GENERAL_INFO |
| EXT007 | extension form signed | SAB_EXCHANGE_CURRENT |
| EXT008 | no guaranteed approval (guardrail) | GOV_FAQ |

## How to verify
1. Save the official page texts in `sources_text/<SOURCE_ID>.txt`, replacing the placeholders.
2. Follow the `draft-rules-from-source` skill for each rule. Add an `evidence_quote` and correct the condition.
   Then set `"status": "verified"`.
3. Run `python -m src.pack validate ikamet_extension` and then `python -m pytest -q`.
