# Consolidated technical mock

## Assessment before changes

Branch: `test/integration-abc`. Baseline: 43 passing offline tests. No branch,
commit, merge, or push operations were performed. No teammate advanced integration
was found in available local branches, worktrees, or project directories.

Implemented: LangGraph routing/state/gates, real document adapter, deterministic
R014 with source metadata, PDF/TXT processing, local Ollama, Pydantic checks,
offline/live CLI, and page-level OCR fallback. Planner and approval are mocks;
reflection is absent. The other 28 knowledge rules are records, not executed.
SQLite support was absent; model defaults disagreed (4B processor versus 8B demo).

Pre-existing modified tracked files were `.gitignore`, `README.md`,
`document_processing/README.md`, `extractor.py`, `knowledge/README.md`,
`src/core/graph.py`, `src/core/state.py`, `src/core/stubs.py`, and
`tests/test_extractor.py`. Untracked work included the document/rules workers,
demo script, integration/OCR/demo tests, two enrollment fixtures, and
`data/samples/`. These were preserved. No personal PDF was used or added.

Mandatory changes: shared model defaults, minimal SQLite lifecycle and reopen
tests, coverage date distinctions and Turkish label instructions, regression
checks, and reproducible documentation. Optional work deferred: separate
document schemas, more rules, page-level field provenance, real approval,
reflection, and teammate implementation adoption when actual files are available.

## Implemented phases and boundaries

1. `config.py` centralizes `qwen3:8b`, local host, and timeout defaults used by
   processor, adapter and demo. Existing function parameters and overrides remain.
2. `src/memory/checkpointer.py` uses the official LangGraph SQLite checkpointer
   and an explicit serializer allowlist for project state models. Connections
   initialize and close inside a context manager. Tests reopen the database,
   restore typed state and source metadata, isolate sessions, and resume pending
   rule evaluation without repeating completed extraction.
3. `DocumentInfo` adds optional `coverage_start_date`, `coverage_end_date`, and
   `source_snippets`. Turkish insurance labels distinguish policy approval/issue,
   document issuance, and coverage. If both issuance labels occur, policy approval
   takes precedence for the existing `issue_date`; the source quote can preserve
   both. There is no additional typed document-issuance field yet. Insurance
   coverage end fills a missing unambiguous `expiry_date` for compatibility.
   Numeric dates are rejected, invalid dates flagged, reversed coverage dates
   and conflicting end/expiry dates flagged. Missing identifiers are never filled
   by Python. Model-provided snippets must appear verbatim in extracted text.
   When both Turkish issuance labels appear, Python adds a manual-review issue:
   the live model picked document issuance instead of policy approval in one
   check. No date is silently corrected. Label spelling variations from OCR can
   evade this conservative check; separate typed issuance fields remain deferred.
4. R014 still checks extracted certificate evidence only. `missing` and
   `manual_review_required` are unsatisfied. Quality issues survive Gate 1.
   `supplied_and_accepted` does not establish authenticity, recency, permit
   eligibility, or legal validity. No additional rule was added.

The state topology and public function signatures are unchanged; the intentional
default-model change is 4B to 8B. Old enrollment and insurance payloads remain
valid because all new extraction fields are optional.

## Reproduction in VS Code PowerShell

Use the repository root and existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m pip check

# Entire offline suite; no Ollama, Tesseract, personal files, or Streamlit needed
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# Original seven offline graph scenarios
.\.venv\Scripts\python.exe -B -m scripts.run_demo --offline
# Equivalent entry point
.\.venv\Scripts\python.exe -B -m src.core.graph

# New offline persistent run, then inspect in a new process/connection
.\.venv\Scripts\python.exe -B -m scripts.persistence_demo --offline --session presentation-offline
.\.venv\Scripts\python.exe -B -m scripts.persistence_demo --inspect --session presentation-offline

# Another ID starts isolated: inspection reports no state
.\.venv\Scripts\python.exe -B -m scripts.persistence_demo --inspect --session different-student

# Resume pending nodes. A completed session has no pending work to resume.
.\.venv\Scripts\python.exe -B -m scripts.persistence_demo --resume --session presentation-offline
```

Default storage is `.local/checkpoints.sqlite`. `--db` selects another local path.
Database and sidecar patterns are ignored by Git. Checkpoints contain extracted
facts and validation messages; keep the local database private. Inspection never
calls the model. Resuming pending extraction can call the real model. The
offline flag starts a synthetic run and mocks only Ollama; it is not a resume mode.

Optional live checks require a running Ollama instance and installed Qwen model:

```powershell
ollama list
# Only if the model is absent:
ollama pull qwen3:8b

# Existing live graph CLI
.\.venv\Scripts\python.exe -B -m scripts.run_demo data/synthetic_samples/student_enrollment.txt --model qwen3:8b
.\.venv\Scripts\python.exe -B -m scripts.run_demo data/synthetic_samples/enrollment_missing.txt --model qwen3:8b

# Live extraction -> graph -> R014 -> persisted final state
.\.venv\Scripts\python.exe -B -m scripts.persistence_demo data/synthetic_samples/student_enrollment.txt --session presentation-live --model qwen3:8b
.\.venv\Scripts\python.exe -B -m scripts.persistence_demo --inspect --session presentation-live

# Synthetic Turkish policy: inspect dates and snippets independently of R014
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/insurance_turkish.txt --model qwen3:8b
```

Missing/uncertain evidence produces an expected exit code 1 in graph demos.
Success is extraction-quality acceptance for R014 only. Planner and approval
remain explicitly MOCKED. Reflection is not implemented.

## Limitations and next milestone

- Offline tests validate contracts and Python behavior, not model accuracy.
  Source substring checks establish quote presence, not semantic entailment.
- OCR remains optional and was not verified live. See the module README for
  Windows Tesseract setup; blank/unreadable pages fail rather than return partial
  text. A page mixing usable digital text and a scanned region is not OCRed.
- SQLite is synchronous local checkpoint storage, not an authenticated multi-user
  service. Use a separate session ID per student; sharing an ID shares its state.
  The demo has no retention policy or encryption. It preserves per-run issue
  history in checkpoints; a new event resets current run issues via `new_run`.
- Document IDs currently use filenames. Within one session a repeated filename
  replaces its prior document/facts. Use distinct names for distinct uploads.
- Live extraction may be repeated when resuming an interrupted extraction node.
  No approval, external submission, or legal decision is performed.
- Invalid/conflicting evidence remains under manual review, not a separate legal
  invalidity category. Applicability beyond the scoped demo is not inferred.
- No new rule, full planner, reflection, frontend, or complex memory was added.

## Validation record

- Baseline/phase 1: 43 tests passed after centralizing defaults.
- Phase 2: 47 tests passed; SQLite reopened typed state and resumed pending rules.
- Initial phases 3/4: 54 tests passed, followed by snippet-schema and Windows
  output regressions discovered during live checks and covered by new tests.
- Live complete enrollment: `supplied_and_accepted`, no issues, exit 0. Stored
  result was reopened from SQLite in another process.
- Live incomplete enrollment: missing student ID and issue date remained null;
  `manual_review_required`, stopped safely, expected exit 1.
- Turkish synthetic live policy: classification and coverage period were correct,
  but the issue date was selected from document issuance. This motivated the
  explicit dual-label review guard. The earlier result is not claimed as correct
  policy approval-date extraction.
- Live OCR: not run; Tesseract was absent from PATH and its usual install path.
- Final offline suite: **58 tests passed**; all seven offline demo scenarios
  matched expectations; `pip check` and `git diff --check` passed.
- Final live complete enrollment (session `consolidation-final-verification`):
  `supplied_and_accepted`, no issues, exit 0, confirmed by reopening the database.
- Latest live Turkish policy retained classification, identifiers and source
  quotes but left dates null. The explicit review guard was present. Coverage
  extraction worked in an earlier run but is not claimed reliable across runs;
  use this as a review-required example rather than a successful insurance demo.

All phases used synthetic files only. The full-suite final count and latest live
checks are reported with the handoff; rerun the commands above to verify locally.

Files changed in this consolidation (distinct from pre-existing uncommitted work):

- Phase 1: `config.py`, default settings in `processor.py`,
  `src/workers/document_analyzer.py`, `scripts/run_demo.py`, and the default-model
  assertion in `tests/test_processor.py`.
- Phase 2: new `src/memory/checkpointer.py`, `scripts/persistence_demo.py`,
  `tests/test_persistence.py`; `requirements.txt` and `.gitignore`.
- Phase 3: `schemas.py`, `processor.py`, new `tests/test_insurance_processing.py`
  and `data/synthetic_samples/insurance_turkish.txt`. Includes snippet-schema,
  dual-label review and Windows CLI encoding checks.
- Phase 4: added missing-issue-date and reversed-date cases in
  `tests/test_rules_integration.py`. R014 implementation was not changed.
- Phase 5: `README.md`, `document_processing/README.md`, and this report; extended
  persistence/insurance tests. Existing demos were exercised, not replaced.

Phase-specific checks (all are also included in full discovery):

```powershell
# Phase 1: existing extraction, processing, A/C, and R014 safety checks
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_processor.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_document_integration.py -v

# Phase 2: database lifecycle, reopen, isolation, resume, persistence CLI
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_persistence.py -v

# Phase 3: Turkish labels, dates, source snippets, Windows console, graph review
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_insurance_processing.py -v

# Phase 4: R014 evidence and provenance checks
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_rules_integration.py -v
```
