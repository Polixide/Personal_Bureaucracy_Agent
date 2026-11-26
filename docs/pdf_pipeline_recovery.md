# Real PDF pipeline recovery — redacted diagnostics

## Observed failures before editing

Three explicitly authorized local PDFs were inspected read-only. No personal
names, identifiers, filenames or full source text are reproduced here. Their
roles are application, declaration/control document, and insurance policy.

| Document | Pages | Extracted characters per page | Earliest observed failure |
| --- | --- | --- | --- |
| Application | 4 | 5734, 6321, 8981, 1196 | Identity-bearing text was extracted but excluded by line/section heuristics. One 72.03-second fallback returned the correct application type with no evidence quotes; the evidence validator correctly refused that unsupported response. Extraction was skipped. |
| Declaration/control | 1 | 2674 | Its main identity is outside the supported categories. Referenced document names were instead treated as conflicting primary titles. No model call occurred. Unknown is appropriate, but the old reason was misleading. |
| Insurance | 1 | 1936 | Its heading spans lines. Whole-text matching found the type, but single-line identity detection did not. The classification fallback timed out after 90 seconds before extraction. No completed raw model decision was available for that attempt. |

All extracted pages were sent to the old fallback; there was no Python text
selection/truncation stage. The active Ollama context was 4096 tokens, making
model-side truncation a risk for the application plus instructions/schema.
That risk is separate from the directly observed missing-quotes rejection;
we do not claim to have measured which original tokens the server dropped.

Numbered self-description paragraphs were mistaken for attachment list items.
Long merged PDF lines containing a true self-description failed heading-length
checks. Wrapped headings were not recognized. Later form-field headings could
then be mistaken for rival document identities. These are reading-order/context
problems, not evidence that the PDFs had no usable text or that thinking mode
alone caused the regression.

## Small corrections

- `document_identity.py` accepts bounded adjacent-line headings and self-purpose
  statements in numbered paragraphs. A mention in a sentence does not start a
  persistent reference section. Wrapped mid-page headings are weaker than an
  independently established title, self-purpose or populated structural roles.
  Strong contradictory identities still require review. Early unsupported
  document headings keep control/declaration/checklist material outside the
  three supported extraction categories.
- `document_classifier.py` uses a small response contract requiring type,
  evidence and ambiguity fields, without inviting free-form prose in `method`.
  Known type without trustworthy quotes is still not blindly accepted.
- `document_context.py` selects bounded source context with identity quotes,
  first-page fields, relevant labels and page heads. Quotes are validated
  against the complete original source, not an invented summary.
- `processor.py` still returns the same normalized object. Unknown types retain
  generic date candidates and source/page provenance as audit evidence, without
  assigning them to a domain field or approving anything. Known types get their
  existing profile. Candidate messages omit unassigned legal/reference dates;
  the full source evidence is retained in the result. Candidates are explicitly
  nonexhaustive; names/institutions still come from the original source.
- `document_profiles.py` requires nullable field keys in the generation contract,
  puts values before metadata and bounds optional quote strings. This does not
  make absent values mandatory or change project/legal field requirements.
- `document_evidence.py` can ground a composed full name against unique labeled
  given/family fields on the same page, at most twelve lines apart. It retains
  the exact source block and a correction note. Unlabeled tokens, different
  people, invented components and cross-page combinations remain unsupported.

The public extraction/processing signatures, LangGraph topology, R014, SQLite,
OCR and Streamlit remain unchanged. No new dependencies or native-security
workarounds were introduced. Existing uncommitted work was preserved.

## Model budget and performance

```powershell
$env:PBA_OLLAMA_NUM_CTX = '8192'
$env:PBA_OLLAMA_MAX_TOKENS = '768'
$env:PBA_CLASSIFIER_CONTEXT_CHARS = '10000'
$env:PBA_EXTRACTION_CONTEXT_CHARS = '6500'
$env:PBA_CLASSIFIER_FALLBACK = 'true'
$env:PBA_CLASSIFIER_THINK = 'false'
```

These are defaults and remain configurable. Per-call timeouts still use existing
arguments. Character caps are not exact tokenizer limits. The server reported
zero VRAM allocation during this session, and the application extraction timed
out at 180 and 240 seconds during intermediate checks. A separate intermediate
policy extraction completed in 64.31 seconds and recovered all five core ID/date
fields with source evidence; holder and insurer were missing, so review remained
required. These intermediate checks are not presented as fully validated results.

A 6500-character application context completed in 48.3 seconds, with four of
five fields accepted before the separate-name grounding fix. Replaying that
captured real response in memory after the fix accepted all five, without
missing/ambiguous fields or validation issues. This replay is a deterministic
regression check, not a second model inference. No private response was saved.

A subsequent fresh end-to-end application run confirmed the final code:
`ApplicationExtraction`, five of five fields populated and source-supported,
no missing/ambiguous/unsupported fields, no validation issues, two correction
notes. It took 130.00 seconds including one 129.85-second Qwen3:8b call with
thinking disabled, 3371 prompt tokens and 245 output tokens (normal stop).
Source selection retained 6497 characters across all four pages. Deterministic
classification inspected every page and accepted three exact identity quotes;
the semantic classifier was not invoked (zero classifier characters sent).

| Final private check | Profile fields / source-supported | Review | Total time / model calls |
| --- | --- | --- | --- |
| Application | 5 / 5 | No extraction-quality review flags | 130.00 s / 1 |
| Insurance | 7 / 7 | No extraction-quality review flags | 128.99 s / 1 |
| Declaration/control | No supported profile; one generic audit candidate | Required | 0.10 s / 0 |

Absence of extraction-quality review flags does not mean legal approval.

The final policy inference completed in 128.99 seconds (one model call), with
all seven profile fields populated and source-supported, no missing/ambiguous
fields and no validation issues. The declaration/control document correctly
remained unknown in 0.10 seconds, with no model calls and one generic audit
candidate retained. Its domain extraction was deliberately skipped.

Offline development (eleven cases), holdout (three cases) and classification
(eight cases) evaluations passed their mocked contracts. They measure pipeline
behavior under controlled model replies, not real-model accuracy. Five test
modules remain unimportable because Windows application control blocks the
installed `ormsgpack` native DLL; this restriction was not changed or bypassed.
The final full discovery run reported 109 passing tests and five loader errors,
with no assertion failures. Nineteen new recovery regressions pass. The blocked
modules are demo, document integration, insurance processing, persistence and
rules integration; their 32 test methods could not execute. Thus graph/SQLite
end-to-end execution is not verified on this machine in this session.

## Files changed in this repair

`config.py`, `document_classifier.py`, `document_identity.py`,
`document_profiles.py`, `processor.py`, `document_evidence.py`;
new `document_context.py`, `scripts/private_document_check.py`,
`tests/test_pdf_pipeline_recovery.py` and this report. Some of these files were
already untracked or modified from the earlier refactor. Other pre-existing
working changes were left in place.

## Remaining limitations

This is evidence from two live supported documents and one unsupported document,
not a general accuracy benchmark. Live student enrollment was not evaluated;
its development and holdout checks use mocked responses. Bounded context can
omit facts on unfamiliar long layouts; increase the configurable source budget
and model context together when necessary. Name-component grounding deliberately
refuses multiple people, distant fields and cross-page combinations. CPU-local
inference remains variable and can exceed the unchanged 120-second public
processor timeout; use an explicit larger timeout for these PDFs. Genuine
identity conflicts and unsupported citations still require review.

The final redacted evaluation is reported in the handoff. No thinking-on
full-document comparison was attempted: the identity failure and missing-quote
contract failure were already observed with thinking disabled, and unnecessary
additional private inference was avoided.

## Reproduction without exposing personal fields

From the repository root, use the existing environment:

```powershell
# Synthetic layout, source retention, schema and diagnostic-privacy regressions
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_pdf_pipeline_recovery.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# Balanced synthetic checks and classifier cases (mocked model replies)
.\.venv\Scripts\python.exe -B -m scripts.evaluate_documents
.\.venv\Scripts\python.exe -B -m scripts.evaluate_documents --holdout
.\.venv\Scripts\python.exe -B -m scripts.evaluate_classification

# Replace paths locally. The report prints counts/field names, not field values.
.\.venv\Scripts\python.exe -B -m scripts.private_document_check 'C:\private\application.pdf' 'C:\private\policy.pdf' 'C:\private\control.pdf' --model qwen3:8b --timeout 480
```

The private checker reports page/character counts, context selections, primary
roles, raw classifier type/quote-check booleans when a fallback runs, schema,
field/source-support counts, missing/ambiguous field names, review status, call
counts and durations. It intentionally omits arbitrary model/error prose and
private field values. It writes no logs or database. Generic evidence is audit
material, not evidence of legal compliance or a supported document identity.

Do not save full `processor.py` JSON output or private extracts in the repository.
No real PDF was copied, modified, included in fixtures, or added to Git.
