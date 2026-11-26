# Primary document identity vs referenced documents

## Diagnosis

The earlier classifier applied `indicator_types()` to the whole document. More
than one matched category caused an immediate `unknown`, before Qwen was called.
Thus `method: indicators` on a residence application mentioning a required
insurance policy described a deterministic false conflict, not an LLM decision.
The old code did not distinguish the form's title from a checklist item.

## Current evidence hierarchy

`document_identity.py` separates primary evidence and contextual references:

- Short title-like headings outside reference sections identify the document.
- Reusable self-description/purpose patterns can identify a form or certificate
  without one fixed title.
- Two distinct populated structural roles can support an untitled document
  (e.g. application number and passport number). Placeholder values and repeated
  occurrences of the same ID do not create independent structural roles.
- Checklists, attachment sections, required-document lists and instructional
  lines are contextual references, not competing identities. Legal references
  do not become authoritative titles merely by mentioning another document type.
- Two different primary identities still return unknown with review; the model
  is not asked to overrule clear contradictory authoritative titles. PDF page
  boundaries allow an actual appended document to establish a second identity.

The category definitions for purpose and structural roles are in
`document_profiles.py`. Extraction schemas and grounding dictionaries are
unchanged. `document_classifier.py` selects a deterministic fast path only from
primary evidence, with supporting source quotes. Uncertain cases get at most
one semantic classification call. Quoted references cannot support a model's
selected identity, even if the model quotes a whole block containing the list.

This is role-based evidence, not a full-document keyword occurrence count.
Repeated identifiers are corroboration; distinct populated fields, purpose,
and titles establish identity. The result still uses the existing
`DocumentClassification` structure and `indicators`/`model` method labels.

## Configuration and calls

No public signatures changed. The classifier uses the supplied Ollama client,
so the existing configurable host and per-request timeout still apply. Known
identities, reference-only checklists, and clear primary conflicts make zero
classification calls. Uncertain cases make at most one. Field extraction remains
a separate existing call only after a supported type is established.

```powershell
# Defaults: fallback on; thinking off
$env:PBA_CLASSIFIER_FALLBACK = 'true'
$env:PBA_CLASSIFIER_THINK = 'false'

# Optional deterministic-only classification
$env:PBA_CLASSIFIER_FALLBACK = 'false'
# Restore normal behavior before evaluation
$env:PBA_CLASSIFIER_FALLBACK = 'true'
```

Set `PBA_CLASSIFIER_THINK=true` only for an explicit slower fallback experiment.
It does not enable thinking for field extraction or add a call to the fast path.
Environment flags are read at classification time; no module reload is required.

## Tests and live evaluation

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_classification_identity.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_document_types.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# Synthetic evaluation with mocked fallback replies
.\.venv\Scripts\python.exe -B -m scripts.evaluate_classification

# Live fallback through Qwen3:8b (deterministic cases still need no model call)
.\.venv\Scripts\python.exe -B -m scripts.evaluate_classification --live --model qwen3:8b --timeout 60

# Full real extraction for the synthetic application with referenced attachments
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/classification_identity/application_attachments.txt --model qwen3:8b
```

The last command tests extraction too; the classification fixture intentionally
lacks some required applicant facts, so missing fields should remain missing.
Successful document classification is not successful field validation or legal
approval. No private PDF or real policy identifier is in the evaluation set.

Results in this session:

- New identity test methods: 11 passed. Existing type-aware methods: 15 passed.
- All importable offline tests: 90 passed, no assertion failures. Full discovery
  also has five import errors because Windows blocks the existing `ormsgpack`
  DLL. Those modules contain 32 additional tests that could not execute; graph
  and SQLite re-verification is not claimed.
- Mocked classification evaluation: 8/8 matched expected types.
- Live Qwen evaluation: 7/8. All seven requested cases passed: application with
  insurance attachments, enrollment with residence requirements, insurance with
  permit legislation, all-type checklist, policy with unrelated references,
  unrelated inventory, and conflicting titles. The extra unfamiliar registrar
  note remained unknown with review via the semantic fallback. It is recorded
  as a false negative, not a successful classification.
- The broader mocked development and holdout evaluations still pass.
- `pip check` and `git diff --check` passed. No Windows settings were changed.

## Boundaries

Section/heading detection is conservative and depends on reading order. An
unmarked attachment list can still be hard to distinguish from a second real
document. Multi-column, heavily damaged OCR or unknown section labels may
require review. The model can still decline or misinterpret an unfamiliar
document; quotes prove source presence, not authenticity or semantic truth.
No extraction schema, graph topology, R014 rule, persistence, OCR or UI change
was needed. The real personal application PDF was not opened for this work.
