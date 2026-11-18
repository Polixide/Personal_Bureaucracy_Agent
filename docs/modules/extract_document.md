# `src/tools/extract_document.py` — documents to proposed facts

## What it does
It reads an uploaded PDF or TXT and returns what the document *shows*: values backed by an exact quote and page.
It maps them to the procedure's fact names with `procedure.json` `documents[].extracts`.

## Where it sits in the flow
- **App:** on upload, the app calls the registered `extract_document` tool.
- **Agent:** the orchestrator or a worker can call it for a pending document.

The registered tool (phase 4) stores the observation in the Twin. Facts are stored with
`provenance=document`, and they are confirmed unless the pack says `requires_confirmation`. This module's
`extract_file()` does no storage.

## Inputs and outputs
- **Input:** a file path, the `Pack`, an optional `doc_type`, and a backend (only needed for the generic path).
- **Output:** `ExtractionResult{observations: [DocumentObservation], error}`. For example:
  ```json
  {"doc_type": "health_insurance_certificate",
   "facts": [{"field": "insurance_end_date", "value": "2027-06-30", "from_field": "coverage_end_date",
              "page": 1, "snippet": "Coverage end: 30.06.2027", "requires_confirmation": false}],
   "ambiguous_fields": [], "issues": []}
  ```

## Why it is built this way
- **Typed path (kept from milestone 1).** The processor classifies the document, extracts with a typed schema,
  and cross-checks every value against deterministic evidence (labels, MRZ check digits, date roles, OCR pages).
  `grounding.py` keeps only values with matching evidence (same rule as milestone 1).
- **Generic path (new).** It handles a document type that exists only in a pack (for example a residence
  permit card). The model proposes `{field, value, page, quote}`. Code keeps a proposal only if the quote is on
  that page, the value is in the quote, and it parses as the fact's type. So a new procedure needs no new Python.
- **Passport bundles.** A passport PDF with an entry stamp becomes two logical documents, as before.

## How to test it
```
python -m pytest -q tests/test_extract_document.py tests/test_processor.py tests/test_extractor.py
```

## Known limits
- **Classification.** The typed classifier only knows the seven milestone-1 kinds. Other kinds need `doc_type`
  from the user or the agent.
- **Generic quote check.** The generic path's check is literal: a quote reworded by OCR is rejected, which is
  safe but may need a manual confirmation.
