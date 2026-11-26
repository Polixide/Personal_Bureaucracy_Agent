# Document-type-aware processing mock

Classification now separates [primary identity from referenced attachments](classification_identity.md).
Competing keywords in a supporting-document list do not by themselves create a
conflict. See that report for the configurable semantic fallback and current
offline/live evaluation results.

## Audit and minimal design

Before this refactor, `processor.py` had one long insurance-heavy prompt and one
normalized Pydantic schema. `document_evidence.py` inferred insurance type and
applied insurance date/ID roles during generic processing. Unknown classification
and field extraction were coupled. Most layout regressions covered insurance;
enrollment had graph coverage but few alternate layouts. No application type
existed. The normal PDF/TXT/OCR extractor was already reusable and is unchanged.

The public `extract_text()` and `process_document()` calling conventions remain
unchanged. The graph receives `DocumentInfo` as before, serialized into facts.
R014, graph edges, SQLite code, and Streamlit are unchanged. The document adapter
adds the application type to its explicit mapping; it does not add a new rule.

Pipeline:

```text
PDF/TXT/OCR -> classify -> profile schema -> Qwen extraction
           -> contextual candidates + reusable grounding -> DocumentInfo
           -> existing graph adapter / checkpoints / R014
```

- `document_classifier.py`: source-backed supported indicators, then a Qwen
  fallback for unfamiliar layouts. A valid quoted passage and category concepts
  are required for fallback classification. Conflicts, unsupported types,
  ungrounded quotes and invalid classification replies yield `unknown` with
  review issues. Strong indicators override a wrong type in an extraction reply.
- `document_profiles.py`: small schema registry, field expectations, per-type
  instructions and field-label dictionaries. Specific extraction schemas expose
  only applicable domain fields. Legacy `institution` maps to insurance `insurer`;
  normalized output retains both. Application `applicant_name` also maps to
  `holder_name` for compatibility.
- `document_evidence.py`: shared date/identifier normalization, source snippets,
  page/line provenance, conflicts, semantic quote grounding and correction notes.
  The processor always supplies the classified type. Insurance coverage roles
  and policy labels execute only in the insurance profile. A model-mapped
  unfamiliar field needs an actual source quote, a supported numeric value, and
  no contradiction with another recognized field role. Names/status need source
  text support too; these checks do not establish authenticity or semantic truth.
- `schemas.py`: normalized compatibility representation and shared deterministic
  date/missing/ambiguity checks, plus classification and optional application
  fields. Internal extraction dates are strings/null so final grounding and
  shared validators control numeric rejection and calendar validity.

## Requirements are project contracts, not legal requirements

| Type | Required for this mock's accepted extraction | Expected, nonblocking | Other fields |
| --- | --- | --- | --- |
| Enrollment | holder name, student ID, institution, issue date | enrollment status | expiry optional |
| Health insurance | holder name, institution/insurer, policy number, issue date, expiry | coverage start/end | explicit expiry supported; coverage end remains its fallback alias |
| Ikamet application | applicant name, application number, application date | passport number, permit type | incomplete/blank forms require review when required fields are absent |
| Unknown | no guessed field requirements | none | always review; no field extraction |

All internal schema fields can be null. The registry's `required()` describes the
normalized required fields; `expected` lists nonblocking expectations. Remaining
schema fields are optional. Existing enrollment and insurance requirements were
preserved. A short/partial application may legitimately omit suggested fields;
the missing result describes this demo's extraction contract, not legal invalidity.

`DocumentInfo` adds `classification`, `expected_missing_fields`, `insurer`,
`enrollment_status`, `applicant_name`, `application_number`, `passport_number`,
`application_date`, and `permit_type`. Existing fields and correction/evidence
metadata remain. Inapplicable model fields are discarded before normalization,
not reported as missing. R014 still uses the same certificate name/date/ID facts.

## Evaluation and reproduction

Use VS Code PowerShell at the repository root, in the existing environment:

```powershell
# Complete suite, including unchanged graph/OCR/SQLite regressions
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# Type-aware tests independent of LangGraph native serialization imports
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_document_types.py -v

# Mocked model replies: development and held-out contracts
.\.venv\Scripts\python.exe -B -m scripts.evaluate_documents
.\.venv\Scripts\python.exe -B -m scripts.evaluate_documents --holdout

# Real Qwen inference, not mocked (requires local Ollama + installed qwen3:8b)
.\.venv\Scripts\python.exe -B -m scripts.evaluate_documents --live --model qwen3:8b
.\.venv\Scripts\python.exe -B -m scripts.evaluate_documents --live --holdout --model qwen3:8b

# Individual live examples from each supported type
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/type_evaluation/student_complete.txt --model qwen3:8b
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/type_evaluation/insurance_multiline.txt --model qwen3:8b
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/type_evaluation/application_complete.txt --model qwen3:8b
```

The manifest at `data/synthetic_samples/type_evaluation/cases.json` contains
privacy-safe synthetic source files, expected fields, mock replies, and review
expectations. Development: three layouts per supported type, one unrelated
document, one mixed-type document (11 cases). Holdout: one unfamiliar layout per
type (3 cases), used to exercise semantic classification/quotes without adding
holdout label patterns. This is a tiny developer-created holdout, not an
independent blind benchmark. Keep future unseen documents separate and evaluate
them with `--live`; do not tune production patterns to their expected values.

Metrics are computed separately by expected document type:

- Classification correctness: exact type matches / cases.
- Field accuracy: exact matches including null / explicitly scored fields.
- Required-field recall: correctly extracted required fields present in the
  fixture / required fields present in the fixture.
- Unsupported-value rate: returned applicable non-null fields that differ from
  the fixture truth or are absent from it / returned applicable non-null fields.
- Manual-review correctness: matches expected missing/ambiguous/issue decisions.

No model accuracy is claimed for mock mode. The mocked development and holdout
checks reached 100% classification/field/required recall/review scores and 0%
unsupported values for the supported types. Unknown has no scored domain fields
so those ratios are null; its classification and review scores are 100%.

Verification in this session: 15 new type-aware test methods passed. Full test
discovery executed 79 passing tests with no assertion failures, plus five module
import errors caused by the baseline DLL restriction. Those modules contain 32
additional test methods which could not execute. The 11 development and three
holdout mocked evaluations passed. No live model evaluation was run for this
refactor. `pip check` and `git diff --check` passed. The Streamlit file was not
modified. This is not an all-green full integration-suite result.

## Environment and limitations

The baseline full-suite attempt was already blocked by Windows application
control denying the installed `ormsgpack` native DLL, including outside the
sandbox. This prevents imports of the demo, graph integration, insurance graph,
persistence, and R014 graph test modules. No Windows controls or native packages
were modified. The final report separates these import failures from passing
document-processing tests; rerun the complete suite once the legitimate
dependency can load in an approved environment. Graph and SQLite regressions
cannot be claimed passing for this session.

Classification indicators and lexical guardrails are conservative but not proof
of document purpose; model-assisted classification may still be wrong. Quotes
prove source presence, not correct interpretation or genuine documents. Reading
order, unknown scripts/labels, damaged OCR, partial names and mixed-document PDFs
can require review. A semantic identifier cannot be borrowed from a different
recognized field label. Insurance heuristics do not execute in other profiles.
Fallback classification can add a second model request; timeout remains per
request. No real private documents or Streamlit flows were evaluated here.
