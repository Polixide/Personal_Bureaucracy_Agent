# Hybrid document extraction

## Diagnosis

The old processor passed the entire extracted text to Ollama, but delegated
classification, date interpretation, and policy identifiers entirely to the
model. Calendar-valid ISO dates could pass Pydantic even if absent from the
source. Optional quotes were checked only for substring presence, not whether
they supported the field. An unknown type had no type-specific missing-field
requirements, although it did carry a review issue. No private PDF was examined
during this change; its exact text layout must be checked locally by its owner.

## Pipeline

1. `extract_text()` retains native PDF extraction and existing OCR fallback. PDF
   pages now have a form-feed separator so page positions survive the string
   interface. TXT behavior is unchanged.
2. `document_evidence.collect_evidence()` identifies numeric dates and recognized
   labels, retaining original source strings, exact snippets, line numbers and
   PDF page numbers. Unlabeled/legal-reference dates are audit candidates only.
3. Qwen receives the unchanged source text and an additional explicitly untrusted
   candidate message. It still performs semantic extraction. Python replaces any
   model-generated `source_evidence` with deterministic evidence.
4. `ground_payload()` resolves only unique labeled values, then Pydantic runs
   calendar, missing-field and ordering checks. Genuine source ambiguity remains
   unresolved; unique insurance evidence can resolve model ambiguity. Unsupported
   fields become null when no reliable source exists.

Numeric dates support ISO YYYY-MM-DD and day/month/year with `/`, `.` or `-`.
An ambiguous date such as `03/04/2028` stays unresolved unless an explicit
DD/MM/YYYY (or MM/DD/YYYY) format or consistent unambiguous nonlegal dates
establish order. Turkish language alone does not establish day/month order.
Malformed/calendar-impossible dates remain null with review issues.

Recognized Turkish coverage-period labels associate exactly two dates with start
and end, including strings containing time suffixes. Times are ignored because
the public fields contain dates. A one-date or multi-date period is unresolved.
Policy approval/issue and generic document issuance are distinct evidence roles.
An explicit expiry is preserved even if it conflicts with coverage end, and the
conflict is flagged. We never select the first date in the PDF as a field value.

Strong Turkish/English health-policy indicators can resolve a model's unknown
classification. Conflicting supported headings stay unknown and require review.
`POLİÇE NO`, `POLİÇE NUMARASI`, `Policy number`, and `Policy no` accept only a
single labeled identifier token containing a digit. Tax/identity numbers are
not searched as policy numbers. Multiple conflicting labeled identifiers stay
null. No real person, policy number, insurer, or policy date is hardcoded.

If a model returns a different insurance date but one reliable labeled source
candidate exists, its output is rejected in favor of that candidate **with an
explicit correction note**, not a silent correction. If the source is ambiguous/conflicting,
the field stays null. Non-ISO model dates may be normalized only under the same
unambiguous source rules. Missing insurance fields can be filled from unique
source evidence; missing enrollment model fields retain the prior review behavior.

Required-field checks run after classification and grounding. Unknown documents
carry an explicit issue that required fields cannot be determined; they are not
successful verified extractions. Presence, extraction quality, and legal
validity remain separate. R014 and the graph/checkpoint integration are unchanged.

## Layout and reconciliation update

The current insurance reconciliation resolves a model's null, ambiguity flag,
or incorrect normalization when a unique labeled source value is established.
It clears stale missing/ambiguous flags and records informational changes in
`correction_notes`. Such corrections alone do not mean unresolved manual review.
Enrollment's existing conservative ambiguity behavior is unchanged. Numeric
model dates still receive the existing strict schema rejection.

Genuine conflicting source values, invalid calendars, uncertain day/month
order, unsupported identifiers, and fabricated quotes still require review.
Identical normalized dates corroborate each other. Empty repeated labels do not
override a valid observation; nonempty invalid observations still prevent a
unique resolution. Policy approval and document issuance remain separate roles.

Continuation scanning accepts up to four neighboring lines consisting of dates,
times, separators or whitespace, stopping at another recognized field, birth/
identity/tax/phone field, legal reference, footer marker or PDF page boundary.
Split Turkish coverage headings are supported. A period must contain exactly two
dates: first is start, second is end. Extra dates are not silently truncated.

Policy prefix punctuation is stripped before validation, including when the
value is on the next line. `original_value` and the source snippet preserve it.
Unicode label offsets are mapped back to the original text, so decomposed Turkish
characters do not shift the start of an identifier. Quotes use NFKC and layout
whitespace normalization, including spaces around date/label punctuation; words
and digits are not changed. Selected values keep precise original source quotes.

Synthetic layout fixture and optional live check:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_evidence_reconciliation.py -v
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/insurance_multiline_layout.txt --model qwen3:8b
```

Automated layout tests mock only Ollama replies. No private documents or real
policy identifiers are part of these fixtures.

Original model quote failures are audited after deterministic grounding and final
Pydantic checks. If the final field equals unique, unambiguous source evidence,
its replacement quote is verified, and no field-related validation conflict
remains, the model citation mismatch is retained in `correction_notes`. It is
not reported as an unsupported final value. Null/ambiguous/conflicting fields
and fields without independent deterministic support retain the citation warning
in `validation_issues`. Expiry aliases also inherit coverage ordering conflicts.
The citation failure itself is never discarded.

The citation-order regression suite passed with 95 total offline tests, including
verified fields with invalid model quotes and genuinely unsupported/fabricated
values. No private PDF was opened or live inference repeated for this change.

Current reconciliation verification: 90 offline tests passed, including existing
OCR, SQLite, R014, and enrollment tests. All seven offline demo scenarios passed.
A separate real Qwen3:8b run on `insurance_multiline_layout.txt` populated the
policy number, issue date, coverage dates and expiry with no missing/ambiguous
fields or validation issues. No personal PDF was inspected. These results do
not establish accuracy for every possible private PDF layout.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_document_evidence.py -v
.\.venv\Scripts\python.exe -B -m scripts.run_demo --offline
```

Automated tests mock Ollama responses. They check deterministic recovery from
missing or wrong model values, ambiguous/invalid/conflicting dates, legal
references, policy IDs, page references, source quotes, and enrollment regression
behavior. They do not measure a model's accuracy on arbitrary real policies.
The generated insurance PDF in an older test now contains the values its mocked
reply claims; accepting values absent from that PDF is no longer permitted.

## Optional real Qwen tests

Ollama must be running locally with `qwen3:8b` installed. Commands are from the
repository root using the existing environment:

```powershell
ollama list
.\.venv\Scripts\python.exe -B processor.py data/synthetic_samples/insurance_numeric_dates.txt --model qwen3:8b
.\.venv\Scripts\python.exe -B -m scripts.run_demo data/synthetic_samples/student_enrollment.txt --model qwen3:8b

# Private local PDF: do not redirect extracted output into repository logs
.\.venv\Scripts\python.exe -B extractor.py 'C:\private\policy.pdf'
.\.venv\Scripts\python.exe -B processor.py 'C:\private\policy.pdf' --model qwen3:8b
```

Inspect the extractor output locally to confirm the headings, issue date, period
and policy-number label survived PDF extraction. Compare `source_evidence`
against each returned date and policy number. Never commit personal PDFs,
extracted text, console output, or checkpoints containing personal data.
The standalone processor prints review issues in JSON; the graph CLI exits 1
when those issues block progression. R014 cannot be satisfied by an insurance
document, regardless of successful extraction.

## Earlier hybrid baseline verification

The full offline suite passed (77 tests), including OCR, R014, graph and SQLite
regressions. All seven offline demo scenarios matched their expectations.
Separate real Qwen3:8b runs on the numeric-date synthetic policy and enrollment
fixture succeeded: policy date/ID fields were source-grounded with no issues;
enrollment returned R014 `supplied_and_accepted` with no issues. No real personal
document or live OCR was tested. These results do not establish accuracy on
arbitrary private policy layouts.

## Limits and manual review

This is a small label-based parser, not a general PDF-layout engine. Unknown
labels, damaged OCR, split headings, multi-column text, multi-token policy IDs,
or multiple policies in one PDF may require manual review. Label continuations
are deliberately short and do not cross PDF pages. A source quote proves
presence and selected numeric interpretation, not document authenticity or the
truth of the original document. Holder/institution semantics still depend on
the LLM. OCR is unchanged and not newly verified live. The source snippet/schema
and SQLite state fields remain backward compatible through optional additions.
