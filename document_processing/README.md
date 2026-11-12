# Document text extraction

Default Ollama settings are centralized in `config.py`: `qwen3:8b`,
`http://127.0.0.1:11434`, 120 seconds. Existing model/host/timeout overrides work.
See [consolidation workflow](../docs/consolidation.md) for SQLite session demos.

Turkish insurance extraction now distinguishes `coverage_start_date` and
`coverage_end_date` from `issue_date`. `POLİÇE ONAY TARİHİ` takes precedence over
`DÜZENLEME TARİHİ` for the existing issue field; exact `source_snippets` can retain
both labels. Coverage end fills a missing unambiguous insurance `expiry_date`.
The hybrid source checks distinguish approval from document issuance. Unique
policy-approval evidence takes priority; document issuance is a fallback only
when no policy issue label is present. A model/source disagreement is retained
as a review issue. Merely having both labels no longer requires a blanket warning.
Conflicting end dates and reversed coverage periods are flagged deterministically.
Source quotes and their field/date association are checked deterministically.
For insurance, uniquely supported values can resolve model ambiguity or bad
normalization. Informational `correction_notes` explain those changes; genuine
source conflicts remain in `validation_issues`. Quote comparison tolerates
layout whitespace and harmless Unicode normalization differences.
`source_evidence` retains original strings, normalized candidates, snippets and
line numbers, plus 1-based PDF page numbers where available (null for TXT).
Separate document-specific models remain future work. Both insurance fixtures
are entirely synthetic. See [hybrid extraction](../docs/hybrid_extraction.md)
for candidate rules, limitations, and live-test commands.

## End-to-end R014 verification

Run from the repository root in the existing environment:

```powershell
# All tests: no Ollama required
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# Deterministic offline demo: only Ollama is mocked
.\.venv\Scripts\python.exe -B -m src.core.graph
# Equivalent CLI entry point
.\.venv\Scripts\python.exe -B -m scripts.run_demo --offline

# Live graph -> real document processor/Ollama -> R014
ollama list
.\.venv\Scripts\python.exe -B -m scripts.run_demo data/synthetic_samples/student_enrollment.txt --model qwen3:8b
```

For live processing, Ollama must be running and `qwen3:8b` installed (run
`ollama pull qwen3:8b` if needed). The CLI accepts a PDF/TXT path and optional
`--model`, `--host`, and `--timeout` settings. It prints extracted facts, document
status, graph issues, R014 result and source metadata, actual node path, and final
response. Live runs exit with status 1 when graph issues are reported.

The offline demo exercises a complete certificate, no supplied certificate,
incomplete fields, ambiguous and impossible dates, unsupported input, and
unavailable Ollama. PASS means the scenario matched its expectation, not that
the document was accepted. Only the complete certificate should satisfy R014.
Unsupported input is intentionally rejected by extension before file access;
it needs no actual `.exe` fixture. All analyzed scenario files are synthetic
fixtures resolved relative to the repository, not invented document filenames.

The offline date scenarios mock the model returning null and ambiguity flags.
They verify graph behavior, not a model's ability to recognize impossible dates.
Use the live command with `enrollment_ambiguous.txt`, `enrollment_invalid_date.txt`,
or `enrollment_missing.txt` to inspect actual model behavior. Planner and approval
are MOCKED in both modes. R014 acceptance is not legal validity or authenticity.

Run these commands in the VS Code PowerShell terminal from the repository root.
Use the existing virtual environment; activation is optional.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe extractor.py data/synthetic_samples/student_enrollment.txt
.\.venv\Scripts\python.exe extractor.py "path\to\document.pdf"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Integration from another Python module:

```python
from extractor import extract_text

text = extract_text("data/synthetic_samples/student_enrollment.txt")
```

`extract_text` accepts a string or `Path` and returns a string. TXT files must
be UTF-8 (an optional BOM is accepted). Empty TXT files return an empty string.
PDFs use PyMuPDF and preserve page order. Missing files raise
`FileNotFoundError`; unsupported extensions or directories raise `ValueError`.
Unreadable, password-protected, or textless PDFs raise `DocumentExtractionError`.
Permission and text decoding errors propagate to callers. The command-line
interface reports input errors and exits with status 1.

PDF pages use normal PyMuPDF extraction first. Pages without usable text (at
least one letter or digit) use PyMuPDF's Tesseract OCR at 300 DPI. Digital and
scanned pages can coexist; page order is preserved. If any page's OCR fails or
returns no usable text, extraction fails rather than returning partial text.
This deliberately includes blank pages. Pages with some digital text plus a
scanned region are not OCRed by this minimal fallback. OCR may misread names,
identifiers, dates, tables, handwriting, or rotated/low-quality scans; review
important fields. No original PDF is modified.

## Windows OCR setup

1. Install Tesseract for Windows using the UB Mannheim installer linked from
   [Tesseract's installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html).
   The usual install location is `C:\Program Files\Tesseract-OCR`.
2. Install English language data (`eng.traineddata`). For Turkish, also install
   `tur.traineddata`. Select the language data in the installer, or obtain it from
   the official [tessdata_fast repository](https://github.com/tesseract-ocr/tessdata_fast)
   and place the actual files in the installation's `tessdata` folder.
3. Configure the current VS Code PowerShell terminal before running Python:

```powershell
$env:TESSDATA_PREFIX = 'C:\Program Files\Tesseract-OCR\tessdata'
$env:PBA_OCR_LANGUAGES = 'eng'  # Default; use 'eng+tur' for English and Turkish
Test-Path "$env:TESSDATA_PREFIX\eng.traineddata"
# Required only when Turkish is requested:
Test-Path "$env:TESSDATA_PREFIX\tur.traineddata"
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --list-langs

# Test extraction without Ollama:
.\.venv\Scripts\python.exe extractor.py 'C:\path\to\scanned_certificate.pdf'

# Test the existing graph with real Ollama/Qwen:
.\.venv\Scripts\python.exe -m scripts.run_demo 'C:\path\to\scanned_certificate.pdf' --model qwen3:8b

# All offline tests (OCR API is mocked):
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Set `TESSDATA_PREFIX` to the folder containing the `.traineddata` files, not
the executable or its parent. These environment settings apply to this terminal
and its child Python processes; repeat them in a new terminal or configure user
environment variables in Windows and restart VS Code. No signature changes to
`extract_text()` or `process_document()` are needed. No extra Python OCR package
is required. PyMuPDF uses its integrated OCR API and Tesseract language data;
the Tesseract CLI is useful for installation diagnostics, not called by this code.
See [PyMuPDF OCR API](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_textpage_ocr)
for the underlying interface. Digital PDFs and TXT files need no OCR setup.

## Structured processing with local Ollama

Install the requirements above into the existing environment. Start Ollama
(`ollama serve` if it is not already running), then prepare and run the model:

```powershell
ollama pull qwen3:8b
.\.venv\Scripts\python.exe processor.py data/synthetic_samples/student_enrollment.txt
# Use a different installed model:
.\.venv\Scripts\python.exe processor.py data/synthetic_samples/health_insurance.txt --model qwen3:8b
```

The processor sends extracted text to `http://127.0.0.1:11434` by default;
`--host` and `--timeout` can override the endpoint and request timeout in seconds.
It never downloads models automatically. No paid API is used. Keep the default
local endpoint when processing documents that should stay on your machine.

`process_document` reuses `extract_text`, requests JSON using the Pydantic schema,
and returns a validated `DocumentInfo`. Temperature is zero, but model extraction
can still be mistaken. The prompt prohibits guessing; missing values are `None`.
Values the model marks ambiguous are cleared to `None` by Python. Ambiguity
detection itself depends on the model; these checks do not prove factual accuracy.

`schemas.py` defines the schema and deterministic checks. Dates must use
YYYY-MM-DD. Numeric and boolean dates are rejected with `ValidationError`.
The model is instructed to return `null` and mark the field ambiguous for invalid
or impossible source dates, without inventing replacements. Invalid date strings
returned despite these instructions become `None` with an issue; reversed date
ordering is flagged while preserving both dates. Unknown types need review.
Python recomputes missing fields rather than trusting the model's summary:

| Document type | Fields expected by this project |
| --- | --- |
| Student enrollment | Holder name, student ID, institution, issue date |
| Health insurance | Holder name, institution, policy number, issue date, expiry date |

Enrollment expiry dates are optional. These are configurable project expectations
in `REQUIRED_FIELDS`, not legal requirements. No legal-validity or expiration
decision is made. Check `missing_fields`, `ambiguous_fields`, and
`validation_issues` before using extracted facts.

Filesystem/extraction errors and Pydantic `ValidationError` propagate to Python
callers. Connection/API failures and malformed JSON raise
`DocumentProcessingError`. The CLI reports errors with exit status 1.

## Tests and future LangGraph integration

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic files, real text extraction, and mocked Ollama replies. They
verify the request contract and Python validation without requiring Ollama. They
do not measure model accuracy; run the processor command above for a live check.

The graph now uses `src/workers/document_analyzer.py` to call the existing
processor for pending documents. It maps enrollment and insurance types to the
graph's names and stores the full JSON-compatible extraction at
`state["extracted_facts"][document.doc_id]`, including all quality flags.
Missing, ambiguous, unknown, or inconsistent results are marked `rejected` with
structured graph issues. Extraction/API failures also reject the document and
clear any previous extraction for that ID. Gate 1 preserves these errors and
routes to the error response before approval. `analyzed` means extraction passed
quality checks, not official document approval. R014 is now evaluated by the
rules worker using the existing knowledge record; all other rules are outside
the demo scope. Planner and approval remain explicitly mocked.

Model settings can be passed without changing the processor API:

```python
from src.core.graph import build_graph
from src.core.state import UserEvent, new_run

state = new_run(UserEvent(kind="upload", file_path="data/synthetic_samples/student_enrollment.txt"),
                user_id="demo-student")
result = build_graph().invoke(state, config={"configurable": {
    "document_model": "qwen3:8b",
    "document_host": "http://127.0.0.1:11434",
    "document_timeout": 120.0,
}})
print(result["final_response"])
```

The offline integration tests mock only Ollama's chat method; file extraction,
Pydantic validation, state merging, and graph execution are real.

The processor can also be imported directly:

```python
from processor import process_document

def document_node(state: dict) -> dict:
    result = process_document(state["document_path"], model="qwen3:8b")
    return {"document_info": result.model_dump(mode="json")}
```

The processor is synchronous; an asynchronous graph can run it in a worker thread.
