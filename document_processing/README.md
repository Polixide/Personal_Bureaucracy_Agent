# Document text extraction

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

Only text-based PDFs are supported. Scanned PDFs need OCR, which is outside this
module. Tests generate synthetic PDFs in temporary directories and do not use
the existing sample PDF.

## Structured processing with local Ollama

Install the requirements above into the existing environment. Start Ollama
(`ollama serve` if it is not already running), then prepare and run the model:

```powershell
ollama pull qwen3:4b
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

A future LangGraph node can import this module without adding a dependency here:

```python
from processor import process_document

def document_node(state: dict) -> dict:
    result = process_document(state["document_path"], model="qwen3:4b")
    return {"document_info": result.model_dump(mode="json")}
```

Agree on state keys and error routing with the orchestration teammate. Route
quality flags to review rather than treating extraction as an approval decision.
The processor is synchronous; an asynchronous graph can run it in a worker thread.
