# Personal Bureaucracy Digital Twin - Milestone 1

Local prototype for Sabancı University incoming exchange students: adults staying
at least three months, preparing a **first short-term residence permit / İkamet
application**. Other procedures and applicant categories are outside scope.
Root `knowledge/` contains the authoritative 29-rule dataset.

## Setup (Windows PowerShell)

Use Python 3.11+ and a local Ollama installation with the desired model installed.
From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contains runtime dependencies; `requirements-dev.txt` also
installs pytest. RapidOCR and ONNX Runtime provide local CPU OCR for scanned PDF
pages without a separate executable or PATH configuration. Text PDFs use
PyMuPDF directly; OCR initializes only when needed. Standalone image uploads
are unsupported. OCR and model text follow the same grounding/validation path.

The default model is `qwen3:4b`. To use another installed model, set the environment
before starting the app:

```powershell
$env:OLLAMA_MODEL="qwen3:8b"
```

Restart the app after changing it. `config.py` centralizes the model, Ollama host,
general/Planner/Reflection timeout (120 seconds), document timeout (180 seconds),
context limits, SQLite path and maximum repair attempts (3).

## Run Streamlit

Start local Ollama, activate the environment, then run:

```powershell
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

1. Enter a thread ID to restore or create a Digital Twin.
2. Upload a PDF/TXT file and select **Analyze document**.
3. Review source-grounded observations and explicitly confirm unresolved values.
4. Repeat for the seven-document checklist, then select **Generate / Update Plan**.
5. Review the proposal and explicitly **Approve** or **Deny** it.

The checklist covers the declaration control document, IKAMET form, passport
identity page, latest Türkiye entry-stamp page, student certificate, health insurance
and card/document-fee receipt. Analysis and confirmation recompute requirements
without generating a plan. Raw observations remain separate from confirmed inputs.

Rules deterministically identify problems. The LLM selects/prioritizes/groups
canonical action candidates and writes titles/reasons; Python supplies structural
metadata. Gate 1 validates each plan, and Reflection repairs rejected proposals
with a three-attempt bound and early stopping for unchanged invalid output.
Transport failures use a deterministic fallback through the same safety gates.

All 29 assessments retain PASS, FAIL or missing_information. Later-stage and
out-of-scope UNKNOWN checks remain UNKNOWN. A completed scoped checklist can
produce a deterministic preparation-completion assessment without an LLM call.
**Document preparation, Gate 1 validation and human plan approval never mean
residence-permit approval or execution of official actions.**

## Local state and private files

SQLite checkpoints live at `data/digital_twin.db`. The same thread ID restores
state across restarts; different threads remain isolated. Reopening a pending
approval never approves it automatically. This is a synchronous local prototype
without authentication or coordination for concurrent changes to one thread.

Database/journal files, `data/uploads/`, `data/local_samples/`, `.local/`, caches
and test artifacts are ignored. Keep private documents out of version control;
new manual samples belong in `data/local_samples/`, not synthetic fixtures.
Existing local administrative PDFs are ignored in place so saved source paths
remain usable. Ignoring/removing files from the index does not erase Git history.

## Verification and offline demo

```powershell
python -m pip check
python -m pytest -q
python -m scripts.run_demo --thread-id demo-user --persistence-demo
```

Automated tests prohibit live Ollama and OCR calls. The default synthetic demo
mocks only model transport and exercises the graph, rules, Gate 1, SQLite and
explicit approval. `--persistence-demo` closes/reopens SQLite before plan review.
Use `--show-state` to inspect a saved thread without resuming it. `--live` is an
explicit manual Ollama smoke test; it never downloads a model automatically.

## Canonical structure

- `src/core/`: shared AgentState, LangGraph orchestration and safe diagnostics.
- `src/document_processing/`: PDF/TXT/RapidOCR extraction, schemas and evidence.
- `src/workers/`: document adaptation, verified-input projection and rule evaluation.
- `src/agents/`: canonical planning policy and LLM planning/repair generation.
- `src/guardrails/`: Gate 0, Gate 1 and human approval.
- `src/memory/`: SQLite checkpoints, repair context and approved-plan bookkeeping.
- `streamlit_app.py`: the single presentation/input layer.
- `tests/` and `data/synthetic_samples/`: offline regressions and synthetic fixtures.

`build_graph()` defaults to isolated in-memory state for tests; applications use
`sqlite_checkpointer()` for persistent state. `docs/INTEGRATION_REPORT.md` is a
historical integration snapshot, not the current runtime specification. Temporary
integration implementations have been removed. No MCP or RAG is included.


## Document providers, diagnostics and local evaluation

Document classification/extraction uses a single provider boundary. `LLM_PROVIDER`
defaults to `ollama`; existing `OLLAMA_MODEL` configuration remains unchanged.
This selector applies to documents only, not Planner or Reflection.
`DOCUMENT_FALLBACK_PROVIDER` defaults to `none`. Ollama-only operation requires
neither an OpenAI SDK installation nor credentials.

Optional OpenAI image fallback uses the official SDK and must be explicitly enabled:

```powershell
python -m pip install -r requirements-openai.txt
$env:DOCUMENT_FALLBACK_PROVIDER="openai"
$env:OPENAI_MODEL="your-compatible-vision-model"
# Set OPENAI_API_KEY locally; never put a real key in repository files.
```

Restart Python after configuration changes. `LLM_PROVIDER=openai` can also select
OpenAI as the primary document provider. Use a model supporting image inputs and
structured JSON output. Enabling cloud fallback permits selected page images/text
to leave the machine, only for inadequate PDFs: poor OCR, unresolved expected fields
or validation issues. It is disabled by default. Fallback makes one bounded attempt;
`DOCUMENT_FALLBACK_MAX_PAGES` defaults to 2 (maximum 4).

PyMuPDF renders at `DOCUMENT_OCR_DPI` (default 200, supported 200-300) with
`DOCUMENT_MAX_IMAGE_SIDE` (default 3000 pixels) limiting oversized pages. Existing
RapidOCR orientation handling remains local. Vision can recover page-labelled text
for poor scans; recovered text is untrusted, not independent proof of authenticity.
The same source candidates, normalization, Pydantic checks and provenance audit
apply. Readable native text is not replaced by vision-generated evidence. A fallback
that does not improve a partial local result leaves that local result in place.

`Document.info.extraction_diagnostics` persists method, provider/model, native text
counts, OCR use/confidence/duration, selected fallback pages, quality signals,
missing expected fields and invalid-field counts. These are technical metadata,
not observations or verified context inputs. Diagnostics log counts/field names to
stderr, never full document text or keys. Existing JSON stdout remains usable.

Store private evaluation manifests/documents in ignored `data/local_samples/` or
`.local/`. For example, a local `cases.json` contains:

```json
[
  {
    "document_path": "insurance.pdf",
    "expected_document_type": "health_insurance_certificate",
    "expected_critical_fields": {"policy_number": "DEMO-123"}
  }
]
```

Paths are relative to the manifest. Run an explicit local/model evaluation with:

```powershell
python -m src.document_processing.processor --evaluate data/local_samples/cases.json
```

Reports contain classification correctness, per-field exact-match booleans,
missing/invalid counts, method, fallback usage and latency; source values and paths
are omitted. This CLI uses the configured providers; automated tests mock all model
and OCR calls. No live OCR/OpenAI accuracy is established by those mocks.


## V2 pre-e-Ikamet collection readiness

Only explicitly initialized V2 threads (`procedure=ProcedureState()`) use
`src/core/procedure.py`. Legacy threads and the seven-document Milestone-1
checklist remain unchanged. Factual/review events refresh only collection
status/blocker references; readiness never advances a stage automatically.

Collection requires accepted passport identity fields (name, number, nationality,
expiry and page presence), latest Turkish entry-page evidence and an explicitly
confirmed latest entry date, an accepted student certificate (name, student ID,
institution, issue date), address-proof category, insurance and biometric-photo
readiness. Source-grounded observations or explicit confirmations count; bare
context flags, unreviewed extraction, action completion and plan approval do not.
Relevant unresolved verification conflicts block readiness. R008 supports the
passport/entry portion only; visa applicability and the full legal R008 remain
separate assessments.

The local `V2_PRE_EIKAMET_INSURANCE_READINESS` policy accepts a confirmed exemption,
or policy end covering either the accepted requested permit end or academic
program end (inclusive). Start date and `insurance_valid_for_ikamet` are not gates
in this specific end-coverage policy. Nonexempt cases also need accepted policy
holder/insurer/number/end evidence and an accepted R016 insurance category.
General/legal R015 and existing Milestone checks are unchanged.

V2 collection adds backend confirmation support through the existing
`confirm_information` / `__declarations__` event for `address_proof_type`
(`dorm_certificate`, `rental_agreement`, `notarized_commitment`, evaluated by R017)
and `biometric_photo` (explicit boolean). These are user-confirmed statements with
user/trace provenance, not document-authenticity checks or ICAO/photo analysis.
No Streamlit controls or photo-analysis system are added in this phase.

An explicit caller can apply the first transition with the existing graph API:

```python
from datetime import datetime, timezone
from src.core.procedure import advance_procedure_if_ready

state = graph.get_state(config).values
update = advance_procedure_if_ready(state, recorded_at=datetime.now(timezone.utc))
if update:
    graph.update_state(config, update)
```

The function recomputes readiness and returns only a ProcedureState update.
Success completes DOCUMENT_COLLECTION, enters EIKAMET_PREPARATION/NOT_STARTED,
removes only collection blocker references and appends a traceable history entry.
The supplied time records the transition, not an official event. No documents,
verified inputs, source evidence or requirements are changed. No later transition,
generated form, payment, fingerprint/GI, declaration, IRO confirmation or submission
is required or implemented here. This readiness decision is not legal approval.
