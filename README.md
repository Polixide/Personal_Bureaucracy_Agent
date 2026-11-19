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
