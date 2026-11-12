# Personal Bureaucracy Agent ? integrated Milestone 1

Local prototype for Sabanc? incoming exchange students (18+, stay of at least three months), first short-term residence permit / ?kamet application. Root `knowledge/` is the authoritative knowledge dataset. Extensions and other permit/student categories are outside scope.

## Run locally

Python 3.11+ and PyMuPDF are required. From the project root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.run_demo
```

The synthetic demo defaults to mocked model responses. It still runs real TXT extraction, Pydantic validation, document adaptation, all 29 deterministic checks, Gate 1, and explicit approval. Enter `y` to approve the proposed plan or anything else to decline. Approval records a plan; it does not mark actions completed or execute them.

For an explicit live smoke test, start local Ollama with the model configured in `config.py`, then run:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_demo --live
```

This flag uses live Ollama for extraction and planning, including any repair requests. There is no automatic model download. Model, host, timeout and the three-repair maximum are centralized in `config.py`. Normal tests prohibit unmocked Ollama calls.

The model defaults to `qwen3:4b`. To use a different model installed on another machine, set the environment before starting Python/Streamlit (restart the app after changing it):

```powershell
$env:OLLAMA_MODEL="qwen3:8b"
```

For scanned PDFs, installing `requirements.txt` supplies RapidOCR 3.10 and CPU ONNX Runtime; no separate OCR executable or PATH edits are needed. The wheel includes default ONNX models (about 32 MB of weights). The extractor imports and initializes OCR only for raster-image pages with fewer than 30 embedded alphanumeric characters, then caches one engine per Python process. It uses the bundled local model files explicitly and reports a controlled error if they are missing or inference fails. Pages are rendered with PyMuPDF at 200 DPI; original page numbers are retained. OCR output enters the same untrusted classification, source-grounding and Pydantic validation path and never establishes legal facts. Native text PDFs do not initialize OCR. CPU inference is synchronous: allow seconds to tens of seconds per scanned page and budget a few hundred MB of additional RAM as a rough, unbenchmarked allowance; actual usage depends on page content and resolution. There is no hard per-page inference deadline in this backend. No live OCR models are loaded by pytest.

## Local Streamlit interface

Start Ollama with the model configured in `config.py`, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Enter a thread ID to load its SQLite-backed Digital Twin. Upload PDF/TXT documents and click **Process document & generate next steps** to run the existing graph. Review summary counts and main blockers; complete rule details, raw facts and full context remain collapsed. **Approve** and **Deny** resume the saved graph approval interrupt; approval records a proposed plan and never executes official actions. Pending processing/approval must be handled before another upload. Files are stored locally under `data/uploads/`, which is ignored by version control.

The interface is a presentation layer: it does not infer legal readiness, edit the residence context, or maintain a second state model. The demo has no authentication or coordination for simultaneous changes to the same thread. Scanned PDFs use the bundled Python OCR fallback; standalone image uploads remain unsupported. Model operations and per-page OCR run synchronously and can take several minutes, including bounded repairs.

## Structure and behavior

- `src/core/state.py`: the single canonical AgentState, typed residence context, schemas and reducers.
- `src/core/graph.py`: Gate 0 ? Supervisor ? selected worker; Planner ? Gate 1; failed plans return through Reflection ? Supervisor ? Planner, at most three times.
- `src/document_processing/`: PDF/TXT extraction, extraction schema and structured local model transport.
- `src/workers/`: document adapter, deterministic R001?R029 engine and root knowledge loader.
- `src/agents/planner.py`: structured, grounded action planning and repair generation.
- `src/guardrails/`: input checks, deterministic plan validation and human approval.
- `src/memory/`: SQLite checkpoint lifecycle, structured repair feedback and deterministic approved-plan bookkeeping.
- `data/synthetic_samples/`: copied C fixtures; `tests/`: migrated offline regression suite.

Workers return partial state updates. Documents and extracted observations merge by stable document ID; the typed residence context is supplied explicitly by the caller. Document observations retain missing/ambiguous fields and never automatically assert legal insurance validity or core-file readiness. Complete extracted text does not establish authenticity or legal acceptance.

Gate 0 accepts existing PDF/TXT files inside the project. Text PDFs use embedded text; raster-image pages with insufficient text use the local OCR fallback. Standalone image uploads remain unsupported. Gate 1 validates schemas, references, ordering, deadlines, document dates and official-action prerequisites. Phrase checks retained from B are supplementary English heuristics, not semantic proof. Every plan requires explicit approval.

The CLI now uses SQLite at `data/digital_twin.db` (configured in `config.py`). Documents, extracted facts, residence context, requirements, plans, blockers and repair/approval checkpoints survive Python restarts. The same `--thread-id` loads the same Digital Twin; different IDs remain independent. Reopening or inspecting a thread never approves a plan: a saved approval interrupt still requires explicit confirmation.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_demo --thread-id demo-user --persistence-demo
# In a new Python process, inspect the saved state without resuming anything:
.\.venv\Scripts\python.exe -m scripts.run_demo --thread-id demo-user --show-state
```

`--persistence-demo` closes/reopens the graph and SQLite connection before requesting approval, checking that the state is identical. Re-running the CLI with a pending approval displays that proposal for explicit review; it does not replace it with a new run. Other unfinished nodes are left untouched by the demo.

Library callers use the existing graph with an explicit connection lifetime:

```python
from src.memory.checkpointer import sqlite_checkpointer
from src.core.graph import build_graph

with sqlite_checkpointer() as saver:
    graph = build_graph(checkpointer=saver)
    snapshot = graph.get_state({"configurable": {"thread_id": "demo-user"}})
```

Keep the context open for all graph operations. `build_graph()` without a supplied saver remains in-memory for ephemeral callers and tests. SQLite stores LangGraph checkpoints of canonical AgentState; there is no second application state schema. This synchronous local prototype is intended for a single application process, not concurrent workers operating on the same thread. Database and journal files are ignored by `.gitignore`. The Streamlit interface uses this same lifecycle. MCP and RAG are not included.

Rule results preserve PASS, FAIL and missing_information. Conditional checks use three-valued logic. Rule condition strings are never executed. The deterministic dependency policy uses the union of root inline dependencies and `knowledge/dependencies.json`. Fees and submission windows must be provided as verified current inputs; the prototype does not fetch or invent them.

See `docs/INTEGRATION_REPORT.md` for migration decisions, verification and the complete source/data file inventory. `integration_sources/` remains untouched as reference material and is excluded from normal test discovery.
