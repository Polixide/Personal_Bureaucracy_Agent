# Personal Bureaucracy Agent

A local procedural Digital Twin for **Sabancı incoming exchange students preparing
an adult first short-term residence-permit (İkamet) application**.

The supported five-step workflow is:

1. Documents
2. e-Ikamet preparation
3. Payment & Fingerprint
4. IRO Review
5. Submission / Tracking

The landing catalogue also shows renewal, Kimlik and other procedures as
**not implemented**. Choosing an entry position never establishes completion.

## Features

- PDF/TXT extraction with PyMuPDF and lazy RapidOCR/ONNX CPU fallback for scanned pages.
- Typed document schemas, source snippets/page provenance, deterministic grounding,
  ambiguity handling and separate explicit user confirmations.
- One canonical AgentState with SQLite persistence and isolated profile/thread IDs.
- A case-aware AI Copilot with typed read tools, bounded conversation context and
  attributed official-source retrieval.
- A bounded local LLM Planner selecting host-generated action candidates, deterministic
  Gate 0/Gate 1 checks, maximum three Reflection repairs and an unchanged-plan breaker.
- Human approval for proposed plans and consequential tool/artifact actions.
- An educational e-Ikamet sandbox with persistent drafts, validation and a clearly
  watermarked simulated PDF; an uploaded real-application review path remains separate.
- Approved email drafts, Gmail/mailto links, EML/copy exports, calendar links and ICS files.
- A small read-only stdio MCP demonstrator and automated outcome/restart evaluations.

## Installation

Use Python 3.11+ and install Ollama separately. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:4b
```

RapidOCR uses bundled ONNX weights and CPU inference. No Tesseract executable or
PATH edit is required. OCR initializes only when a PDF page lacks usable text.

The supported runtime uses Ollama. An isolated lazy OpenAI provider extension
remains dormant and unsupported; its SDK is not part of the installation.

## Local model configuration

`config.py` is the central boundary. The default model is `qwen3:4b`:

```powershell
$env:OLLAMA_MODEL="qwen3:4b"
$env:ASSISTANT_MODEL="qwen3:4b"
$env:PLANNER_MODEL="qwen3:4b"
```

Assistant and Planner overrides are optional and otherwise inherit `OLLAMA_MODEL`.
Document processing uses `OLLAMA_MODEL`. Select another already-installed model
through the environment and restart Python. General Ollama calls default to a
120-second timeout; document calls default to 180 seconds.

## Run

```powershell
python -m streamlit run streamlit_app.py
```

Enter a profile ID to create or resume its saved case. The app and CLI use
`data/digital_twin.db`; the same ID restores state and pending approval after restart.

A manual CLI persistence demo is also available:

```powershell
python -m scripts.run_demo --thread-id demo-user --persistence-demo
```

## Architecture and safety

The controller is a **state-aware Router / Supervisor** over the existing LangGraph
workers, Planner, deterministic gates and bounded Reflection. Procedure sequencing
is deterministic. Rules determine requirement outcomes; the model prioritizes and
explains current action candidates. UNKNOWN information is kept distinct from FAIL.

`src/core/state.py` is the only Digital Twin. Checkpoints are factual authority;
the typed memory record and accepted repair lessons share the same SQLite file.
Conversation, retrieved prose, OCR and simulated artifacts cannot silently become
verified legal facts. Exact dates, comparisons, permissions and writes stay in code.
Per-profile locking coordinates one Streamlit server process.

Document analysis and confirmation reevaluate requirements without generating a
plan automatically. Obvious Copilot state questions bypass the Planner and model.
Current package readiness is shared by progress, blockers, attachments, plans and
IRO transitions. Receipt evidence records payment readiness; a real reviewed
application and actual fingerprint/GI progress are required before later steps.

Plan validation/approval means approval of the proposed plan, **not approval of a
residence permit**. Email/calendar links open drafts for the user to review and
confirm; attachments are not automatically added by Gmail/mailto links. The app
does not operate the government portal, pay fees or submit an official application.

## Knowledge and project materials

- [Active V2 rules](docs/active_rules_v2.md): 24 active rules; stable IDs and policy rationale.
- `knowledge/`: authoritative rule definitions, dependencies, versions and source metadata.
- `data/official_guidance.json`: attributed, bounded local RAG guidance. Retrieval never updates rules.
- [External reference documents](data/reference_samples/SOURCES.md): source links and attribution;
  raw external PDFs are not bundled.
- [Initial course proposal](docs/Personal_Bureaucracy_Agent_pres1.pdf): historical design material;
  the implemented controller is the state-aware Router described above.

The source packages are `src/core`, `src/workers`, `src/document_processing`,
`src/agents`, `src/guardrails`, `src/memory`, `src/tools`, `src/rag`, `src/evaluation`,
`src/integrations` and `src/ui`. Streamlit remains an input/presentation layer.

## Tests and evaluation

Install the development requirements to run the offline suite:

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
python scripts/run_evaluation.py --repeats 3 --memory
```

Automated tests mock model/OCR/external transports and require no live model or
OCR download. Coverage includes document grounding, rules, planning/gates/repair,
approval, tools, sandbox, RAG/MCP, UI workflows and SQLite restart isolation.

Explicit live local benchmarks require local PDFs for document evaluation. Put
private PDFs in `data/local_samples/` or supply a private manifest.

```powershell
python scripts/benchmark_documents.py --mode local --output .local/document-benchmark.json
python scripts/run_evaluation.py --copilot-benchmark --output .local/copilot-benchmark.json
```

The document benchmark discovers PDFs in ignored `data/local_samples/` and in
`data/synthetic_samples/`. A private ignored manifest can supply gold labels via
`--manifest`; no labelled accuracy is claimed without gold data. The Copilot
benchmark uses synthetic cases and already-installed configured models only;
it downloads no model and executes no approved external action.

The read-only MCP demonstrator can be started separately:

```powershell
python -m src.integrations.mcp.server
```

## Privacy and limitations

**Never commit real student documents or credentials.** Keep private PDFs in
`data/local_samples/`; runtime uploads are stored in ignored `data/uploads/`.
SQLite state, uploads, caches, exports and local reports are plaintext and local-only.
The public TXT fixtures are synthetic. Audit tracked files as well as ignore rules
before publishing; ignoring a path does not remove existing Git history.

Only first Ikamet preparation is implemented. There is no authentication,
encryption, deployment or real government-portal automation. Multi-server writes
to one profile are unsupported. MCP is demonstrative; RAG uses a bounded local
lexical corpus, not a production vector database or live legal feed. Sources,
fees and calendars need current human review. Local LLM latency and wording can
vary, and OCR/unseen layouts or ambiguous evidence may require user confirmation.
