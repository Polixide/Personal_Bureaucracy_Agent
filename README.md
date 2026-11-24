# Personal Bureaucracy Agent

An AI agent that helps people complete Turkish bureaucratic procedures (CS58012 Agentic AI Systems, Sabancı
University). You state a goal in plain language, for example *"I want to apply for İkamet"*. The agent then:
- identifies the procedure,
- tells you what to upload,
- reads your documents and works out where you are, what is missing and what looks wrong,
- prepares artifacts (calendar file, e-mail draft, filled PDF, checklist), trying official steps in a
  sandbox first.

Nothing leaves the Digital Twin before you approve it, and the agent never acts on a real service.

**Principle: the LLM proposes, the code verifies.** Rules, dates and state are checked by deterministic
Python. Unknown facts stay unknown.

## Architecture
- **Orchestrator-Workers with ReAct** (`src/agent/`). The orchestrator (LangGraph cycle
  `decide → act → decide`) decomposes the goal into the procedure's sections, delegates each one with
  `delegate_section` to a worker (a small bounded ReAct loop with read-only tools), and synthesizes one answer.
- **Gate 1** (`src/agent/gate1.py`) checks the answer deterministically:
  - every claim equals the latest `evaluate_rules`, and failing rules are addressed;
  - nothing is said to be submitted, paid or approved;
  - every date comes from facts or rules;
  - every action cites a rule, finding or proposal;
  - unverified rules are labelled.

  Failures go back to the model at most 3 times, then the answer is flagged for human review. Two invalid
  decisions in a row switch to a deterministic workflow, so the app never dies.
- **Digital Twin as an environment** (`src/twin/`). SQLite stores facts with provenance and validity dates,
  events, proposals, simulations and episodes:
  - **real → twin:** documents, statements, confirmations, reported outcomes, rule updates;
  - **twin → real:** approved artifacts only (`src/sandbox/approval.py`), after the mock world
    (`src/sandbox/mock_world.py`) accepted the stage.
- **Procedure Packs** (`procedures/<id>/`). A procedure is data: `procedure.json`, `rules.json` (JSON condition
  trees, three-valued logic), `sources.json`, `sources_text/` and `eval/scenarios.json`. Adding a procedure
  changes no engine code.
- **Tools** (`src/tools/`), with Pydantic-validated arguments:
  - read: `read_twin`, `search_procedures`, `evaluate_rules`, `extract_document`, `record_fact`, `search_sources`;
  - ask: `ask_user`;
  - sandbox: `simulate_submission`;
  - artifact (proposal only): `draft_email`, `create_ics`, `fill_pdf`, `build_checklist`.
- **The assistant in the chat** (`src/agent/chat.py`, `src/agent/briefing.py`): the same ReAct graph without
  workers, started from the plan that code computes from the Twin (where you are, what to do next, step by step).
  It answers questions, records what the user says (unconfirmed until confirmed), prepares proposals, and every
  reply passes Gate 1; without the model, the reply is the plan written by code.
- **Documents** (`src/document_processing/`, `src/tools/extract_document.py`): one upload may hold several
  documents (split where a title or a machine-readable zone opens one); every page is read; MRZ values are checked
  by their digits; photos and scans go through OCR, and pictures nothing else can name (a border stamp) through a
  small vision model whose values code compares with the OCR text. What code cannot check waits for the user.
- **Knowledge base and RAG** (`src/rag/`): the official pages and documents of each pack are downloaded as
  citable text (`python -m src.rag fetch`), searched by the agent (`search_sources`), checked for changes
  (`check`, which names the rules to review) and used to propose rule drafts with verified quotes
  (`draft-rules`). It cites; it never decides a rule. **Web scraping** (`python -m src.rag crawl`) collects
  official pages for procedures without a pack (robots.txt obeyed); the chat answers from them with citations and
  says that nothing was checked. **Memory**: the Twin keeps every run with its rule
  verdicts, and `src/memory/episodes.py` keeps lessons from repairs that Gate 1 accepted.

Each module has a one-page explanation in `docs/modules/`. The migration from the milestone-1 prototype is
recorded in `docs/migration_plan.md`.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
ollama pull qwen3:4b              # agent team and documents (4 GB GPU); qwen3:8b or gemma4:e4b on bigger machines
ollama pull qwen2.5:3b            # the chat assistant: 16 s per reply here against 42 s for qwen3:4b (docs/eval_chat.md)
ollama pull qwen3-vl:2b-instruct  # vision model for photos and scans (docs/eval_documents.md)
ollama pull bge-m3                # embeddings, chosen by measurement (docs/eval_rag.md)
```
All settings live in `config.py`. Models can also be chosen with `OLLAMA_MODEL`, `OLLAMA_CHAT_MODEL` and
`OLLAMA_VISION_MODEL`; `VISION_ENABLED=0` switches the vision model off.

## Use
```powershell
streamlit run app.py                     # the app
python -m pytest -q                      # tests (the LLM is always mocked; real Ollama calls fail the test)
python -m eval.run_eval                  # offline evaluation -> docs/eval_results.md
python -m eval.run_eval --live           # with the local model
python -m src.pack validate ikamet_first # check a Procedure Pack
python -m src.rag check tc_kimlik_first  # did an official page change? which rules to review?
python -m src.rag fetch tc_kimlik_first  # save the current official texts
python -m src.rag draft-rules tc_kimlik_first   # rule drafts with checked quotes -> rule_drafts.json
python -m eval.rag_eval --models qwen3-embedding:0.6b bge-m3   # compare embedding models
python -m src.rag crawl                  # web scraping for procedures without a pack (knowledge/seeds.json)
python -m src.rag search-web "How do I get a tax number?"
python -m eval.document_eval             # how well uploads are read (live) -> docs/eval_documents.md with --out
python -m eval.chat_eval                 # the chat assistant on 8 typical messages (live) -> docs/eval_chat.md
```

### The app
The layout comes from Neda's interface; the agent, the Twin and the rules behind it are ours.
- **Sidebar.** Your profile name (the same name continues where you left off), the procedure, today's date, and
  *Reset* (clear the conversation, or delete the profile).
- **Status bar.** Documents ready, checks passed, and the agent's last status: *where am I?* at a glance.
- **Waiting for your decision.** Actions to approve or deny (nothing is sent: approving writes a file you
  download), before anything else.
- **Assistant.** A conversation that opens with your plan, step by step. Ask anything, tell it something new
  (it asks you to confirm), ask it to prepare a checklist or an e-mail, or attach documents to a message (a PDF
  with several documents is fine). Beside it: your documents, *Run a full check* (the orchestrator and its workers,
  a few minutes), *Prepare my checklist*, *Try this step* in the sandbox, and *Something happened?*.
- **Documents** (pages and how each one was recognised, *Read it again as …*), **Your details** (open questions,
  values to confirm with their quote), **Checks** (every requirement, its status and the official sources) and
  **Technical** (agent steps, Gate 1, run log, lessons, knowledge base, web corpus).

One agent run or document reading per profile runs at a time; other tabs wait and refresh by themselves.
To try it without personal documents, upload the files in `data/synthetic_samples/` (for example
`health_insurance.txt`) or the public specimens in `data/reference_samples/` (see `SOURCES.md` there). To keep a
demo apart from your own Twin, set `TWIN_DATABASE` to another file:
`$env:TWIN_DATABASE = "data/demo_twin.sqlite"; streamlit run app.py`.

## Procedures
- `ikamet_first`: first short-term residence permit for Sabancı incoming exchange students. It has 29 verified
  rules, converted from the milestone-1 `knowledge/` data (that folder was removed once the pack replaced it).
- `ikamet_extension`: short-term residence permit extension. Its rules are **unverified** until a person checks
  them against the official text.
- `tc_kimlik_first` and `tc_kimlik_renewal`: the Turkish identity card (T.C. Kimlik Kartı) for citizens, first
  card (birth, new citizenship, old nüfus cüzdanı) and renewal (expired, lost, damaged). Their rules quote the
  NVİ pages they come from and are **unverified** until a person checks them (review lists in their README).

## Data and privacy
Synthetic data only. `data/uploads/`, `data/outbox/` and the Twin database are git-ignored and never committed.
`sources_text/` holds the text of public official pages (downloaded with `python -m src.rag fetch`, a generic
client name and no personal data). `knowledge/corpus/` holds public pages collected by `python -m src.rag crawl`.
`data/reference_samples/` holds public specimen documents (licences in its `SOURCES.md`). The document evaluation
writes its synthetic files to a temporary folder.
