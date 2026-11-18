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
- **RAG** (`src/rag/`): it cites the official sources and never decides a rule. **Episodic memory**
  (`src/memory/episodes.py`): lessons from repairs that Gate 1 accepted.

Each module has a one-page explanation in `docs/modules/`. The migration from the milestone-1 prototype is
recorded in `docs/migration_plan.md`.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
ollama pull qwen3:4b        # default in config.py (4 GB GPU); qwen3:8b or gemma4:e4b on bigger machines
ollama pull bge-m3          # embeddings; qwen3-embedding:0.6b if slow
```
All settings live in `config.py`. The model can also be chosen with the environment variable `OLLAMA_MODEL`.

## Use
```powershell
streamlit run app.py                     # the app
python -m pytest -q                      # tests (the LLM is always mocked; real Ollama calls fail the test)
python -m eval.run_eval                  # offline evaluation -> docs/eval_results.md
python -m eval.run_eval --live           # with the local model
python -m src.pack validate ikamet_first # check a Procedure Pack
```

## Procedures
- `ikamet_first`: first short-term residence permit for Sabancı incoming exchange students. It has 29 verified
  rules, converted from the milestone-1 `knowledge/` data.
- `ikamet_extension`: short-term residence permit extension. Its rules are **unverified** until a person checks
  them against the official text.

## Data and privacy
Synthetic data only. `data/uploads/`, `data/outbox/` and the Twin database are git-ignored and never committed.
`sources_text/` files are compiled summaries, not copies of the official pages; replace them with the saved
page text.
