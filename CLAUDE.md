# Personal Bureaucracy Agent: rules for Claude Code

## What this project is
An AI agent that helps people complete Turkish bureaucratic procedures. The user states a goal in plain
language ("I want to apply for İkamet"); the agent tells them what to upload, understands from the uploaded
documents where they are, what is missing and what is wrong, and uses tools (calendar file, email draft,
PDF forms) to move them forward. First procedure: first short-term residence permit (İkamet) for incoming
exchange students at Sabancı University. Course: CS58012 Agentic AI Systems.

Core principle: **the LLM proposes, the code verifies.** The LLM understands the goal, chooses the next step
and tool, and writes the wording. Deterministic Python checks rules, dates and state, and has the last word.
Unknown facts stay unknown (None), never guessed.

## Architecture in one paragraph
A generic engine in `src/` plus one folder per procedure in `procedures/<id>/` (a "Procedure Pack"); adding a
procedure never changes engine code. Pattern: Orchestrator-Workers with ReAct. The orchestrator (LLM) decomposes
the goal into sub-goals, delegates each section to a worker (a small bounded ReAct loop with fewer tools), and
synthesizes one answer. Every ReAct loop is: the model picks a tool and arguments (schema-validated), the code runs
it, the model sees the observation, repeat (max `MAX_STEPS`). The final answer goes through Gate 1 (deterministic
verifier); if it fails, the structured errors go back to the model (max 3 repairs). The Digital Twin is an
environment with two-way flow: real -> twin (documents, user-reported outcomes) and twin -> real (artifacts the
user carries out), with a mock-world sandbox in between.

## Code style (the whole team must understand every line)
- Python 3.11, type hints everywhere, Pydantic v2 for data crossing module boundaries.
- Small functions (under 40 lines), one job each, plain names, a docstring on each public function (what and why).
- No metaprogramming, no async, no clever one-liners, no eval/exec, no deep inheritance.
- Allowed dependencies: langgraph, pydantic, ollama, numpy, pymupdf, pytest, streamlit (plus what the existing
  repo already uses, such as rapidocr, if kept). Ask before adding anything else. Prefer the standard library.
- All LLM and embedding calls go through `src/llm.py`: schema-constrained output, Pydantic validation, one retry,
  then a deterministic fallback. The app must never crash because the model produced bad output.
- All settings live in `config.py`.

## Hard rules
- An LLM never decides whether a rule is satisfied and never computes a date. It calls `evaluate_rules`.
- Side-effect tools (draft_email, create_ics, fill_pdf, build_checklist) only create PROPOSED actions. They run
  after explicit user approval. Nothing is sent, booked, paid or submitted on a real service.
- Every rule lists `source_ids` that exist in `sources.json`. Unverifiable rules are `"status": "unverified"`, and
  any answer that uses them must say so.
- Tool arguments are validated with Pydantic before the tool runs. Invalid calls are returned to the model as an
  error observation, not executed.
- Synthetic data only. Never commit personal documents.
- Never change expected values in tests or evals to make them pass. Fix the code or explain why.

## Working on the existing repository
This repository already contains a working prototype. Prefer adapting it over rewriting it. Before deleting or
rewriting a module, record the decision (keep / adapt / replace / delete, and why) in `docs/migration_plan.md`.
Use the existing synthetic cases as a regression oracle: the new generic rule engine must reproduce the old
verdicts before the old engine is removed.

## Workflow
- Work phase by phase. Run `python -m pytest -q` before saying a phase is done. One commit per phase.
- After each module, use the `document-module` skill.
- Use a new branch for this work; never push to main.

## Commands
- `python -m pytest -q`
- `python -m eval.run_eval`
- `python -m src.pack validate <procedure_id>`
- `streamlit run app.py` (or the existing Streamlit entry point)
