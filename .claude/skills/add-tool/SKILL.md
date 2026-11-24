---
name: add-tool
description: Use when adding or changing a tool that the ReAct agent can call (for example draft_email, create_ics, fill_pdf, simulate_submission). Defines the tool contract, safety kind and required tests.
---

# Add a tool

Every tool lives in `src/tools/<name>.py` and is registered in `src/tools/registry.py` as a `ToolSpec`:

- `name` (snake_case) and `description` (one or two plain sentences; the LLM reads this to decide when to use it)
- `input_model` and `output_model` (Pydantic)
- `fn` (a plain function: input model in, output model out)
- `kind`: one of
  - `read`: only reads state (read_twin, evaluate_rules, search_sources, extract_document, search_procedures)
  - `sandbox`: acts only on the mock world (simulate_submission)
  - `artifact`: prepares something for the real world (draft_email, create_ics, fill_pdf, build_checklist).
    It creates a PROPOSED action stored in the twin; it runs only after user approval.
  - `ask`: asks the user a question (ask_user)

Rules:
1. Never add a tool that contacts a real service (sending email, booking, submitting forms) without asking the user first.
2. Output must be small and factual. Include source ids and dates when relevant, so the model can cite them.
3. Errors are returned as an `error` field with a short message; do not raise exceptions into the agent loop.
4. Write tests: valid input, invalid input (returns error), and one realistic example.
5. If the tool needs a new third-party library, stop and ask.
6. Run `python -m pytest -q`, then the `document-module` skill.
