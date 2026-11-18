# `src/tools/` — the tools the agent can call

## What it does
Each tool is a small function with a Pydantic input model and output model, described by a `ToolSpec`. The
registry (`registry.py`) is the only place a tool runs. It validates the arguments first and turns every
failure into an *error observation* instead of an exception.

## Where it sits in the flow
In each ReAct step the model returns a `Decision{thought, action, args}`. The agent calls
`run_tool(registry, action, args, ctx, role)` and appends the returned `Observation` to the conversation.

| Tool | Kind | Who | What |
|---|---|---|---|
| `read_twin` | read | both | documents, confirmed facts, unconfirmed statements, conflicts, proposals, simulations |
| `search_procedures` | read | orchestrator | picks the matching pack; says "ambiguous" or "nothing available" |
| `evaluate_rules` | read | both | deterministic verdicts (optionally one section) |
| `extract_document` | read | both | runs extraction on an uploaded document and stores facts |
| `record_fact` | read | orchestrator | stores a chat statement as **unconfirmed** |
| `ask_user` | ask | orchestrator | asks one question and ends the turn |
| `delegate_section` | orchestrator | orchestrator | runs a worker on one section (phase 5) |
| `simulate_submission` | sandbox | orchestrator | tries a stage in the mock world (phase 6) |
| `draft_email`, `create_ics`, `fill_pdf`, `build_checklist` | artifact | orchestrator | create PROPOSED actions only (phase 6) |
| `search_sources` | read | both | cites official text (phase 7) |

## Inputs and outputs
```python
run_tool(reg, "record_fact", {"field": "age"}, ctx, "orchestrator")
# Observation(tool="record_fact", ok=False, error="Invalid arguments: value: Field required")
run_tool(reg, "evaluate_rules", {"section_id": "passport_photo"}, ctx, "worker").output
# {"procedure_id": "ikamet_first", "results": [{"rule_id": "R006", "status": "fail", ...}],
#  "counts": {"pass": 0, "fail": 1, "unknown": 2}}
```
`ToolContext` carries the store, the loaded packs, the user, `today`, the backend and the active procedure. It
also collects what the run created: proposals, findings and the question.

## Why it is built this way
- **Invalid calls never run.** The model learns from the error text. After two consecutive invalid decisions
  the agent switches to the deterministic fallback.
- **Roles restrict tools.** Workers can only read and evaluate. Only the orchestrator can record statements,
  ask the user, simulate and propose artifacts.
- **No real-world side effects.** No tool sends, books, pays or submits anything; artifact tools only propose.

## How to test it
```
python -m pytest -q tests/test_tools.py
```

## Known limits
- **`search_procedures` matches words.** It matches alias phrases literally; synonyms need to be added as
  aliases in the pack.
- **Short tool descriptions.** Tool descriptions are kept short so they fit in small models' context; long
  tool lists would need trimming per role.
