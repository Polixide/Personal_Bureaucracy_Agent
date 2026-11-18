# `app.py` — the Streamlit user interface

## What it does
It is the user's window on the Digital Twin. The user states a goal for the agent, uploads documents, and sees
what the Twin knows and where they are rule by rule. They can try a step in the sandbox, approve or deny the
proposed actions, and report what happened in the real world.

## Where it sits in the flow
`streamlit run app.py`. Each rerun opens the Twin at `config.DATABASE_PATH`. The panels call the same
functions the agent uses:
- `run_agent`,
- the `extract_document` tool,
- `twin.service` (confirm, correct, resolve, outcome),
- `rules.evaluate_rules`,
- `sandbox.mock_world.run_simulation`,
- `sandbox.approval.approve_action` / `deny_action`.

The app keeps no state of its own except the last agent answer for display.

## Inputs and outputs
| Panel | Real → twin | Twin → real |
|---|---|---|
| 1. Goal | the goal text | the agent's checked answer |
| 2. Documents | uploads (saved under content-hashed names in `data/uploads/`) | — |
| 3. What the Twin knows | confirmations, conflict choices, typed values | — |
| 4. Where you are | — | rule status per section |
| 5. Sandbox | — | mock-portal result (nothing submitted) |
| 6. Proposed actions | approve / deny | approved files in `data/outbox/<user>/` (download) |
| 7. What happened | reported outcome (confirmed facts) | — |

## Why it is built this way
- **The UI holds no state.** Everything is in the Twin, so restarting the app loses nothing, and the app and
  the agent see the same facts.
- **Documents, confirmations and outcomes bypass the LLM** (migration plan, decision D8). They are explicit user
  actions and go straight to the Twin as deterministic operations.
- **Fallback when Ollama is down.** The agent answers from the deterministic workflow, and values can be typed
  in panel 3.

## How to test it
```
python -m pytest -q tests/test_app.py      # helpers + a headless run with streamlit.testing.AppTest
streamlit run app.py
```

## Known limits
- **Single-user prototype.** It has no login, and one id means one Twin.
- **Slow on a laptop.** Long agent runs block the page while they work.
