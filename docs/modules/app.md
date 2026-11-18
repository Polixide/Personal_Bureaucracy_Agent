# `app.py` — the Streamlit user interface

## What it does
It is the user's window on the Digital Twin, organised around the three questions of our presentation:
*Where am I? What is missing? What should I do next?* The user states a goal for the agent at the top; three
tabs below show the Twin and let the user add documents, answer questions, approve actions and report outcomes.

## Where it sits in the flow
`streamlit run app.py`. Each rerun opens the Twin at `config.DATABASE_PATH` (env `TWIN_DATABASE` overrides it).
The page calls the same functions the agent uses: `run_agent`, the `extract_document` and `build_checklist`
tools (through `run_tool`), `twin.service`, `rules.evaluate_rules`, `sandbox.mock_world.run_simulation` and
`sandbox.approval.approve_action` / `deny_action`. It keeps no state of its own except the last agent answer,
the open tab and one-shot messages.

## Inputs and outputs
| Part | Real → twin | Twin → real |
|---|---|---|
| Goal box | the goal; the reply to the agent's question (sent with the goal, the agent has no chat history) | the Gate-1-checked answer and its steps |
| Where am I? | remove a document | requirements met per section (✅ ❌ ⏳), documents, all facts |
| What is missing? | upload (content-hashed name in `data/uploads/`), confirmations, conflict choices, typed answers | problems, open questions per section |
| What next? | approve / deny, sandbox try, *something happened* (outcome event + confirmed fact) | approved files in `data/outbox/<user>/` (download) |

Example: on an empty Twin the İkamet "Eligibility and legal stay" section shows `⏳ 0/4`; its open questions
ask the program start/end dates instead of the derived `planned_stay_months`, and suggest the student
certificate upload for `student_certificate_present`.

## Why it is built this way
- **Three questions, three tabs.** Each tab answers one question; counts and actions sit where they are used
  (a sandbox check is offered next to the action that needs it). Tab names are fixed so Streamlit keeps the open
  tab after a click.
- **Plain words, generic code.** Labels come from the pack (`question`, section, stage and document titles,
  rule descriptions); fields without a question get a short one built from the name. No procedure is named in
  the code, and guardrail sections (`delegate: false`) are not shown as user progress.
- **Explicit actions bypass the LLM** (migration plan, decision D8). Uploads, answers, confirmations and outcomes
  go straight to the Twin; typed answers are confirmed `user_stated` facts. Verdicts always come from
  `evaluate_rules`, and unverified rules are labelled.

## How to test it
```
python -m pytest -q tests/test_app.py      # helpers + real page runs with streamlit.testing.AppTest
streamlit run app.py
```
The page tests use a fake `run_agent` and a temporary Twin; the agent question/answer, typed answers, the
checklist approval and the sandbox are exercised through the real widgets.

## Known limits
- **Single-user prototype.** There is no login; one id means one Twin.
- **Slow on a laptop.** An agent run blocks the page for 2-4 minutes with `qwen3:4b`.
- **Free-text values in *Something happened?*** are coerced by the pack's field type; a bad value is refused
  with the coercion error.
