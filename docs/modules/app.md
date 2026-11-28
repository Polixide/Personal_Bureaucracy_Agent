# `app.py` — the Streamlit user interface

## What it does
One thing at a time. On top, the **journey**: the official steps of the procedure (online application →
fingerprints → scans to IRO → submission) with "you are here", and one progress bar (documents ready, requirements
met). Below, three views:

| View | What you see | What you do |
|---|---|---|
| **Assistant** | the chat in a fixed-size window that scrolls; beside it the **Now** box (the ONE next step of the plan), "then…", *Ready for you*, *The agent can prepare*, *Full check by the agent team* | answer in the Now box (Yes / No, a choice, a date, upload this document, approve this draft, check this value), *Skip for now*, chat, attach documents with **+**, open approved items in Gmail / Google Calendar |
| **Documents** | each document of the procedure: status, what was read from it, values to check | upload (the type is found automatically), *Confirm*, *Wrong document? / Delete* |
| **Details** | for examiners: every requirement with its source, the sandbox, how the agent works, the last answer step by step (ReAct + reflection), history, the Twin, the knowledge base | try an official step, tell the Twin what happened outside, test a question against the sources |

Before a procedure is chosen, the page is the conversation alone.

## Where it sits in the flow
`streamlit run app.py`. Each rerun opens the Twin at `config.DATABASE_PATH` (env `TWIN_DATABASE` overrides it).
The Now box shows `briefing.build_briefing(...).steps[0]` (skipped steps move to the end, per session). Answers given
with a click or typed are confirmed facts (`service.record_app_value`); a corrected value closes the unsure one
(`service.correct_unconfirmed`). The sidebar's *AI model* (Local / OpenAI, Claude when its key is set) is passed to
every model call: the chat calls `run_chat` (`agent_backend("chat", provider())`), the full check `run_agent`
(`agent_backend("team", provider())`); tools read documents with `document_backend(provider(), documents_cloud())`,
local unless the sidebar switch lets the cloud model read them. A cloud choice without its key shows a warning and
keeps the local model. Uploads and the profile name pass Gate 0 (`src/agent/gate0.py`) first; a chat message refused
by Gate 0 is shown as a warning. Details → How it works ends with the model usage (requests, tokens, cost). *The agent can prepare* lists `actions.suggestions`;
each one is a proposal shown in the Now box with a preview, and after approval *Ready for you* opens it in Gmail or
Google Calendar already filled in (`src/tools/open_links.py`, which also offers the default e-mail app).

## Why it is built this way
- **One step at a time.** A list of 27 requirements and 20 questions is not an agentic experience. The plan is
  computed by code and shown one step at a time, with the control the step needs; everything else is one click
  away in Documents or Details.
- **Gmail and Google Calendar without giving the agent your account.** The links open a compose window or a new
  event already filled in; you press Send or Save. Nothing is sent or booked by the agent (CLAUDE.md hard rule), and
  no Google login, key or library is needed. The IRO address comes from the official Sabancı page.
- **The model is not sure → you decide.** Unsure values come one by one in the Now box (and in Documents) with
  their quote: *Yes, that's right* or *Change it*.
- **One run per profile** (`profile_lock`, `run_locked`, `wait_for_run`, Neda's guard).

## How to test it
```
python -m pytest -q tests/test_app.py tests/test_open_links.py tests/test_gate0.py
streamlit run app.py
```
Page tests (AppTest, fake assistant and orchestrator, temporary Twin) render every view of every procedure, answer a
question in the Now box with one click, approve a checklist and an e-mail (the Gmail link is checked), run the
sandbox, confirm a fact from the chat, run the full check and show the run log.

## Known limits
- **Single-user prototype.** No login: a profile name is not a password.
- **Speed with local models on a 4 GB GPU.** A chat turn takes 15–60 s (`qwen2.5:3b`), a document 20–60 s.
- **English replies** (Gate 1's text checks are in English).
