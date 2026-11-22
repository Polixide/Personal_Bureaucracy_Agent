# `app.py` — the Streamlit user interface

## What it does
It is the user's window on the Digital Twin. The layout is Neda's (status bar, the decision first, tabs); behind it
run our agents, the Twin, the rule engine and the sandbox. The main tab is now a **conversation with the assistant**:
it opens with the plan (where you are, what to do next, step by step), answers questions in plain words, records
what you say, prepares artifacts for approval and reads documents attached to a message. The other tabs keep the
detailed, point-by-point views.

## Where it sits in the flow
`streamlit run app.py`. Each rerun opens the Twin at `config.DATABASE_PATH` (env `TWIN_DATABASE` overrides it).
The page calls the same functions the agents use: `run_chat` (the assistant), `run_agent` (the full check by the
orchestrator and its workers), the `extract_document` and `build_checklist` tools (through `run_tool`),
`twin.service`, `rules.evaluate_rules`, the sandbox and `approve_action` / `deny_action`. The conversation is kept
in the Twin (`chat_messages`), so it continues across sessions and devices.

## Inputs and outputs
| Part | Real → twin | Twin → real |
|---|---|---|
| Sidebar | profile name (one Twin each), procedure, today's date, clear the conversation, delete the profile (typed) | "saved progress restored", the models in use |
| Status bar | – | documents ready, checks passed (user sections), the agent's last status |
| Decision card | Approve / Deny / Check in sandbox | actions waiting for approval |
| **Assistant** | messages, one-click suggestions, documents attached in the chat (PDF, TXT, JPG, PNG; several documents in one PDF), *Confirm* under a fact the assistant recorded, *Run a full check*, *Prepare my checklist*, *Try this step*, *Something happened?* | the plan, checked replies with their sources, what each upload gave (types, pages, values, what to confirm), the document checklist, downloads |
| Documents | upload, confirm values (with their quote), *Read it again as …* (the same pages, the type you choose), delete | each document with its pages, how it was recognised, notes and warnings |
| Your details | typed answers (confirmed facts), confirmations, conflict decisions | open questions per section, computed values |
| Checks | – | passed / failed / unknown, problems, recommendations, every rule per section, official sources |
| Technical | – | the last answer's steps and Gate 1 issues, run log with verdicts, lessons, facts, results, knowledge base and web corpus size |

Before a procedure is chosen, the page is the conversation alone: the assistant finds the procedure from what you
write, or answers from the official web pages of the corpus and says that nothing was checked.

## Why it is built this way
- **The static points, explained.** The tabs list requirements, questions and documents one by one; the assistant
  aggregates them into one explanation and lets you ask "why?". Its plan is computed by code (`briefing.py`), so it
  is right even when the model is slow or wrong; replies pass Gate 1 (`docs/modules/chat.md`).
- **Neda's layout, our engine.** Status bar and decision card stay on top; the Twin is the single source of truth.
- **One run per profile.** There is one local model, so a second click, tab or device waits instead of queueing
  (`profile_lock`, `run_locked`, `wait_for_run`, Neda's guard with its tests). Chat turns, document readings and the
  full check use the lock; direct Twin edits do not need it.
- **Plain words, generic code.** Labels come from the pack. No procedure is named in the code; unverified rules are
  labelled; explicit actions (approve, confirm, typed answers) never go through a model.

## How to test it
```
python -m pytest -q tests/test_app.py      # helpers, the lock, and real page runs with streamlit.testing.AppTest
streamlit run app.py
```
Page tests use a fake assistant and a fake orchestrator with a temporary Twin. Every tab of every procedure is
rendered; a chat message, a suggestion, a fact confirmed from the chat, the full check, typed answers, checklist
approval, the sandbox and the run log go through the real widgets.

## Known limits
- **Single-user prototype.** No login: a profile name is not a password.
- **Speed on a 4 GB GPU.** A chat turn takes 15–60 s (`qwen2.5:3b`), a document 20–60 s (more for photos read by the
  vision model), a full check by the agent team a few minutes (`qwen3:4b`). Steps are shown while they run.
- **English replies** (Gate 1's text checks are in English).
