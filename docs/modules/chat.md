# `src/agent/chat.py` and `src/agent/briefing.py` — the assistant in the chat

## What it does
The app's main tab is a conversation. It always opens with **the plan**: where the user is and what to do next,
computed by code from the Digital Twin. The user can ask anything ("why is my passport a problem?", "how do I get
a tax number?"), tell the assistant something new ("I lost my card last week"), ask it to prepare something
("make me a checklist"), or attach a document. Each reply is a human explanation that gathers what the other
tabs show point by point — and it is checked by code before it is shown.

## The briefing (`briefing.py`, no model)
`build_briefing(pack, documents, proposals, view, results)` returns one ordered plan:

| order | kind | what | example |
|---|---|---|---|
| 1 | decide | actions waiting for approval | Approve or deny: checklist |
| 2 | fix | blocking requirements that fail (with their source) | Fix: the passport must be valid 60 days beyond the permit |
| 3 | confirm | values to check, documents that disagree | Check the latest entry date: 2026-09-15 (from you) |
| 4 | upload | documents that answer open questions (most useful first) | Upload, if you have it: the ID card — it answers: birth date |
| 5 | answer | the remaining questions | Why do you need a new card? |
| 6 | improve | advisory requirements not met | Recommended: book an appointment |
| 7 | try | when nothing blocks: the official step in the sandbox | Try "card application" in the sandbox |

`briefing_markdown` writes it in plain words (the chat's first message, and the reply when the model is not usable).
The app's helpers that the views share (questions, documents for a question, document status) moved here.

## The assistant (`chat.py`)
A ReAct agent built on the orchestrator's graph (`src/agent/orchestrator.py`), set up for a conversation:
- **Starts from the state**: the briefing, the last 4 exchanges (`chat_messages` in the Twin) and official passages
  retrieved for the message — 3 from the procedure's sources and up to 2 from the web corpus when they are about
  the question (`CORPUS_MIN_SCORE`). So a question usually needs **one** model call.
- **Tools, chosen per message by code** (`offered_tools`; validated like every tool). With a **cloud model**
  (`LLM_PROVIDER`) every chat tool is offered and STATE lists what can be prepared now (`can_prepare`: reminder
  dates, form ids, contact roles), so the assistant may prepare an e-mail, a reminder or a form for the next step on
  its own (always a proposal). With a small local model: `search_procedures` and
  `search_sources` always; `record_fact` only when the message tells something (a date, or a sentence in the first
  person: "I lost…", "My passport…"), never for a question; the artifact tools (`build_checklist`, `create_ics`,
  `draft_email`, `fill_pdf`: proposals for approval) and `simulate_submission` only when the message asks to prepare
  or try something. Without a procedure, only `search_procedures` (the answer comes from the web passages).
  `final_answer` is the first option. No workers, and no `evaluate_rules`: facts from the chat stay unconfirmed, so
  it would change nothing. Small models used every tool they were given, unasked (seen live), hence this choice.
- **Facts must come from the user's words** (`stated.py`). A recorded fact is accepted only if the message shows it:
  the same date (digits or a month name), the digits of a number, a word of the field's question for yes/no, a word
  of the value for text. Otherwise the call is refused and the model is told to ask. It stays unconfirmed until the
  user presses *Confirm*.
- **Spelling tidied by code, meaning never**: "not met" → `fail`, "met" → `pass`, "missing" → `unknown`; a ref that
  contains exactly one valid ref ("fix: … [ref R006] …") becomes that ref; only the first 5 next actions are kept
  (the summary still lists every step). Anything else is left for Gate 1.
- **Gate 1** checks every reply like the orchestrator's answer (statuses copied from `evaluate_rules`, no completion
  or approval claims, no unknown called failure, no invented dates or rule ids, refs that exist), except
  `MISSING_FAILED_RULE` (the plan beside the chat already lists every problem). Dates in the given passages and in
  values waiting for the user's check are allowed. A section id given as a ref becomes the first plan step of that
  section (seen live).
- **Reflection** (at most `CHAT_MAX_REPAIRS` = 2 rewrites):
  1. a reply Gate 1 refuses goes back to the model with the structured errors (and past lessons);
  2. a reply Gate 1 accepts is re-read once by the model against STATE and SOURCES (**self-review**,
     `REVIEW_SYSTEM`): it quotes the sentences that state something STATE does not support, change a status, invent a
     fee or date, contradict the plan or miss the question. Code keeps only quotes of 12–300 characters that really
     are in the reply, then the model rewrites. `SELF_REVIEW=auto`: on with a cloud model, off with Ollama (measured:
     `qwen2.5:3b` also criticised correct sentences, and the extra call costs about 40 s on 4 GB).
  Gate 1 keeps the last word: a reply it still refuses after the last rewrite is replaced by the plan; doubts of the
  self-review left after the last rewrite are recorded with the reply. Every round is stored (`reflections`) and
  shown under the reply and in *Technical*.
- **Labels added by code, not by the model**: "some rules are unverified" when the reply uses unverified rules;
  "general information … nothing was checked against your documents" when there is no procedure and the reply
  cites web pages.
- **Progress**: each step is shown while it runs ("Rechecking the requirements", "Checking the answer (Gate 1)").
- **Model**: `agent_backend("chat")`: Claude or GPT when `LLM_PROVIDER` says so, else `CHAT_MODEL` (default
  `qwen2.5:3b`), separate from the agent team's model. Tools (documents, search) always use the local backend. Measured on the 4 GB GPU:
  one reply took 16 s with `qwen2.5:3b` (all on the GPU) and 42 s with `qwen3:4b` (45% on the CPU); a smaller
  context (4096) only brought `qwen3:4b` to 37 s. On 8 typical messages (`docs/eval_chat.md`) `qwen2.5:3b` passed
  Gate 1 on 6 in about 50 s per turn, `qwen3:4b` on 4 in about 210 s.

The full **Orchestrator-Workers** run is still one click away ("Run a full check" beside the chat): the orchestrator
delegates each section to a worker and posts its checked answer in the conversation.

## Inputs and outputs
`run_chat(message, store, packs, user_id, today, backend, embedder=..., progress=...) -> ChatReply` with
`text`, `status` (answered / fallback / needs_human_review), `sources` (id, title, url), `facts` recorded this
turn (to confirm), `proposals`, `steps`, `llm_calls`, `gate_issues`, `reflections`, `model`. Both messages are stored in the Twin and the
turn is logged in the run log (memory across sessions).

## How to test it
```
python -m pytest -q tests/test_chat.py tests/test_reflection.py
python -m eval.chat_eval --models qwen2.5:3b qwen3:4b --out docs/eval_chat.md   # live, needs Ollama
```

## Known limits
- **English replies.** Gate 1's text checks (completion claims, "unverified") are written for English, so the
  assistant answers in English.
- **One model on one small GPU.** A turn takes 15–60 s; a turn that records a fact needs 2–3 calls.
- **Facts from the chat never count until confirmed** (by design): the user presses *Confirm* under the reply.
