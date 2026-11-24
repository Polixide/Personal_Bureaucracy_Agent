# `eval/` — comparing the agent with baselines

## What it does
It runs every pack's `eval/scenarios.json`, plus `eval/goal_scenarios.json` for procedure identification,
against three systems:
- **A**, a static checklist;
- **B**, a single LLM call with no state, rules or tools;
- **D**, the full agent.

It writes a results table to `docs/eval_results.md`. Two more checks sit next to it:
- **memory across sessions** (`eval/memory_eval.py`, Neda's scenarios M1-M3 on our architecture): the Twin file is
  closed and reopened between a proposal and its approval or denial, and one profile is reset next to another;
- **retrieval** (`eval/rag_eval.py`): hit@1, hit@3 and MRR of each embedding model on the saved official texts,
  with one query per rule (its description) and the rule's quoted passage as the expected hit.

## Where it sits in the flow
`python -m eval.run_eval` (offline) or `python -m eval.run_eval --live` (Ollama). See the `run-evals` skill.
Each run uses a fresh in-memory Twin seeded with the scenario's facts and `today`.

## Inputs and outputs
A scenario (generated from milestone-1 `tests/test_evaluation.py` and never edited to improve a score):
```json
{"scenario_id": "02-passport-buffer", "kind": "date_conflict", "goal": "I want to apply for İkamet",
 "today": "2026-11-11", "facts": {...}, "expected": {"procedure_id": "ikamet_first", "rule_status": {"R006": "fail", ...}}}
```

The table columns are:
- **verdict accuracy:** claims compared with the expected statuses; a rule the answer doesn't mention counts as "pass";
- **date errors:** Gate 1 `UNSUPPORTED_DATE`;
- **invented rules:** `CLAIM_UNKNOWN_RULE` + `INVENTED_RULE_ID`;
- **valid-final-answer rate:** the share of answers that pass Gate 1;
- **pass^k:** all k runs of a scenario succeed;
- **average LLM calls and average seconds.**

D's behaviour metrics are the valid tool-call rate, average steps, loops, fallback rate and Gate 1 first
rejections. Failed D runs are classified as planning, tool/schema, observation misread or memory loss.

## Why it is built this way
- **The offline D is a pipeline test.** Offline, D uses `eval/policy.py:RuleFollowingPolicy`, a scripted policy
  that follows the orchestrator prompt literally. That checks the pipeline (tools, workers, Gate 1, fallback)
  without a model, so the offline numbers are *not* model quality. B needs a real model, so offline it is
  reported as "not run".
- **One judge for every system.** All of them are judged by the same deterministic Gate 1 and the same expected
  verdicts.

## How to test it
```
python -m pytest -q tests/test_eval.py
python -m eval.run_eval            # offline, writes docs/eval_results.md
python -m eval.run_eval --live     # needs `ollama pull qwen3:4b` and the EMBED_MODEL of config.py
python -m pytest -q tests/test_memory_sessions.py
python -m eval.rag_eval --models qwen3-embedding:0.6b bge-m3 --out docs/eval_rag.md   # live, embeddings only
python -m eval.document_eval --label "my change" --out docs/eval_documents.md       # live: reading uploads
python -m eval.chat_eval --models qwen2.5:3b qwen3:4b --out docs/eval_chat.md        # live: the chat assistant
```
- **`document_eval`** writes 13 files (synthetic text PDFs, scans, a phone photo, files holding two documents, and
  the public passport and stamp specimens of `data/reference_samples/`) with known values, reads them with the real
  pipeline and counts per file: documents found, values read correctly (and how many the user must confirm),
  values read wrong (believed, or shown for confirmation), values not read, seconds. "Believed wrong" must stay 0.
- **`chat_eval`** sends 8 typical messages (where am I, a stated fact, a checklist request, a procedure without
  pack, naming a procedure, "why", an unknown date, a question for the sources) and checks by code whether the reply
  passed Gate 1 at the first try and whether the expected effect happened (fact recorded, proposal, procedure,
  source cited). It reuses only the embedding cache of the app's Twin.

## Known limits
- **Six or seven scenarios per pack.** The multi-session checks use the offline policy and one pack
  (`ikamet_first`); they test the memory plumbing, not what a live model does with it.
- **Retrieval queries are rule descriptions.** They are written in English like the rules, so the Turkish ID-card
  sources make them cross-lingual; real users' wording may be harder.
- **No live numbers yet.** No live run has been made yet on this machine; the live numbers are still to be
  measured.
