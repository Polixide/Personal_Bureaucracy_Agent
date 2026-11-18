# `eval/` — comparing the agent with baselines

## What it does
It runs every pack's `eval/scenarios.json`, plus `eval/goal_scenarios.json` for procedure identification,
against three systems:
- **A**, a static checklist;
- **B**, a single LLM call with no state, rules or tools;
- **D**, the full agent.

It writes a results table to `docs/eval_results.md`.

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
python -m eval.run_eval --live     # needs `ollama pull qwen3:4b` and `ollama pull bge-m3`
```

## Known limits
- **Seven scenarios per pack, single session.** "Memory loss" can only appear in multi-session scenarios,
  which have not been written yet.
- **No live numbers yet.** No live run has been made yet on this machine; the live numbers are still to be
  measured.
