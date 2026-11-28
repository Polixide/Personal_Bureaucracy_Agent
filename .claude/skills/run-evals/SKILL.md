---
name: run-evals
description: Use when the user asks to evaluate the agent, compare it with baselines, or check reliability. Runs tests and the scenario evaluation and reports a short results table, including ReAct behaviour metrics.
---

# Run the evaluation

1. Run `python -m pytest -q` and note failures.
2. Run `python -m eval.run_eval` (offline, LLM mocked) and, only if the user asks, `python -m eval.run_eval --live`
   (uses the local Ollama model; add `--provider openai --model <name> --effort <level>` for the cloud model, only
   with the user's key and consent: it costs money).
3. Report a table with one row per system:
   - A: static checklist
   - B: single LLM, no state, no rules, no tools
   - D: full agent (orchestrator + ReAct workers + verifiers)
   Columns: rule-verdict accuracy, unknown kept, fail recall, date errors, invented rules or deadlines,
   valid-final-answer rate, pass^k (k = 3 runs of the same scenario must all succeed), pass@k, average LLM calls,
   average and p95 seconds, tokens and estimated cost.
4. For D also report agent behaviour: valid tool-call rate, average steps, loops detected, fallback-to-workflow
   rate, human-review rate, number of times Gate 1 rejected the first final answer, issues Gate 1 caught and the
   reflection recovery rate. Every metric is defined in `docs/metrics.md` (RAG, chat and document metrics too).
5. List the scenarios where D failed and classify each failure: planning, tool/schema, observation misread, memory loss.
6. Never edit `eval/scenarios.json` expected values to improve a score. If a scenario looks wrong, explain and ask.
7. Save the table to `docs/eval_results.md` with the date and the model names used.
