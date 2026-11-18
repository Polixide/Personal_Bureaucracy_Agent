# Evaluation results (2026-11-18)

- Mode: offline; models: offline: eval.policy.RuleFollowingPolicy (no language model); k = 3 runs per scenario.
- A = static checklist, B = single LLM (no state/rules/tools), D = orchestrator + ReAct workers + Gate 1.
- Offline D measures the pipeline with a rule-following scripted policy, not model quality.

| system | runs | verdict_accuracy | date_errors | invented | valid_answer_rate | pass_k | avg_llm_calls | avg_seconds |
|---|---|---|---|---|---|---|---|---|
| A | 21 | 0.828 | 0 | 0 | 0.429 | 1/7 | 0.0 | 0.011 |
| B | not run offline (needs --live) |  |  |  |  |  |  |  |
| D | 21 | 1.0 | 0 | 0 | 1.0 | 7/7 | 4.29 | 0.042 |

## Agent behaviour (D)

| valid_tool_call_rate | avg_steps | loops_detected | fallback_rate | gate1_first_rejections |
|---|---|---|---|---|
| 1.0 | 3.14 | 0 | 0.0 | 0 |

## Procedure identification (D)

| scenario_id | expected | selected | status | ok |
|---|---|---|---|---|
| G01-first-ikamet | ikamet_first | ikamet_first | answered | True |
| G02-no-procedure | None | None | answered | True |
| G03-other-wording | ikamet_first | ikamet_first | answered | True |

## Scenarios where D failed

- none
