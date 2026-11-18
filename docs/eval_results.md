# Evaluation results (2026-11-18)

- Mode: offline; models: offline: eval.policy.RuleFollowingPolicy (no language model); k = 3 runs per scenario.
- A = static checklist, B = single LLM (no state/rules/tools), D = orchestrator + ReAct workers + Gate 1.
- Offline D measures the pipeline with a rule-following scripted policy, not model quality.

| system | runs | verdict_accuracy | date_errors | invented | valid_answer_rate | pass_k | avg_llm_calls | avg_seconds |
|---|---|---|---|---|---|---|---|---|
| A | 39 | 0.782 | 0 | 0 | 0.231 | 1/13 | 0.0 | 0.005 |
| B | not run offline (needs --live) |  |  |  |  |  |  |  |
| D | 39 | 1.0 | 0 | 0 | 1.0 | 13/13 | 4.31 | 0.025 |

## Agent behaviour (D)

| valid_tool_call_rate | avg_steps | loops_detected | fallback_rate | gate1_first_rejections |
|---|---|---|---|---|
| 1.0 | 3.15 | 0 | 0.0 | 0 |

## Procedure identification (D)

| scenario_id | expected | selected | status | ok |
|---|---|---|---|---|
| G01-first-ikamet | ikamet_first | ikamet_first | answered | True |
| G02-no-procedure | None | None | answered | True |
| G03-other-wording | ikamet_first | ikamet_first | answered | True |
| G04-extension | ikamet_extension | ikamet_extension | answered | True |

## Scenarios where D failed

- none
