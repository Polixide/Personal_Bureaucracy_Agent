# Evaluation results (2026-11-24)

- Mode: live; models: qwen3:4b / bge-m3; k = 1 runs per scenario.
- A = static checklist, B = single LLM (no state/rules/tools), D = orchestrator + ReAct workers + Gate 1.
- Offline D measures the pipeline with a rule-following scripted policy, not model quality.

| system | runs | verdict_accuracy | date_errors | invented | valid_answer_rate | pass_k | avg_llm_calls | avg_seconds |
|---|---|---|---|---|---|---|---|---|
| A | 6 | 0.833 | 0 | 0 | 0.0 | 0/6 | 0.0 | 0.002 |
| B | 6 | 0.822 | 0 | 3 | 0.0 | 0/6 | 1.0 | 7.279 |
| D | 6 | 1.0 | 0 | 0 | 0.833 | 5/6 | 10.67 | 13.298 |

## Agent behaviour (D)

| valid_tool_call_rate | avg_steps | loops_detected | fallback_rate | gate1_first_rejections |
|---|---|---|---|---|
| 0.818 | 7.33 | 8 | 0.5 | 3 |

## Procedure identification (D)

| scenario_id | expected | selected | status | ok |
|---|---|---|---|---|
| G02-no-procedure | None | None | answered | True |
| G06b-lost-turkish-id-card | tc_kimlik_renewal | tc_kimlik_renewal | fallback | True |
| G06-lost-id-card | tc_kimlik_renewal | tc_kimlik_renewal | fallback | True |
| G07-renew-kimlik | tc_kimlik_renewal | tc_kimlik_renewal | fallback | True |
| G08-first-or-renewal-is-ambiguous | None | tc_kimlik_renewal | fallback | False |

## Memory across sessions (Twin closed and reopened; offline policy)

- 0/0 checks passed

## Scenarios where D failed

- R05-child-lost-card-parent-applies: planning (gate: UNGROUNDED_ACTION, accuracy 1.0)
