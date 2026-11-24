# Evaluation results (2026-11-24)

- Mode: live; models: qwen3:4b / bge-m3; k = 1 runs per scenario.
- A = static checklist, B = single LLM (no state/rules/tools), D = orchestrator + ReAct workers + Gate 1.
- Offline D measures the pipeline with a rule-following scripted policy, not model quality.

| system | runs | verdict_accuracy | date_errors | invented | valid_answer_rate | pass_k | avg_llm_calls | avg_seconds |
|---|---|---|---|---|---|---|---|---|
| A | 6 | 0.833 | 0 | 0 | 0.0 | 0/6 | 0.0 | 0.004 |
| B | 6 | 0.822 | 0 | 0 | 0.0 | 0/6 | 1.0 | 6.807 |
| D | 6 | 0.845 | 0 | 0 | 0.667 | 4/6 | 11.0 | 16.44 |

## Agent behaviour (D)

| valid_tool_call_rate | avg_steps | loops_detected | fallback_rate | gate1_first_rejections |
|---|---|---|---|---|
| 0.717 | 7.67 | 13 | 0.333 | 3 |

## Procedure identification (D)

| scenario_id | expected | selected | status | ok |
|---|---|---|---|---|
| G02-no-procedure | None | None | answered | True |
| G05-first-id-card | tc_kimlik_first | tc_kimlik_first | answered | True |
| G08-first-or-renewal-is-ambiguous | None | tc_kimlik_first | answered | False |
| G09-replace-paper-id | tc_kimlik_first | tc_kimlik_first | answered | True |

## Memory across sessions (Twin closed and reopened; offline policy)

- 0/0 checks passed

## Scenarios where D failed

- F01-adult-replaces-nufus-cuzdani: observation misread (gate: CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, CLAIM_MISMATCH, UNGROUNDED_ACTION, accuracy 0.067)
- F02-newborn-free-card: planning (gate: UNGROUNDED_ACTION, accuracy 1.0)
