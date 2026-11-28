# Evaluation results (2026-11-28)

- Mode: offline; models: offline: eval.policy.RuleFollowingPolicy (no language model); k = 3 runs per scenario.
- A = static checklist, B = single LLM (no state/rules/tools), D = orchestrator + ReAct workers + Gate 1.
- Offline D measures the pipeline with a rule-following scripted policy, not model quality.
- pass^k: scenarios where all k runs succeed; pass@k: chance that at least one of k runs succeeds. Tokens and cost come from the providers' usage reports (n/a offline or without a price).
- reflection_recovery_rate: answers refused by Gate 1 that the model then repaired (not the fallback).
- unknown_kept_rate: expected-unknown rules still called unknown (never guessed); fail_recall: failing rules the answer reports.

| system | runs | verdict_accuracy | unknown_kept_rate | fail_recall | date_errors | invented | valid_answer_rate | pass^k | pass@k | avg_llm_calls | avg_seconds | p95_seconds | avg_tokens | cost_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 75 | 0.807 | 0.0 | 0.0 | 0 | 0 | 0.16 | 1/25 | 0.04 | 0.0 | 0.009 | 0.017 | n/a | n/a |
| B | not run offline (needs --live) |  |  |  |  |  |  |  |  |  |  |  |  |  |
| D | 75 | 1.0 | 1.0 | 1.0 | 0 | 0 | 1.0 | 25/25 | 1.0 | 4.64 | 0.049 | 0.076 | n/a | n/a |

## Agent behaviour (D)

| valid_tool_call_rate | avg_steps | loops_detected | fallback_rate | human_review_rate | gate1_first_rejections | gate1_issues_caught | reflection_recovery_rate |
|---|---|---|---|---|---|---|---|
| 1.0 | 3.32 | 0 | 0.0 | 0.0 | 0 | 0 | n/a |

## Procedure identification (D)

| scenario_id | expected | selected | status | ok |
|---|---|---|---|---|
| G01-first-ikamet | ikamet_first | ikamet_first | answered | True |
| G02-no-procedure | None | None | answered | True |
| G03-other-wording | ikamet_first | ikamet_first | answered | True |
| G04-extension | ikamet_extension | ikamet_extension | answered | True |
| G05-first-id-card | tc_kimlik_first | tc_kimlik_first | answered | True |
| G06b-lost-turkish-id-card | tc_kimlik_renewal | tc_kimlik_renewal | answered | True |
| G06-lost-id-card | tc_kimlik_renewal | tc_kimlik_renewal | answered | True |
| G07-renew-kimlik | tc_kimlik_renewal | tc_kimlik_renewal | answered | True |
| G08-first-or-renewal-is-ambiguous | None | None | needs_user_input | True |
| G09-replace-paper-id | tc_kimlik_first | tc_kimlik_first | answered | True |

## Memory across sessions (Twin closed and reopened; offline policy)

- 10/10 checks passed

## Scenarios where D failed

- none
