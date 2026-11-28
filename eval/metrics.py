"""Per-run measurements and the aggregate table (metrics from the run-evals skill).

Verdict accuracy: for each expected rule status, the answer's claim for that rule; a rule the answer
does not mention counts as "pass" (answers must mention every failing rule and Gate 1 enforces it).

Unknown kept: of the rules expected "unknown" (information missing), the share the answer still calls unknown,
the project's core promise (unknown facts stay unknown, never guessed). Fail recall: of the failing rules, the share
the answer reports (a missed failure is the costly error for the user).

Reliability: pass^k (every one of the k runs of a scenario succeeds: the user can rely on it) and pass@k (at least
one of k runs succeeds; the unbiased estimator of Chen et al. 2021). Cost: tokens and estimated USD per run, from
the usage log of src/llm.py. Reflection: how often an answer refused by Gate 1 was repaired by the model. The last
three are adapted from Andrea's metrics (feature/marco-integration-review, src/evaluation/metrics.py).
"""
from math import ceil, comb

from pydantic import BaseModel, Field

from src.agent.answer import AgentResult, FinalAnswer
from src.agent.gate1 import GateContext, verify

INVENTION_CODES = ("CLAIM_UNKNOWN_RULE", "INVENTED_RULE_ID")


class RunMeasure(BaseModel):
    """One run of one system on one scenario."""
    scenario_id: str
    system: str
    accuracy: float
    date_errors: int
    inventions: int
    valid: bool
    success: bool
    llm_calls: int
    seconds: float
    gate_codes: list[str] = Field(default_factory=list)
    agent: AgentResult | None = None
    unknown_expected: int = 0            # expected "unknown" rules (information missing) ...
    unknown_kept: int = 0                # ... that the answer still calls unknown (never guessed into pass/fail)
    fail_expected: int = 0               # expected failing rules ...
    fail_found: int = 0                  # ... that the answer reports as failing
    input_tokens: int | None = None      # None: no token count (offline policy, or a provider that reports none)
    output_tokens: int | None = None
    cost_usd: float | None = None        # None: a local model, or a cloud model without a price in config.py


def verdict_accuracy(answer: FinalAnswer | None, expected: dict[str, str]) -> float:
    if not expected:
        return 1.0
    claims = {claim.rule_id: claim.status for claim in (answer.claims if answer else [])}
    return sum(claims.get(rule_id, "pass") == status for rule_id, status in expected.items()) / len(expected)


def status_kept(answer: FinalAnswer | None, expected: dict[str, str], status: str) -> tuple[int, int]:
    """(expected rules with this status, of which the answer claims the same status)."""
    claims = {claim.rule_id: claim.status for claim in (answer.claims if answer else [])}
    wanted = [rule_id for rule_id, value in expected.items() if value == status]
    return len(wanted), sum(claims.get(rule_id) == status for rule_id in wanted)


def measure(scenario_id: str, system: str, answer: FinalAnswer | None, expected: dict[str, str], gate: GateContext,
            llm_calls: int, seconds: float, agent: AgentResult | None = None, usage: dict | None = None) -> RunMeasure:
    """Check the delivered answer with Gate 1 and compare its claims with the expected verdicts.

    `usage` is src/llm.py `usage_summary` of the run's model calls (tokens, cost), when the run made any.
    """
    issues = verify(answer, gate) if answer is not None else []
    codes = [issue.code for issue in issues]
    accuracy = verdict_accuracy(answer, expected)
    valid = answer is not None and not issues
    usage = usage if usage and usage.get("requests") else {}
    unknown_expected, unknown_kept = status_kept(answer, expected, "unknown")
    fail_expected, fail_found = status_kept(answer, expected, "fail")
    return RunMeasure(scenario_id=scenario_id, system=system, accuracy=round(accuracy, 3),
                      unknown_expected=unknown_expected, unknown_kept=unknown_kept, fail_expected=fail_expected,
                      fail_found=fail_found,
                      date_errors=codes.count("UNSUPPORTED_DATE"), inventions=sum(codes.count(c) for c in INVENTION_CODES),
                      valid=valid, success=valid and accuracy == 1.0, llm_calls=llm_calls, seconds=round(seconds, 3),
                      gate_codes=codes, agent=agent, input_tokens=usage.get("input_tokens"),
                      output_tokens=usage.get("output_tokens"), cost_usd=usage.get("cost_usd"))


def pass_k(runs: list[RunMeasure]) -> int:
    """Scenarios where every one of the k runs succeeded."""
    by_scenario: dict[str, list[bool]] = {}
    for run in runs:
        by_scenario.setdefault(run.scenario_id, []).append(run.success)
    return sum(all(results) for results in by_scenario.values())


def pass_at_k(runs: list[RunMeasure], k: int) -> float:
    """Mean over scenarios of 1 - C(n-c, k) / C(n, k): the chance that at least one of k runs succeeds.

    n runs of a scenario, c of them successful. With exactly k runs it is the share of scenarios with a success.
    """
    by_scenario: dict[str, list[bool]] = {}
    for run in runs:
        by_scenario.setdefault(run.scenario_id, []).append(run.success)
    if not by_scenario:
        return 0.0
    values = []
    for results in by_scenario.values():
        n, c, draws = len(results), sum(results), min(k, len(results))
        values.append(1.0 if n - c < draws else 1.0 - comb(n - c, draws) / comb(n, draws))
    return sum(values) / len(values)


def share(part: int, whole: int) -> float | str:
    """part / whole rounded, or "n/a" when nothing was expected."""
    return round(part / whole, 3) if whole else "n/a"


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (fraction 0.5 = median, 0.95 = p95); 0.0 for no values."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, ceil(fraction * len(ordered)) - 1)]


def per_run_cost(runs: list[RunMeasure]) -> tuple[str, str]:
    """Average tokens and total estimated cost, as table text ("n/a" when unknown)."""
    tokens = [r.input_tokens + r.output_tokens for r in runs if r.input_tokens is not None and r.output_tokens is not None]
    costs = [r.cost_usd for r in runs if r.cost_usd is not None]
    average = f"{round(sum(tokens) / len(tokens)):,}" if tokens else "n/a"
    total = f"${sum(costs):.4f}" if costs and len(costs) == len(tokens) else "n/a"
    return average, total


def system_row(system: str, runs: list[RunMeasure], k: int = 1) -> dict:
    """One table row: averages over runs, pass^k and pass@k over scenarios, latency, tokens and cost."""
    count = len(runs) or 1
    scenarios = len({run.scenario_id for run in runs})
    tokens, cost = per_run_cost(runs)
    seconds = [r.seconds for r in runs]
    return {"system": system, "runs": len(runs),
            "verdict_accuracy": round(sum(r.accuracy for r in runs) / count, 3),
            "date_errors": sum(r.date_errors for r in runs), "invented": sum(r.inventions for r in runs),
            "valid_answer_rate": round(sum(r.valid for r in runs) / count, 3),
            "unknown_kept_rate": share(sum(r.unknown_kept for r in runs), sum(r.unknown_expected for r in runs)),
            "fail_recall": share(sum(r.fail_found for r in runs), sum(r.fail_expected for r in runs)),
            "pass^k": f"{pass_k(runs)}/{scenarios}", "pass@k": round(pass_at_k(runs, k), 3),
            "avg_llm_calls": round(sum(r.llm_calls for r in runs) / count, 2),
            "avg_seconds": round(sum(seconds) / count, 3), "p95_seconds": round(percentile(seconds, 0.95), 3),
            "avg_tokens": tokens, "cost_usd": cost}


def agent_behaviour(runs: list[RunMeasure]) -> dict:
    """ReAct behaviour of system D."""
    results = [run.agent for run in runs if run.agent is not None]
    steps = [step for result in results for step in result.steps]
    count = len(results) or 1
    rejected = [r for r in results if r.first_answer_rejected]
    recovered = sum(r.status == "answered" for r in rejected)
    caught = sum(len(round_["issues"]) for r in results for round_ in r.reflections)
    return {"valid_tool_call_rate": round(sum(s.ok for s in steps) / (len(steps) or 1), 3),
            "avg_steps": round(len(steps) / count, 2),
            "loops_detected": sum(r.loops_detected for r in results),
            "fallback_rate": round(sum(r.fallback_used for r in results) / count, 3),
            "human_review_rate": round(sum(r.needs_human_review for r in results) / count, 3),
            "gate1_first_rejections": sum(r.first_answer_rejected for r in results),
            "gate1_issues_caught": caught,
            "reflection_recovery_rate": f"{recovered}/{len(rejected)}" if rejected else "n/a"}


def classify_failure(run: RunMeasure) -> str:
    """planning | tool/schema | observation misread | memory loss (run-evals skill categories)."""
    result = run.agent
    if result is not None and result.fallback_used:
        return "tool/schema"
    if "CLAIM_MISMATCH" in run.gate_codes or (result is not None and any(
            i.code == "CLAIM_MISMATCH" for i in result.gate_issues)):
        return "observation misread"
    return "planning"
