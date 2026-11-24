"""Per-run measurements and the aggregate table (metrics from the run-evals skill).

Verdict accuracy: for each expected rule status, the answer's claim for that rule; a rule the answer
does not mention counts as "pass" (answers must mention every failing rule and Gate 1 enforces it).
"""
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


def verdict_accuracy(answer: FinalAnswer | None, expected: dict[str, str]) -> float:
    if not expected:
        return 1.0
    claims = {claim.rule_id: claim.status for claim in (answer.claims if answer else [])}
    return sum(claims.get(rule_id, "pass") == status for rule_id, status in expected.items()) / len(expected)


def measure(scenario_id: str, system: str, answer: FinalAnswer | None, expected: dict[str, str], gate: GateContext,
            llm_calls: int, seconds: float, agent: AgentResult | None = None) -> RunMeasure:
    """Check the delivered answer with Gate 1 and compare its claims with the expected verdicts."""
    issues = verify(answer, gate) if answer is not None else []
    codes = [issue.code for issue in issues]
    accuracy = verdict_accuracy(answer, expected)
    valid = answer is not None and not issues
    return RunMeasure(scenario_id=scenario_id, system=system, accuracy=round(accuracy, 3),
                      date_errors=codes.count("UNSUPPORTED_DATE"), inventions=sum(codes.count(c) for c in INVENTION_CODES),
                      valid=valid, success=valid and accuracy == 1.0, llm_calls=llm_calls, seconds=round(seconds, 3),
                      gate_codes=codes, agent=agent)


def pass_k(runs: list[RunMeasure]) -> int:
    """Scenarios where every one of the k runs succeeded."""
    by_scenario: dict[str, list[bool]] = {}
    for run in runs:
        by_scenario.setdefault(run.scenario_id, []).append(run.success)
    return sum(all(results) for results in by_scenario.values())


def system_row(system: str, runs: list[RunMeasure]) -> dict:
    """One table row: averages over runs, pass^k over scenarios."""
    count = len(runs) or 1
    scenarios = len({run.scenario_id for run in runs})
    return {"system": system, "runs": len(runs),
            "verdict_accuracy": round(sum(r.accuracy for r in runs) / count, 3),
            "date_errors": sum(r.date_errors for r in runs), "invented": sum(r.inventions for r in runs),
            "valid_answer_rate": round(sum(r.valid for r in runs) / count, 3),
            "pass_k": f"{pass_k(runs)}/{scenarios}",
            "avg_llm_calls": round(sum(r.llm_calls for r in runs) / count, 2),
            "avg_seconds": round(sum(r.seconds for r in runs) / count, 3)}


def agent_behaviour(runs: list[RunMeasure]) -> dict:
    """ReAct behaviour of system D."""
    results = [run.agent for run in runs if run.agent is not None]
    steps = [step for result in results for step in result.steps]
    count = len(results) or 1
    return {"valid_tool_call_rate": round(sum(s.ok for s in steps) / (len(steps) or 1), 3),
            "avg_steps": round(len(steps) / count, 2),
            "loops_detected": sum(r.loops_detected for r in results),
            "fallback_rate": round(sum(r.fallback_used for r in results) / count, 3),
            "gate1_first_rejections": sum(r.first_answer_rejected for r in results)}


def classify_failure(run: RunMeasure) -> str:
    """planning | tool/schema | observation misread | memory loss (run-evals skill categories)."""
    result = run.agent
    if result is not None and result.fallback_used:
        return "tool/schema"
    if "CLAIM_MISMATCH" in run.gate_codes or (result is not None and any(
            i.code == "CLAIM_MISMATCH" for i in result.gate_issues)):
        return "observation misread"
    return "planning"
