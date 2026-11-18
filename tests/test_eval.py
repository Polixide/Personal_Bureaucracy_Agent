"""eval/: offline evaluation of A and D, metrics, report, and B with a scripted model."""
from fakes import ScriptedBackend
from eval.metrics import pass_k, verdict_accuracy
from eval.policy import RuleFollowingPolicy
from eval.run_eval import evaluate, main, report, run_b
from src.agent.answer import Claim, FinalAnswer
from src.pack import load_all_packs, load_scenarios

PACKS = {"ikamet_first": load_all_packs()["ikamet_first"]}


def test_offline_full_agent_passes_every_pack_scenario_and_beats_the_static_checklist():
    outcome = evaluate(PACKS, RuleFollowingPolicy(), k=1, live=False)
    d_runs = [r for r in outcome["runs"] if r.system == "D"]
    a_runs = [r for r in outcome["runs"] if r.system == "A"]
    assert d_runs and all(r.success for r in d_runs)
    assert sum(r.accuracy for r in a_runs) < sum(r.accuracy for r in d_runs)
    assert not any(r.agent.fallback_used for r in d_runs)
    assert all(item["ok"] for item in outcome["identification"])


def test_verdict_accuracy_counts_unstated_rules_as_pass():
    expected = {"R001": "pass", "R006": "fail", "R009": "unknown"}
    answer = FinalAnswer(summary="x", claims=[Claim(rule_id="R006", status="fail")])
    assert verdict_accuracy(answer, expected) == 2 / 3
    assert verdict_accuracy(None, {"R001": "pass"}) == 1.0


def test_pass_k_needs_every_run_to_succeed():
    from eval.metrics import RunMeasure
    def run(scenario, success):
        return RunMeasure(scenario_id=scenario, system="D", accuracy=1.0, date_errors=0, inventions=0, valid=success,
                          success=success, llm_calls=0, seconds=0.0)
    assert pass_k([run("s1", True), run("s1", True), run("s2", True), run("s2", False)]) == 1


def test_single_llm_baseline_is_judged_by_the_same_gate():
    pack = PACKS["ikamet_first"]
    scenario = next(s for s in load_scenarios(pack) if s["scenario_id"] == "02-passport-buffer")
    invented = {"summary": "Everything is fine; submit on 2026-12-24.", "claims": [{"rule_id": "R099", "status": "pass"}]}
    measure = run_b(pack, PACKS, scenario, ScriptedBackend([invented]))
    assert not measure.valid and measure.date_errors == 1 and measure.inventions == 1 and measure.llm_calls == 1


def test_report_lists_systems_behaviour_and_identification():
    text = report(evaluate(PACKS, RuleFollowingPolicy(), k=1, live=False), k=1, live=False)
    assert "| A |" in text and "| D |" in text and "not run offline" in text
    assert "## Agent behaviour (D)" in text and "## Procedure identification (D)" in text


def test_cli_writes_the_results_file(tmp_path, capsys):
    out = tmp_path / "results.md"
    assert main(["--k", "1", "--out", str(out), "--procedure", "ikamet_first"]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Evaluation results")
