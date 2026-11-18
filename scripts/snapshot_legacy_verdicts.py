"""HISTORICAL ONE-OFF (phase 1). It imports milestone-1 modules that were deleted in phase 9, so it only
runs on a checkout of commit 486917b or earlier. Its output is committed and must not be edited by hand.

Write tests/data/legacy_verdicts.json: the milestone-1 engine's verdicts on fixed contexts.

Run once, before the old engine is deleted: python -m scripts.snapshot_legacy_verdicts
The file is generated, never edited by hand. tests/test_rules_oracle.py compares the new
evaluator (src/rules.py) against it.

Contexts: the 15 knowledge/test_cases.json inputs, the 29 pass/fail branches and boundaries of
tests/test_requirement_engine.py, the 10 scenario contexts of tests/test_evaluation.py, the empty
context, and 250 seeded random perturbations of the "ready" context.
"""
import json
import random
import sys
from datetime import date, timedelta

from config import PROJECT_ROOT
from src.core.state import ResidencePermitContext
from src.workers.rules_engine import evaluate_requirements

OUT = PROJECT_ROOT / "tests" / "data" / "legacy_verdicts.json"
SEED = 58012
PERTURBATIONS = 250
STATUS = {"passed": "pass", "failed": "fail", "missing_information": "unknown"}
STRING_ALTERNATIVES = {
    "permit_type": ["short_term", "student"],
    "application_channel": ["https://e-ikamet.goc.gov.tr/", "unofficial"],
    "insurance_evidence_type": ["private_health_insurance", "sgk_provision", "travel_only"],
    "address_proof_type": ["dorm_certificate", "rental_agreement", "airbnb"],
    "approval_status": ["pending", "guaranteed"],
    "decision_authority": ["Migration Management", "Sabancı IRO"],
    "tracking_channel": ["e_ikamet", "unofficial"],
    "application_form_window_id": ["synthetic-nov", "synthetic-oct"],
    "current_iro_window_id": ["synthetic-nov", "synthetic-oct"],
}
DAY_SHIFTS = [-120, -61, -60, -59, -1, 1, 59, 60, 61, 120]
INT_SHIFTS = [-7, -3, -1, 1, 2, 6]


def typed(values: dict) -> dict:
    """Validate through the old context model so values have their real types."""
    return {k: v for k, v in ResidencePermitContext(**values).model_dump().items() if v is not None}


def fixed_contexts() -> list[tuple[str, dict]]:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    import test_evaluation
    import test_requirement_engine as engine_tests
    contexts = [("empty", {})]
    contexts += [(f"knowledge-{case['id']}", case["input"]) for case in engine_tests.CASES]
    for rule_id, positive, negative in engine_tests.RULE_BRANCHES:
        contexts.append((f"branch-{rule_id}-pass", positive))
        contexts.append((f"branch-{rule_id}-fail", {**positive, **negative}))
    contexts += [("boundary-R006-60-days", {"requested_permit_end_date": "2027-06-01", "passport_expiry_date": "2027-07-31"}),
                 ("boundary-R006-59-days", {"requested_permit_end_date": "2027-06-01", "passport_expiry_date": "2027-07-30"}),
                 ("boundary-R004-equal", {"application_date": "2026-11-15", "legal_stay_expiry_date": "2026-11-15"})]
    for scenario in test_evaluation.SCENARIOS:
        base = {} if scenario.get("empty_context") else test_evaluation.READY_CONTEXT
        contexts.append((f"scenario-{scenario['id']}", {**base, **scenario["changes"]}))
    return [(name, typed(values)) for name, values in contexts]


def mutate(name: str, value, rng: random.Random):
    """One small change of a known value, keeping its type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + rng.choice(INT_SHIFTS)
    if isinstance(value, date):
        return value + timedelta(days=rng.choice(DAY_SHIFTS))
    return rng.choice(STRING_ALTERNATIVES.get(name, [value, value + "-changed"]))


def perturbed_contexts() -> list[tuple[str, dict]]:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    import test_evaluation
    rng = random.Random(SEED)
    ready = typed(test_evaluation.READY_CONTEXT)
    contexts = []
    for index in range(PERTURBATIONS):
        values = {}
        for name, value in ready.items():
            roll = rng.random()
            if roll < 0.12:
                continue
            values[name] = mutate(name, value, rng) if roll < 0.22 else value
        contexts.append((f"perturbed-{index:03}", values))
    return contexts


def verdicts(values: dict) -> dict[str, str]:
    """Old engine, default (unscoped) mode: 'pass' | 'fail' | 'unknown:<missing fields>'."""
    results = evaluate_requirements({"residence_permit_context": ResidencePermitContext(**values)})
    output = {}
    for result in results:
        status = STATUS[result.status]
        output[result.rule_id] = status if status != "unknown" else "unknown:" + ",".join(result.missing_fields)
    return output


def main() -> None:
    cases = []
    for name, values in fixed_contexts() + perturbed_contexts():
        context = ResidencePermitContext(**values).model_dump(mode="json")
        context = {k: v for k, v in context.items() if v is not None}
        cases.append({"id": name, "context": context, "verdicts": verdicts(values)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases]
    header = {"generated_by": "src/workers/rules_engine.py (milestone-1, default scope)",
              "generated_on": date.today().isoformat(), "seed": SEED, "cases": len(cases)}
    OUT.write_text(json.dumps(header) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
