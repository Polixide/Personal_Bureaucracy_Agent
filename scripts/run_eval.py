"""Run the synthetic scenario evaluation and print a report.

    python -m scripts.run_eval
Exit code 0 when every scenario passes, 1 otherwise (usable in CI).
"""
import sys

from src.evaluation.harness import evaluate, format_report

if __name__ == "__main__":
    report = evaluate()
    print(format_report(report))
    sys.exit(0 if report["scenarios_passed"] == report["scenarios_total"] else 1)
