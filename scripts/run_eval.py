"""Run the evaluation scenarios offline and print a scored report.

    python -m scripts.run_eval
Exit code 0 when every scenario passes, 1 otherwise (usable in CI).
"""
import logging
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from src.evaluation.harness import evaluate, format_report

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    # hide the per-call [LLM]/[PLANNER]/[GATE1] diagnostics
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        report = evaluate()
    print(format_report(report))
    sys.exit(0 if report["scenarios_passed"] == report["scenarios_total"] else 1)
