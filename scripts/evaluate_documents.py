"""Balanced synthetic evaluation. Mock mode checks contracts, live mode measures Qwen."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from processor import process_document
from document_profiles import REGISTRY

ROOT = Path(__file__).resolve().parents[1] / "data/synthetic_samples"
CASES = ROOT / "type_evaluation/cases.json"


def evaluate(*, live: bool = False, holdout: bool = False, model: str = "qwen3:8b") -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    groups = {}
    details = []
    for case in cases:
        if bool(case["holdout"]) != holdout:
            continue
        def reply(**request):
            value = case["classification"] if request["format"].get("title") == "DocumentClassification" else case["extraction"]
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))
        with nullcontext() if live else patch("processor.Client.chat", side_effect=reply):
            result = process_document(ROOT / case["file"], model=model).model_dump(mode="json")
        kind = case["document_type"]
        counts = groups.setdefault(kind, dict(cases=0, classification_correct=0, field_correct=0,
            fields=0, required_correct=0, required_present=0, unsupported=0, returned=0, review_correct=0))
        counts["cases"] += 1
        counts["classification_correct"] += result["document_type"] == kind
        review = bool(result["missing_fields"] or result["ambiguous_fields"] or result["validation_issues"])
        counts["review_correct"] += review == case["review"]
        for field, expected in case["expected"].items():
            counts["fields"] += 1
            counts["field_correct"] += result.get(field) == expected
        required = REGISTRY[kind].required(kind)
        aliases = {"institution": "insurer"} if kind == "health_insurance_certificate" else {}
        for field in required:
            expected = case["expected"].get(aliases.get(field, field))
            if expected is not None:
                counts["required_present"] += 1
                counts["required_correct"] += result.get(field) == expected
        for field in REGISTRY[kind].schema.model_fields:
            if field in {"source_snippets", "ambiguous_fields"} or result.get(field) is None:
                continue
            counts["returned"] += 1
            counts["unsupported"] += field not in case["expected"] or result[field] != case["expected"][field]
        details.append(dict(id=case["id"], classification=result["document_type"], review=review,
                            missing_fields=result["missing_fields"], validation_issues=result["validation_issues"]))
    metrics = {}
    def ratio(a, b):
        return round(a / b, 4) if b else None
    for kind, c in groups.items():
        metrics[kind] = {**c, "classification_accuracy": ratio(c["classification_correct"], c["cases"]),
            "field_accuracy": ratio(c["field_correct"], c["fields"]),
            "required_field_recall": ratio(c["required_correct"], c["required_present"]),
            "unsupported_value_rate": ratio(c["unsupported"], c["returned"]),
            "manual_review_accuracy": ratio(c["review_correct"], c["cases"])}
    return {"mode": "LIVE Qwen" if live else "MOCKED model responses: contract checks, not model accuracy",
            "split": "holdout" if holdout else "development", "metrics": metrics, "cases": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--model", default="qwen3:8b")
    arguments = parser.parse_args()
    print(json.dumps(evaluate(live=arguments.live, holdout=arguments.holdout, model=arguments.model), indent=2))
