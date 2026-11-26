"""Identity-vs-attachments checks: explicit mock/live modes, synthetic sources only."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ollama import Client

from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT
from document_classifier import classify_document

ROOT = Path(__file__).resolve().parents[1] / "data/synthetic_samples/classification_identity"
EXPECTED = {
    "application_attachments": "ikamet_application_form",
    "student_requirements": "student_enrollment_certificate",
    "insurance_legislation": "health_insurance_certificate",
    "checklist": "unknown",
    "insurance_misleading": "health_insurance_certificate",
    "unknown": "unknown",
    "conflicting_titles": "unknown",
    "semantic_student": "student_enrollment_certificate",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--timeout", type=float, default=DEFAULT_OLLAMA_TIMEOUT)
    args = parser.parse_args()
    client = Client(host=args.host, timeout=args.timeout)
    results = []
    for name, expected in EXPECTED.items():
        text = (ROOT / (name + ".txt")).read_text(encoding="utf-8")
        reply = ({"document_type": expected, "evidence": [text.splitlines()[-1]]}
                 if name == "semantic_student" else {"document_type": "unknown"})
        mock = nullcontext() if args.live else patch.object(client, "chat", return_value=SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(reply))))
        with mock:
            result = classify_document(text, client, args.model)
        results.append({"case": name, "expected": expected, "correct": result.document_type == expected,
                        **result.model_dump(mode="json")})
    print(json.dumps({"mode": "LIVE classifier; deterministic fast paths do not call Qwen" if args.live else "MOCKED fallback",
                      "correct": sum(item["correct"] for item in results), "total": len(results), "results": results}, indent=2))
    if not all(item["correct"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
