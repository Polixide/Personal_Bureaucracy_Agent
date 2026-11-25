"""How well uploads are read, on synthetic documents with known values (eval/document_samples.py).

    python -m eval.document_eval --label "qwen3:4b" --out docs/eval_documents.md

For each file: were the right documents found (and split), and for each expected value: read correctly,
read wrong, or not read. "Wrong" is the number that must stay at zero: a missing value is asked to the user,
a wrong one is believed. Needs Ollama running; the model comes from config (OLLAMA_MODEL). With
DOCUMENT_PROVIDER=cloud and LLM_PROVIDER=openai (or anthropic) the cloud model reads the files instead (it costs
a few cents per run):

    $env:LLM_PROVIDER="openai"; $env:DOCUMENT_PROVIDER="cloud"; $env:OPENAI_MODEL="gpt-6.1-terra"
    python -m eval.document_eval --label "gpt-6.1-terra" --out docs/eval_documents.md
"""
import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel

from eval.document_samples import ExpectedDocument, Sample, synthetic_samples
from src.document_processing.evidence import normalized
from src.document_processing.observations import DocumentObservation
from src.llm import document_backend
from src.pack import load_pack
from src.tools.extract_document import extract_file


class SampleScore(BaseModel):
    sample_id: str
    kind: str
    documents_expected: int
    documents_found: int
    extra_documents: int
    correct: int
    to_confirm: int          # of the correct values, those the user is asked to confirm
    wrong: int
    wrong_to_confirm: int = 0   # of the wrong values, those shown to the user for confirmation (not believed)
    missing: int
    seconds: float
    error: str = ""
    wrong_values: list[str] = []


def comparable(value) -> str:
    """Dates as ISO text, booleans as true/false, text without case, accents or extra spaces."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return str(value).lower()
    return " ".join(normalized(str(value)).split())


def facts_of(observation: DocumentObservation) -> dict[str, tuple[str, bool]]:
    """field -> (value, needs the user's confirmation)."""
    return {fact.field: (comparable(fact.value), fact.requires_confirmation) for fact in observation.facts}


def score_document(expected: ExpectedDocument, found: dict[str, tuple[str, bool]]) -> tuple[int, int, int, int, int, list[str]]:
    """(correct, of which to confirm, wrong, of which to confirm, missing, wrong values) for one expected document."""
    correct = to_confirm = wrong = wrong_to_confirm = missing = 0
    wrong_values = []
    for field, value in expected.facts.items():
        if field not in found:
            missing += 1
        elif found[field][0] == comparable(value):
            correct += 1
            to_confirm += found[field][1]
        else:
            wrong += 1
            wrong_to_confirm += found[field][1]
            shown = " (shown for confirmation)" if found[field][1] else " (believed)"
            wrong_values.append(f"{field}: read {found[field][0]!r}, expected {comparable(value)!r}{shown}")
    return correct, to_confirm, wrong, wrong_to_confirm, missing, wrong_values


def score_sample(sample: Sample, observations: list[DocumentObservation], seconds: float, error: str = "") -> SampleScore:
    """Match each expected document to an observation of the same type, then count the values."""
    unused = list(observations)
    found_documents = correct = to_confirm = wrong = wrong_to_confirm = missing = 0
    wrong_values: list[str] = []
    for expected in sample.expected:
        match = next((obs for obs in unused if obs.doc_type == expected.doc_type), None)
        facts = {}
        if match is not None:
            unused.remove(match)
            found_documents += 1
            facts = facts_of(match)
        c, k, w, wk, m, details = score_document(expected, facts)
        correct, to_confirm, wrong, missing = correct + c, to_confirm + k, wrong + w, missing + m
        wrong_to_confirm += wk
        wrong_values += details
    return SampleScore(sample_id=sample.sample_id, kind=sample.kind, documents_expected=len(sample.expected),
                       documents_found=found_documents, extra_documents=len(unused), correct=correct,
                       to_confirm=to_confirm, wrong=wrong, wrong_to_confirm=wrong_to_confirm,
                       missing=missing, seconds=round(seconds, 1), error=error, wrong_values=wrong_values)


def run_sample(sample: Sample, backend) -> SampleScore:
    pack = load_pack(sample.procedure_id)
    started = perf_counter()
    result = extract_file(sample.file, pack, backend=backend)
    return score_sample(sample, result.observations, perf_counter() - started, result.error or "")


def totals(scores: list[SampleScore]) -> dict[str, float]:
    values = sum(s.correct + s.wrong + s.missing for s in scores) or 1
    return {"documents found": f"{sum(s.documents_found for s in scores)}/{sum(s.documents_expected for s in scores)}",
            "values read correctly": round(sum(s.correct for s in scores) / values, 3),
            "of which to confirm": sum(s.to_confirm for s in scores),
            "values read wrong": sum(s.wrong for s in scores),
            "of which believed (not shown for confirmation)": sum(s.wrong - s.wrong_to_confirm for s in scores),
            "values not read": sum(s.missing for s in scores),
            "seconds per file": round(sum(s.seconds for s in scores) / max(len(scores), 1), 1)}


def table(label: str, scores: list[SampleScore]) -> str:
    """Markdown section: one row per file, then the totals."""
    lines = [f"## {label}", "", "| file | kind | documents | correct (to confirm) | wrong | not read | seconds |",
             "|---|---|---|---|---|---|---|"]
    for s in scores:
        docs = f"{s.documents_found}/{s.documents_expected}" + (f" (+{s.extra_documents} extra)" if s.extra_documents else "")
        note = f" ({s.error[:60]})" if s.error else ""
        correct = f"{s.correct} ({s.to_confirm})" if s.to_confirm else str(s.correct)
        lines.append(f"| {s.sample_id}{note} | {s.kind} | {docs} | {correct} | {s.wrong} | {s.missing} | {s.seconds} |")
    lines += ["", " · ".join(f"{name}: {value}" for name, value in totals(scores).items())]
    wrong = [f"- {s.sample_id}: {detail}" for s in scores for detail in s.wrong_values]
    if wrong:
        lines += ["", "Values read wrong:"] + wrong
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="current code")
    parser.add_argument("--out", type=Path, default=None, help="append the markdown section to this file")
    parser.add_argument("--only", nargs="*", default=None, help="sample ids to run")
    args = parser.parse_args()
    backend = document_backend()          # Ollama, or the cloud model with DOCUMENT_PROVIDER=cloud
    with tempfile.TemporaryDirectory() as folder:
        samples = [s for s in synthetic_samples(Path(folder)) if not args.only or s.sample_id in args.only]
        scores = []
        for sample in samples:
            score = run_sample(sample, backend)
            print(json.dumps(score.model_dump(), ensure_ascii=False), flush=True)
            scores.append(score)
    section = table(args.label, scores)
    print(section)
    if args.out:
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write("\n" + section)


if __name__ == "__main__":
    sys.exit(main())
