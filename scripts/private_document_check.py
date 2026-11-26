"""Read-only private evaluation: output counts/field names, never extracted values."""

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from ollama import Client

from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL
from document_context import select_context
from document_evidence import snippet_in_source
from document_identity import identity_evidence
from document_profiles import REGISTRY
from extractor import extract_text
from processor import process_document


def check(path: Path, *, model: str, host: str, timeout: float) -> dict:
    started = time.perf_counter()
    text = extract_text(path)
    pages = text.split("\n\f\n")
    identity = identity_evidence(text)
    calls = []
    selections = []
    original = Client.chat
    def tracked(client, **request):
        entry = {"stage": request["format"].get("title"), "think": request.get("think"),
                 "input_characters": sum(len(item["content"]) for item in request["messages"] if item["role"] == "user"),
                 "num_ctx": request.get("options", {}).get("num_ctx")}
        calls.append(entry)
        call_started = time.perf_counter()
        try:
            response = original(client, **request)
            entry["prompt_tokens_evaluated"] = getattr(response, "prompt_eval_count", None)
            entry["output_tokens"] = getattr(response, "eval_count", None)
            reason = getattr(response, "done_reason", None)
            entry["stop_reason"] = reason if reason in {"stop", "length", "load", "unload"} else "other"
            if entry["stage"] == "DocumentClassification":
                try:
                    raw = json.loads(response.message.content or "{}")
                    kind = raw.get("document_type")
                    entry["raw_type"] = kind if kind in REGISTRY else "invalid_or_absent"
                    entry["raw_ambiguous"] = raw.get("ambiguous") if isinstance(raw.get("ambiguous"), bool) else None
                    quotes = raw.get("evidence", [])
                    entry["raw_evidence_count"] = len(quotes) if isinstance(quotes, list) else 0
                    entry["raw_quotes_source_backed"] = bool(quotes) and all(
                        isinstance(quote, str) and snippet_in_source(quote, text) for quote in quotes)
                except (ValueError, TypeError):
                    entry["raw_type"] = "invalid_response"
            else:
                try:
                    raw = json.loads(response.message.content or "{}")
                    allowed = {field for profile in REGISTRY.values() for field in profile.schema.model_fields}
                    entry["raw_field_presence"] = {field: "null" if value is None else "present"
                        for field, value in raw.items() if field in allowed and field not in {"source_snippets", "ambiguous_fields"}}
                except (ValueError, TypeError):
                    entry["raw_field_presence"] = "invalid_response"
            return response
        except Exception as error:
            entry["error_type"] = type(error).__name__
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - call_started, 2)
    report = {"pages": len(pages), "characters_per_page": [len(page) for page in pages],
              "deterministic_primary_types": identity["types"],
              "primary_roles": sorted({item["role"] for item in identity["primary"]}),
              "other_identity_heading_count": len(identity["other_identity"]),
              "calls": calls, "context_selections": selections}
    def selected(stage):
        def track(*args, **kwargs):
            context, metadata = select_context(*args, **kwargs)
            selections.append({"stage": stage, **metadata})
            return context, metadata
        return track
    try:
        with patch("processor.Client.chat", new=tracked), \
                patch("processor.select_context", new=selected("extraction")), \
                patch("document_classifier.select_context", new=selected("classification")):
            result = process_document(path, model=model, host=host, timeout=timeout)
        data = result.model_dump(mode="json")
        profile = REGISTRY[result.document_type]
        fields = [name for name in profile.schema.model_fields if name not in {"source_snippets", "ambiguous_fields"}]
        present = [name for name in fields if data.get(name) is not None]
        supported = [name for name in present if any(
            evidence.normalized_value == data[name] and evidence.field == name and
            snippet_in_source(evidence.snippet, text) for evidence in result.source_evidence)]
        if result.document_type == "health_insurance_certificate" and data.get("expiry_date") == data.get("coverage_end_date"):
            if "expiry_date" in present and "coverage_end_date" in supported and "expiry_date" not in supported:
                supported.append("expiry_date")
        report.update(document_type=result.document_type, method=result.classification.method,
            classification_ambiguous=result.classification.ambiguous,
            classification_quote_count=len(result.classification.evidence),
            classification_quotes_source_backed=bool(result.classification.evidence) and all(
                snippet_in_source(quote, text) for quote in result.classification.evidence),
            schema_selected=profile.schema.__name__ if result.document_type != "unknown" else None,
            applicable_field_count=len(fields), extracted_field_count=len(present), supported_field_count=len(supported),
            populated_field_names=present, unsupported_field_names=sorted(set(present) - set(supported)),
            missing_field_names=result.missing_fields, ambiguous_field_names=result.ambiguous_fields,
            validation_issue_count=len(result.validation_issues), correction_note_count=len(result.correction_notes),
            generic_evidence_count=len(result.source_evidence),
            manual_review=bool(result.validation_issues or result.ambiguous_fields or result.missing_fields),
            extraction_skipped_for_unknown=result.document_type == "unknown")
        if result.document_type != "unknown":
            report["classification_reason"] = "primary_identity_source_verified" if result.classification.method == "indicators" else "model_type_and_quotes_source_verified"
        elif identity["other_identity"]:
            report["classification_reason"] = "unsupported_primary_identity"
        elif len(identity["types"]) > 1:
            report["classification_reason"] = "conflicting_primary_identities"
        else:
            report["classification_reason"] = "model_or_evidence_insufficient"
        context_call = next((call for call in calls if call["stage"] == "DocumentClassification"), None)
        report["classifier_characters_sent"] = context_call["input_characters"] if context_call else 0
        selection = next((item for item in selections if item["stage"] == "classification"), {})
        report["classifier_selected_pages"] = selection.get("pages", []) if context_call else []
        report["deterministic_pages_inspected"] = list(range(1, len(pages) + 1))
        if result.document_type != "unknown":
            report["extraction_selected_context"] = next((item for item in selections if item["stage"] == "extraction"), {})
    except Exception as error:
        # ValidationError/error strings can contain personal input. Never print them.
        report["error_type"] = type(error).__name__
    report["total_seconds"] = round(time.perf_counter() - started, 2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    for index, path in enumerate(args.paths, 1):
        try:
            report = check(path, model=args.model, host=args.host, timeout=args.timeout)
        except Exception as error:
            report = {"error_type": type(error).__name__}
        print(json.dumps({"document_index": index, **report}), flush=True)
