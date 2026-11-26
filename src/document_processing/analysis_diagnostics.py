"""Content-free metadata for one document attempt and its provider requests."""


def summarize_document_analysis(info, calls, *, cloud, seconds, failed=None):
    diagnostics = info.extraction_diagnostics if info is not None else getattr(failed, "diagnostics", {})
    cache_hit = diagnostics.get("cache_hit", False)
    requests = [
        {
            "task": call.task,
            "provider": call.provider,
            "requested_model": call.model,
            "responding_model": call.response_model,
            "request_success": call.request_success,
            "outcome": call.error_code or ("success" if call.request_success else "timeout" if call.timed_out else "failed"),
            "image_count": call.image_count if call.provider == "openai" else 0,
            "images": call.image_inputs or [],
            "text_pages": call.document_text_pages,
            "partial_text_pages": call.document_partial_text_pages,
            "input_tokens": call.prompt_tokens,
            "output_tokens": call.completion_tokens,
            "total_tokens": call.total_tokens,
            "latency_seconds": call.wall_seconds,
            "structured_output_valid": call.structured_output_valid,
            "estimated_cost_usd": call.estimated_cost_usd,
        } for call in calls
    ]
    responded = [row for row in requests if row["request_success"]]
    submitted_pages = sorted({image["page"] for row in requests for image in row["images"]})
    page_count = diagnostics.get("pdf_page_count")
    local_ocr = diagnostics.get("ocr_used")
    if any(row["provider"] == "openai" and row["image_count"] for row in requests):
        mode = "cloud vision"
    elif any(row["provider"] == "openai" for row in requests):
        mode = "cloud text"
    elif local_ocr:
        mode = "local OCR"
    elif info is not None:
        mode = "native PDF text" if page_count is not None else "native text"
    else:
        mode = None

    def distinct(rows, key):
        return ", ".join(dict.fromkeys(row[key] for row in rows if row[key])) or None

    def total(key):
        values = [row[key] for row in requests]
        return sum(values) if values and all(value is not None for value in values) else None

    return {
        "outcome": "failed" if failed is not None else "analyzed",
        "selected_provider": "openai" if cloud else "local",
        "provider_responded": distinct(responded, "provider") or ("local" if info is not None and not cache_hit and not requests else None),
        "requested_model": distinct(requests, "requested_model") or (diagnostics.get("model") if cloud else None),
        "responding_model": distinct(responded, "responding_model"),
        "processing_mode": mode,
        "pdf_page_count": page_count,
        "native_text_pages": [row["page_number"] for row in diagnostics["page_quality"]
                              if row["native_text_character_count"] > 0] if "page_quality" in diagnostics else None,
        "ocr_used": local_ocr,
        "evidence_pages": info.selected_pages if info is not None else None,
        "image_pages": submitted_pages,
        "pdf_pages_without_images": (sorted(set(range(1, page_count + 1)) - set(submitted_pages))
                                    if page_count is not None else None),
        "image_count": sum(row["image_count"] or 0 for row in requests),
        "pdf_input": False,  # The current transport uses page PNGs and text.
        "validated_extraction": info is not None,
        "classification": diagnostics.get("classification"),
        "fields_not_found_in_processed_source": diagnostics.get("fields_not_found_in_processed_source"),
        "validation_issue_count": len(info.validation_issues) if info is not None else None,
        "validation_issue_fields": sorted(
            name for name in info.observation_fields() if any(issue.startswith(name) for issue in info.validation_issues)
        ) if info is not None else None,
        "missing_fields": info.missing_fields if info is not None else None,
        "ambiguous_fields": info.ambiguous_fields if info is not None else None,
        "cache_hit": cache_hit,
        "reused_result_provider": (
            diagnostics.get("provider") if diagnostics.get("model_calls") else "local"
        ) if cache_hit else None,
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "total_tokens": total("total_tokens"),
        "estimated_cost_usd": total("estimated_cost_usd"),
        "failure_stage": getattr(failed, "stage", None) or ("document_processing" if failed is not None else None),
        "failure_category": getattr(failed, "category", None),
        "failure_categories": diagnostics.get("failure_categories", []),
        "fallback_reasons": diagnostics.get("fallback_trigger_reasons", []),
        "fallback_outcome": diagnostics.get("fallback_outcome") if diagnostics.get("fallback_attempted") else None,
        "latency_seconds": seconds,
        "requests": requests,
    }

def document_analysis_findings(info, summary):
    """Interpret persisted safe codes for both current and older readings."""
    details = info.extraction_diagnostics if info is not None else {}
    classification = summary.get("classification") or details.get("classification") or {}
    category = summary.get("failure_category")
    stage = summary.get("failure_stage")
    findings = []
    if summary.get("outcome") == "failed":
        if category == "timeout":
            findings.append("API timeout")
        elif category == "connection":
            findings.append("API connection error")
        elif category in {"invalid_output", "SCHEMA_VALIDATION_FAILURE"}:
            findings.append("Validation or schema rejection")
        elif stage in {"text_extraction", "transcription", "rendering"} and category in {
            "OCR_FAILURE", "TEXT_EXTRACTION_FAILURE", None,
        }:
            findings.append("OCR or Vision extraction failure")
        elif category == "CLASSIFICATION_FAILURE" or stage == "classification" and not category:
            findings.append("Classification failed")
        elif summary.get("selected_provider") == "openai":
            findings.append("OpenAI request or configuration failed")
        else:
            findings.append("Document analysis failed")
    elif info is not None and info.document_type == "unknown":
        if classification.get("proposed_type") not in {None, "unknown"}:
            findings.append("Classification rejected by source validation")
        elif "TEXT_EXTRACTION_FAILURE" in summary.get("failure_categories", ()):
            findings.append("OCR or Vision extraction failure")
        else:
            findings.append("Unsupported document type")
    elif info is not None:
        if info.missing_fields:
            findings.append("Missing required information")
        if set(summary.get("failure_categories", ())) & {"SCHEMA_VALIDATION_FAILURE", "GROUNDING_FAILURE"}:
            findings.append("Some readings failed field validation")
        if summary.get("fields_not_found_in_processed_source"):
            findings.append("Extraction completed with fields not found in the processed source")
        if not findings:
            findings.append("Extraction completed")
    return classification or None, findings
