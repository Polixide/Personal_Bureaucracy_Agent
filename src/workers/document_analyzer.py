"""Bridge the standalone document processor to the existing graph state."""

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT
from processor import DocumentProcessingError, process_document
from src.core.state import AgentState, ValidationIssue


DOCUMENT_TYPES = {
    "student_enrollment_certificate": "student_certificate",
    "health_insurance_certificate": "health_insurance",
}


def document_analyzer(state: AgentState, config: RunnableConfig) -> dict:
    """Analyze pending files, retaining facts and rejecting uncertain results.

    Optional configurable keys: document_model, document_host, document_timeout.
    Analyzed means extraction passed quality checks, not official approval.
    """
    settings = config.get("configurable", {})
    updated, facts = [], {}
    issues = list(state.get("validation_errors", []))
    for document in state.get("documents", []):
        if document.status != "pending":
            continue
        changes = dict(status="rejected", doc_type=None, issue_date=None, expiry_date=None)
        try:
            result = process_document(
                document.file_path,
                model=settings.get("document_model", DEFAULT_OLLAMA_MODEL),
                host=settings.get("document_host", DEFAULT_OLLAMA_HOST),
                timeout=settings.get("document_timeout", DEFAULT_OLLAMA_TIMEOUT),
            )
            facts[document.doc_id] = result.model_dump(mode="json")
            doc_type = DOCUMENT_TYPES.get(result.document_type)
            uncertain = (not doc_type or result.missing_fields or result.ambiguous_fields
                         or result.validation_issues)
            changes.update(doc_type=doc_type, issue_date=result.issue_date,
                           expiry_date=result.expiry_date)
            if uncertain:
                issues.append(ValidationIssue(
                    code="SCHEMA_ERROR", source="document_analyzer", item=document.doc_id,
                    message=f"Document needs review: missing={result.missing_fields}, "
                            f"ambiguous={result.ambiguous_fields}, issues={result.validation_issues}",
                ))
            else:
                changes["status"] = "analyzed"
        except (OSError, ValueError, DocumentProcessingError) as error:
            code = "SCHEMA_ERROR" if isinstance(error, (ValidationError, DocumentProcessingError)) else "INVALID_INPUT"
            issues.append(ValidationIssue(
                code=code, source="document_analyzer", item=document.doc_id,
                message=f"Document extraction failed: {error}",
            ))
            # Replace any facts from an earlier upload with the same document ID.
            facts[document.doc_id] = None
        updated.append(document.model_copy(update=changes))
    return {"documents": updated, "extracted_facts": facts, "validation_errors": issues}
