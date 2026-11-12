"""Adapt semantic extraction to canonical documents without legal conclusions."""
from src.core.state import AgentState
from src.document_processing.processor import process_document

TYPE_MAP = {"student_enrollment_certificate": "student_certificate",
            "health_insurance_certificate": "health_insurance",
            "ikamet_application_form": "ikamet_application_form", "unknown": "unknown"}


def document_analyzer(state: AgentState) -> dict:
    documents, facts = [], {}
    for doc in state.get("documents", []):
        if doc.status != "pending": continue
        try:
            info = process_document(doc.file_path)
        except (OSError, ValueError, RuntimeError) as error:
            documents.append(doc.model_copy(update={"status": "rejected", "processing_error": str(error)}))
            continue
        documents.append(doc.model_copy(update={"status": "analyzed", "doc_type": TYPE_MAP[info.document_type],
            "issue_date": getattr(info, "issue_date", None),
            "expiry_date": getattr(info, "coverage_end_date", None), "info": info}))
        # Namespaced observations retain provenance and uncertainty. They do not
        # set legal context booleans or assume issue_date means coverage_start.
        facts[doc.doc_id] = info.model_dump(mode="json")
    return {"documents": documents, "extracted_facts": facts, "relevant_requirements": None,
            "candidate_plan": None}
