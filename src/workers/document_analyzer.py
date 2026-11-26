"""Adapt semantic extraction to canonical documents without legal conclusions."""
from time import perf_counter

from src.core.llm_diagnostics import collect_model_calls
from src.core.state import AgentState, VerifiedInput
from src.core.model_provider import ai_options
from src.document_processing.analysis_diagnostics import summarize_document_analysis
from src.document_processing.providers import DocumentProcessingError
from src.document_processing.processor import cached_process_document as process_document

_cached_analyze = process_document

TYPE_MAP = {"student_enrollment_certificate": "student_certificate",
            "health_insurance_certificate": "health_insurance",
            "ikamet_application_form": "ikamet_application_form", "passport_identity_page":"passport_copy", "turkiye_entry_stamp":"turkiye_entry_stamp", "declaration_control_document":"declaration_control_document", "card_fee_receipt":"card_fee_receipt", "utility_bill":"utility_bill", "address_form":"address_form", "tax_id_certificate":"tax_id_certificate", "unknown": "unknown"}


from src.document_processing.schemas import ADDRESS_EVIDENCE_SCHEMAS

TYPE_MAP.update({kind: kind for kind in ADDRESS_EVIDENCE_SCHEMAS})


def document_analyzer(state: AgentState, *, processor=None) -> dict:
    # Optional processor callable keeps the same typed observation contract;
    # document-provider selection/fallback stays inside the canonical processor.
    analyze = processor or process_document
    documents, facts = [], {}
    existing=state.get("documents", [])
    records = dict(state.get("verification_records", {}))
    event = state.get("event")
    for doc in existing:
        if doc.status != "pending": continue
        reprocess = event and event.kind == "reprocess_document" and event.document_id == doc.doc_id
        hint = event.document_type_hint if reprocess else None
        cloud_pages = None
        options = {"document_type_hint": hint} if hint else {}
        if processor is None and analyze is _cached_analyze:
            options["cache_scope"] = state.get("user_id")
            permitted = ai_options().cloud_document
            target = (
                ("reprocess_document", doc.doc_id) if reprocess else
                ("analyze_document", doc.file_path) if event and event.kind == "analyze_document"
                and event.file_path == doc.file_path else None
            )
            if permitted and target == permitted:
                cloud_pages = []
                options.update(mode="cloud", prefer_native=True, page_sink=cloud_pages)
        started = perf_counter()
        with collect_model_calls(propagate=True) as calls:
            try:
                info = analyze(doc.file_path, **options)
                infos = [info]
                if info.document_type in {"passport_identity_page", "turkiye_entry_stamp"} and str(doc.file_path).lower().endswith(".pdf"):
                    from src.document_processing.processor import process_passport_pages

                    if cloud_pages is None:
                        infos = process_passport_pages(doc.file_path, info)
                    else:
                        infos = process_passport_pages(doc.file_path, info, pages=cloud_pages, mode="cloud")
            except (OSError, ValueError, RuntimeError) as error:
                analysis = summarize_document_analysis(None, calls, cloud=cloud_pages is not None,
                                                       seconds=perf_counter() - started, failed=error)
                message = (str(error) if isinstance(error, DocumentProcessingError) else
                           "Document analysis failed. Try reprocessing or upload a clearer file.")
                documents.append(doc.model_copy(update={"status": "rejected", "processing_error": message,
                                                        "analysis_diagnostics": analysis}))
                continue
        analysis = summarize_document_analysis(info, calls, cloud=cloud_pages is not None,
                                               seconds=perf_counter() - started)
        prior = (doc.info.model_dump(mode="json", include=doc.info.observation_fields(), exclude_none=True)
                 if reprocess and doc.info is not None else {})
        if reprocess and len(infos) > 1:
            # Reprocessing a logical document in a bundle must leave its companion alone.
            infos = [next((part for part in infos if part.document_type == info.document_type), info)]
        for index,observation in enumerate(infos):
            doc_id=doc.doc_id if index==0 else doc.doc_id+"_entry"
            kind=TYPE_MAP[observation.document_type]
            if reprocess and doc_id in records:
                # Rebuild observations from the new reading. A correction remains
                # authoritative only for the same logical document and type.
                if doc.doc_type == kind:
                    records[doc_id] = {
                        name: entry for name, entry in records[doc_id].items()
                        if VerifiedInput.model_validate(entry).origin == "user_confirmed"
                    }
                else:
                    records.pop(doc_id)
            duplicate=kind not in {"unknown", "host_support"} and any(d.doc_id!=doc_id and d.doc_type==kind and d.status=="analyzed" for d in existing+documents)
            status="replacement_pending" if duplicate else "analyzed"
            documents.append(doc.model_copy(update={"doc_id":doc_id,"status":status,"doc_type":kind,
                "issue_date":getattr(observation,"issue_date",None),
                "expiry_date":getattr(observation,"coverage_end_date",getattr(observation,"expiry_date",None)),"info":observation,"processing_error":None,
                "analysis_diagnostics": analysis, "prior_observations": prior}))
            facts[doc_id]=observation.model_dump(mode="json")
    updates = {"documents": documents, "extracted_facts": facts, "relevant_requirements": None,
               "candidate_plan": None}
    if records != state.get("verification_records", {}):
        updates["verification_records"] = records
    return updates
