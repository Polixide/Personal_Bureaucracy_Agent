"""Adapt semantic extraction to canonical documents without legal conclusions."""
from src.core.state import AgentState
from src.document_processing.processor import cached_process_document as process_document

TYPE_MAP = {"student_enrollment_certificate": "student_certificate",
            "health_insurance_certificate": "health_insurance",
            "ikamet_application_form": "ikamet_application_form", "passport_identity_page":"passport_copy", "turkiye_entry_stamp":"turkiye_entry_stamp", "declaration_control_document":"declaration_control_document", "card_fee_receipt":"card_fee_receipt", "utility_bill":"utility_bill", "address_form":"address_form", "unknown": "unknown"}


def document_analyzer(state: AgentState, *, processor=None) -> dict:
    # Optional processor callable keeps the same typed observation contract;
    # document-provider selection/fallback stays inside the canonical processor.
    analyze = processor or process_document
    documents, facts = [], {}
    existing=state.get("documents", [])
    for doc in existing:
        if doc.status != "pending": continue
        try:
            hint=state.get("event").document_type_hint if state.get("event") and state["event"].kind=="reprocess_document" and state["event"].document_id==doc.doc_id else None
            info = analyze(doc.file_path,document_type_hint=hint) if hint else analyze(doc.file_path)
        except (OSError, ValueError, RuntimeError) as error:
            documents.append(doc.model_copy(update={"status": "rejected", "processing_error": str(error)}))
            continue
        infos=[info]
        if info.document_type in {"passport_identity_page","turkiye_entry_stamp"} and str(doc.file_path).lower().endswith(".pdf"):
            from src.document_processing.processor import process_passport_pages
            try: infos=process_passport_pages(doc.file_path,info)
            except (OSError,ValueError,RuntimeError) as error:
                documents.append(doc.model_copy(update={"status":"rejected","processing_error":str(error)}));continue
        for index,observation in enumerate(infos):
            doc_id=doc.doc_id if index==0 else doc.doc_id+"_entry"
            kind=TYPE_MAP[observation.document_type]
            duplicate=any(d.doc_id!=doc_id and d.doc_type==kind and d.status=="analyzed" for d in existing+documents)
            status="replacement_pending" if duplicate else "analyzed"
            documents.append(doc.model_copy(update={"doc_id":doc_id,"status":status,"doc_type":kind,
                "issue_date":getattr(observation,"issue_date",None),
                "expiry_date":getattr(observation,"coverage_end_date",getattr(observation,"expiry_date",None)),"info":observation,"processing_error":None}))
            facts[doc_id]=observation.model_dump(mode="json")
    return {"documents": documents, "extracted_facts": facts, "relevant_requirements": None,
            "candidate_plan": None}
