"""Deterministic input and local-path checks."""
from datetime import date
from pydantic import TypeAdapter
from hashlib import sha256
from pathlib import Path
from config import PROJECT_ROOT
from src.core.state import Document, ResidencePermitContext, UserEvent, ValidationIssue


def gate_0(state):
    def reject(message):
        return {"validation_errors": [ValidationIssue(code="INVALID_INPUT", source="gate_0", message=message)], "run_status": "INVALID_INPUT"}
    try:
        event = UserEvent.model_validate(state["event"])
        context_input = state.get("residence_permit_context", {})
        # Revalidate older typed checkpoints through the current input boundary.
        context = ResidencePermitContext.model_validate(context_input.model_dump() if isinstance(context_input, ResidencePermitContext) else context_input)
        deadlines = TypeAdapter(dict[str, date]).validate_python(state.get("deadlines", {}))
    except (ValueError, KeyError, TypeError) as error:
        return reject(str(error))
    if state.get("active_procedure", "ikamet") != "ikamet": return reject("Unsupported procedure")
    if event.kind not in {"analyze_document", "confirm_information", "resolve_conflict", "record_progress", "reevaluate", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction"} and (context.institution not in (None, "Sabanc\u0131 University") or context.applicant_type not in (None, "incoming_exchange")
        or context.application_kind not in (None, "first") or context.permit_type not in (None, "short_term")
        or (context.age is not None and context.age < 18)
        or context.exchange_student is False or (context.planned_stay_months is not None and context.planned_stay_months < 3)):
        return reject("Outside Sabanc\u0131 adult incoming exchange first short-term application scope")
    result = {"event": event, "residence_permit_context": context, "deadlines": deadlines}
    if event.kind in {"delete_document","replace_document","cancel_document","reprocess_document","remove_correction"}:
        from src.workers.verification import document_lifecycle
        try: return {**result,**document_lifecycle(state)}
        except ValueError as error: return reject(str(error))
    if event.kind == "assistant_message":
        return result if event.text.strip() else reject("Empty assistant message")
    if event.kind == "run_tool":
        from src.tools.schemas import ToolCall
        try: ToolCall.model_validate(event.tool_call)
        except (ValueError, TypeError) as error: return reject("Invalid typed tool call")
        return result
    if event.kind == "generate_plan":
        from src.core.address_proof import address_proof_status
        from src.workers.verification import conflict_details

        recorded = next((rule for rule in state.get("relevant_requirements") or [] if rule.rule_id == "R017"), None)
        stale_address = recorded is not None and (recorded.passed is True) != address_proof_status(state)["ready"]
        if stale_address or state.get("verification_conflicts") or any(not detail["resolved"] for detail in conflict_details(state).values()):
            result["relevant_requirements"] = None
        return result
    if event.kind in {"resolve_conflict", "record_progress", "reevaluate", "delete_document", "replace_document", "cancel_document", "reprocess_document", "remove_correction"}:
        return result
    if event.kind == "confirm_information":
        if not event.document_id or not event.confirmed_fields:
            return reject("Choose a document and explicitly confirm at least one field")
        return result
    if event.kind == "message":
        return result if event.text.strip() else reject("Empty message")
    path = Path(event.file_path or "").resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
        return reject("Upload must be an existing local PDF/TXT inside the project")
    result["documents"] = [Document(doc_id=sha256(str(path).encode()).hexdigest()[:20], file_path=str(path), original_filename=Path(event.original_filename).name if event.original_filename else None)]
    return result
