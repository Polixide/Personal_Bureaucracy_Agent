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
        context = ResidencePermitContext.model_validate(state.get("residence_permit_context", {}))
        deadlines = TypeAdapter(dict[str, date]).validate_python(state.get("deadlines", {}))
    except (ValueError, KeyError, TypeError) as error:
        return reject(str(error))
    if state.get("active_procedure", "ikamet") != "ikamet": return reject("Unsupported procedure")
    if (context.institution not in (None, "Sabanc\u0131 University") or context.applicant_type not in (None, "incoming_exchange")
        or context.application_kind not in (None, "first") or context.permit_type not in (None, "short_term")
        or (context.age is not None and context.age < 18)
        or context.exchange_student is False or (context.planned_stay_months is not None and context.planned_stay_months < 3)):
        return reject("Outside Sabanc\u0131 adult incoming exchange first short-term application scope")
    result = {"event": event, "residence_permit_context": context, "deadlines": deadlines}
    if event.kind == "message":
        return result if event.text.strip() else reject("Empty message")
    path = Path(event.file_path or "").resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
        return reject("Upload must be an existing local PDF/TXT inside the project")
    result["documents"] = [Document(doc_id=sha256(str(path).encode()).hexdigest()[:20], file_path=str(path))]
    return result
