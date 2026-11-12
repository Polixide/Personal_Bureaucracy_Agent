"""Stub nodes for Milestone 1: deterministic mock responses.

Every stub already has its final signature: it receives the whole AgentState
and returns ONLY the fields it changes. Each owner replaces a stub with the real
implementation (same name, same inputs/outputs) in the module named in its
docstring, then updates the import in graph.py. The wiring does not change.
"""
from datetime import date
from pathlib import Path

from src.core.state import AgentState, Document, PlanStep, Requirement, ValidationIssue

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Canned Document Analyzer output, keyed by file name.
MOCK_EXTRACTIONS = {
    "insurance_valid.pdf": {"doc_type": "health_insurance",
                            "issue_date": date(2026, 11, 1), "expiry_date": date(2027, 11, 1)},
    "insurance_expiring.pdf": {"doc_type": "health_insurance",
                               "issue_date": date(2025, 11, 10), "expiry_date": date(2026, 11, 10)},
    "student_certificate.pdf": {"doc_type": "student_certificate", "issue_date": date(2026, 10, 15)},
}

# Mock rule base: document types required by each procedure.
MOCK_REQUIRED_DOCS = {"ikamet": ["health_insurance", "student_certificate"]}


def gate_0(state: AgentState) -> dict:
    """Zero-token input check -> src/guardrails/gate_0.py.

    On PASS an uploaded file is registered in the Digital Twin as 'pending'.
    """
    event = state["event"]
    if event.kind == "upload":
        ok = bool(event.file_path) and Path(event.file_path).suffix.lower() in ALLOWED_EXTENSIONS
    else:
        ok = bool(event.text.strip())
    if not ok:
        issue = ValidationIssue(code="INVALID_INPUT", source="gate_0", item=event.file_path,
                                message=f"Rejected {event.kind}: empty or unsupported input.")
        return {"validation_errors": [issue]}
    if event.kind == "upload":
        return {"documents": [Document(doc_id=Path(event.file_path).name, file_path=event.file_path)]}
    return {}


def document_analyzer(state: AgentState) -> dict:
    """Mock of parser + LLM + Pydantic extraction -> src/workers/document_analyzer.py."""
    updated = []
    for doc in state["documents"]:
        if doc.status == "pending":
            fields = MOCK_EXTRACTIONS.get(doc.doc_id)
            changes = {**fields, "status": "analyzed"} if fields else {"status": "rejected"}
            updated.append(doc.model_copy(update=changes))
    return {"documents": updated}


def rules_engine(state: AgentState) -> dict:
    """Mock of the deterministic Requirements Engine -> src/workers/rules_engine.py."""
    available = {d.doc_type for d in state.get("documents", []) if d.status == "analyzed"}
    requirements = [
        Requirement(rule_id=f"REQ_{doc_type.upper()}",
                    description=f"Provide a valid {doc_type.replace('_', ' ')}",
                    satisfied=doc_type in available,
                    source_id="mock_rule_base")
        for doc_type in MOCK_REQUIRED_DOCS[state["active_procedure"]]
    ]
    return {"relevant_requirements": requirements}


def planner(state: AgentState) -> dict:
    """Mock of the Action / Deadline Planner -> src/agents/planner.py."""
    # The connected R014 demo produces no claims about the other rules.
    if all(r.rule_id == "R014" for r in state["relevant_requirements"]):
        return {"candidate_plan": [PlanStep(
            step_id="review_r014_demo", action="[MOCK planner] Review the R014 certificate result") ]}
    steps = [PlanStep(step_id=f"obtain_{r.rule_id.lower()}", action=r.description)
             for r in state["relevant_requirements"] if not r.satisfied]
    steps.append(PlanStep(
        step_id="attend_appointment",
        action="Attend the residence-permit appointment",
        due_date=state.get("extracted_facts", {}).get("appointment_date"),
        depends_on=[s.step_id for s in steps],
        required_documents=MOCK_REQUIRED_DOCS[state["active_procedure"]],
        requires_approval=True,
    ))
    return {"candidate_plan": steps}


def gate_1(state: AgentState) -> dict:
    """Deterministic plan validator -> src/guardrails/gate_1.py.

    The stub implements one check: every document a step needs must still be
    valid on the date of that step.
    """
    # Keep analyzer failures so the existing FAIL route bypasses approval.
    if state.get("validation_errors"):
        return {"validation_errors": state["validation_errors"]}
    expiry = {d.doc_type: d.expiry_date for d in state.get("documents", [])
              if d.status == "analyzed" and d.expiry_date}
    errors = []
    for step in state["candidate_plan"]:
        for doc_type in step.required_documents:
            if step.due_date and doc_type in expiry and expiry[doc_type] < step.due_date:
                errors.append(ValidationIssue(
                    code="DATE_CONFLICT", source="gate_1", item=step.step_id,
                    message=f"{doc_type} expires on {expiry[doc_type]}, "
                            f"before '{step.action}' on {step.due_date}."))
    return {"validation_errors": errors}


def human_approval(state: AgentState) -> dict:
    """Stub: auto-approves the plan.

    Real version: langgraph.types.interrupt() pauses the graph until the user
    confirms (needs a checkpointer); on refusal it returns an APPROVAL_DENIED issue.
    """
    return {}


def state_update(state: AgentState) -> dict:
    """Deterministic write-back to the Digital Twin -> SQLite layer (src/memory/checkpointer.py)."""
    return {"blockers": [r.rule_id for r in state["relevant_requirements"] if not r.satisfied]}
