"""R014-only demonstration: explicit Python checks, never JSON expression execution."""

import json
from pathlib import Path

from schemas import DocumentInfo
from src.core.state import AgentState, Requirement, ValidationIssue

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


def load_r014() -> tuple[dict, list[dict]]:
    """Load the existing rule and resolve every provenance reference locally."""
    rules = json.loads((KNOWLEDGE_DIR / "rules.json").read_text(encoding="utf-8"))
    sources = json.loads((KNOWLEDGE_DIR / "sources.json").read_text(encoding="utf-8"))
    rule = next(rule for rule in rules["rules"] if rule["rule_id"] == "R014")
    by_id = {source["source_id"]: source for source in sources["sources"]}
    return rule, [by_id[source_id] for source_id in rule["source_ids"]]


def evaluate_r014(state: AgentState) -> Requirement:
    """Accept only analyzed certificates with complete, unambiguous extraction.

    Acceptance here concerns extracted evidence only, not authenticity, legal
    validity, or certificate recency (no recency threshold is defined here).
    """
    rule, sources = load_r014()
    outcome = "missing"
    facts = state.get("extracted_facts", {})
    for document in state.get("documents", []):
        raw = facts.get(document.doc_id)
        certificate = (document.doc_type == "student_certificate" or
                       isinstance(raw, dict) and raw.get("document_type") == "student_enrollment_certificate")
        if not certificate:
            if document.status != "analyzed" or document.doc_type is None:
                outcome = "manual_review_required"
            continue
        outcome = "manual_review_required"
        if document.status != "analyzed" or not isinstance(raw, dict):
            continue
        # Preserve original flags as well as recomputing checks from the facts.
        if any(raw.get(name) for name in ("missing_fields", "ambiguous_fields", "validation_issues")):
            continue
        try:
            extraction = DocumentInfo.model_validate(raw)
        except (ValueError, TypeError):
            continue
        if (document.doc_type == "student_certificate" and
                extraction.document_type == "student_enrollment_certificate" and
                not extraction.missing_fields and not extraction.ambiguous_fields and
                not extraction.validation_issues):
            outcome = "supplied_and_accepted"
            break
    return Requirement(
        rule_id=rule["rule_id"], description=rule["description"],
        satisfied=outcome == "supplied_and_accepted", evaluation_result=outcome,
        rule_version=rule["version"], source_id=rule["source_ids"][0], source_metadata=sources,
    )


def rules_engine(state: AgentState) -> dict:
    """Evaluate only R014; retain analyzer issues and block unverified evidence."""
    requirement = evaluate_r014(state)
    issues = list(state.get("validation_errors", []))
    if not requirement.satisfied:
        issues.append(ValidationIssue(
            code="MISSING_PREREQUISITE" if requirement.evaluation_result == "missing" else "SCHEMA_ERROR",
            source="rules_engine", rule_id="R014",
            message=f"R014: {requirement.evaluation_result} (source: {requirement.source_id}).",
        ))
    return {"relevant_requirements": [requirement], "validation_errors": issues}
