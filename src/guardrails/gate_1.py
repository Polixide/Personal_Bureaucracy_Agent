"""Deterministic acceptance gate: no model calls or state mutation."""
import re
from pydantic import ValidationError
from src.core.state import ActionPlan, ValidationIssue, ResidencePermitContext
from src.workers.rule_loader import load_dependencies

# These intentionally small patterns define exactly what the text checks detect.
FAILURE_CLAIM = r"\b(failed|invalid|insufficient|not satisfied|not met|noncompliant)\b"
COMPLETION_CLAIM = (
    r"\b(?:already|successfully)\s+(?:been\s+)?(?:submitted|completed|paid|booked|renewed|uploaded)\b"
    r"|\b(?:has|have)\s+(?:been\s+)?(?:submitted|completed|paid|booked|renewed|uploaded)\b"
    r"|\b(?:was|were|is|are)\s+(?:submitted|completed|paid|booked|renewed|uploaded)\b"
    r"|\b(?:i|we)\s+(?:submitted|completed|paid|booked|renewed|uploaded)\b"
)
FINAL_ACTION = (
    r"\bsubmit\b.{0,60}\b(application|file|document set|documents|submission)\b"
    r"|\b(?:complete|finalize|finish)\b.{0,40}\b(application|procedure|process|submission)\b"
)



def validate_plan(state, plan):
    errors = []
    def add(code, message, step=None, rule=None):
        errors.append(ValidationIssue(code=code, message=message, source="gate_1", item=step, rule_id=rule))
    try:
        plan = ActionPlan.model_validate(plan.model_dump() if isinstance(plan, ActionPlan) else plan)
    except (ValueError, TypeError, AttributeError) as error:
        add("SCHEMA_ERROR", str(error))
        return errors
    results = {r.rule_id: r for r in state.get("relevant_requirements") or []}
    failed = {key for key,r in results.items() if r.status == "failed"}
    unknown = {key for key,r in results.items() if r.status == "missing_information"}
    dependencies = load_dependencies()
    ids = [a.step_id for a in plan.actions]
    if len(ids) != len(set(ids)): add("DUPLICATE_STEP_ID", "Step IDs must be unique")
    priorities = [a.priority for a in plan.actions]
    if priorities != sorted(set(priorities)): add("AMBIGUOUS_ORDER", "Use unique increasing priorities")
    completed = set(state.get("completed_steps", []))
    seen = set(completed)
    steps_by_id = {a.step_id: a for a in plan.actions}
    context = ResidencePermitContext.model_validate(state.get("residence_permit_context", {}))
    deadlines = dict(state.get("deadlines", {}))
    if context.legal_stay_expiry_date: deadlines["legal_stay_expiry_date"] = context.legal_stay_expiry_date
    if context.iro_window_end: deadlines["iro_window_end"] = context.iro_window_end
    def ready(rule, action):
        required = set()
        def visit(key):
            if key in required: return
            required.add(key)
            for parent in dependencies.get(key, []): visit(parent)
        visit(rule)
        for key in sorted(required):
            if key not in results or results[key].passed is not True:
                add("MISSING_PREREQUISITE", "Official action requires a verified passed prerequisite", action.step_id, key)
    for action in plan.actions:
        rid = action.related_rule_id
        text = action.action + " " + action.reason
        if rid is not None and rid not in results: add("UNKNOWN_RULE_ID", "Use a supplied rule ID", action.step_id, rid)
        if rid in unknown and re.search(FAILURE_CLAIM, text, re.I): add("UNKNOWN_AS_FAILURE", "Ask for information without asserting failure", action.step_id, rid)
        if re.search(COMPLETION_CLAIM, text, re.I): add("COMPLETION_CLAIM", "Do not claim external execution", action.step_id)
        if not action.requires_approval: add("APPROVAL_REQUIRED", "All proposed plans require explicit human approval", action.step_id)
        if any(dep not in seen for dep in action.depends_on): add("MISSING_PREREQUISITE", "Dependencies must precede this step or be completed", action.step_id)
        for dep in action.depends_on:
            prerequisite = steps_by_id.get(dep)
            if prerequisite and prerequisite.due_date and action.due_date and prerequisite.due_date > action.due_date:
                add("DATE_CONFLICT", "A prerequisite is scheduled after its dependent action", action.step_id)
        seen.add(action.step_id)
        if action.due_date is not None:
            if action.deadline_key not in deadlines or deadlines[action.deadline_key] != action.due_date:
                add("UNSUPPORTED_DEADLINE", "Use a matching explicit deadline key and date, or null", action.step_id)
        if action.action_type == "official_application":
            for key in ["R001", "R002", "R003", "R004", "R005", "R006"]: ready(key, action)
        elif action.action_type == "iro_preverification": ready("R020", action)
        elif action.action_type == "fingerprint":
            for key in ["R005", "R007"]: ready(key, action)
        elif action.action_type == "iro_submission":
            for key in ["R023", "R024", "R025", "R026"]: ready(key, action)
        # Phrase checks supplement structured categories; they never authorize execution.
        if re.search(FINAL_ACTION, action.action, re.I):
            blockers = [key for key,r in results.items() if r.severity == "blocking" and r.passed is not True]
            if blockers: add("PREMATURE_SUBMISSION", "Resolve and recheck blocking requirements before submission", action.step_id)
        for kind in action.required_documents:
            candidates = [d for d in state.get("documents", []) if d.doc_type == kind and d.status == "analyzed" and d.info and not d.info.missing_fields and not d.info.validation_issues]
            if not candidates: add("MISSING_DOCUMENT", "No complete extracted document for " + kind, action.step_id)
            elif action.due_date and not any(
                (getattr(d.info, "coverage_start_date", d.issue_date) is None
                 or getattr(d.info, "coverage_start_date", d.issue_date) <= action.due_date)
                and (d.expiry_date is None or d.expiry_date >= action.due_date)
                for d in candidates
            ):
                add("DATE_CONFLICT", "Document dates conflict with action date", action.step_id)
    for rid in sorted(failed - {a.related_rule_id for a in plan.actions}):
        add("FAILED_RULE_IGNORED", "Include a corrective action for the failed rule", rule=rid)
    if re.search(COMPLETION_CLAIM, plan.summary, re.I): add("COMPLETION_CLAIM", "Summary must not claim execution")
    for rid in unknown:
        if re.search(rf"\b{rid}\b\s+(?:is\s+|has\s+)?{FAILURE_CLAIM}", plan.summary, re.I): add("UNKNOWN_AS_FAILURE", "Summary must preserve unknown status", rule=rid)
    return errors


def gate_1(state):
    errors = validate_plan(state, {"actions": state.get("candidate_plan"), "summary": state.get("plan_summary", "")})
    if state.get("candidate_plan") is None:
        errors = [*state.get("validation_errors", []), *errors]
    return {"validation_errors": errors, "run_status": "PLAN_INVALID" if errors else "VALIDATED"}
