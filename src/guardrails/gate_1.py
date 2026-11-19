"""Deterministic acceptance gate: no model calls or state mutation."""
import re
from pydantic import ValidationError
from src.core.state import ActionPlan, ValidationIssue, ResidencePermitContext
from src.workers.rule_loader import load_dependencies
from src.core.llm_diagnostics import log_validation_rejections

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
    scoped_active=None
    if state.get("planning_scope")=="milestone_1_document_preparation":
        from src.agents.planner import actionable_needs
        scoped_active={n["rule_id"] for n in actionable_needs(state) if n["category"]=="immediate"}
    for action in plan.actions:
        if not action.related_rule_id: add("UNTARGETED_REQUEST","Each action must reference a supplied rule",action.step_id)
        normalized_title=re.sub(r"[\W_]+"," ",(action.title or action.action).casefold()).strip()
        if normalized_title in {"information","prepare","information prepare","process","validate","review","guidance","correct information","request document","request user confirmation","review document"} or len(normalized_title.split())<2:
            add("VAGUE_TITLE","Use a concrete instruction naming the document or information",action.step_id,action.related_rule_id)
        if action.action_type=="CORRECT_INFORMATION" and not re.search(r"\b(?:review|correct\w*|extend\w*|update\w*|resolve\w*|renew\w*|replace\w*|arrange\w*|amend\w*|verify|check|request|obtain)\b",normalized_title):
            add("NON_CORRECTIVE_TITLE","Name what must be reviewed, corrected or updated",action.step_id,action.related_rule_id)
        for rid in action.related_rule_ids:
            if rid not in results:add("UNKNOWN_RULE_ID","Use supplied rule IDs",action.step_id,rid)
        for field in action.requested_fields:
            if field not in type(context).model_fields:add("UNKNOWN_FIELD","Use canonical fields",action.step_id)
        if action.grounding_ids:
            from src.agents.planner import canonical_candidates
            allowed={c["candidate_id"]:c for c in canonical_candidates(state)}
            selected=[allowed[key] for key in action.grounding_ids if key in allowed]
            if len(selected)!=len(action.grounding_ids):add("UNSUPPORTED_GROUNDING","Candidate is absent from current actionable needs",action.step_id)
            elif action.related_rule_id not in {rid for c in selected for rid in c["rule_ids"]} or action.action_type not in {c["action_type"] for c in selected} or set(action.requested_fields)!={f for c in selected for f in c["fields"]} or set(action.requested_documents)!={d for c in selected for d in c["documents"]}:
                add("UNSUPPORTED_GROUNDING","Preserve candidate rule, type and targets",action.step_id)
        if scoped_active is not None and action.related_rule_id not in scoped_active and not (action.request_kind=="PROVIDE_GUIDANCE" and action.execution_type=="information" and not action.requested_fields and not action.requested_documents):
            add("OUTSIDE_MILESTONE_SCOPE","Keep active requests within the supported document-preparation scope",action.step_id,action.related_rule_id)
        if action.request_kind:
            if not action.related_rule_id:
                add("UNTARGETED_REQUEST", "Link the request to a requirement", action.step_id)
            if action.request_kind in {"REQUEST_USER_CONFIRMATION","REVIEW_EXTRACTED_EVIDENCE","RESOLVE_CONFLICT"} and not action.requested_fields:
                add("UNTARGETED_REQUEST", "Name the specific fields needing input", action.step_id)
            if action.request_kind == "REQUEST_DOCUMENT" and not action.requested_documents:
                add("UNTARGETED_REQUEST", "Name the requested document or page", action.step_id)
            for field in action.requested_fields:
                if field not in type(context).model_fields:
                    add("UNKNOWN_FIELD", "Use a canonical context field", action.step_id)
                elif action.request_kind == "REQUEST_USER_CONFIRMATION" and getattr(context,field) is not None:
                    add("REDUNDANT_REQUEST", "Already accepted: " + field, action.step_id)
        if re.fullmatch(r"(?:identify missing information|gather required data|validate data|update submission)[.!]?", action.action.strip(), re.I):
            add("GENERIC_ACTION", "Request a specific field or document linked to a rule", action.step_id)
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
        if action.execution_type == "official_application":
            for key in ["R001", "R002", "R003", "R004", "R005", "R006"]: ready(key, action)
        elif action.execution_type == "iro_preverification": ready("R020", action)
        elif action.execution_type == "fingerprint":
            for key in ["R005", "R007"]: ready(key, action)
        elif action.execution_type == "iro_submission":
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
    coverage_failed=failed
    if state.get("planning_scope")=="milestone_1_document_preparation" and all(a.execution_type in {"information","prepare"} for a in plan.actions):
        from src.agents.planner import actionable_needs
        active={n["rule_id"] for n in actionable_needs(state) if n["category"]=="immediate"}
        coverage_failed=failed & active
    for rid in sorted(coverage_failed - {rid for a in plan.actions for rid in [a.related_rule_id]+a.related_rule_ids}):
        add("FAILED_RULE_IGNORED", "Include a corrective action for the failed rule", rule=rid)
    if re.search(COMPLETION_CLAIM, plan.summary, re.I): add("COMPLETION_CLAIM", "Summary must not claim execution", "plan_summary")
    approval_claim=r"(?:permit|ikamet).{0,60}(?:officially approved|is approved|approval guaranteed)|guaranteed.{0,30}(?:permit|approval)"
    action_text=" ".join(a.action + " " + a.reason for a in plan.actions)
    if re.search(approval_claim, plan.summary + " " + action_text, re.I):
        # A summary-only rejection must not discard otherwise valid actions.
        summary_only=re.search(approval_claim,plan.summary,re.I) and not re.search(approval_claim,action_text,re.I)
        add("UNSUPPORTED_APPROVAL_CLAIM", "Only the authority can approve a residence permit", "plan_summary" if summary_only else None)
    for rid in unknown:
        if re.search(rf"\b{rid}\b\s+(?:is\s+|has\s+)?{FAILURE_CLAIM}", plan.summary, re.I): add("UNKNOWN_AS_FAILURE", "Summary must preserve unknown status", "plan_summary", rule=rid)
    return errors


def gate_1(state):
    if state.get("candidate_plan") is None:
        errors=state.get("validation_errors") or [ValidationIssue(code="NO_CANDIDATE_PLAN",message="No candidate was produced",source="gate_1")]
        log_validation_rejections(errors)
        return {"validation_errors":errors,"run_status":state.get("run_status","PLAN_INVALID")}
    errors = [e for e in state.get("validation_errors",[]) if e.source=="planner"] + validate_plan(state, {"actions": state.get("candidate_plan"), "summary": state.get("plan_summary", "")})
    if state.get("candidate_plan") is None:
        errors = [*state.get("validation_errors", []), *errors]
    log_validation_rejections(errors)
    return {"validation_errors": errors, "run_status": "PLAN_INVALID" if errors else "VALIDATED"}
