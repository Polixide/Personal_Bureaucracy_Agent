"""Deterministic stage/action selection; no model calls and no legal state writes."""
from src.core.state import ResidencePermitContext, ActionPlan, PlanStep
from src.workers.rule_loader import load_dependencies

FIELD_REQUESTS={
    "application_form_generated":"Prepare or upload the online residence-permit application form",
    "application_form_signed":"Review and sign the printed application form when appropriate",
    "biometric_photo":"Provide a recent biometric photograph",
    "icao_compliant":"Have the photograph checked against the ICAO standard referenced by R009",
    "photo_age_months":"Confirm when the photograph was taken",
    "fee_payment_required":"Confirm with IRO whether the card fee applies to your case",
    "fee_receipt_present":"Provide the card-fee payment receipt if required",
    "student_certificate_present":"Provide a current student certificate",
    "turkey_address_present":"Provide your address in Turkiye",
    "phone_present":"Provide a contact telephone number",
    "email_present":"Provide a contact email address",
}

STAGES = ("INITIAL_INFORMATION", "DOCUMENT_PREPARATION", "APPLICATION_PREPARATION", "IRO_PREVERIFICATION", "PHYSICAL_SUBMISSION", "STATUS_TRACKING")

def workflow_stage(state):
    c=ResidencePermitContext.model_validate(state.get("residence_permit_context",{}))
    from src.workers.verification import MILESTONE_SCOPE, milestone_checklist
    if state.get("planning_scope")==MILESTONE_SCOPE:
        return STAGES[2] if all(row["status"]=="complete" for row in milestone_checklist(state)) else STAGES[1]
    if c.submitted_to_iro is True: return STAGES[5]
    if c.online_application_completed is True and c.iro_preverification_confirmed is True and c.gi_number_present is True and c.all_core_documents_present is True: return STAGES[4]
    if c.all_core_documents_present is True: return STAGES[3]
    if c.student_certificate_present is True and c.passport_id_page_present is True and c.insurance_valid_for_ikamet is True: return STAGES[2]
    if c.exchange_student is True and c.permit_type == "short_term" and c.is_in_turkiye is True: return STAGES[1]
    return STAGES[0]


def prioritize(state, needs):
    from src.workers.verification import MILESTONE_SCOPE
    limited=state.get("planning_scope")==MILESTONE_SCOPE
    stage=workflow_stage(state)
    results={r.rule_id:r for r in state.get("relevant_requirements") or []}
    dependencies=load_dependencies()
    for need in needs:
        rid=need["rule_id"];number=int(rid[1:])
        need["source_ids"]=results[rid].source_ids
        need["knowledge_basis"]="Canonical local rules; legal currency not independently verified at runtime"
        need["declared_progress"]=[row for row in state.get("action_progress",{}).values() if row.get("related_rule_id")==rid]
        need["waiting_for"]=[key for key in sorted(dependencies.get(rid,())) if key not in results or results[key].passed is not True]
        if limited and rid=="R008" and not ({"passport_id_page_present","turkey_entry_page_present"} & set(need["missing_fields"])):
            category="outside_scope"
            need["scope_note"]="Identity and entry pages are supported; visa applicability/page evidence is outside this milestone"
        elif limited and rid in {"R009","R011","R012","R017","R018","R020","R025","R026"}: category="outside_scope"
        elif limited and rid in {"R005","R021","R022","R023","R024","R029"}: category="future"
        elif rid in {"R027","R028"}: category="guardrail_only"
        elif rid=="R005" and stage in STAGES[:2]: category="future"
        elif rid=="R029" and stage!="STATUS_TRACKING": category="future"
        elif number>=20 and stage in STAGES[:3]: category="future"
        elif number>=20 and need["waiting_for"]: category="waiting"
        elif results[rid].severity!="blocking": category="advisory"
        else: category="immediate"
        need["category"]=category
        need["stage"]=stage
        if limited: need["milestone_scope"]=True
    # Basic eligibility and legal-stay dates precede parallel document preparation.
    return sorted(needs,key=lambda n:({"immediate":0,"waiting":1,"future":2,"advisory":3,"guardrail_only":4,"outside_scope":5}[n["category"]], n["rule_id"]))


def selected_needs(needs, limit=5):
    limited=any(n.get("milestone_scope") for n in needs)
    failed=[n for n in needs if n["status"]=="failed" and (not limited or n["category"]=="immediate")]
    immediate=[n for n in needs if n["category"]=="immediate" and n["status"]!="failed"]
    selected=failed+immediate[:max(0,limit-len(failed))]
    if not selected and not limited:
        selected=[n for n in needs if n["category"] in {"waiting","future","advisory"}][:1]
    return selected


def fallback_plan(state, needs):
    """Conservative preparation/guidance only. Gate 1 still owns acceptance."""
    from src.workers.verification import CONTEXT_QUESTIONS, CONTEXT_FIELDS
    selected=selected_needs(needs)
    actions=[]
    conflicts=state.get("verification_conflicts",[])
    if conflicts:
        rid=next((r.rule_id for r in state.get("relevant_requirements",[]) if r.rule_id=="R014"),None)
        if rid:
            actions.append(PlanStep(step_id="resolve_identity",priority=1,action="Review the holder-name observations and explicitly resolve the identity conflict",reason="Conflicting identity observations need human review",related_rule_id=rid,request_kind="RESOLVE_CONFLICT",requested_fields=["holder_name"],available_in_prototype=True,action_type="information"))
    for need in selected:
        rid=need["rule_id"];fields=need["missing_fields"]
        kind="PROVIDE_GUIDANCE";docs=[];available=False
        if need["category"] in {"future","waiting","advisory"} and need["status"]!="failed":
            text=f"Review the prerequisites for {rid} before starting that later stage"
            fields=[]
        elif need["status"]=="failed":
            # Keep any official submission language in the reason, not an instruction.
            text=f"Review the recorded inputs for {rid} with IRO before proceeding"
        elif need.get("declared_progress"):
            text=f"Review supporting evidence for your declared action for {rid}; its requirement still needs assessment"
            kind="REVIEW_EXTRACTED_EVIDENCE" if fields else "PROVIDE_GUIDANCE"
            available=bool(fields)
        elif need["existing_evidence_to_review"]:
            text="Review the existing document evidence for "+", ".join(f.replace("_"," ") for f in fields)
            kind="REVIEW_EXTRACTED_EVIDENCE";available=True
        elif need["documents"]:
            docs=need["documents"];text="Provide "+" and ".join(docs)+" for manual review"
            kind="REQUEST_DOCUMENT";available=True
        elif rid=="R005":
            text="Review the official e-ikamet channel referenced by R005 before entering application details"
            fields=[]
        elif rid=="R017":
            docs=["Address proof: dormitory certificate, rental agreement or notarized commitment"]
            text="Provide the address proof required by R017 for review"
            kind="REQUEST_DOCUMENT";available=True
        elif fields:
            text="; ".join(CONTEXT_QUESTIONS.get(f, FIELD_REQUESTS.get(f,"Provide the "+f.replace("_"," ")+f" to assess {rid}")) for f in fields)
            kind="REQUEST_USER_CONFIRMATION";available=all(f in CONTEXT_FIELDS for f in fields)
        else:
            text=f"Review the requirements and prerequisites for {rid} with IRO"
        if kind=="REVIEW_EXTRACTED_EVIDENCE" and not fields: kind="PROVIDE_GUIDANCE"
        actions.append(PlanStep(step_id="next_"+rid,priority=len(actions)+1,action=text,
            reason=need["description"]+"; status: "+need["status"],related_rule_id=rid,
            request_kind=kind,requested_fields=fields,requested_documents=docs,
            available_in_prototype=available,action_type="information"))
    from src.workers.verification import MILESTONE_SCOPE, milestone_checklist
    if not actions and state.get("planning_scope")==MILESTONE_SCOPE and all(row["status"]=="complete" for row in milestone_checklist(state)):
        if any(r.rule_id=="R020" for r in state.get("relevant_requirements",[])):
            actions.append(PlanStep(step_id="review_full_file",priority=1,action="Review the remaining full-file requirements with IRO before later workflow steps",reason="Seven-document preparation does not establish full-file readiness under R020",related_rule_id="R020",request_kind="PROVIDE_GUIDANCE",action_type="information"))
    kinds={"REQUEST_DOCUMENT":"REQUEST_DOCUMENT","REQUEST_USER_CONFIRMATION":"REQUEST_USER_CONFIRMATION","REVIEW_EXTRACTED_EVIDENCE":"REVIEW_DOCUMENT","RESOLVE_CONFLICT":"CORRECT_INFORMATION"}
    actions=[PlanStep.model_validate({**action.model_dump(),"action_type":kinds.get(action.request_kind,"PROVIDE_GUIDANCE"),"execution_type":action.execution_type or "information"}) for action in actions]
    return ActionPlan(actions=actions,summary="Conservative next steps from current requirements; no official action executed")



def milestone_needs(state, needs):
    from src.workers.verification import MILESTONE_SCOPE, MAPPINGS, milestone_checklist, document_review
    if state.get("planning_scope")!=MILESTONE_SCOPE:return needs
    results={r.rule_id:r for r in state.get("relevant_requirements") or []}
    output=[];used=set()
    order=("passport_copy","turkiye_entry_stamp","student_certificate","health_insurance","ikamet_application_form","declaration_control_document","card_fee_receipt")
    checklist={r["document_type"]:r for r in milestone_checklist(state)}
    for kind in order:
        row=checklist[kind]
        if row["status"]=="complete":continue
        rid=(row["rule_ids"] or ["R008"])[0]
        if rid not in results:continue
        if kind=="health_insurance" and results.get("R015") and results["R015"].passed is True and results.get("R016") and results["R016"].passed is not True:
            rid="R016"
        rule=results[rid]
        docs=[d for d in state.get("documents",[]) if getattr(d,"doc_id",None) in row["document_ids"]]
        fields=[]
        for doc in docs:
            for field in document_review(state,doc)["fields"]:
                if field["status"] not in {"source_grounded","user_confirmed"}:
                    fields.extend(MAPPINGS[kind].get(field["name"],()))
        need={"rule_id":rid,"status":rule.status,"description":rule.description,
            "missing_fields":list(dict.fromkeys(fields)),"questions":{},
            "documents":[row["label"]] if row["status"]=="missing" else [],
            "existing_evidence_to_review":[] if kind=="health_insurance" and rid=="R016" else [{"document_id":d.doc_id,"type":kind} for d in docs],
            "prototype_confirmation_fields":fields,"waiting_for":[],"category":"immediate",
            "stage":workflow_stage(state),"source_ids":rule.source_ids,"milestone_scope":True,
            "checklist_status":row["status"],"declared_progress":[],"document_type":kind}
        output.append(need);used.add(rid)
    output.extend(n for n in needs if n["category"]=="immediate" and n["rule_id"] not in used)
    output.extend(n for n in needs if n["category"]!="immediate")
    return output



def canonical_candidates(state):
    """Determine WHAT is actionable. Never generate user-facing final prose here."""
    from src.agents.planner import actionable_needs
    from src.core.state import ResidencePermitContext
    from src.workers.verification import PROFILE_TYPES, MAPPINGS
    context=ResidencePermitContext.model_validate(state.get("residence_permit_context",{}))
    output=[]
    for need in actionable_needs(state):
        if need["category"]!="immediate":continue
        rid=need["rule_id"];fields=list(need["missing_fields"])
        target=need.get("document_type")
        if not target:target={"R006":"passport_copy","R008":"turkiye_entry_stamp","R019":"declaration_control_document","R015":"health_insurance"}.get(rid)
        available=any(getattr(d,"doc_type",None)==target and getattr(d,"status",None)=="analyzed" for d in state.get("documents",[]))
        conflict_fields=set(state.get("verification_conflicts",[])) & {field for targets in MAPPINGS.get(target,{}).values() for field in targets}
        if conflict_fields:
            kind="CORRECT_INFORMATION";fields=sorted(conflict_fields)
        elif need["status"]=="failed":
            kind="CORRECT_INFORMATION"
            if rid=="R015":fields=["insurance_end_date"]
            if not fields:fields={"R015":["insurance_end_date"],"R006":["passport_expiry_date"],"R019":["declaration_control_document_signed"]}.get(rid,[])
        elif need["documents"] or (target and not available and ((rid=="R006" and "passport_expiry_date" in fields) or rid in {"R008","R019"})):
            kind="REQUEST_DOCUMENT"
        elif need["existing_evidence_to_review"]:kind="REVIEW_DOCUMENT" if fields else "PROVIDE_GUIDANCE"
        elif fields:kind="REQUEST_USER_CONFIRMATION"
        else:kind="PROVIDE_GUIDANCE"
        documents=[]
        if kind=="REQUEST_DOCUMENT":
            if target:documents=[PROFILE_TYPES.get(target,target)]
            else:documents=list(need["documents"])
        output.append({"candidate_id":rid,"rule_ids":[rid],"status":need["status"],"issue":"coverage_end_insufficient" if rid=="R015" and need["status"]=="failed" else "missing_document_or_information", "waiting_for":need.get("waiting_for",[]),"action_type":kind,"target_document":PROFILE_TYPES.get(target,target),"fields":fields,"documents":documents,"message":need["description"],"policy":getattr(next((r for r in state.get("relevant_requirements",[]) if r.rule_id==rid),None),"assessment_policy",None)})
    return output
