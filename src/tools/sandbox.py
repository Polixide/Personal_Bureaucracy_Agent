"""Deterministic environment for simulated actions; never a second fact store."""
from copy import deepcopy
from src.core.procedure import EIKAMET_FIELD_CATALOGUE, build_eikamet_preparation, eikamet_field_value
from src.tools.schemas import SandboxForm, SandboxField, SandboxResult, SimulatedApplicationResult


def read_form(state):
    return SandboxForm.model_validate(deepcopy(state.get("sandbox_form") or {}))


def _stage_allows_fill(state):
    procedure = state.get("procedure")
    stage = procedure.get("current_stage") if isinstance(procedure, dict) else getattr(procedure, "current_stage", None)
    if stage not in {"EIKAMET_PREPARATION", "EIKAMET_READY"}:
        raise ValueError("Enter e-Ikamet preparation explicitly before simulating form actions")


def fill_field(state, field_id, value=None):
    _stage_allows_fill(state)
    preparation = build_eikamet_preparation(state)
    field = preparation.fields.get(field_id)
    if field is None:
        raise ValueError("Unsupported preparation field")
    if field.status == "CONFLICT":
        raise ValueError("Conflicting source information prevents automated filling")
    if field.status not in {"VERIFIED", "USER_CONFIRMED"}:
        raise ValueError("Only accepted source evidence or user-confirmed information may fill a field")
    grounded = eikamet_field_value(state, field)
    if grounded is None or not field.source_ref:
        raise ValueError("The accepted field source is missing or stale")
    if value is not None and value != grounded:
        raise ValueError("Proposed value differs from accepted canonical evidence")
    before = read_form(state)
    after = before.model_copy(deep=True)
    item = SandboxField(value=grounded, source_ref=field.source_ref, source_status=field.status)
    if before.fields.get(field_id) != item:
        after.fields[field_id] = item
        after.revision += 1
        after.status = "DRAFT"
        after.validated_revision = None
        after.validation_issues = []
        after.simulated_application_ready=False
        after.artifact_path=after.artifact_sha256=None
        after.artifact_revision=None
    result = validate_form({**state, "sandbox_form": after.model_dump(mode="json")})
    result.form = after
    return result, {field_id: {"before": before.fields[field_id].model_dump() if field_id in before.fields else None,
                               "after": item.model_dump()}}


def validate_form(state):
    form = read_form(state)
    preparation = build_eikamet_preparation(state)
    missing, conflicts, stale = [], [], []
    method=preparation.fields.get('contact_method')
    method_value=eikamet_field_value(state,method) if method and method.status in {'VERIFIED','USER_CONFIRMED'} else None
    optional_contact={'turkish_phone'} if method_value=='email' else {'email'} if method_value=='turkish_phone' else set()
    for definition in EIKAMET_FIELD_CATALOGUE:
        field_id = definition.field_id
        if field_id in optional_contact and field_id not in form.fields:continue
        accepted = preparation.fields[field_id]
        item = form.fields.get(field_id)
        if accepted.status == "CONFLICT":
            conflicts.append(field_id)
        elif item is None:
            missing.append(field_id)
        elif (accepted.status not in {"VERIFIED", "USER_CONFIRMED"} or item.source_ref != accepted.source_ref
              or item.value != eikamet_field_value(state, accepted)):
            stale.append(field_id)
    # Unknown fields cannot make validation succeed or hide a missing catalogue field.
    stale.extend(sorted(set(form.fields) - {item.field_id for item in EIKAMET_FIELD_CATALOGUE}))
    invalid=[]
    definitions={item.context_field:item.field_id for item in EIKAMET_FIELD_CATALOGUE if item.context_field}
    from src.core.procedure import _accepted_snapshot
    context=_accepted_snapshot(state)['residence_permit_context'].model_dump(mode='json')
    for key,value in form.draft_answers.items():
        if key in definitions and str(value)!=str(context.get(key)) and value is not None:invalid.append(definitions[key])
    if context.get('simulation_declaration_acknowledged') is not True:invalid.append('declaration_acknowledged')
    from src.core.procedure import supported_exchange_reason
    if context.get('reason_of_stay') is not None and not supported_exchange_reason(context['reason_of_stay']):invalid.append('reason_of_stay')
    from datetime import date
    def day(key):
        item=form.fields.get(key)
        if item is None:return None
        try:return date.fromisoformat(item.value)
        except ValueError:invalid.append(key);return None
    dates={key:day(key) for key in ('birth_date','passport_issue_date','passport_expiry_date','latest_entry_date','requested_permit_start','requested_permit_end','insurance_end_date')}
    for first,last in (('requested_permit_start','requested_permit_end'),('passport_issue_date','passport_expiry_date')):
        if dates[first] and dates[last] and dates[last]<dates[first]:invalid.extend((first,last))
    if dates['birth_date'] and dates['birth_date']>date.today():invalid.append('birth_date')
    from src.workers.rules_engine import evaluate_requirements
    targets={'R006':'passport_expiry_date','R015':'insurance_end_date','R016':'insurance_category'}
    invalid.extend(targets[row.rule_id] for row in evaluate_requirements(state) if row.rule_id in targets and row.passed is False)
    valid = not (missing or conflicts or stale or invalid)
    return SandboxResult(form=form, valid=valid, missing_field_ids=missing,
                         conflict_field_ids=conflicts, stale_field_ids=stale,invalid_field_ids=sorted(set(invalid)))


def validate_field(state, field_id):
    result = validate_form(state)
    if field_id not in {item.field_id for item in EIKAMET_FIELD_CATALOGUE}:
        raise ValueError("Unsupported preparation field")
    result.valid = field_id not in result.missing_field_ids + result.conflict_field_ids + result.stale_field_ids + result.invalid_field_ids
    return result


def apply_validation(state, *, submit=False):
    _stage_allows_fill(state)
    result = validate_form(state)
    form = result.form.model_copy(deep=True)
    form.validation_issues = (["missing:" + key for key in result.missing_field_ids]
        + ["conflict:" + key for key in result.conflict_field_ids]
        + ["stale:" + key for key in result.stale_field_ids] + ["invalid:" + key for key in result.invalid_field_ids])
    form.validated_revision = form.revision if result.valid else None
    form.status = ("SIMULATED_SUBMISSION" if submit else "VALIDATED") if result.valid else "DRAFT"
    result.form = form
    if submit:
        result.message = "Local simulated submission only; no government website, application or payment was accessed."
    return result


def needs_prefill(state):
    preparation=build_eikamet_preparation(state)
    form=read_form(state)
    accepted={key:item for key,item in preparation.fields.items() if item.status in {'VERIFIED','USER_CONFIRMED'}}
    if set(form.fields)-set(accepted):return True
    return any(form.fields.get(key)!=SandboxField(value=eikamet_field_value(state,item),source_ref=item.source_ref,source_status=item.status)
        for key,item in accepted.items() if eikamet_field_value(state,item) is not None)


def prefill_fields(state,answers=None):
    """One safe sandbox operation; values resolve from canonical records only."""
    _stage_allows_fill(state)
    preparation=build_eikamet_preparation(state)
    before=read_form(state);after=before.model_copy(deep=True)
    if answers:
        from src.workers.verification import FORM_CONTEXT_FIELDS
        if set(answers)-set(FORM_CONTEXT_FIELDS):raise ValueError('Unsupported educational form field')
        if any(value is not None and (not isinstance(value,(str,bool)) or len(str(value))>500) for value in answers.values()):raise ValueError('Form answers must be short text, Yes/No or empty')
        after.draft_answers.update(answers)
    # Remove only drafts now accepted by the canonical confirmation boundary.
    from src.core.procedure import _accepted_snapshot
    accepted_context=_accepted_snapshot(state)['residence_permit_context'].model_dump(mode='json')
    after.draft_answers={key:value for key,value in after.draft_answers.items() if value!=accepted_context.get(key)}
    fields={}
    for key,item in preparation.fields.items():
        if item.status not in {'VERIFIED','USER_CONFIRMED'}:continue
        value=eikamet_field_value(state,item)
        if value is not None:fields[key]=SandboxField(value=value,source_ref=item.source_ref,source_status=item.status)
    if fields!=after.fields or after.draft_answers!=before.draft_answers:
        after.fields=fields;after.revision+=1;after.status='DRAFT';after.validated_revision=None;after.validation_issues=[]
        after.simulated_application_ready=False;after.artifact_path=after.artifact_sha256=None;after.artifact_revision=None
    result=validate_form({**state,'sandbox_form':after.model_dump(mode='json')})
    result.message=f"Prepared {len(fields)} fields from your accepted information."
    return result,{'before':before.model_dump(mode='json'),'after':after.model_dump(mode='json')}


def simulated_application_preview(state):
    _stage_allows_fill(state)
    result=validate_form(state)
    if not result.valid or result.form.validated_revision!=result.form.revision:
        raise ValueError('Review all required application fields before generating the simulation')
    labels={item.field_id:item.label for item in EIKAMET_FIELD_CATALOGUE}
    from src.core.procedure import EDUCATIONAL_CHOICES
    values={labels[key]:result.form.fields[key].value for key in labels if key in result.form.fields}
    values.update(EDUCATIONAL_CHOICES)
    return SimulatedApplicationResult(prepared_values=values,form_revision=result.form.revision)


def generate_simulated_pdf(state,call_id):
    """Existing PyMuPDF creates an educational artifact, never an official Document."""
    from hashlib import sha256
    from html import escape
    from pathlib import Path
    import os
    import pymupdf
    from config import PROJECT_ROOT
    result=simulated_application_preview(state)
    rows=''.join('<tr><th align="left">'+escape(label)+'</th><td>'+escape(value)+'</td></tr>' for label,value in result.prepared_values.items())
    content='<h2>SIMULATED E-IKAMET APPLICATION</h2><p><b>Educational sandbox artifact<br>NOT AN OFFICIAL GOVERNMENT FORM<br>NOT SUBMITTED</b></p><table>'+rows+'</table><p>Prepared from document observations and explicit user confirmations. This artifact does not establish an official application or permit decision.</p>'
    with pymupdf.open() as pdf:
        page=pdf.new_page()
        spare,scale=page.insert_htmlbox(pymupdf.Rect(35,35,page.rect.width-35,page.rect.height-35),content,css='body {font-family:sans-serif;font-size:10pt} th,td {padding:4px}')
        if spare<0:raise ValueError('The simulated form is too long to export; shorten unusually long input values')
        data=pdf.tobytes(garbage=4,deflate=True)
    relative=Path('.local/artifacts')/sha256(str(state.get('user_id','')).encode()).hexdigest()[:16]/('simulated-'+sha256(call_id.encode()).hexdigest()[:24]+'.pdf')
    target=PROJECT_ROOT/relative
    io_target=Path(chr(92)*2+'?'+chr(92)+str(target.resolve())) if os.name=='nt' else target
    io_target.parent.mkdir(parents=True,exist_ok=True);io_target.write_bytes(data)
    result.artifact_path=relative.as_posix();result.artifact_sha256=sha256(data).hexdigest()
    form=read_form(state);form.simulated_application_ready=True;form.artifact_path=result.artifact_path;form.artifact_sha256=result.artifact_sha256;form.artifact_revision=form.revision
    return result,form


def read_simulated_pdf(state):
    """Download only this profile's current, hash-checked simulation artifact."""
    from config import PROJECT_ROOT
    from pathlib import Path
    from hashlib import sha256
    import os
    form=read_form(state)
    if not form.simulated_application_ready or form.artifact_revision!=form.revision or not form.artifact_path or not validate_form(state).valid:return None
    target=(PROJECT_ROOT/form.artifact_path).resolve()
    if not target.is_relative_to((PROJECT_ROOT/'.local/artifacts').resolve()):return None
    io_target=Path(chr(92)*2+'?'+chr(92)+str(target)) if os.name=='nt' else target
    if not io_target.is_file():return None
    data=io_target.read_bytes()
    return data if sha256(data).hexdigest()==form.artifact_sha256 else None
