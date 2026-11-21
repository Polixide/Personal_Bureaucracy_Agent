CHECKLIST_ICONS = {"complete": ":green[:material/check_circle:]", "needs confirmation": ":orange[:material/rate_review:]",
                   "missing": ":gray[:material/radio_button_unchecked:]", "ambiguous": ":orange[:material/help:]"}
CHECKLIST_LABELS = {"complete": "Ready", "needs confirmation": "Needs your review", "missing": "Missing", "ambiguous": "Needs your review"}



"""Focused Streamlit presentation; all decisions use canonical backend functions."""
from src.ui.components import *


def review_form(review, document, graph, config, thread_key, disabled):
    if not review['can_confirm']:
        st.caption('Reprocess this saved document before confirming its information.');return
    unresolved=[item for item in review['fields'] if item['status'] not in {'source_grounded','user_confirmed'}]
    if unresolved:st.markdown('**Needs your attention**')
    st.caption('Your confirmed answers remain separate from the original document observations.')
    edit=st.checkbox('Edit information already read or confirmed',key=f'edit_{thread_key}_{document.doc_id}',disabled=disabled)
    with st.form(f'review_{thread_key}_{document.doc_id}'):
        inputs={}
        for item in review['fields']:
            if item['status'] in {'source_grounded','user_confirmed'} and not edit:continue
            name=item['name'];value=item['value'];label=field_label(name)
            if name=='nationality' and value is None:st.caption('We could not read nationality confidently from your passport. Confirm it using the original document, for example Italy (ITA).')
            elif item['status']=='ambiguous':st.caption('This value is unclear or conflicting. Check the original document before confirming it.')
            if name=='insurance_evidence_type':
                options=['Not confirmed',*INSURANCE_TYPES]
                names={'private_health_insurance':'Private health insurance','sgk_provision':'SGK - manual review','sgk_application':'SGK application - manual review','bilateral_social_security':'Bilateral cover - manual review'}
                choice=st.selectbox('Insurance category',options,index=options.index(value) if value in options else 0,format_func=lambda x:names.get(x,x),disabled=disabled)
                if choice!='Not confirmed':inputs[name]=choice
            elif item['boolean']:
                choice=st.selectbox(label,['Not confirmed','Yes','No'],index=0 if value is None else 1 if value else 2,disabled=disabled)
                if choice!='Not confirmed':inputs[name]=choice=='Yes'
            else:
                text=st.text_input(label+(' (YYYY-MM-DD)' if 'date' in name else ''),value='' if value is None else str(value),disabled=disabled)
                if text.strip():inputs[name]=text.strip()
            if item.get('observation') is not None and value is None:st.caption('The original reading is retained in Technical; it has not been accepted automatically.')
        acknowledged=st.checkbox('I confirm the entered information is accurate',disabled=disabled)
        submit=st.form_submit_button('Confirm information',disabled=disabled or not (unresolved or edit))
    if submit:
        if not acknowledged:st.warning('Confirm that your answers are accurate before saving.')
        elif not inputs:st.info('Enter an answer to confirm, or keep reviewing the source document.')
        elif invoke(graph,ui_run(UserEvent(kind='confirm_information',document_id=document.doc_id,confirmed_fields=inputs),config['configurable']['thread_id']),config):st.rerun()


def identity_review(state, graph, config, thread_key, disabled, technical=True):
    detail=conflict_details(state).get('holder_name')
    if not detail or detail['resolved']:return
    st.warning('The holder names in your documents do not match. Review the originals and choose the correct identity.')
    rows=detail['observations']
    for row in rows:st.write(str(row['value'])+' - '+document_label(row['document_type']))
    with st.form(f"identity_{thread_key}_{state.get('trace_id','initial')}"):
        choice=st.selectbox('Correct identity after reviewing the documents',range(len(rows)),format_func=lambda i:str(rows[i]['value'])+' - '+document_label(rows[i]['document_type']),disabled=disabled)
        confirmed=st.checkbox('I explicitly resolve these identity observations',disabled=disabled)
        submit=st.form_submit_button('Save identity decision',disabled=disabled)
    if submit and confirmed and invoke(graph,ui_run(UserEvent(kind='resolve_conflict',confirmed_fields={'holder_name':rows[choice]['value']}),config['configurable']['thread_id']),config):st.rerun()


def declaration_form(state, graph, config, thread_key, disabled):
    context = state.get("residence_permit_context")
    edit = st.checkbox("Edit saved information", key=f"edit_declarations_{thread_key}")
    situation={"is_in_turkiye","entered_with_passport","application_date"}
    unresolved = [name for name in CONTEXT_QUESTIONS if name in situation and (edit or field(context,name) is None)]
    if any(getattr(d,"doc_type",None)=="turkiye_entry_stamp" and getattr(d,"status",None)=="analyzed" for d in state.get("documents",[])):
        unresolved=[name for name in unresolved if name!="latest_entry_date"]  # Review the page-specific candidate above.
    if not unresolved:
        st.caption("Your current location and passport declarations are recorded.")
        return
    st.caption("Answers are your statements, saved separately from document observations.")
    for group,names in CONTEXT_GROUPS.items():
        questions=[name for name in names if name in unresolved]
        if not questions:continue
        with st.expander(group+" confirmations",expanded=False):
            with st.form(f"declarations_{thread_key}_{group}_{state.get('trace_id','initial')}"):
                answers={}
                for name in questions:
                    value=field(context,name);kind=CONTEXT_FIELDS[name]
                    if kind=="bool":
                        choice=st.selectbox(CONTEXT_QUESTIONS[name],["Not answered","Yes","No"],index=0 if value is None else 1 if value else 2,disabled=disabled)
                        if choice!="Not answered":answers[name]=choice=="Yes"
                        elif edit and value is not None:answers[name]=None
                    elif kind=="insurance_type":
                        choices=["Not answered"]+list(INSURANCE_TYPES)
                        choice=st.selectbox(CONTEXT_QUESTIONS[name],choices,index=choices.index(value) if value in choices else 0,disabled=disabled)
                        if choice!="Not answered":answers[name]=choice
                        elif edit and value is not None:answers[name]=None
                    else:
                        entered=st.text_input(CONTEXT_QUESTIONS[name]+" (YYYY-MM-DD)",value="" if value is None else str(value),disabled=disabled)
                        if entered.strip():answers[name]=entered.strip()
                        elif edit and value is not None:answers[name]=None
                submit=st.form_submit_button("Save answers and recheck requirements",disabled=disabled)
            if submit:
                if answers:
                    event=UserEvent(kind="confirm_information",document_id="__declarations__",confirmed_fields=answers)
                    if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config):st.rerun()
                else:st.info("Answer at least one question to save.")


def render_checklist(state):
    from src.core.procedure import stage_document_checklist
    checklist=stage_document_checklist(state)
    if not checklist:return
    left,right=st.columns(2)
    for index,row in enumerate(checklist):
        status=row['status'];label=CHECKLIST_LABELS.get(status,'Needs your review')
        (left if index%2==0 else right).markdown(f"{CHECKLIST_ICONS.get(status,CHECKLIST_ICONS['missing'])} {row['label']}  \n:gray[{label}]")
    if any(row.get('key')=='address_proof' for row in checklist):
        from src.core.procedure import address_proof_status
        address=address_proof_status(state)
        if address['route']=='address_form_and_utility_bill':
            st.markdown('**Address proof: Address Form + Utility bill**')
            for part in address['parts']:
                st.write(('? ' if part['status']=='complete' else '? ')+part['label']+(' ? still required' if part['status']=='missing' else ' ? needs review' if part['status']!='complete' else ''))
            st.caption('Status: Ready' if address['ready'] else 'Status: Incomplete')
    st.caption('Ready means prepared for this step; it is not residence-permit approval.')



def lifecycle_button(graph,config,thread_key,doc_id,kind,label,disabled=False,**kwargs):
    if st.button(label,key=f"{kind}_{thread_key}_{doc_id}",disabled=disabled):
        target=kwargs.pop("target_document_id",doc_id)
        event=UserEvent(kind=kind,document_id=target,**kwargs)
        if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config):st.rerun()


def render_document(document,state,graph,config,thread_key,disabled):
    record=display_record(document);doc_id=record.get("doc_id","");kind=record.get("doc_type")
    review=document_card(state,document)
    title=document_label(kind)
    status="Replacement awaiting decision" if record.get("status")=="replacement_pending" else review["status"].replace("_"," ")
    with st.container(border=True):
        st.write(f"{title} - {status}")
        if kind=="health_insurance":st.caption("Coverage is compared with your requested permit end or academic-program end.")
        shared=sum(field(d,"file_path")==record.get("file_path") for d in state.get("documents",[]))>1
        st.caption(record.get("original_filename") or Path(record.get("file_path","")).name+(" | logical item from shared upload" if shared else ""))
        if record.get("processing_error"):st.error(friendly_error(str(record["processing_error"])))
        if record.get("status")=="analyzed" and sum(getattr(d,"doc_type",None)==kind and getattr(d,"status",None)=="analyzed" for d in state.get("documents",[]))>1:
            st.warning("Saved duplicate versions exist. Choose the current version explicitly.")
            lifecycle_button(graph,config,thread_key,doc_id,"replace_document","Use this version and replace other versions",disabled=disabled,confirmed=True)
        if record.get("status")=="replacement_pending":
            st.warning("Another current document has this type. Replace it explicitly or cancel this upload; pending evidence is not used.")
            if not disabled:
                lifecycle_button(graph,config,thread_key,doc_id,"replace_document","Replace current document",confirmed=True)
                lifecycle_button(graph,config,thread_key,doc_id,"cancel_document","Cancel",confirmed=True)
        info=field(document,"info")
        observations,legacy=observation_view(info)
        if info is not None and legacy:st.warning("Legacy extraction: reprocess this document before accepting its observations.")
        diagnostics=field(info,"extraction_diagnostics",{}) if info else {}
        method=diagnostics.get("extraction_route") if isinstance(diagnostics,dict) else None
        if method:st.caption("Text recovered from a scan" if method=="RAPIDOCR" else "Document text read for review")
        with st.expander("Details / Edit",expanded=st.session_state.get("review_document_"+thread_key)==doc_id):
            st.caption("Extracted document fields are observations and are not automatically treated as verified legal facts.")
            for label,key in (("Information still needed","missing_fields"),("Please confirm","ambiguous_fields")):
                names=[name for name in (observations.get(key) or []) if isinstance(name,str)]
                if names:st.caption(label+': '+', '.join(field_label(name) for name in names))
            for message in dict.fromkeys(validation_message(issue) for issue in observations.get('validation_issues',[]) or []):st.caption(message)
            pages=[number for number in (observations.get('selected_pages') or []) if type(number) is int]
            if pages:st.caption('Selected pages: '+', '.join(str(number) for number in pages))
            source_pages=observations.get('source_pages',{})
            if isinstance(source_pages,dict):
                for name,number in source_pages.items():
                    if type(number) is int:st.caption(field_label(name)+' - source page '+str(number))
            for item in review["fields"]:
                if item["value"] is not None:
                    origin={"source_grounded":"source-grounded","user_confirmed":"confirmed by you"}.get(item["origin"],"observation awaiting confirmation")
                    st.write(f"{field_label(item['name'])}: {user_value(item['name'],item['value'])} ({origin})")
            if isinstance(document,Document) and document.status=="analyzed":
                review_form(review,document,graph,config,thread_key,disabled)
                for name,entry in state.get("verification_records",{}).get(doc_id,{}).items():
                    if field(entry,"origin")=="user_confirmed" and not disabled:
                        lifecycle_button(graph,config,thread_key,doc_id+name,"remove_correction","Use original "+name.replace("_"," ")+" observation",confirmed_fields={name:None},target_document_id=doc_id)
            if isinstance(document,Document) and not disabled and document.status!="replacement_pending":
                from src.workers.verification import PROFILE_TYPES
                profiles={**PROFILE_TYPES,"student_certificate":"student_enrollment_certificate","health_insurance":"health_insurance_certificate","ikamet_application_form":"ikamet_application_form"}
                choices=list(profiles)
                selected=st.selectbox("Reclassify / Reprocess as",choices,index=choices.index(kind) if kind in choices else 0,format_func=document_label,key=f"profile_{thread_key}_{doc_id}")
                lifecycle_button(graph,config,thread_key,doc_id,"reprocess_document","Reprocess selected profile",document_type_hint=profiles[selected])
            st.caption('Full observations, source snippets and original diagnostics are retained in Technical.')
        checked=st.checkbox("I want to delete this document",key=f"delete_confirm_{thread_key}_{doc_id}_{state.get('trace_id','initial')}",disabled=disabled)
        lifecycle_button(graph,config,thread_key,doc_id,"delete_document","Delete",disabled=disabled or not checked,confirmed=True)



def collection_declarations(state, graph, config, thread_key, disabled):
    assessment=assess_document_collection(state)
    c=state.get('residence_permit_context')
    with st.expander('Address proof and photo readiness',expanded=False):
        edit=st.checkbox('Update my address/photo answers',key=f'collection_edit_{thread_key}',disabled=disabled)
        if assessment.prerequisites['address_proof']=='READY' and assessment.prerequisites['biometric_photo']=='READY' and not edit:
            st.success('Address proof and photo readiness are recorded.');return
        st.caption('These are your statements. Keep the supporting evidence for IRO review.')
        with st.form(f'collection_{thread_key}'):
            inputs={}
            if edit or assessment.prerequisites['address_proof']!='READY':
                options=['Not confirmed',*ACCEPTED_ADDRESS_PROOF_TYPES,'utility_bill_and_address_form']
                names={'dorm_certificate':'Dormitory certificate','rental_agreement':'Rental agreement','notarized_commitment':'Notarized undertaking','utility_bill_and_address_form':'Utility bill + completed Address Form'}
                current=field(c,'address_proof_type')
                choice=st.selectbox('Address-proof path',options,index=options.index(current) if current in options else 0,format_func=lambda x:names.get(x,x),disabled=disabled)
                if choice!='Not confirmed':inputs['address_proof_type']=choice
                st.caption('The utility-bill path needs both the bill and completed Address Form. Upload them separately; unclear signature details need confirmation.')
            if edit or assessment.prerequisites['biometric_photo']!='READY':
                value=field(c,'biometric_photo')
                photo=st.selectbox('Do you have a biometric photo ready?',['Not confirmed','Yes','No'],index=0 if value is None else 1 if value else 2,disabled=disabled)
                if photo!='Not confirmed':inputs['biometric_photo']=photo=='Yes'
            confirmed=st.checkbox('I confirm these preparation statements',disabled=disabled)
            submit=st.form_submit_button('Save preparation answers',disabled=disabled)
        if submit and confirmed and inputs and invoke(graph,ui_run(UserEvent(kind='confirm_information',document_id='__declarations__',confirmed_fields=inputs),config['configurable']['thread_id']),config):st.rerun()






def view_documents(state, snapshot, graph, config, thread_key, thread_id):
    from src.core.procedure import stage_document_checklist
    busy=bool(snapshot.next)
    st.markdown('##### Required now')
    render_checklist(state)
    with st.form(f'upload_{thread_key}',border=True):
        uploaded=st.file_uploader('Add a document (PDF or TXT)',type=['pdf','txt'],key=f'file_{thread_key}',disabled=busy)
        submit=st.form_submit_button('Upload and analyze',type='primary',disabled=busy)
    if submit:
        if uploaded is None:st.warning('Choose a PDF or TXT file first.')
        else:
            path=save_upload(uploaded.name,uploaded.getvalue(),thread_id)
            run_event(graph,config,UserEvent(kind='analyze_document',file_path=str(path),original_filename=uploaded.name))
    if field(state.get('procedure'),'current_stage') in {'DOCUMENT_COLLECTION','IRO_REVIEW'}:collection_declarations(state,graph,config,thread_key,busy)
    for message in conflict_messages(state):st.warning(message['message'])
    identity_review(state,graph,config,thread_key,busy,technical=False)
    st.markdown('##### All uploaded documents')
    documents=state.get('documents') or []
    if not documents:st.caption('No documents yet. Upload what you have, then review any information needing confirmation.')
    for document in documents:render_document(document,state,graph,config,thread_key,busy)



def view_details(state, snapshot, graph, config, thread_key):
    identity_review(state,graph,config,thread_key,bool(snapshot.next),technical=False)
    declaration_form(state,graph,config,thread_key,bool(snapshot.next))
    entry=state.get('verification_records',{}).get('__derived__',{}).get('planned_stay_months')
    if entry:st.caption(f"Academic stay: approximately {field(entry,'value')} complete months, calculated from your exchange dates.")



def view_checks(state, snapshot, graph, config, thread_key):
    from src.core.procedure import STAGE_RULE_IDS
    stage=field(state.get('procedure'),'current_stage')
    ids=STAGE_RULE_IDS.get(stage,set())
    rows=[r for r in display_records(state.get('relevant_requirements')) if r.get('rule_id') in ids]
    st.markdown('##### Preparation for your current step')
    render_checklist(state)
    failed=[r for r in rows if r.get('status')=='failed']
    if failed:
        st.markdown('**Problems to review**')
        for row in failed:st.warning(user_text(row.get('description','Review the relevant information.')))
    st.caption('The complete rule assessment, including later-stage and advisory items, is available in Technical.')
    with st.expander('Sources and explanations',expanded=False):
        from src.rag.ingestion import source_catalogue
        catalogue=source_catalogue();sources={sid for row in rows for sid in row.get('source_ids',[])}
        for sid in sorted(sources):
            source=catalogue.get(sid,{})
            if source.get('url'):st.markdown('['+source['title']+']('+source['url']+')')
            elif source.get('title'):st.caption(source['title']+' - team-reported direct IRO instruction.')
    if st.button('Re-check saved information',disabled=bool(snapshot.next),key=f'recheck_{thread_key}'):run_event(graph,config,UserEvent(kind='reevaluate'))



def render_procedure_confirmations(snapshot,graph,config,thread_key):
    if field(snapshot.values.get("procedure"),"current_stage")=="DOCUMENT_COLLECTION":
        collection_declarations(snapshot.values,graph,config,thread_key,bool(snapshot.next))


def render_legal_stay_technical(snapshot,graph,config,thread_key):
    from src.core.legal_stay import assess_legal_stay
    state=snapshot.values
    assessment=assess_legal_stay(state)
    with st.container(border=True):
        st.markdown("##### Legal stay and preparation target")
        if assessment.estimated_or_calculated_end:
            st.write("Estimated / supplied stay limit:",assessment.estimated_or_calculated_end)
            st.write("Agent recommended completion target:",assessment.recommended_target)
        else:st.info("More entry information is needed before estimating a stay limit.")
        st.caption(assessment.confidence_type.replace("_"," "))
        for assumption in assessment.assumptions:st.caption(assumption)
        st.caption("The recommended target is not an official deadline. An estimate does not update your confirmed legal-expiry field.")
    c=state.get("residence_permit_context")
    with st.expander("Confirm legal-stay calculation inputs",expanded=False):
        st.caption("Confirm actual entry conditions. Only Italian/German ordinary visa-exempt passports and explicitly supplied visa duration/validity are supported. Unknown past stays produce an estimate.")
        with st.form(f"stay_inputs_{thread_key}"):
            answers={}
            for name,label in (("nationality","Nationality / ISO code"),("latest_entry_date","Latest entry date (YYYY-MM-DD)"),("visa_valid_until","Visa validity end (YYYY-MM-DD)")):
                value=field(c,name)
                text=st.text_input(label,value=str(value) if value is not None else "",disabled=bool(snapshot.next))
                answers[name]=text.strip() or None
            for name,label,choices in (("passport_type","Passport type",["Not confirmed","ordinary","other"]),("visa_status","Actual entry status",["Not confirmed","visa_exempt","visa","evisa","other"])):
                current=field(c,name)
                choice=st.selectbox(label,choices,index=choices.index(current) if current in choices else 0,disabled=bool(snapshot.next))
                answers[name]=None if choice=="Not confirmed" else choice
            days=st.text_input("Permitted days shown on visa (if applicable)",value=str(field(c,"visa_stay_days") or ""),disabled=bool(snapshot.next))
            history=st.text_area("Previous Türkiye stays: JSON start/end pairs; [] means confirmed none; blank means unknown",value=field(c,"previous_stay_history") or "",disabled=bool(snapshot.next))
            answers["previous_stay_history"]=history.strip() or None
            confirmed=st.checkbox("I confirm these entry conditions and travel-history statements",disabled=bool(snapshot.next))
            submitted=st.form_submit_button("Save entry conditions",disabled=bool(snapshot.next))
        if submitted:
            if not confirmed:st.warning("Confirm your statements before saving.")
            elif days and not days.isdigit():st.warning("Permitted visa days must be a whole number.")
            else:
                answers["visa_stay_days"]=int(days) if days else None
                if invoke(graph,ui_run(UserEvent(kind="confirm_information",document_id="__declarations__",confirmed_fields=answers),config["configurable"]["thread_id"]),config):st.rerun()



def render_legal_stay(snapshot,graph,config,thread_key):
    from src.tools.calendar import declared_reminder_candidates
    from src.document_processing.evidence import normalized_date
    c=snapshot.values.get('residence_permit_context')
    with st.container(border=True):
        st.markdown('##### Legal stay expiry date')
        st.caption('Not sure about this date? Check the official Turkish Ministry of Foreign Affairs visa information.')
        st.markdown('[Official MFA visa information](https://www.mfa.gov.tr/visa-information-for-foreigners.en.mfa)')
        with st.form(f'declared_expiry_{thread_key}'):
            current=field(c,'legal_stay_expiry_date')
            value=st.text_input('Legal stay expiry date (YYYY-MM-DD)',value=str(current) if current else '',disabled=bool(snapshot.next))
            acknowledged=st.checkbox('I confirm this is my declared legal-stay expiry date',disabled=bool(snapshot.next))
            saved=st.form_submit_button('Save legal-stay expiry',disabled=bool(snapshot.next))
        if saved:
            if not acknowledged:st.warning('Confirm the date before saving it.')
            elif normalized_date(value) is None:st.warning('Enter a valid date as YYYY-MM-DD.')
            elif invoke(graph,ui_run(UserEvent(kind='confirm_information',document_id='__declarations__',confirmed_fields={'legal_stay_expiry_date':value}),config['configurable']['thread_id']),config):st.rerun()
        rows=declared_reminder_candidates(snapshot.values)
        for row in rows:
            if row['candidate_id']=='declared_insurance_expiry':continue
            day=date.fromisoformat(row['date'])
            st.write(('Your declared legal-stay expiry is: ' if row['candidate_id']=='declared_legal_stay' else 'Recommended preparation reminder: ')+day.strftime('%d %B %Y'))
            if row['candidate_id']=='declared_preparation_target':st.caption('30 days earlier. Agent recommendation - not an official deadline.')
            label='Add reminder' if row['candidate_id']=='declared_legal_stay' else 'Add recommended preparation reminder'
            if st.button(label,key=f"declared_add_{thread_key}_{row['candidate_id']}",disabled=bool(snapshot.next)):
                from src.ui.actions import propose_reminder
                if propose_reminder(graph,config,row):st.rerun()
