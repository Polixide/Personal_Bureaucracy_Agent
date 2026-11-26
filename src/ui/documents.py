from src.workers.verification import (
    FORM_CONTEXT_FIELDS, accepted_sources, cached_conflict_fields, comparison_key, declaration_fields,
    document_editable_fields, document_field_targets,
)

CHECKLIST_ICONS = {"complete": ":green[:material/check_circle:]", "needs confirmation": ":orange[:material/rate_review:]",
                   "missing": ":gray[:material/radio_button_unchecked:]", "ambiguous": ":orange[:material/help:]",
                   "uploaded": ":gray[:material/upload_file:]", "conflict": ":orange[:material/warning:]",
                   "not_applicable": ":gray[:material/remove:]"}
CHECKLIST_LABELS = {"complete": "Ready", "needs confirmation": "Needs review", "missing": "Missing",
                    "ambiguous": "Needs review", "uploaded": "Uploaded", "conflict": "Conflict detected",
                    "warning": "Warning", "not_applicable": "Not applicable"}
from src.core.address_proof import ADDRESS_OPTIONS, HOST, RESIDENCE_DOCUMENTS
DETAIL_CONFLICT_FIELDS = {
    "passport_expiry_date", "requested_permit_end_date", "latest_entry_date", "insurance_evidence_type",
    "visa_required", "visa_page_present", "passport_id_page_present", "turkey_entry_page_present",
} | FORM_CONTEXT_FIELDS.keys()



"""Focused Streamlit presentation; all decisions use canonical backend functions."""
from html import escape

from src.ui.components import *


DOCUMENT_KEY_FIELDS = {
    "passport_copy": ("holder_name", "passport_number", "nationality", "expiry_date"),
    "turkiye_entry_stamp": ("latest_entry_date", "entry_stamp_present"),
    "health_insurance": ("holder_name", "insurer", "policy_number", "coverage_end_date"),
    "student_certificate": ("holder_name", "institution", "program_start_date", "program_end_date"),
    "address_form": ("first_name", "last_name", "passport_number", "declared_address", "signature_present"),
    "utility_bill": ("holder_name", "service_address", "provider", "billing_date"),
    "ikamet_application_form": ("applicant_name", "application_number", "requested_start_date", "requested_end_date"),
    "dorm_certificate": ("holder_name", "declared_address", "dorm_certificate_format"),
    "rental_agreement": ("holder_name", "declared_address", "landlord_name", "notary_certification"),
    "notarized_commitment": ("holder_name", "declared_address", "host_name", "notary_certification"),
    "host_support": ("host_name", "declared_address"),
    "utility_subscription": ("holder_name", "service_address", "provider"),
    "card_fee_receipt": ("payment_code", "amount", "payment_date", "reference_number", "taxpayer_id", "holder_name"),
    "tax_id_certificate": ("tax_identification_number", "holder_name", "passport_number"),
}
ACCEPTED_FIELD_STATUSES = {"source_grounded", "user_confirmed"}
DOCUMENT_READ_METHODS = {
    "native_text": "Native PDF text", "pdf_form": "Filled PDF field", "rapidocr": "RapidOCR",
    "openai_vision": "OpenAI Vision",
}
DOCUMENT_FIELD_GROUPS = {
    "Identity": {"holder_name", "applicant_name", "first_name", "last_name", "birth_date", "nationality",
                 "passport_number", "passport_or_foreign_id", "foreigner_id", "tax_identification_number",
                 "host_name", "landlord_name"},
    "Address and contact": {"declared_address", "service_address", "phone", "provider", "billing_date"},
    "Studies and coverage": {"student_id", "institution", "exchange_student", "program_start_date", "program_end_date",
                             "insurer", "policy_number", "coverage_start_date", "coverage_end_date",
                             "insurance_evidence_type", "residence_permit_purpose"},
    "Dates, application and payment": {"issue_date", "expiry_date", "form_date", "latest_entry_date", "application_number",
                                      "permit_type", "requested_start_date", "requested_end_date",
                                      "reference_number", "amount", "payment_date", "taxpayer_id",
                                      "payment_code", "document_reference", "issuing_authority"},
    "Review confirmations": {"signature_present", "signed", "identity_page_present", "entry_stamp_present",
                             "application_form_signed", "online_application_completed", "insurance_valid_for_ikamet",
                             "document_present", "receipt_present"},
}


def fields_to_review(review):
    return [item for item in review["fields"] if item["issue"]]


def document_status(document, review, requirement, *, conflicted=False, duplicate=False):
    status = field(document, "status")
    if status == "pending":
        return "Analyzing"
    if status == "rejected":
        return "Analysis failed"
    if field(document, "doc_type") == "unknown":
        return "Unsupported"
    if status != "analyzed" or duplicate or not review["can_confirm"]:
        return "Needs review"
    if conflicted or review.get("status") == "conflict" or requirement == "conflict":
        return "Conflict detected"
    if requirement == "not_applicable":
        return "Not applicable"
    return "Ready" if review.get("status") == "complete" and requirement in (None, "complete") else "Needs review"


def render_document_values(fields, *, original=False, include_missing=False):
    if not fields:
        return
    facts = []
    for item in fields:
        name = item["name"]
        reading = item["observation"]
        confirmed = item["origin"] == "user_confirmed"
        value = item["value"]
        if not confirmed and item["status"] != "source_grounded":
            value = reading
        if value is None and not include_missing:
            continue
        if value is None:
            provenance = "Cleared by you" if confirmed else "Uncertain reading" if item["status"] == "ambiguous" else "Not recorded"
        elif confirmed:
            provenance = "Confirmed in Your details" if item.get("from_declaration") else "Confirmed by you"
        elif item["status"] == "source_grounded":
            provenance = "From document"
            if item.get("source_page"):
                provenance += f" · Page {item['source_page']}"
        else:
            provenance = "Uncertain reading" if item["status"] == "ambiguous" else "Read from document · verify"
        method = DOCUMENT_READ_METHODS.get(item.get("observation_method"))
        if method and reading is not None and not confirmed:
            provenance = f"{method} · Page {item['source_page']}" if item.get("source_page") else method
            if item["status"] != "source_grounded":
                provenance += " · Verify against the original"
        if item.get("identifier_untyped"):
            provenance += " · Passport or foreign ID type unclear"
        original_reading = ""
        if original and confirmed and reading is None and item.get("document_field_absent"):
            original_reading = "<small>Document reading · No value captured</small>"
        if original and reading is not None and (confirmed or reading != item["value"]):
            reading_label = "Original reading"
            observation_field = item.get("observation_field", name)
            if observation_field != item.get("display_field", name):
                reading_label += f" ({field_label(observation_field)})"
            source = f" · {method}" if method else ""
            original_reading = f"<small>{escape(reading_label + source)} · {escape(user_value(name, reading))}</small>"
        displayed = user_value(name, value) if value is not None else "—"
        facts.append(
            f"<div><dt>{escape(field_label(item.get('display_field', name)))}</dt><dd>{escape(displayed)}</dd>"
            f"<small>{escape(provenance)}</small>{original_reading}</div>"
        )
    if not facts:
        return
    st.html("""<style>
    .pba-document-values {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(min(11rem, 100%), 1fr));
        gap: 0.65rem 1rem; margin: 0; padding: 0;
    }
    .pba-document-values > div { padding: 0.55rem 0; min-width: 0; }
    .pba-document-values dt { font-size: 0.82rem; }
    .pba-document-values dd { margin: 0.12rem 0 0; font-weight: 620; overflow-wrap: anywhere; }
    .pba-document-values small { display: block; font-weight: 400; opacity: 0.88; }
    </style>""" + '<dl class="pba-document-values">' + "".join(facts) + "</dl>")


def render_full_document_information(state, document):
    with st.expander("View all information"):
        review = document_review(state, document, full=True)
        remaining = {item["name"]: item for item in review["fields"]}
        if not remaining:
            st.caption("No structured information is saved for this file.")
            return
        if field(document, "status") != "analyzed":
            st.caption("The saved readings are available for review while this file is inactive.")
        for label, names in DOCUMENT_FIELD_GROUPS.items():
            items = [remaining.pop(name) for name in list(remaining) if name in names]
            if items:
                st.markdown(f"**{label}**")
                render_document_values(items, original=True, include_missing=True)
        if remaining:
            st.markdown("**Other saved information**")
            render_document_values(list(remaining.values()), original=True, include_missing=True)
        for message in dict.fromkeys(validation_message(issue)
                                     for issue in field(field(document, "info"), "validation_issues", [])):
            st.caption(message)
        info = document.info
        if info is not None:
            uncertain = {
                (item.field, item.page_number, item.original_value) for item in info.source_evidence
                if item.field in info.ambiguous_fields and item.original_value
            }
            for name, page, reading in sorted(uncertain):
                st.text(f"{field_label(name)} · Uncertain reading on page {page} · {reading}")
        prior = document.prior_observations
        if prior and info is not None:
            current = info.model_dump(mode="json", include=info.observation_fields(), exclude_none=True)
            changes = [
                {"Field": field_label(name), "Previous reading": user_value(name, prior[name]) if name in prior else "—",
                 "Latest reading": user_value(name, current[name]) if name in current else "—"}
                for name in sorted(prior.keys() | current.keys()) if prior.get(name) != current.get(name)
            ]
            if changes:
                st.markdown("**Readings changed by reanalysis**")
                st.table(changes)
            else:
                st.caption("The latest reading matches the previous saved extraction.")


def _passport_reference(state, document, name):
    field_name = "passport_or_foreign_id"
    entry = state.get("verification_records", {}).get(document.doc_id, {}).get(name)
    if name not in {"passport_number", field_name} or document_field_targets(document, name, entry) != (field_name,):
        return None
    sources = accepted_sources(state)
    rows = sources.get(field_name, ())
    if field_name in cached_conflict_fields(state, sources=sources) or len({
        comparison_key(field_name, row["value"]) for row in rows
    }) != 1:
        return None
    passports = [row["value"] for row in rows
                 if row["document_type"] == "passport_copy" and row["document_id"] != document.doc_id]
    return passports[0] if len(passports) == 1 else None


def review_form(review, document, graph, config, thread_key, disabled, *, state, checkpoint=None,
                detailed=False, draft_only=False, conflicting_fields=()):
    scope = f"{thread_key}_document_{document.doc_id}"
    draft = editor_draft(scope)
    if not review["can_confirm"]:
        st.caption("Reprocess this saved document before confirming its information.")
        return
    edit = not draft_only and st.checkbox(
        "Edit accepted fields", key=f"edit_{thread_key}_{document.doc_id}", disabled=disabled,
    )
    if draft or edit:
        optional = set(draft) | (document_editable_fields(document) if edit else set())
        review = document_review(state, document, include_optional=optional)
    unresolved = fields_to_review(review)
    if unresolved and not draft_only:
        st.markdown("**Missing information**" if any(item["issue"] == "missing" for item in unresolved)
                    else "**Needs your attention**")
    fields = unresolved + [
        item for item in review["fields"] if not item["issue"] and (
            draft_only or (edit and item["name"] in optional) or item["name"] in draft or item["name"] in conflicting_fields
        )
    ]
    unsupported = sorted(set(draft) - {item["name"] for item in review["fields"]})
    if unsupported:
        st.warning("Draft entries kept from another document type · " + ", ".join(field_label(name) for name in unsupported)
                   + ". Choose the matching type to review them.")
    if fields and document.doc_type in {HOST, "host_support"}:
        st.caption("For supporting pages, choose Yes when the page is included in this file. You can upload other pages separately.")
    elif fields and document.doc_type in RESIDENCE_DOCUMENTS:
        st.caption("Check the original before confirming certification or QR markings.")
    inputs, prefilled = {}, {}
    for item in fields:
        name = item["name"]
        already_accepted = item["status"] in ACCEPTED_FIELD_STATUSES
        value = item["value"] if already_accepted and not draft_only else None
        default_value = item["suggestion"] if not draft_only else None
        label = field_label(item.get("display_field", name))
        if item["message"]:
            st.caption(item["message"])
        elif item["issue"] == "missing":
            st.caption(f"{label} is missing. Check the original document.")
        elif item["issue"] == "conflict" or name in conflicting_fields:
            st.caption(f"Compare {label.lower()} across the listed sources before correcting it.")
        reference = _passport_reference(state, document, name) if default_value is None else None
        if reference:
            st.caption(f"Accepted passport reference · {reference}. Compare it with this document before saving.")
        if already_accepted and name in draft:
            st.caption("Accepted value · " + user_value(name, value))
        if name == "insurance_evidence_type":
            options = ["Not confirmed", *INSURANCE_TYPES]
            widget = draft_widget(scope, draft, name, default_value if default_value in options else options[0])
            chosen = st.selectbox(label, options, index=None, format_func=lambda item: user_value(name, item),
                                  disabled=disabled, **widget)
            answer = None if chosen == "Not confirmed" else chosen
        elif item["boolean"]:
            default = "Not confirmed" if type(default_value) is not bool else "Yes" if default_value else "No"
            widget = draft_widget(scope, draft, name, default)
            choice = st.selectbox(label, ["Not confirmed", "Yes", "No"], index=None, disabled=disabled, **widget)
            answer = {"Yes": True, "No": False}.get(choice)
        else:
            widget = draft_widget(scope, draft, name, "" if default_value is None else str(default_value))
            text = st.text_input(label + (" (YYYY-MM-DD)" if "date" in name else ""),
                                 value=None, disabled=disabled, **widget)
            answer = (text or "").strip() or None
        if name in draft and answer != jsonable(value):
            inputs[name] = answer
        elif (item["issue"] in {"uncertain", "declaration"} and item["observation"] is not None
              and answer == item["observation"] and answer not in (None, False)):
            prefilled[name] = answer

    if draft:
        st.caption("Unconfirmed edits stay in this browser session. Save and confirm resolved values before reloading.")
        st.button("Discard draft", key=f"discard_{scope}", on_click=draft.clear, disabled=disabled)
    if not fields or draft_only:
        return
    if prefilled:
        st.caption("Check the prefilled readings against the original. Saving records your edits or confirms the prefilled readings if you leave them as shown.")
    if not st.button("Save and confirm", key=f"save_review_{scope}", type="primary", disabled=disabled):
        return
    inputs = inputs or prefilled
    if not inputs:
        st.info("Enter or change an answer before saving.")
        return
    event = UserEvent(kind="confirm_information", document_id=document.doc_id, confirmed_fields=inputs)
    payload = ui_run(event, config["configurable"]["thread_id"])
    if invoke(graph, payload, config, expected_checkpoint=checkpoint):
        latest = graph.get_state(config).values
        if latest.get("trace_id") == payload["trace_id"] and not latest.get("validation_errors"):
            for name in inputs:
                draft.pop(name, None)
            st.session_state[f"confirmed_document_{thread_key}"] = payload["trace_id"]
        st.rerun()


def identity_review(state, graph, config, thread_key, disabled, technical=True):
    detail = conflict_details(state).get("holder_name")
    if not detail or detail["resolved"]:
        return
    st.warning("The holder names in your documents do not match. Compare the originals and choose the correct name.")
    rows = detail["observations"]

    def choice_label(index):
        row = rows[index]
        return f"{row['value']} — {document_label(row['document_type'])}"

    for index in range(len(rows)):
        st.write(choice_label(index))
    with st.form(f"identity_{thread_key}_{state.get('trace_id', 'initial')}"):
        choice = st.selectbox("Name to use for your case", range(len(rows)), index=None,
                              placeholder="Choose a name after checking the documents",
                              format_func=choice_label, disabled=disabled)
        submit = st.form_submit_button("Confirm chosen identity", disabled=disabled)
    if submit:
        if choice is None:
            st.warning("Choose the correct name from the observations above.")
        else:
            run_event(graph, config, UserEvent(kind="resolve_conflict", confirmed_fields={"holder_name": rows[choice]["value"]}))


def declaration_form(state, graph, config, thread_key, disabled):
    context = state.get("residence_permit_context")
    declarations = (state.get("verification_records") or {}).get("__declarations__", {})
    conflicts = set(state.get("verification_conflicts") or [])
    regular_fields = ("is_in_turkiye", "entered_with_passport", "application_date")
    allowed = declaration_fields(state)
    conflict_fields = sorted(conflicts & declarations.keys() & DETAIL_CONFLICT_FIELDS & allowed.keys())
    saved = {name: jsonable(field(declarations.get(name), "value") if name in conflicts else field(context, name))
             for name in (*regular_fields, *conflict_fields)}
    st.markdown("##### Entry and application plans")
    with st.form(f"declarations_{thread_key}_{state.get('trace_id', 'initial')}"):
        answers = {}
        for name in regular_fields[:2]:
            value = saved[name]
            choice = st.selectbox(CONTEXT_QUESTIONS[name], ["Not answered", "Yes", "No"],
                                  index=0 if value is None else 1 if value else 2, disabled=disabled)
            answers[name] = {"Yes": True, "No": False}.get(choice)
        value = st.text_input("Planned application date (YYYY-MM-DD)", value=saved["application_date"] or "", disabled=disabled)
        answers["application_date"] = value.strip() or None
        if conflict_fields:
            st.markdown("**Saved answers that conflict with document evidence**")
            st.caption("Compare the original documents. Edit the saved answer here or clear it if you cannot confirm it.")
        for name in conflict_fields:
            value = saved[name]
            label = field_label(name)
            if allowed[name] == "date":
                answer = st.text_input(label + " (YYYY-MM-DD)", value=value or "", disabled=disabled).strip() or None
            elif allowed[name] == "str":
                answer = st.text_input(label, value=value or "", disabled=disabled).strip() or None
            elif allowed[name] == "insurance_type":
                options = ["Not answered", *INSURANCE_TYPES]
                choice = st.selectbox(label, options, index=options.index(value) if value in options else 0,
                                      format_func=lambda item: user_value("insurance_evidence_type", item), disabled=disabled)
                answer = None if choice == "Not answered" else choice
            else:
                choice = st.selectbox(label, ["Not answered", "Yes", "No"],
                                      index=0 if value is None else 1 if value else 2, disabled=disabled)
                answer = {"Yes": True, "No": False}.get(choice)
            answers[name] = answer
        submit = st.form_submit_button("Save and confirm answers", type="primary", disabled=disabled)
    if submit:
        changes = {name: answer for name, answer in answers.items() if answer != saved[name]}
        if changes:
            form_changes = {name: value for name, value in changes.items() if name in FORM_CONTEXT_FIELDS}
            if form_changes:
                from src.core.procedure import checked_form_answers

                try:
                    changes.update(checked_form_answers(state, form_changes))
                except ValueError:
                    st.warning("Compare the saved answer with your document and enter a valid value.")
                    return
            run_event(graph, config, UserEvent(kind="confirm_information", document_id="__declarations__", confirmed_fields=changes))
        else:
            st.info("Your saved answers are already up to date.")


def evidence_status(row, documents_by_type, conflict_ids):
    status = row["status"]
    kind = row.get("document_type")
    if not kind:
        return status
    documents = documents_by_type.get(kind, [])
    if any(field(doc, "doc_id") in conflict_ids for doc in documents):
        return "conflict"
    if status in {"missing", "needs confirmation"} and documents:
        return "uploaded" if all(field(doc, "status") == "pending" for doc in documents) else "needs confirmation"
    return status


def document_requirements(state, checklist):
    rows = list(checklist)
    declarations = state.get("verification_records", {}).get("__declarations__", {})
    exemption = declarations.get("insurance_exempt")
    if field(exemption, "origin") == "user_confirmed" and field(exemption, "value") is True:
        rows = [{**row, "status": "not_applicable"} if row.get("key") == "health_insurance"
                and row["status"] == "complete" else row for row in rows]
    return rows


def lifecycle_button(graph,config,thread_key,doc_id,kind,label,disabled=False,**kwargs):
    if st.button(label,key=f"{kind}_{thread_key}_{doc_id}",disabled=disabled):
        target=kwargs.pop("target_document_id",doc_id)
        event=UserEvent(kind=kind,document_id=target,**kwargs)
        if invoke(graph,ui_run(event,config["configurable"]["thread_id"]),config):st.rerun()


def render_analysis_status(document, thread_key):
    if not isinstance(document, Document):
        return
    analysis = document.analysis_diagnostics
    route = analysis.get("processing_mode")
    cloud = analysis.get("selected_provider") == "openai"
    if analysis.get("outcome") == "failed":
        st.caption("OpenAI analysis failed." if cloud else "Local extraction failed.")
        if document.info is not None:
            st.caption("The earlier extraction is still available below.")
    elif document.info is not None:
        if route == "cloud vision":
            st.caption("Saved extraction · OpenAI Vision analyzed PDF page images.")
        elif route == "cloud text":
            st.caption("Saved extraction · OpenAI analyzed text.")
        elif route == "local OCR":
            st.caption("Saved extraction · Local OCR.")
        elif route in {"native PDF text", "native text"}:
            st.caption(f"Saved extraction · {route[0].upper() + route[1:]}.")
        else:
            st.caption("Saved extraction · Detailed request metadata is unavailable for this older result.")
        if analysis.get("cache_hit"):
            st.caption("Reused a previously saved local extraction.")
    if cloud_documents_available(thread_key):
        kind = "OpenAI Vision is available for this PDF" if Path(document.file_path).suffix.lower() == ".pdf" else "OpenAI text analysis is available"
        st.caption(f"{kind}. Choose the OpenAI action in Document options to authorize another analysis.")


def render_document(document, state, graph, config, thread_key, disabled, *, requirement=None,
                    conflicted=False, checkpoint=None, conflicting_fields=()):
    doc_id = field(document, "doc_id")
    kind = field(document, "doc_type")
    status = field(document, "status")
    info = field(document, "info")
    review = document_card(state, document)
    attention = fields_to_review(review)
    duplicate = kind not in {"unknown", "host_support"} and status == "analyzed" and sum(
        field(doc, "doc_type") == kind and field(doc, "status") == "analyzed" for doc in state.get("documents", [])
    ) > 1
    label = document_status(document, review, requirement, conflicted=conflicted, duplicate=duplicate)
    can_review = status == "analyzed" and review["can_confirm"]
    with st.container(border=True):
        st.markdown(f"### {document_label(kind)}")
        color = {"Ready": "green", "Conflict detected": "red", "Needs review": "orange",
                 "Analysis failed": "red"}.get(label, "gray")
        st.markdown(f":{color}[**{label}**]")
        filename = field(document, "original_filename") or Path(field(document, "file_path") or "").name
        if filename:
            st.text(filename)
        render_analysis_status(document, thread_key)
        saved_key = f"confirmed_document_{thread_key}"
        event = state.get("event")
        if (st.session_state.get(saved_key) == state.get("trace_id") and st.session_state.get(saved_key)
                and field(event, "document_id") == doc_id and field(event, "kind") == "confirm_information"):
            st.session_state.pop(saved_key)
            if not state.get("validation_errors"):
                st.success("Your changes were saved and confirmed.")
        if field(document, "processing_error"):
            st.error(friendly_error(str(field(document, "processing_error"))))
        if duplicate:
            st.warning("More than one saved document has this type. Choose the current version.")
            lifecycle_button(graph, config, thread_key, doc_id, "replace_document",
                             "Use this version and replace other versions", disabled=disabled, confirmed=True)
        if status == "replacement_pending":
            st.warning("Choose whether this file replaces the current document.")
            lifecycle_button(graph, config, thread_key, doc_id, "replace_document",
                             "Replace current document", disabled=disabled, confirmed=True)
            lifecycle_button(graph, config, thread_key, doc_id, "cancel_document",
                             "Cancel replacement", disabled=disabled, confirmed=True)
        if info is not None and not isinstance(info, tuple(DOCUMENT_SCHEMAS.values())):
            st.warning("Reprocess this older extraction before confirming its values.")
        if label == "Unsupported":
            st.caption("Choose a document type to prepare corrections, or upload a clearer copy.")
        fields = {item["name"]: item for item in review["fields"]}
        keys = DOCUMENT_KEY_FIELDS.get(kind, tuple(fields)[:3])
        visible = [
            fields[name] for name in keys
            if name in fields and fields[name] not in (attention if can_review else ())
            and (fields[name]["value"] is not None or fields[name]["observation"] is not None)
        ]
        if visible:
            st.markdown("**Key information**")
            render_document_values(visible, original=True)
        if attention and not can_review:
            st.caption(f"{len(attention)} fields need review after processing.")
        if label == "Needs review" and review["status"] == "complete" and requirement not in (None, "complete"):
            st.caption("The document has been read. Check what this step still requires.")
            st.button("See checks and warnings", key=f"document_checks_{thread_key}_{doc_id}",
                      on_click=request_view, args=(thread_key, "checks"))
        if kind == "address_form" and any(item["name"] == "signature_present" for item in attention):
            st.warning("Review the signature against the original, or upload a completed, signed copy.")
        if label == "Conflict detected":
            st.caption("Compare the conflicting readings above before confirming changes.")
        if not doc_id:
            return False
        if isinstance(document, Document):
            render_full_document_information(state, document)
        if can_review:
            review_form(review, document, graph, config, thread_key, disabled, state=state, checkpoint=checkpoint,
                        detailed=True, conflicting_fields=conflicting_fields)
        if info is not None and st.toggle("View extraction evidence", key=f"evidence_{thread_key}_{doc_id}"):
            render_extraction_evidence(info, thread_key, doc_id)
        if label == "Unsupported" or status == "rejected" or st.toggle("Document options", key=f"document_options_{thread_key}_{doc_id}"):
            render_document_options(document, state, graph, config, thread_key, disabled, checkpoint=checkpoint)
    return label == "Ready"


def render_document_options(document, state, graph, config, thread_key, disabled, *, checkpoint=None):
    doc_id = field(document, "doc_id")
    kind = field(document, "doc_type")
    status = field(document, "status")
    if isinstance(document, Document) and status == "analyzed":
        for name, entry in state.get("verification_records", {}).get(doc_id, {}).items():
            if field(entry, "origin") == "user_confirmed":
                lifecycle_button(graph, config, thread_key, doc_id + name, "remove_correction",
                                 f"Use document reading for {field_label(name).lower()}",
                                 disabled=disabled, confirmed_fields={name: None}, target_document_id=doc_id)
    if isinstance(document, Document) and (status != "replacement_pending" or kind == "unknown"):
        from src.workers.verification import PROFILE_TYPES

        profiles = {**PROFILE_TYPES, "student_certificate": "student_enrollment_certificate",
                    "health_insurance": "health_insurance_certificate",
                    "ikamet_application_form": "ikamet_application_form"}
        type_scope = f"{thread_key}_document_type_{doc_id}"
        picker = draft_widget(type_scope, editor_draft(type_scope), "type", kind if kind in profiles else None)
        selected = st.selectbox("Document type for reprocessing", list(profiles), index=None, disabled=disabled,
                                format_func=document_label, placeholder="Choose document type", **picker)
        draft = editor_draft(f"{thread_key}_document_{doc_id}")
        reprocess = st.button("Reprocess document", key=f"reprocess_document_{thread_key}_{doc_id}",
                              disabled=disabled or selected is None)
        cloud = False
        if cloud_documents_available(thread_key):
            from src.ui.model_settings import DOCUMENT_CONSENT

            st.caption(DOCUMENT_CONSENT)
            label = ("Analyze this document with OpenAI Vision" if Path(document.file_path).suffix.lower() == ".pdf"
                     else "Analyze this document with OpenAI")
            cloud = st.button(label, key=f"reprocess_cloud_{thread_key}_{doc_id}", disabled=disabled or selected is None)
        if reprocess or cloud:
            event = UserEvent(kind="reprocess_document", document_id=doc_id, document_type_hint=profiles[selected])
            payload = ui_run(event, config["configurable"]["thread_id"])
            if invoke(graph, payload, config, expected_checkpoint=checkpoint, allow_document_cloud=cloud):
                latest = graph.get_state(config).values
                updated = next((doc for doc in latest.get("documents", []) if doc.doc_id == doc_id), None)
                if updated and updated.doc_type != kind and latest.get("trace_id") == payload["trace_id"]:
                    for name, entry in state.get("verification_records", {}).get(doc_id, {}).items():
                        value = field(entry, "value")
                        if field(entry, "origin") == "user_confirmed" and value is not None:
                            draft.setdefault(name, ("Yes" if value else "No") if isinstance(value, bool) else str(value))
                st.rerun()
        if (kind == "unknown" or status == "rejected") and selected is not None:
            st.caption("Prepare draft answers for this type. Reprocess the document before confirming them.")
            preview = document.model_copy(update={"doc_type": selected, "status": "analyzed",
                                                  "info": DOCUMENT_SCHEMAS[profiles[selected]]()})
            preview_state = {**state, "verification_records": {}}
            review_form(document_review(preview_state, preview), preview, graph, config, thread_key, disabled,
                        state=preview_state, draft_only=True)
        elif (kind == "unknown" or status == "rejected") and draft:
            st.caption("Choose a document type to review the draft saved in this browser session.")
            st.button("Discard draft", key=f"discard_{thread_key}_document_{doc_id}",
                      on_click=draft.clear, disabled=disabled)
    checked = st.checkbox("I want to delete this document", disabled=disabled,
                          key=f"delete_confirm_{thread_key}_{doc_id}_{state.get('trace_id','initial')}")
    lifecycle_button(graph, config, thread_key, doc_id, "delete_document", "Delete",
                     disabled=disabled or not checked, confirmed=True)


def render_extraction_evidence(info, thread_key, doc_id):
    from src.ui.technical import open_advanced

    st.caption("Original readings from this upload. Confirm values against your document.")
    evidence = field(info, "source_evidence", []) or []
    if evidence:
        for entry in evidence:
            name = field(entry, "field")
            page = field(entry, "page_number")
            st.caption(f"{field_label(name) if name else 'Document text'} · Page {page}")
            st.text(field(entry, "snippet", ""))
    else:
        pages = field(info, "source_pages", {}) or {}
        for name, snippet in (field(info, "source_snippets", {}) or {}).items():
            location = f" · Page {pages[name]}" if name in pages else ""
            st.caption(field_label(name) + location)
            st.text(snippet)
        if not field(info, "source_snippets"):
            st.caption("No source excerpts are available for this upload.")
    st.button("Open full saved evidence", key=f"full_evidence_{thread_key}_{doc_id}",
              on_click=open_advanced, args=(thread_key, "Saved state and document evidence"))


def collection_declarations(state, graph, config, thread_key, disabled):
    from src.core.procedure import address_proof_status

    declarations = state.get("verification_records", {}).get("__declarations__", {})
    ready = field(declarations.get("biometric_photo"), "value")
    address = address_proof_status(state)
    proof = address["route"]
    guided = state.get("procedure") is not None

    with st.container(border=True):
        st.subheader("Address proof and photo readiness" if guided else "Address proof")
        if proof in ADDRESS_OPTIONS:
            st.markdown(f"**Saved address option** · {ADDRESS_OPTIONS[proof]}")
        if address["ready"]:
            st.success("Address evidence ready for this step")
        if proof is not None:
            st.caption(address["message"])
        for part in address["parts"]:
            st.write(f"{part['label']} · {CHECKLIST_LABELS[part['status']]}")
            if part["status"] != "complete":
                action = address_evidence_action(part)
                st.button(action["label"], key=f"address_evidence_{thread_key}_{part['key']}",
                          on_click=request_view, args=(thread_key, "documents", action["document_id"]),
                          kwargs={"document_type": part["document_type"]})
        if ready is True:
            st.caption("Biometric photo ready")
        elif ready is False:
            st.caption("Biometric photo still needed")

        with st.form(f"collection_{thread_key}_{state.get('trace_id', 'initial')}"):
            options = [None, *ADDRESS_OPTIONS]
            choice = st.selectbox("Address-proof route", options, index=options.index(proof) if proof in options else 0,
                                  format_func=lambda option: ADDRESS_OPTIONS.get(option, "Choose address evidence"),
                                  help="Save your choice to see the documents and review required for that method.",
                                  disabled=disabled)
            photo = None
            if guided:
                photo = st.selectbox("Do you have a biometric photo ready?", ["Not confirmed", "Yes", "No"],
                                     index=0 if ready is None else 1 if ready else 2, disabled=disabled)
            submit = st.form_submit_button("Save and confirm preparation answers", disabled=disabled)
        if submit:
            answers = {}
            if choice is not None and choice != proof:
                answers["address_proof_type"] = choice
            photo_answer = {"Yes": True, "No": False}.get(photo)
            if photo_answer is not None and photo_answer != ready:
                answers["biometric_photo"] = photo_answer
            if answers:
                run_event(graph, config, UserEvent(kind="confirm_information", document_id="__declarations__",
                                                   confirmed_fields=answers))
            elif proof is None and ready is None:
                st.warning("Choose an address option or record photo readiness before saving.")
            else:
                st.info("These answers are already saved.")


def view_documents(state, snapshot, graph, config, thread_key, *, summary):
    busy = bool(snapshot.next)
    conflicts = conflict_messages(state)
    conflict_ids = {item["document_id"] for item in conflicts if item.get("document_id")}
    conflicting_fields = {}
    for item in conflicts:
        conflict_ids.update(item.get("related_document_ids", []))
        for observation in item.get("observations", []):
            if observation.get("document_id"):
                owner = observation["document_id"]
                conflict_ids.add(owner)
                conflicting_fields.setdefault(owner, set()).add(observation["field_name"])
    documents = [doc for doc in state.get("documents") or [] if field(doc, "status") not in {"deleted", "cancelled"}]
    by_type = {}
    for doc in documents:
        by_type.setdefault(field(doc, "doc_type"), []).append(doc)
    rows = document_requirements(state, summary["items"])

    identity_review(state, graph, config, thread_key, busy, technical=False)
    for conflict in conflicts:
        if conflict["field"] == "holder_name":
            continue
        st.warning(conflict["message"])
        for observation in conflict.get("observations", []):
            provenance = "Confirmed by you" if observation["origin"] == "user_confirmed" else "From document"
            st.write(f"{observation['label']} · {user_value(conflict['field'], observation['value'])} · {provenance}")
        if conflict["document_id"]:
            st.button(conflict["label"], key=f"review_conflict_{thread_key}_{conflict['field']}",
                      on_click=request_view, args=(thread_key, "documents", conflict["document_id"]))
        editable = DETAIL_CONFLICT_FIELDS | {"is_in_turkiye", "entered_with_passport", "application_date", "legal_stay_expiry_date"}
        if conflict["field"] in editable and any(row["label"] == "Your details" for row in conflict.get("observations", [])):
            st.button("Review your saved answers", key=f"conflict_details_{thread_key}_{conflict['field']}",
                      on_click=request_view, args=(thread_key, "details"), kwargs={"detail_section": "entry"})

    if rows:
        workspace, requirements = st.columns([3, 2], gap="large")
        with requirements, st.container(border=True, key="document_tasks"):
            st.subheader("Required for this step")
            for index, row in enumerate(rows):
                if row["status"] == "not_applicable":
                    continue
                render_requirement_row(row, by_type, conflict_ids, thread_key, str(index))
                parts = [part for part in row.get("parts", ()) if part.get("document_type")]
                if parts:
                    with st.container(key=f"address_components_{thread_key}"):
                        if len(parts) > 1:
                            st.caption("Required together for this address-proof option")
                        for part in parts:
                            render_requirement_row(part, by_type, conflict_ids, thread_key, part["key"])
            st.caption("Ready means the evidence is prepared for this step.")
    else:
        workspace = st.container()
    action = primary_action(state, checklist=summary["items"], conflicts=conflicts, collection=summary["collection"])
    ready_transition = action and (
        action["kind"] in {"start_application", "review_application"}
        or (action["kind"] == "advance" and summary["stage"] == "PAYMENT")
    )
    document_rows = rows + [part for row in rows for part in row.get("parts", ()) if part.get("document_type")]
    with workspace:
        if ready_transition:
            render_ready_transition(action, snapshot, graph, config, thread_key)
        reviewed = render_document_workspace(state, graph, config, thread_key, busy, document_rows, documents, conflict_ids,
                                             checkpoint=checkpoint_id(snapshot), conflicting_fields=conflicting_fields)
        if not ready_transition and reviewed and any(
            row.get("document_type") == reviewed and row["status"] == "complete" for row in document_rows
        ):
            render_following_task(action, snapshot, graph, config, thread_key, reviewed)
        render_document_upload(state, graph, config, thread_key, busy)


def render_requirement_row(row, by_type, conflict_ids, thread_key, row_key):
    kind = row.get("document_type")
    status = row["status"] if "key" in row and "document_ids" in row else evidence_status(row, by_type, conflict_ids)
    key = f"document_task_{thread_key}_{row_key}"
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        if kind:
            document_id = next(iter(row.get("document_ids", [])), None)
            st.button(row["label"], type="tertiary", key=key, on_click=request_view,
                      args=(thread_key, "documents", document_id), kwargs={"document_type": kind})
        elif row.get("key") in {"address_proof", "biometric_photo"}:
            st.button(row["label"], type="tertiary", key=key, on_click=request_view,
                      args=(thread_key, "details"), kwargs={"detail_section": "address"})
        else:
            st.write(row["label"])
        icon = CHECKLIST_ICONS.get(status, CHECKLIST_ICONS["missing"])
        st.caption(f"{icon} {CHECKLIST_LABELS.get(status, 'Needs review')}")


def render_document_workspace(state, graph, config, thread_key, busy, rows, documents, conflict_ids, *,
                              checkpoint=None, conflicting_fields=None):
    target_key = f"document_target_{thread_key}"
    requested = st.session_state.pop(f"review_type_{thread_key}", None)
    focus = st.session_state.pop(f"review_document_{thread_key}", None)
    upload = st.session_state.pop(f"review_upload_{thread_key}", None)
    outstanding = [row.get("document_type") for row in rows
                   if row["status"] not in {"complete", "not_applicable"} and row.get("document_type")]
    documents.sort(key=lambda doc: (field(doc, "status") != "replacement_pending",
                                    field(doc, "doc_type") not in outstanding and field(doc, "status") == "analyzed"))
    options = {field(doc, "doc_id"): doc for doc in documents if field(doc, "doc_id")}
    if upload is not None:
        focus = next((doc_id for doc_id, doc in options.items() if field(doc, "file_path") == upload), focus)
    if focus in options:
        requested = field(options[focus], "doc_type")
    if requested:
        st.session_state[target_key] = requested
    target = st.session_state.get(target_key) or next(iter(outstanding), None)
    selection = f"selected_document_{thread_key}"
    remembered = f"last_selected_document_{thread_key}"
    previous = st.session_state.get(remembered)
    if requested or remembered not in st.session_state or (previous is not None and previous not in options):
        matching = next((doc_id for doc_id, doc in options.items() if field(doc, "doc_type") == target), None)
        previous = focus if focus in options else matching if target else next(iter(options), None)
    if selection not in st.session_state or st.session_state[selection] != previous:
        st.session_state[selection] = previous

    def remember_selection():
        st.session_state[remembered] = st.session_state[selection]

    st.subheader("Upload or review a document")
    selected = None
    if options:
        selected = st.selectbox("Saved document", list(options), index=None, format_func=lambda key: document_option(options[key]),
                                key=selection, placeholder="Choose a saved file", on_change=remember_selection)
        st.session_state[remembered] = selected
    if selected is None and target:
        st.caption(f"Working on {document_label(target)}")
    if selected in options:
        document = options[selected]
        kind = field(document, "doc_type")
        related = [row for row in rows if row.get("document_type") == kind
                   and ("document_ids" not in row or selected in row["document_ids"])]
        requirement = next((row["status"] for row in related if row["status"] != "complete"),
                           related[0]["status"] if related else None)
        if render_document(document, state, graph, config, thread_key, busy, requirement=requirement,
                           conflicted=selected in conflict_ids, checkpoint=checkpoint,
                           conflicting_fields=(conflicting_fields or {}).get(selected, ())):
            return kind


def render_document_upload(state, graph, config, thread_key, busy):
    with st.form(f"upload_{thread_key}", border=True):
        uploaded = st.file_uploader("Add a document (PDF or TXT)", type=["pdf", "txt"],
                                    key=f"file_{thread_key}", disabled=busy)
        submit = st.form_submit_button("Upload and analyze", type="primary", disabled=busy)
        cloud = False
        if cloud_documents_available(thread_key):
            from src.ui.model_settings import DOCUMENT_CONSENT

            st.caption(DOCUMENT_CONSENT)
            cloud = st.form_submit_button("Upload and analyze with OpenAI", disabled=busy)
    if submit or cloud:
        analyzed = submit_document_upload(uploaded, state, graph, config, save_file=save_upload,
                                          allow_document_cloud=cloud)
        if analyzed:
            st.session_state[f"review_upload_{thread_key}"] = str(analyzed)
            st.rerun()


def render_ready_transition(action, snapshot, graph, config, thread_key):
    from src.ui.actions import perform_primary_action

    with st.container(border=True):
        st.success(action["message"])
        if st.button(action["label"], type="primary", key=f"ready_transition_{thread_key}", disabled=bool(snapshot.next)):
            perform_primary_action(action, graph, config, thread_key)


def render_following_task(action, snapshot, graph, config, thread_key, reviewed):
    from src.ui.actions import perform_primary_action

    if not action or action.get("document_type") == reviewed:
        return
    st.markdown("**Next in your preparation**")
    if action["kind"] == "navigate":
        st.button(action["label"], key=f"follow_document_{thread_key}", disabled=bool(snapshot.next),
                  on_click=request_view, args=(thread_key, action["view"], action.get("document_id")),
                  kwargs={"document_type": action.get("document_type"), "detail_section": action.get("detail_section")})
    elif st.button(action["label"], key=f"follow_document_{thread_key}", disabled=bool(snapshot.next)):
        perform_primary_action(action, graph, config, thread_key)


def document_option(document):
    label = document_label(field(document, "doc_type"))
    name = field(document, "original_filename") or Path(field(document, "file_path") or "").name
    status = {"replacement_pending": "Replacement", "rejected": "Analysis failed",
              "pending": "Analyzing"}.get(field(document, "status"))
    return " · ".join(value for value in (label, name, status) if value)


def view_details(state, snapshot, graph, config, thread_key):
    identity_review(state,graph,config,thread_key,bool(snapshot.next),technical=False)
    declaration_form(state,graph,config,thread_key,bool(snapshot.next))
    entry=state.get('verification_records',{}).get('__derived__',{}).get('planned_stay_months')
    if entry:st.caption(f"Academic stay: approximately {field(entry,'value')} complete months, calculated from your exchange dates.")



def view_checks(state, snapshot, graph, config, thread_key, *, summary):
    from src.core.legal_stay import insurance_explanation
    from src.core.procedure import STAGE_RULE_IDS
    from src.ui.technical import open_advanced

    stage = field(state.get("procedure"), "current_stage")
    ids = STAGE_RULE_IDS.get(stage, set())
    checklist = summary["items"]
    st.subheader("Checks for this step")
    address_row = next((row for row in checklist if row.get("key") == "address_proof"), None)
    if address_row:
        with st.container(border=True):
            st.markdown("**Address proof**")
            st.caption(CHECKLIST_LABELS[address_row["status"]])
            if address_row["status"] != "complete":
                st.info(address_proof_status(state)["message"])
            for part in address_row["parts"]:
                st.write(f"{part['label']} · {CHECKLIST_LABELS[part['status']]}")
    warning = insurance_explanation(state) if "R015" in ids else None
    if warning:
        st.warning(warning)
    failed = [row for row in display_records(state.get("relevant_requirements"))
              if row.get("rule_id") in ids and row.get("status") == "failed"
              and (not warning or row["rule_id"] != "R015")
              and (not address_row or row["rule_id"] != "R017")]
    for row in failed:
        st.warning(user_text(row.get("description", "Review the relevant information.")))
    if not failed and not warning:
        if any(row["status"] != "complete" for row in checklist):
            st.info("Some information still needs review. Follow the next action to update your case.")
        elif checklist:
            st.success("The listed prerequisites for this step are ready.")
        else:
            st.caption("Record actual updates for your current step on its preparation page.")

    action = primary_action(state, checklist=checklist, collection=summary["collection"])
    if action and action["kind"] == "navigate":
        st.button(action["label"], key=f"check_next_{thread_key}", on_click=request_view,
                  args=(thread_key, action["view"], action.get("document_id")),
                  kwargs={"document_type": action.get("document_type"), "detail_section": action.get("detail_section")})
    st.button("See rule details and sources", key=f"check_sources_{thread_key}", on_click=open_advanced,
              args=(thread_key, "Rules and sources"))


def render_details_workspace(state, snapshot, graph, config, thread_key, *, summary):
    from src.ui.actions import render_reminders

    st.subheader("Your details")
    labels = {"entry": "Entry and dates", "reminders": "Reminders", "knowledge": "Knowledge base"}
    stage = field(state.get("procedure"), "current_stage")
    if stage in {"DOCUMENT_COLLECTION", "IRO_REVIEW"}:
        labels = {"address": "Address and photo", **labels}
    elif stage is None:
        labels["address"] = "Address proof"
    key = f"details_section_{thread_key}"
    requested = st.session_state.pop(f"requested_details_{thread_key}", None)
    if requested in labels:
        st.session_state[key] = requested
    elif st.session_state.get(key) not in labels:
        st.session_state[key] = next(iter(labels))
    section = st.radio("Details section", list(labels), format_func=labels.get, horizontal=True,
                       label_visibility="collapsed", key=key)
    if section == "address":
        if summary["collection"] and summary["collection"].ready:
            action = primary_action(state, checklist=summary["items"], collection=summary["collection"])
            if action and action["kind"] == "start_application":
                render_ready_transition(action, snapshot, graph, config, thread_key)
        collection_declarations(state, graph, config, thread_key, bool(snapshot.next))
    elif section == "entry":
        view_details(state, snapshot, graph, config, thread_key)
        render_legal_stay(snapshot, graph, config, thread_key)
    elif section == "knowledge":
        from src.ui.knowledge import render_knowledge

        render_knowledge(state, thread_key, busy=bool(snapshot.next))
    else:
        render_reminders(snapshot, graph, config, thread_key)


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
            submitted=st.form_submit_button("Confirm entry conditions and travel history",disabled=bool(snapshot.next))
        if submitted:
            if days and not days.isdigit():st.warning("Permitted visa days must be a whole number.")
            else:
                answers["visa_stay_days"]=int(days) if days else None
                if invoke(graph,ui_run(UserEvent(kind="confirm_information",document_id="__declarations__",confirmed_fields=answers),config["configurable"]["thread_id"]),config):st.rerun()



def render_legal_stay(snapshot,graph,config,thread_key):
    from src.document_processing.evidence import normalized_date
    c=snapshot.values.get('residence_permit_context')
    with st.container(border=True):
        st.markdown('##### Legal stay expiry date')
        st.caption('Not sure about this date? Check the official Turkish Ministry of Foreign Affairs visa information.')
        st.markdown('[Official MFA visa information](https://www.mfa.gov.tr/visa-information-for-foreigners.en.mfa)')
        with st.form(f'declared_expiry_{thread_key}'):
            current=field(c,'legal_stay_expiry_date')
            value=st.text_input('Legal stay expiry date (YYYY-MM-DD)',value=str(current) if current else '',disabled=bool(snapshot.next))
            saved=st.form_submit_button('Confirm my legal-stay expiry date',disabled=bool(snapshot.next))
        if saved:
            if normalized_date(value) is None:st.warning('Enter a valid date as YYYY-MM-DD.')
            elif normalized_date(value) == jsonable(current):st.info('This expiry date is already saved.')
            elif invoke(graph,ui_run(UserEvent(kind='confirm_information',document_id='__declarations__',confirmed_fields={'legal_stay_expiry_date':value}),config['configurable']['thread_id']),config):st.rerun()
