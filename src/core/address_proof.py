"""Address evidence for the supported first-application and IRO preparation flow."""
from src.core.state import ResidencePermitContext, VerifiedInput

DORM = "dorm_certificate"
RENTAL = "rental_agreement"
HOST = "notarized_commitment"
IRO_PAIR = "utility_bill_and_address_form"
RESIDENCE_DOCUMENTS = (DORM, RENTAL, HOST)
ADDRESS_OPTIONS = {
    DORM: "Dormitory certificate",
    RENTAL: "Rental agreement",
    HOST: "Notarized undertaking",
    IRO_PAIR: "Utility bill + completed Address Form",
}
ADDRESS_GUIDANCE = {
    DORM: "Ask Sabancı Dormitory Management for a Turkish certificate with a QR code. Use SU Ticket or visit the D2 Dorm Administration.",
    RENTAL: "Upload the notarized agreement in your name with the rental address and the landlord's details.",
    HOST: "Ask your host for a notarized undertaking and their supporting documents. The host can take their notarized lease or title deed to the notary.",
    IRO_PAIR: "Provide a utility bill and a completed, signed Address Form. Use this local preparation route when Sabancı IRO gives this instruction for your case. The public first-application checklist lists dormitory, rental and hosted evidence separately.",
}
REVIEW_LABELS = {
    "dorm_certificate_format": "Turkish certificate from Sabancı Dormitory Management with a QR code and an e-signature or a signature and stamp",
    "notary_certification": "Notary certification is visible on this document",
    "host_married": "Is the host married?",
    "host_is_turkish": "Is the host a Turkish citizen?",
    "host_utility_current": "Current electricity, water, gas or landline bill, or a current new subscription contract",
    "host_identity_copy": "Host's ID copy",
    "host_address_certificate": "Host's registered-address certificate",
    "host_population_record": "Turkish host's civil registry extract",
    "host_permit_copy": "International host's valid residence or work permit copy",
    "spouse_undertaking": "Spouse's notarized undertaking and ID copy",
}
HOST_SUPPORT_FIELDS = (
    "host_identity_copy", "host_address_certificate", "host_population_record",
    "host_permit_copy", "spouse_undertaking",
)
ADDRESS_ATTESTATIONS = {
    DORM: ("dorm_certificate_format",),
    RENTAL: ("notary_certification",),
    HOST: ("notary_certification", "host_married", "host_is_turkish", *HOST_SUPPORT_FIELDS),
    "host_support": HOST_SUPPORT_FIELDS,
    "utility_bill": ("host_utility_current",),
    "utility_subscription": ("host_utility_current",),
}
ADDRESS_REQUIRED_YES = frozenset({"dorm_certificate_format", "notary_certification", "host_utility_current"})
ADDRESS_PART_LABELS = {
    **{kind: ADDRESS_OPTIONS[kind] for kind in RESIDENCE_DOCUMENTS},
    "address_form": "Address Form",
    "utility_bill": "Utility bill",
    "utility_subscription": "Utility subscription contract",
    "host_support": "Host supporting pages",
}


def selected_address_method(state):
    """Read the confirmed choice, with a legacy context value as a selection only."""
    raw = state.get("verification_records", {}).get("__declarations__", {}).get("address_proof_type")
    if raw is not None:
        record = VerifiedInput.model_validate(raw)
        if (record.origin == "user_confirmed" and record.confirmed_by and record.trace_id
                and record.document_id == "__declarations__" and record.field_name == "address_proof_type"):
            return record.value
    context = state.get("residence_permit_context") or {}
    return getattr(context, "address_proof_type", None) if isinstance(context, ResidencePermitContext) else context.get("address_proof_type")


def host_support_fields(married, turkish):
    fields = ["host_identity_copy", "host_address_certificate"]
    if turkish is not None:
        fields.append("host_population_record" if turkish else "host_permit_copy")
    if married is True:
        fields.append("spouse_undertaking")
    return fields


def compare_host_evidence(host, support, *, name_field, address_field):
    from src.workers.verification import comparison_key

    checks = {
        "host_name": ("holder_name", host.get("host_name"), support.get(name_field)),
        "host_address": ("residential_address", host.get("declared_address"), support.get(address_field)),
    }
    return {field: comparison_key(role, left) == comparison_key(role, right) if left is not None and right is not None else None
            for field, (role, left, right) in checks.items()}


def _part(kind, documents, reviews, conflicted, *, label=None, accepted=None):
    active = [doc for doc in documents if doc.status == "analyzed"]
    ids = [doc.doc_id for doc in active or documents]
    if len(active) > 1:
        status = "ambiguous"
    elif active:
        doc, = active
        valid = reviews[doc.doc_id]["status"] == "complete" if accepted is None else doc.doc_id in accepted
        status = "conflict" if doc.doc_id in conflicted else "complete" if valid else "needs confirmation"
    elif not documents:
        status = "missing"
    else:
        status = "uploaded" if all(doc.status == "pending" for doc in documents) else "needs confirmation"
    return {"key": kind, "document_type": kind, "label": label or ADDRESS_PART_LABELS[kind],
            "document_ids": ids, "status": status}


def address_proof_status(state):
    """Derive document requirements and readiness from the active route and file reviews."""
    from src.workers.verification import conflict_details, document_review

    method = selected_address_method(state)
    context = ResidencePermitContext.model_validate(state.get("residence_permit_context") or {})
    supported = (state.get("active_procedure", "ikamet") == "ikamet"
                 and context.application_kind in (None, "first")
                 and context.permit_type in (None, "short_term")
                 and context.institution in (None, "Sabancı University")
                 and context.applicant_type in (None, "incoming_exchange"))
    result = {"route": method, "label": ADDRESS_OPTIONS.get(method, "Address proof"),
              "ready": False, "status": "missing", "parts": [], "supported": supported and method in ADDRESS_OPTIONS,
              "message": "Choose the address evidence you plan to provide."}
    if not supported or (method is not None and method not in ADDRESS_OPTIONS):
        result["status"] = "needs confirmation"
        result["message"] = "Review the address evidence for your application with IRO. This workflow covers Sabancı exchange students' first applications."
    if not result["supported"]:
        return result

    kinds = ({method} if method in RESIDENCE_DOCUMENTS else {"address_form", "utility_bill"})
    if method == HOST:
        kinds.update(("utility_bill", "utility_subscription", "host_support"))
    documents = {kind: [] for kind in kinds}
    for doc in state.get("documents", ()):
        if doc.doc_type in documents and doc.status != "deleted":
            documents[doc.doc_type].append(doc)
    conflicts = conflict_details(state)
    reviews = {doc.doc_id: document_review(state, doc, conflicts=conflicts)
               for copies in documents.values() for doc in copies if doc.status == "analyzed"}
    # Treat the normalized, accepted values as inputs. A stored checkbox or
    # legacy flag without the document and matching provenance cannot pass.
    values = {owner: {field["name"]: field["value"] for field in review["fields"]
                      if field["origin"] in {"source_grounded", "user_confirmed"}}
              for owner, review in reviews.items()}
    affected = {row["document_id"] for detail in conflicts.values() if not detail["resolved"]
                for row in detail["observations"] if row["document_id"] in reviews} if reviews else set()

    parts = result["parts"]
    if method == IRO_PAIR:
        signed_forms = {owner for owner, rows in values.items()
                        if rows.get("signature_present") is True and reviews[owner]["status"] == "complete"}
        parts.extend((_part("address_form", documents["address_form"], reviews, affected, accepted=signed_forms),
                      _part("utility_bill", documents["utility_bill"], reviews, affected)))
    else:
        parts.append(_part(method, documents[method], reviews, affected))
    if method == HOST:
        host = parts[0]
        entries = values.get(host["document_ids"][0], {}) if len(host["document_ids"]) == 1 else {}
        def matches(rows, name_field, address_field):
            return all(same is True for same in compare_host_evidence(
                entries, rows, name_field=name_field, address_field=address_field,
            ).values())

        bill_label = "Host utility bill or subscription contract"
        bills = []
        for kind in ("utility_bill", "utility_subscription"):
            accepted = {doc.doc_id for doc in documents[kind]
                        if (rows := values.get(doc.doc_id)) and reviews[doc.doc_id]["status"] == "complete"
                        and rows.get("host_utility_current") is True and matches(rows, "holder_name", "service_address")}
            bills.append(_part(kind, documents[kind], reviews, affected, label=bill_label, accepted=accepted))
        bill = next((row for row in bills if row["status"] == "complete"),
                    next((row for row in bills if row["status"] != "missing"), bills[0]))
        parts.append({**bill, "key": "host_utility", "message": "Check that the evidence is current and shows the undertaking host and the same address."})
        support = documents["host_support"]
        for field in host_support_fields(entries.get("host_married"), entries.get("host_is_turkish")):
            # Each supporting item may be included in the undertaking upload
            # or contained in a separate file reviewed for this host/address.
            accepted_id = next((doc.doc_id for doc in support if (rows := values.get(doc.doc_id))
                                and reviews[doc.doc_id]["status"] == "complete" and rows.get(field) is True
                                and matches(rows, "host_name", "declared_address")), None)
            if host["status"] == "complete" and entries.get(field) is True:
                accepted_id = host["document_ids"][0]
            ids = [accepted_id] if accepted_id else [doc.doc_id for doc in support]
            mismatch = any(owner in affected and values.get(owner, {}).get(field) is True for owner in ids)
            parts.append({"key": field, "document_type": "host_support", "label": REVIEW_LABELS[field],
                          "status": "complete" if accepted_id else "conflict" if mismatch
                                    else "needs confirmation" if ids else "missing",
                          "document_ids": ids,
                          "message": "Review the supporting pages in the undertaking or add a separate file for the same host and address."})
    statuses = {part["status"] for part in parts}
    result["ready"] = statuses == {"complete"}
    result["status"] = next((status for status in ("conflict", "ambiguous") if status in statuses),
                            "complete" if result["ready"] else "missing" if statuses == {"missing"} else "needs confirmation")
    result["message"] = ADDRESS_GUIDANCE[method]
    return result
