# Active V2 rule catalogue

## R001 ? exchange_stay_requires_ikamet

Incoming Sabancı exchange students staying 3 months or longer enter the residence-permit workflow.

Sources: SAB_EXCHANGE_OVERVIEW



## R002 ? permit_type_short_term

For the scoped Sabancı incoming exchange-student case, the workflow uses a short-term residence permit.

Sources: GOV_RESIDENCE_TYPES, SAB_EXCHANGE_OVERVIEW



## R003 ? application_after_entry

A first application is started only after entry into Türkiye; Sabancı instructs entry with passport.

Sources: SAB_EXCHANGE_OVERVIEW



## R004 ? apply_before_legal_stay_expiry

The online first application must be completed before visa/visa-exemption/current legal stay expires.

Sources: SAB_EXCHANGE_OVERVIEW, METU_RESIDENCE_2026



## R005 ? official_eikamet_channel

Use the official e-İkamet channel for the online application.

Sources: GOV_GENERAL_INFO, GOV_RESIDENCE_INFO, SAB_EXCHANGE_CURRENT



## R006 ? passport_validity_buffer

Passport validity must extend at least 60 days beyond the requested residence-permit end date.

Sources: SAB_EXCHANGE_OVERVIEW, GOV_FAQ



## R007 ? application_form_signed

The generated residence-permit application form must be signed before it counts as ready.

Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_CURRENT



## R008 ? passport_copy_pages_complete

Required passport copy pages must be present; Sabancı also requires the Türkiye entry page and visa page if applicable.

Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_OVERVIEW



## R009 ? biometric_photo_current

The biometric photo must be within 6 months and ICAO-compliant.

Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_CURRENT



## R010 ? residence_fee_evidence

Required residence-permit/document-fee payment evidence must be present. Exact amounts are dynamic.

Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_CURRENT

Do not hard-code TL amounts.

## R011 ? single_entry_visa_fee_conditional

Single-entry visa fee evidence is required only when current official rules make it applicable.

Sources: GOV_SHORT_TERM_FORM_2026



## R013 ? exchange_purpose_evidence

Official evidence of exchange/education purpose and duration must be available.

Sources: GOV_RESIDENCE_TYPES, GOV_SHORT_TERM_FORM_2026



## R014 ? student_certificate_present

Sabancı requires a current student certificate in the file.

Sources: SAB_EXCHANGE_OVERVIEW



## R015 ? insurance_covers_period

Insurance end covers the requested permit end OR the accepted academic program end; a confirmed exemption also satisfies this V2 operational check.

Sources: GOV_SHORT_TERM_FORM_2026, GOV_RESIDENCE_TYPES, SAB_EXCHANGE_OVERVIEW, METU_RESIDENCE_2026

Selected local Sabanci V2 end-coverage policy, not a universal statement of national insurance law. Start date is retained as observation but is not a readiness blocker. An end before the requested end can still satisfy the academic-period target.

## R016 ? insurance_evidence_type

The supported insurance path is private health insurance; SGK and bilateral/other alternatives require manual review.

Sources: GOV_SHORT_TERM_FORM_2026, GOV_RESIDENCE_TYPES

Unsupported official alternatives remain missing_information/manual review, not an automatically validated alternative or a legal rejection.

## R017 ? address_proof_sabanci

Provide one supported address-proof path: dorm certificate, rental agreement, notarized undertaking, OR a utility bill together with a completed Address Form.

Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW, SAB_IRO_TEAM_ADDRESS_INSTRUCTION

Utility bill + completed Address Form is the scoped Sabanci operational path reported by the team from direct IRO instruction. It is not asserted as universal Turkish national law.

## R018 ? address_contact_complete

Türkiye address and contact information must be complete and accurate.

Sources: GOV_SHORT_TERM_FORM_2026



## R019 ? declaration_control_document

Sabancı declaration/control document must be present and signed.

Sources: SAB_EXCHANGE_CURRENT



## R020 ? core_file_complete

The core Sabancı submission file must be complete before physical IRO submission.

Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW



## R021 ? iro_remote_preverification

Scanned documents must be sent to IRO and IRO confirmation received before physical submission.

Sources: SAB_EXCHANGE_CURRENT



## R022 ? fingerprint_gi_required

After online application, fingerprint/GI processing and a GI number are required in the Sabancı workflow.

Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW



## R023 ? iro_submission_dependencies

Physical IRO submission is allowed only after online application, complete/preverified file, and GI/fingerprint requirements.

Sources: SAB_EXCHANGE_CURRENT



## R026 ? missing_documents_block_iro

Sabancı IRO does not accept files with missing required documents.

Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW



## R028 ? migration_is_final_authority

Final evaluation/approval belongs to Migration Management, not Sabancı IRO.

Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW, METU_RESIDENCE_2026
