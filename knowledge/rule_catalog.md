# Rule Catalog — IKAMET v1.0.0

## R001 — exchange_stay_requires_ikamet
Incoming Sabancı exchange students staying 3 months or longer enter the residence-permit workflow.
- Category: eligibility
- Severity: blocking
- Volatility: low
- Condition: `exchange_student == true AND planned_stay_months >= 3`
- Sources: SAB_EXCHANGE_OVERVIEW
- Dependencies: none


## R002 — permit_type_short_term
For the scoped Sabancı incoming exchange-student case, the workflow uses a short-term residence permit.
- Category: eligibility
- Severity: blocking
- Volatility: low
- Condition: `permit_type == 'short_term'`
- Sources: GOV_RESIDENCE_TYPES, SAB_EXCHANGE_OVERVIEW
- Dependencies: none


## R003 — application_after_entry
A first application is started only after entry into Türkiye; Sabancı instructs entry with passport.
- Category: eligibility
- Severity: blocking
- Volatility: low
- Condition: `is_in_turkiye == true AND entered_with_passport == true`
- Sources: SAB_EXCHANGE_OVERVIEW
- Dependencies: none


## R004 — apply_before_legal_stay_expiry
The online first application must be completed before visa/visa-exemption/current legal stay expires.
- Category: deadline
- Severity: blocking
- Volatility: medium
- Condition: `application_date <= legal_stay_expiry_date`
- Sources: SAB_EXCHANGE_OVERVIEW, METU_RESIDENCE_2026
- Dependencies: none


## R005 — official_eikamet_channel
Use the official e-İkamet channel for the online application.
- Category: channel
- Severity: blocking
- Volatility: low
- Condition: `application_channel == 'https://e-ikamet.goc.gov.tr/'`
- Sources: GOV_GENERAL_INFO, GOV_RESIDENCE_INFO, SAB_EXCHANGE_CURRENT
- Dependencies: none


## R006 — passport_validity_buffer
Passport validity must extend at least 60 days beyond the requested residence-permit end date.
- Category: date_constraint
- Severity: blocking
- Volatility: low
- Condition: `passport_expiry_date >= requested_permit_end_date + 60 days`
- Sources: SAB_EXCHANGE_OVERVIEW, GOV_FAQ
- Dependencies: none


## R007 — application_form_signed
The generated residence-permit application form must be signed before it counts as ready.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `application_form_generated == true AND application_form_signed == true`
- Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_CURRENT
- Dependencies: R005


## R008 — passport_copy_pages_complete
Required passport copy pages must be present; Sabancı also requires the Türkiye entry page and visa page if applicable.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `passport_id_page_present == true AND turkey_entry_page_present == true AND (visa_required == false OR visa_page_present == true)`
- Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_OVERVIEW
- Dependencies: none


## R009 — biometric_photo_current
The biometric photo must be within 6 months and ICAO-compliant.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `photo_age_months <= 6 AND biometric_photo == true AND icao_compliant == true`
- Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_CURRENT
- Dependencies: none


## R010 — residence_fee_evidence
Required residence-permit/document-fee payment evidence must be present. Exact amounts are dynamic.
- Category: payment
- Severity: blocking
- Volatility: high
- Condition: `fee_payment_required == false OR fee_receipt_present == true`
- Sources: GOV_SHORT_TERM_FORM_2026, SAB_EXCHANGE_CURRENT
- Dependencies: none
- Notes: Do not hard-code TL amounts.

## R011 — single_entry_visa_fee_conditional
Single-entry visa fee evidence is required only when current official rules make it applicable.
- Category: payment
- Severity: blocking
- Volatility: high
- Condition: `single_entry_visa_fee_required == false OR single_entry_visa_fee_receipt_present == true`
- Sources: GOV_SHORT_TERM_FORM_2026
- Dependencies: none


## R012 — financial_capacity_declaration
The short-term permit form requires declaration/evidence of sufficient regular financial means.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `financial_capacity_declared == true`
- Sources: GOV_SHORT_TERM_FORM_2026
- Dependencies: none


## R013 — exchange_purpose_evidence
Official evidence of exchange/education purpose and duration must be available.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `exchange_program_evidence_present == true`
- Sources: GOV_RESIDENCE_TYPES, GOV_SHORT_TERM_FORM_2026
- Dependencies: none


## R014 — student_certificate_present
Sabancı requires a current student certificate in the file.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `student_certificate_present == true`
- Sources: SAB_EXCHANGE_OVERVIEW
- Dependencies: none


## R015 — insurance_covers_period
Health-insurance coverage must cover the requested permit period unless a valid official exemption/alternative applies.
- Category: insurance
- Severity: blocking
- Volatility: low
- Condition: `insurance_exempt == true OR (insurance_valid_for_ikamet == true AND insurance_start_date <= requested_start_date AND insurance_end_date >= requested_end_date)`
- Sources: GOV_SHORT_TERM_FORM_2026, GOV_RESIDENCE_TYPES, SAB_EXCHANGE_OVERVIEW, METU_RESIDENCE_2026
- Dependencies: none


## R016 — insurance_evidence_type
Accepted evidence is bilateral coverage, SGK provision/application evidence, or valid private health insurance.
- Category: insurance
- Severity: blocking
- Volatility: low
- Condition: `insurance_evidence_type in ['bilateral_social_security','sgk_provision','sgk_application','private_health_insurance']`
- Sources: GOV_SHORT_TERM_FORM_2026, GOV_RESIDENCE_TYPES
- Dependencies: none


## R017 — address_proof_sabanci
For Sabancı operational submission, accepted address proof is dorm certificate, rental agreement, or notarized undertaking; Airbnb alone is not accepted.
- Category: address
- Severity: blocking
- Volatility: low
- Condition: `address_proof_type in ['dorm_certificate','rental_agreement','notarized_commitment']`
- Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW
- Dependencies: none
- Notes: Government also recognizes other cases; outside this scoped workflow they should trigger manual verification.

## R018 — address_contact_complete
Türkiye address and contact information must be complete and accurate.
- Category: address
- Severity: blocking
- Volatility: low
- Condition: `turkey_address_present == true AND phone_present == true AND email_present == true`
- Sources: GOV_SHORT_TERM_FORM_2026
- Dependencies: none


## R019 — declaration_control_document
Sabancı declaration/control document must be present and signed.
- Category: document
- Severity: blocking
- Volatility: low
- Condition: `declaration_control_document_present == true AND declaration_control_document_signed == true`
- Sources: SAB_EXCHANGE_CURRENT
- Dependencies: none


## R020 — core_file_complete
The core Sabancı submission file must be complete before physical IRO submission.
- Category: prerequisite_set
- Severity: blocking
- Volatility: low
- Condition: `all_core_documents_present == true`
- Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW
- Dependencies: R007, R008, R009, R010, R014, R015, R017, R019


## R021 — iro_remote_preverification
Scanned documents must be sent to IRO and IRO confirmation received before physical submission.
- Category: workflow
- Severity: blocking
- Volatility: low
- Condition: `scans_sent_to_iro == true AND iro_preverification_confirmed == true`
- Sources: SAB_EXCHANGE_CURRENT
- Dependencies: R020


## R022 — fingerprint_gi_required
After online application, fingerprint/GI processing and a GI number are required in the Sabancı workflow.
- Category: workflow
- Severity: blocking
- Volatility: low
- Condition: `online_application_completed == true AND fingerprint_step_completed == true AND gi_number_present == true`
- Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW
- Dependencies: R005, R007


## R023 — iro_submission_dependencies
Physical IRO submission is allowed only after online application, complete/preverified file, and GI/fingerprint requirements.
- Category: workflow
- Severity: blocking
- Volatility: low
- Condition: `online_application_completed == true AND iro_preverification_confirmed == true AND gi_number_present == true AND all_core_documents_present == true`
- Sources: SAB_EXCHANGE_CURRENT
- Dependencies: R020, R021, R022


## R024 — current_submission_window
Sabancı submission must occur inside a currently published IRO window; actual dates must be refreshed.
- Category: deadline
- Severity: blocking
- Volatility: very_high
- Condition: `current_date within current_iro_submission_window`
- Sources: SAB_EXCHANGE_CURRENT
- Dependencies: none


## R025 — application_form_current_window
Application forms from previous months/windows are not accepted for the current IRO submission window.
- Category: deadline
- Severity: blocking
- Volatility: very_high
- Condition: `application_form_window_id == current_iro_window_id`
- Sources: SAB_EXCHANGE_CURRENT
- Dependencies: none


## R026 — missing_documents_block_iro
Sabancı IRO does not accept files with missing required documents.
- Category: workflow
- Severity: blocking
- Volatility: low
- Condition: `missing_required_documents_count == 0`
- Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW
- Dependencies: R020


## R027 — additional_documents_possible
Migration authorities may request additional information/documents; complete initial file does not guarantee approval.
- Category: evaluation
- Severity: guardrail
- Volatility: low
- Condition: `approval_status != 'guaranteed'`
- Sources: GOV_SHORT_TERM_FORM_2026, GOV_FAQ, SAB_EXCHANGE_OVERVIEW
- Dependencies: none


## R028 — migration_is_final_authority
Final evaluation/approval belongs to Migration Management, not Sabancı IRO.
- Category: authority
- Severity: guardrail
- Volatility: low
- Condition: `decision_authority == 'Migration Management'`
- Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW, METU_RESIDENCE_2026
- Dependencies: none


## R029 — track_after_submission
After IRO submission, application status should be tracked through official e-İkamet.
- Category: post_submission
- Severity: advisory
- Volatility: medium
- Condition: `submitted_to_iro == false OR tracking_channel == 'e_ikamet'`
- Sources: SAB_EXCHANGE_CURRENT, SAB_EXCHANGE_OVERVIEW, GOV_GENERAL_INFO
- Dependencies: none

