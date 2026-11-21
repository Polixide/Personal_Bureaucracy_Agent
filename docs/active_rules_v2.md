# Active V2 rules - 2026-11-20

24 active rules. Removed from the active dataset: R012, R024, R025, R027, R029. No renumbering and no hidden evaluation of removed rules. Tracking and time-sensitive windows remain procedure/advisory/RAG concerns.

| Rule | Class | Why retained |
|---|---|---|
| R001 | CORE | Incoming Sabancı exchange students staying 3 months or longer enter the residence-permit workflow. |
| R002 | CORE | For the scoped Sabancı incoming exchange-student case, the workflow uses a short-term residence permit. |
| R003 | CORE | A first application is started only after entry into Türkiye; Sabancı instructs entry with passport. |
| R004 | GUARDRAIL | The online first application must be completed before visa/visa-exemption/current legal stay expires. |
| R005 | GUARDRAIL | Use the official e-İkamet channel for the online application. |
| R006 | CORE | Passport validity must extend at least 60 days beyond the requested residence-permit end date. |
| R007 | STAGE_SPECIFIC | The generated residence-permit application form must be signed before it counts as ready. |
| R008 | CORE | Required passport copy pages must be present; Sabancı also requires the Türkiye entry page and visa page if applicable. |
| R009 | CORE | The biometric photo must be within 6 months and ICAO-compliant. |
| R010 | STAGE_SPECIFIC | Required residence-permit/document-fee payment evidence must be present. Exact amounts are dynamic. |
| R011 | STAGE_SPECIFIC | Single-entry visa fee evidence is required only when current official rules make it applicable. |
| R013 | CORE | Official evidence of exchange/education purpose and duration must be available. |
| R014 | CORE | Sabancı requires a current student certificate in the file. |
| R015 | CORE | Insurance end covers the requested permit end OR the accepted academic program end; a confirmed exemption also satisfies this V2 operational check. |
| R016 | CORE | The supported insurance path is private health insurance; SGK and bilateral/other alternatives require manual review. |
| R017 | CORE | Provide one supported address-proof path: dorm certificate, rental agreement, notarized undertaking, OR a utility bill together with a completed Address Form. |
| R018 | CORE | Türkiye address and contact information must be complete and accurate. |
| R019 | STAGE_SPECIFIC | Sabancı declaration/control document must be present and signed. |
| R020 | PROCEDURAL | The core Sabancı submission file must be complete before physical IRO submission. |
| R021 | STAGE_SPECIFIC | Scanned documents must be sent to IRO and IRO confirmation received before physical submission. |
| R022 | PROCEDURAL | After online application, fingerprint/GI processing and a GI number are required in the Sabancı workflow. |
| R023 | PROCEDURAL | Physical IRO submission is allowed only after online application, complete/preverified file, and GI/fingerprint requirements. |
| R026 | GUARDRAIL | Sabancı IRO does not accept files with missing required documents. |
| R028 | GUARDRAIL | Final evaluation/approval belongs to Migration Management, not Sabancı IRO. |

R015 is the selected V2 operational end-coverage policy (exemption OR requested-period end OR academic-period end). It does not require start-date coverage or a generic legal-validity attestation. R016 supports private health insurance without a positive validity attestation; an explicit statement of invalidity remains a failure. SGK/bilateral paths remain manual review. R017 accepts one public documented path OR both accepted utility bill and completed Address Form. The latter is sourced to team-supplied direct Sabanci IRO instruction (date/URL unavailable), not universal national law.

Legal-stay arithmetic is separate from deterministic rule satisfaction; estimates never silently populate the legal-expiry input for R004. Human review/regression testing is required before any source-change candidate can become a new active rule version.
