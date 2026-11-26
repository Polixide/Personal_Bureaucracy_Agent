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
| R017 | CORE | Review the actual address evidence for the selected first-application or local IRO preparation route. |
| R018 | CORE | Türkiye address and contact information must be complete and accurate. |
| R019 | STAGE_SPECIFIC | Sabancı declaration/control document must be present and signed. |
| R020 | PROCEDURAL | The core Sabancı submission file must be complete before physical IRO submission. |
| R021 | STAGE_SPECIFIC | Scanned documents must be sent to IRO and IRO confirmation received before physical submission. |
| R022 | PROCEDURAL | After online application, fingerprint/GI processing and a GI number are required in the Sabancı workflow. |
| R023 | PROCEDURAL | Physical IRO submission is allowed only after online application, complete/preverified file, and GI/fingerprint requirements. |
| R026 | GUARDRAIL | Sabancı IRO does not accept files with missing required documents. |
| R028 | GUARDRAIL | Final evaluation/approval belongs to Migration Management, not Sabancı IRO. |

R015 uses exemption, requested-period end coverage, or academic-period end coverage. R016 supports private health insurance, and an explicit statement of invalidity remains a failure. SGK and bilateral alternatives remain under manual review.

### R017 in the supported first-application workflow

The address method selects the checklist. Readiness comes from reviewed files for that method. File review records visible markings and the user's confirmations with the existing provenance. The IRO and authorities determine authenticity and acceptance.

| Method | Evidence used by the application | Conditional guidance |
| --- | --- | --- |
| Dormitory | Certificate from Sabancı Dormitory Management naming the student. In document review the student confirms Turkish, a visible QR code, and the signing or stamping described by the government. A readable address can be carried into the form. | Request it through SU Ticket or at the D2 Dorm Administration. |
| Rental | Notarized agreement in the student's name showing the address and landlord's name. The student reviews visible notary certification. | The official documents guide also explains ID copies and contract details for address registration after receiving the card. The first-application checklist specifically asks for the notarized copy with the lessor's details. |
| Hosted accommodation | Host's notarized undertaking naming the student and residence. Review the host ID copy, registered-address certificate, and a current electricity, water, gas or landline bill in the host's name, or the corresponding new subscription contract. | A married host supplies the spouse's notarized undertaking and ID copy. Sabancı asks Turkish hosts for a civil-registry extract or international hosts for a valid residence or work permit copy. The host brings a notarized rental contract or title deed when obtaining the undertaking. Supporting pages can be included in the reviewed undertaking upload or reviewed separately for the same host and address. |
| Utility bill + completed Address Form | Both reviewed files for the local IRO preparation route documented under `SAB_IRO_TEAM_ADDRESS_INSTRUCTION`. The form must be completed and signed. | Use it when Sabancı IRO provides that instruction for the case. The published first-application list gives the three accommodation paths above. The repository's team-supplied instruction has no published URL or exact issue date. |

Scope and sources checked on 2026-11-25. The first-application options follow the [Sabancı exchange-student guidance](https://iro.sabanciuniv.edu/en/exchange-students-short-term-residence-permit-first-and-extension-application-guidelines) and the accommodation section of the [Migration Management student-permit checklist](https://e-ikamet.goc.gov.tr/Ikamet/BasvuruIstenenBelgeler/BasvuruFormuIstenenBelgeler?tur=5). Host supporting pages and the dormitory request instructions come from the [Sabancı official documents guide](https://iro.sabanciuniv.edu/en/official-documents). The registered residence certificate with utility evidence in the government checklist is described separately for extension and transfer applications. Sabancı's [address registration guidance](https://iro.sabanciuniv.edu/en/address-registration) covers the process after receiving the permit card.


Legal-stay arithmetic is separate from deterministic rule satisfaction; estimates never silently populate the legal-expiry input for R004. Human review/regression testing is required before any source-change candidate can become a new active rule version.
