# Procedure Pack: `tc_kimlik_renewal` (UNVERIFIED draft)

A new Turkish identity card (T.C. Kimlik Kartı) for citizens in Türkiye whose chip card expired, was lost or
stolen, or is damaged. A first card (birth, citizenship, old paper nüfus cüzdanı) is the sibling pack
`tc_kimlik_first`. Added on 2026-11-19 with **no change in the rule engine**.

## Status
- **Draft rules.** All 14 rules are `"status": "unverified"`; each rule's `notes` quote the saved official text.
- **Sources.** The 3 NVİ pages of `tc_kimlik_first` plus NVİ's instruction of 30.10.2020 on damaged cards
  (`NVI_DAMAGED_CARD_2020`, a PDF), all downloaded on 2026-11-19 with `python -m src.rag fetch tc_kimlik_renewal`.
- **Not modelled yet.** A change of registered information (for example a new surname): the saved pages do not
  state its requirements, so it is listed as excluded instead of guessed. Fee amounts and the validity period are
  not stored (see `tc_kimlik_first`).
- **Choosing this pack.** Both ID-card packs share the generic aliases ("tc kimlik", "turkish id card"...);
  the specific words decide (lost / expired / renew / kayıp / yenileme here; first / birth / citizenship /
  nüfus cüzdanı in `tc_kimlik_first`). A goal with neither is a tie, and the agent asks which one is meant.

## Review list for a person (before switching any rule to `verified`)
| Rule | Draft | Check against |
|---|---|---|
| TCR001 | card holder is a Turkish citizen | NVI_TCKK_PROCEDURES |
| TCR002 | reason is expired, lost or damaged | NVI_TCKK_FAQ |
| TCR003 | a lost card is reported (Alo 199 or a population directorate) | NVI_TCKK_FAQ (regulation art. 19) |
| TCR004 | expired or damaged: bring the current card | NVI_APPOINTMENT_FAQ ("değiştirme" documents) |
| TCR005 | damaged card still readable, otherwise identity check (advisory) | NVI_DAMAGED_CARD_2020 |
| TCR006 | lost card: bring another photo ID in case biometrics fail (advisory) | NVI_APPOINTMENT_FAQ |
| TCR007 | biometric photo of the last 6 months (15+, or travel use) | NVI_TCKK_PROCEDURES, NVI_APPOINTMENT_FAQ |
| TCR008 | 15+: apply in person | NVI_TCKK_FAQ |
| TCR009 | under 15: parent / guardian applies | NVI_TCKK_FAQ, NVI_APPOINTMENT_FAQ |
| TCR010 | fee (lost card: loss fee) paid before applying | NVI_APPOINTMENT_FAQ |
| TCR011 | fee of the application year | NVI_TCKK_FAQ |
| TCR012 | appointment (advisory) | **conflict** between NVI_APPOINTMENT_FAQ and NVI_TCKK_PROCEDURES (see `tc_kimlik_first`) |
| TCR013 | appointment date not passed (advisory) | NVI_APPOINTMENT_FAQ |
| TCR014 | no guaranteed issuance (guardrail) | NVI_TCKK_PROCEDURES |

## How to verify
Same steps as `tc_kimlik_first/README.md`, with `tc_kimlik_renewal` as the procedure id.
