# Procedure Pack: `tc_kimlik_first` (UNVERIFIED draft)

First Turkish identity card (T.C. Kimlik Kartı) for Turkish citizens applying in Türkiye: after a birth, after
acquiring citizenship, or to replace an old paper nüfus cüzdanı. Renewals (expired, lost, damaged chip cards)
are the sibling pack `tc_kimlik_renewal`. Added on 2026-11-19 with **no change in the rule engine**.

## Status
- **Draft rules.** All 15 rules are `"status": "unverified"`. Each rule's `notes` quote the sentence of the saved
  official text it comes from. The agent must say the rules are unverified (Gate 1 enforces it).
- **Sources.** 3 pages of the Nüfus ve Vatandaşlık İşleri Genel Müdürlüğü (NVİ), downloaded on 2026-11-19 with
  `python -m src.rag fetch tc_kimlik_first`. The texts are in `sources_text/`; `content_sha256` in
  `sources.json` lets `python -m src.rag check tc_kimlik_first` detect a changed page.
- **Not stored on purpose.** The fee amount (published every year in the Resmî Gazete) and the card validity
  period (printed on the card; not stated on the saved pages).
- **Foreigners** are out of scope: they receive a foreigner identity number with their residence permit
  (see the İkamet packs).
- **Choosing this pack.** Both ID-card packs share the generic aliases ("tc kimlik", "turkish id card"...);
  the specific words decide (first / birth / citizenship / nüfus cüzdanı here; lost / expired / renew / kayıp /
  yenileme in `tc_kimlik_renewal`). A goal with neither is a tie, and the agent asks which one is meant.

## Review list for a person (before switching any rule to `verified`)
| Rule | Draft | Check against |
|---|---|---|
| TCF001 | card holder is a Turkish citizen | NVI_TCKK_PROCEDURES (definition of the card) |
| TCF002 | reason is birth, citizenship or replacing a nüfus cüzdanı | NVI_TCKK_FAQ (list of reasons) |
| TCF003 | new citizens: citizenship registered first | NVI_TCKK_FAQ |
| TCF004 | new citizens: MERNİS address registered first | NVI_APPOINTMENT_FAQ |
| TCF005 | 15+: photo ID, or a relative confirms the identity | NVI_TCKK_PROCEDURES, NVI_APPOINTMENT_FAQ (the relative rule is quoted from the "değiştirme" answer) |
| TCF006 | replacing a cüzdan: bring the cüzdan | NVI_APPOINTMENT_FAQ |
| TCF007 | new citizens: citizenship announcement document | NVI_APPOINTMENT_FAQ, NVI_TCKK_FAQ |
| TCF008 | biometric photo of the last 6 months (15+, or travel use) | NVI_TCKK_PROCEDURES |
| TCF009 | 15+: apply in person | NVI_TCKK_FAQ |
| TCF010 | under 15: parent / guardian applies | NVI_TCKK_FAQ |
| TCF011 | fee paid before applying (free after a birth notified in time) | NVI_APPOINTMENT_FAQ |
| TCF012 | fee of the application year | NVI_TCKK_FAQ |
| TCF013 | appointment (advisory) | **conflict**: NVI_APPOINTMENT_FAQ says no service without an appointment except birth notifications; NVI_TCKK_PROCEDURES says "randevulu veya randevusuz" |
| TCF014 | appointment date not passed (advisory) | NVI_APPOINTMENT_FAQ (15-day block) |
| TCF015 | no guaranteed issuance (guardrail) | NVI_TCKK_PROCEDURES |

Ages use whole months (`applicant_age_months`, 180 = 15 years) computed by code from the birth date and the
application date; the photo age is computed the same way.

## How to verify
1. Read the rule's quote in `sources_text/<SOURCE_ID>.txt` and the live page.
2. Correct the condition if needed (`add-rule` skill), then set `"status": "verified"`.
3. Run `python -m src.pack validate tc_kimlik_first` and `python -m pytest -q`.
4. `python -m src.rag draft-rules tc_kimlik_first` proposes extra requirements from the texts (with checked
   quotes) in `rule_drafts.json`, for comparison.
