# Reference test documents

Public documents for testing document recognition: no real person's data. In the app, choose the
`ikamet_first` procedure; the document types are `declaration_control_document`, `ikamet_application_form`,
`passport_identity_page` and `turkiye_entry_stamp`. The blank e-İkamet form is also the saved text of the
source `GOV_SHORT_TERM_FORM_2026` (`python -m src.rag fetch ikamet_first GOV_SHORT_TERM_FORM_2026 --file ...`).
Retrieved 2026-11-19. Images were converted to one-page, image-only PDFs, so
they exercise the OCR path. PDF metadata (author, producer, dates) was removed.

| File | Checklist type | Expected behaviour | Source | Licence |
|---|---|---|---|---|
| `declaration_control_document.pdf` | Declaration control document | Recognised; blank, so no filled fields | Sabancı University IRO, <https://iro.sabanciuniv.edu/sites/iro.sabanciuniv.edu/files/2024-06/Declaration%20Control%20Document.pdf> | Official blank form, used for testing only |
| `ikamet_application_form_blank.pdf` | İkamet application form | Recognised; blank, so no applicant data | Presidency of Migration Management, e-İkamet, <https://e-ikamet.goc.gov.tr/Ikamet/BasvuruFormuOrnegi/BasvuruFormuBos?tur=0> (also `GOV_SHORT_TERM_FORM_2026` in `knowledge/sources.json`) | Official blank form, used for testing only |
| `passport_specimen_norway.pdf` | Passport identity page | Specimen "Åsamund Specimen"; issued 15 Apr 2020, expires 15 Apr 2030 | Government of Norway, via Wikimedia Commons, <https://commons.wikimedia.org/wiki/File:Norwegian_Passport_Specimen.jpg> | Public domain |
| `passport_specimen_germany_2007.pdf` | Passport identity page | Specimen "Erika Mustermann"; **expired 31 Oct 2017** (expired-passport case) | Bundesministerium des Innern, via Wikimedia Commons, <https://commons.wikimedia.org/wiki/File:Mustermann_Reisepass_2007.jpg> | Public domain |
| `entry_stamp_istanbul_airport.pdf` | Türkiye entry stamp | Entry 26.02.22, İstanbul, sharp scan | Photo by aveaoz, Wikimedia Commons, <https://commons.wikimedia.org/wiki/File:Entry_stamp_at_Istanbul_Airport.jpg> | CC BY-SA 4.0 |
| `entry_stamp_istanbul.pdf` | Türkiye entry stamp | Entry 18.07.20, İstanbul, noisy scan | Photo by Webshark25, Wikimedia Commons, <https://commons.wikimedia.org/wiki/File:Istanbul_entry_stamp.jpg> | CC BY-SA 4.0 |

The two CC BY-SA 4.0 images require attribution (above) and share-alike: the
converted PDFs are shared under the same licence
(<https://creativecommons.org/licenses/by-sa/4.0/>).

Not available publicly without personal data, so to be generated synthetically:
student certificate, health insurance certificate, residence card fee receipt.
