---
name: add-procedure
description: Use when the user wants to add a new bureaucratic procedure (for example İkamet extension, tax number, bank account) as a new Procedure Pack under procedures/. Never edits engine code.
---

# Add a new procedure

A procedure is a folder, not code. Create `procedures/<id>/` with:

1. `procedure.json`: `id`, `title`, `scope` (who it is for, what is excluded),
   `aliases` (phrases that name what the procedure is about: "kimlik", "residence permit") and `qualifiers`
   (words that only tell it apart from a sibling pack: "lost", "first", "renew"; they count only when an alias
   matched, so never put a generic word in `aliases`; check with `python -m eval.routing_eval`),
   `sections` (list of `{id, title}` such as identity, insurance, housing, appointment),
   `fields` (list of `{name, type, question}`; type is one of bool, int, str, date). Write the question as you would
   ask a person ("When does your passport expire?"), never with a rule id. Optional: `ask` = `"later"` for steps after
   the application, `"never"` for values computed by code, read from documents only or published by an office (IRO
   windows, counts); `choices` = `{"text shown": "stored value"}` for a text field the rules compare with codes.
   Sections about steps after the application get `"later": true`.
   `documents` (list of `{doc_type, description, extracts, titles, keywords, mrz, optional}`); `optional: true` for a
   document needed only in some cases (say which in its description):
   - `extracts`: field names (or `{field, from_field, requires_confirmation, lines}`; `lines: 3` for an address);
   - `titles`: phrases printed at the top of the document's first page (any language; compared without case or
     accents). A page whose top lines show one of them starts this document, so one PDF can hold several documents.
     Use real titles ("Health Insurance Certificate", "nüfus cüzdanı"), never a label with a value;
   - `keywords`: phrases found anywhere in the document; they name a document that has no title;
   - `mrz`: `"passport"` or `"id_card"` when the document has a machine-readable zone (code reads it with its
     check digits and fills `holder_name`, `birth_date`, `expiry_date`, `document_number`, `nationality`).
   Check a new document with `python -m eval.document_eval` after adding a synthetic sample to
   `eval/document_samples.py`.
2. `rules.json`: list of rules (see the `add-rule` skill for the format).
3. `sources.json`: list of `{source_id, title, url, retrieved_at, authority}`.
   Do NOT invent URLs or dates. Ask the user for them, or mark rules `"status": "unverified"`.
4. `sources_text/`: plain-text copies of the official pages the user downloaded (used by the RAG index).
5. `eval/scenarios.json`: at least 5 synthetic scenarios (pass, fail, missing information, date conflict).

Steps:
1. Ask the user which sections and rules exist, and where each rule comes from.
2. Write the files. Keep rule ids stable (`<PREFIX>001`, ...).
3. Run `python -m src.pack validate <id>`. Fix every error it reports (unknown field, unknown source,
   dependency cycle, section without rules).
4. Add tests in `tests/` that load the pack and check one pass, one fail and one missing case per rule.
5. Run `python -m pytest -q`.

If the new procedure needs something the engine cannot express, STOP. Explain what is missing and
propose a small engine change; do not hack it into the pack.
