---
name: draft-rules-from-source
description: Use when the user gives official text (a government or university page saved in a Procedure Pack's sources_text/) and wants draft rules extracted from it. Produces unverified drafts with quotes for a human to review.
---

# Draft rules from an official source

1. Read the source file in `procedures/<id>/sources_text/`. Do not fetch anything from the web unless the user asks.
2. For each requirement you find (a document, a deadline, a validity condition, a channel), write a draft rule in
   the format of the `add-rule` skill with:
   - `"status": "unverified"`
   - `source_ids` pointing to the right entry in `sources.json` (ask the user to add the entry if missing)
   - an extra field `"evidence_quote"`: the exact sentence from the source (copied, not paraphrased)
3. Use only field names that exist in `procedure.json`; propose new fields in a separate list and ask the user
   to approve them.
4. Do NOT guess numbers, dates or conditions that the text does not state. If the text is ambiguous, write the
   rule with the ambiguity described in `"notes"` and leave the condition out.
5. Save drafts to `procedures/<id>/rules_draft.json`, never directly to `rules.json`.
6. Give the user a short review list: rule id, one-line summary, the quote, and what to check.
   A person must verify each draft against the source before it moves to `rules.json` with `"status": "verified"`.
7. Run `python -m src.pack validate <id>` on the final `rules.json` and add the three tests per rule.
