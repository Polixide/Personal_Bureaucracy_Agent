---
name: draft-rules-from-source
description: Use when the user gives official text (a government or university page saved in a Procedure Pack's sources_text/) and wants draft rules extracted from it. Produces unverified drafts with quotes for a human to review.
---

# Draft rules from an official source

0. Get the text into the knowledge base first (never paste it by hand):
   - a page already in `sources.json`: `python -m src.rag fetch <id> <SOURCE_ID>`
   - a new page or PDF: `python -m src.rag add <id> <SOURCE_ID> --url URL --title T --authority A [--file PATH]`
     (ask the user for the URL, title and authority; never invent them)
   - a site that refuses downloads: save the page, then `python -m src.rag fetch <id> <SOURCE_ID> --file PATH`
   Do not fetch anything from the web unless the user asks.
1. Optional first pass with the model: `python -m src.rag draft-rules <id> [--topic "..."]` writes
   `procedures/<id>/rule_drafts.json`. Every draft's quote is checked against the saved text (`quote_found`,
   `problems`); `similar_rules` lists existing rules that probably cover it. Treat these drafts as leads only.
2. Read the source file in `procedures/<id>/sources_text/`. For each requirement you find (a document, a deadline,
   a validity condition, a channel), write a draft rule in the format of the `add-rule` skill with:
   - `"status": "unverified"`
   - `source_ids` pointing to the right entry in `sources.json`
   - the exact sentence from the source (copied, not paraphrased) in `"notes"`, as `Quote: "..."`
     (`python -m eval.rag_eval` uses these quotes to measure retrieval)
3. Use only field names that exist in `procedure.json`; propose new fields in a separate list and ask the user
   to approve them.
4. Do NOT guess numbers, dates or conditions that the text does not state. If two sources disagree, say so in
   `"notes"` (`CONFLICT: ...`) and prefer `"severity": "advisory"` until a person decides.
5. Never write directly to `rules.json` without the user's go-ahead; give the user a short review list first:
   rule id, one-line summary, the quote, and what to check. A person must verify each draft against the source
   before it becomes `"status": "verified"`.
6. Run `python -m src.pack validate <id>` on the final `rules.json` and add the three tests per rule.
7. Later, `python -m src.rag check <id>` lists the sources that changed and the rules that cite them.
