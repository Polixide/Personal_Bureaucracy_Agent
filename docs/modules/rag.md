# `src/rag/` — the knowledge base: official sources, retrieval, rule drafts, web corpus

## What it does
It keeps each pack's official web pages and documents as citable text, finds the passages closest to a question
(for the agent's `search_sources` tool), tells when an official page changed, and proposes rule drafts from the
texts for a person to review. It **cites and explains**; it never decides a rule result and never edits rules.

## Where it sits in the flow
| Module | Job | Called by |
|---|---|---|
| `ingest.py` | page/PDF/TXT → readable text (menus, scripts, footers dropped); header + SHA-256 | `knowledge.py` |
| `knowledge.py` | `fetch_sources`, `fetch_from_file`, `add_source`, `check_sources` on `sources.json` | CLI |
| `index.py` | chunks (800 chars), embeddings in batches of 16, cosine search with numpy | `search_sources`, `drafts.py` |
| `drafts.py` | retrieve → model proposes `{description, quote, source_id, fields}` → code checks | CLI |
| `crawl.py` | web scraping for procedures **without** a pack: seeds → same-site content links → `knowledge/corpus/` | CLI |
| `__main__.py` | `python -m src.rag fetch / add / check / index / search / draft-rules <procedure_id>`, `crawl`, `search-web` | a person |

## Inputs and outputs
```text
python -m src.rag fetch tc_kimlik_first
OK   NVI_TCKK_FAQ: 11824 characters (changed)      -> sources_text/NVI_TCKK_FAQ.txt, sources.json hash + date
python -m src.rag check tc_kimlik_renewal
CHANGED      NVI_TCKK_FAQ                            (exit code 1)
             review rules: TCR002, TCR003, ...
             -Son altı ay ...  / +Son üç ay ...
```
A draft in `rule_drafts.json` (model output checked by code):
```json
{"draft_id": "D001", "description": "A recent biometric photo is needed.", "source_id": "NVI_TCKK_PROCEDURES",
 "quote": "Son altı ay içinde çekilmiş biyometrik fotoğraf", "fields": ["photo_date"], "quote_found": true,
 "problems": [], "similar_rules": ["TCF008"], "status": "unverified"}
```

## Why it is built this way
- **The code checks the model.** A draft's quote must really be in the saved text (case, Turkish ı/İ and spacing
  ignored, at least 12 characters), its source must be among the retrieved passages, and its fields must exist.
  Anything else is listed under `problems`. `rules.json` is only changed by a person (`add-rule` skill).
- **Change detection by hash.** Rules cite pages that change every year (fees, forms). `check` compares the new
  text with the saved one and names the rules to review; nothing is overwritten until `fetch`.
- **Standard library only.** `html.parser` + httpx (already used by ollama) + PyMuPDF; no new dependency. A
  generic client name is sent, never user or project details. Sites that refuse downloads use `--file`.
- **Embedding model `bge-m3`**, chosen by measurement (`python -m eval.rag_eval`, `docs/eval_rag.md`): 66 queries
  on 567 real passages, hit@3 0.758 against 0.667 for `qwen3-embedding:0.6b`, and 0.60-0.64 against 0.33-0.36 on
  the Turkish NVİ pages, where users ask in English. It indexes slower (392 s against 175 s, once, then cached)
  but needs less VRAM while running (689 MB against 1.2 GB). Models that expect a query instruction get it
  (`EMBED_QUERY_INSTRUCTIONS`); passages never do. Vectors are cached per model in the Twin's `rag_vectors` table.

## Where RAG is useful, and where it is not (decided 2026-11-20)
- **Not for deciding.** Rules stay in `rules.json`, written and checked by people, evaluated by code. Retrieval misses
  the right passage about 1 time in 4 (hit@3 0.76), so it can never be the place a requirement comes from.
- **Not for explaining a rule.** Each rule already carries its quote and `source_ids`; the chat gets them directly in
  the briefing, which is exact. Retrieval would only add noise there.
- **Yes for questions the rules do not cover** ("how long does it take?", "do I need an appointment?"): the chat
  retrieves 3 passages of the procedure's sources for every message and cites them.
- **Yes for procedures without a pack**: the web corpus below.
- **Yes for writing new rules** (`draft-rules`), and for noticing that an official page changed (`check`).

## Web scraping for other procedures (`crawl.py`)
```text
python -m src.rag crawl           # follows knowledge/seeds.json, saves knowledge/corpus/*.txt + corpus.json, embeds them
python -m src.rag search-web "How do I get a tax number as a foreigner?"
[0.743] WEB_IRO_EN_HOW_OBTAIN_TAX_ID: The tax ID number is necessary for important payments ...
```
- **Polite and limited**: same site and path prefix as the seed, links found in the page's content only (menus,
  headers, footers and sidebars are skipped, otherwise the site menu fills the page budget — seen on the first run),
  `max_depth` / `max_pages` per seed, robots.txt obeyed (NVİ and Göç disallow `/kurumlar`, where their PDFs are, so the
  crawler skips them), one second between requests, a generic client name. BTK refuses automatic clients: not used.
- **First corpus (2026-11-20)**: 25 pages of Sabancı University's International Relations Office guides for exchange
  and full-time students — tax ID, bank account, SIM card, IMEI registration, health insurance, address registration,
  e-Devlet, residence permit and extension, official documents, transport card. These are the procedures an incoming
  student meets after İkamet, and the pages are written for exactly these users.
- **How it is used**: the chat adds up to 2 web passages to a turn when their similarity is at least 0.45. Without an
  active procedure, a reply that cites them gets a note, added by code, that nothing was checked against the user's
  documents. A web page never becomes a rule by itself; a frequent procedure can become a pack (`add-procedure` skill,
  then `draft-rules`).

## How to test it
```
python -m pytest -q tests/test_rag.py tests/test_knowledge_base.py tests/test_vision_and_crawl.py
python -m eval.rag_eval --models qwen3-embedding:0.6b bge-m3      # live retrieval comparison (Ollama)
```
The tests use a fake fetcher, `ScriptedBackend` and a bag-of-words embedder; `test_rag.py` runs on texts pinned in
`tests/data/placeholder_sources/` so a government website cannot change its results.

## Known limits
- **Main-content guess.** The HTML filter keeps the largest "content" container. A page built by JavaScript only
  gives no text (`no readable text found`); save the page and use `fetch ... --file`.
- **Noisy hashes.** A page that prints a date or a counter shows as `changed` on every check; the diff preview tells.
- **Drafts are leads, not rules.** Live with `qwen3:4b` on `tc_kimlik_renewal` (3 topics): 10 drafts, every quote
  found in the saved text, 3 linked to existing rules, one requirement the pack lacks (re-applying for a lost card
  through e-Devlet), and one wrong field (a photo rule tied to `renewal_reason`). The first prompt copied the topic
  as the description and listed six fields; both are now flagged as problems. No conditions are proposed.
