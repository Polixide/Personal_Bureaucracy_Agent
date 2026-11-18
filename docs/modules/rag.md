# `src/rag/index.py` + `search_sources` — citing the official sources

## What it does
It splits each pack's `sources_text/<SOURCE_ID>.txt` into chunks and embeds them with `EMBED_MODEL` through
`src/llm.py`. It then finds the chunks closest to a question by cosine similarity, computed with numpy.
It **cites and explains**; it never decides a rule result.

## Where it sits in the flow
The `search_sources` tool is available to the orchestrator and to workers. A typical use is a worker
explaining why a rule fails, citing the passage and its `source_id`.

## Inputs and outputs
```python
search(store, pack, backend, "How long must my passport be valid?", k=3)
# [Hit(source_id="GOV_FAQ", title="Frequently Asked Questions about Foreigners", chunk_id="GOV_FAQ#2",
#      text="- R006 (passport_validity_buffer): Passport validity must extend at least 60 days ...", score=0.41), ...]
```
Vectors are cached in the Twin database (table `rag_vectors`). Each is keyed by pack, chunk and embedding model,
and recomputed only when the chunk's text hash changes.

## Why it is built this way
- **Plain numpy cosine** over a few dozen chunks is enough and has no extra dependencies. The cache makes
  repeated searches cost a single query embedding.
- **One file per `source_id`.** Every hit carries a citable source that exists in `sources.json`;
  `src.pack validate` reports any text file that matches no source.

## How to test it
```
python -m pytest -q tests/test_rag.py
```
- **10 queries.** Each expects a source that actually states the requirement (from the rule's `source_ids`).
- **Offline embedder.** The tests use the deterministic bag-of-words embedder in `tests/fakes.py`, so they run
  without Ollama.

## Known limits
- **The text files are placeholders.** `sources_text/` holds summaries compiled from the pack's `rules.json`,
  not copies of the official pages; each file says so. Retrieval only becomes truly useful after someone saves
  the real page texts there.
- **No live retrieval measurement yet.** Quality with `bge-m3` (or `qwen3-embedding:0.6b`) still needs a run
  with `python -m eval.run_eval --live`.
