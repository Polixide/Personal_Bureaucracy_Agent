# Every metric of the project, in one place

Each evaluation script prints a Markdown report; every one takes the same model options
(`--provider ollama|openai|anthropic`, `--model <name>`, `--effort <level>`), so the local model and GPT can be
measured on the same cases. Tokens and cost come from the providers' usage reports (`USAGE` in `src/llm.py`) and
the price list in `config.py` (`CLOUD_PRICES`); they are "n/a" offline or for a model without a price.
Metrics marked **(A)** are adapted from Andrea's framework (`feature/marco-integration-review`).

## 1. The agent against baselines — `python -m eval.run_eval` → `docs/eval_results.md`
Systems on the same scenarios (expected verdicts never edited): **A** static checklist, **B** one LLM call without
state, rules or tools, **D** our agent (orchestrator + ReAct workers + Gate 1). Every answer is judged by Gate 1.

| Metric | Definition | Why it matters |
|---|---|---|
| verdict_accuracy | expected rule statuses equal to the answer's claims (an unmentioned rule counts as pass) | does it tell the user the right thing |
| unknown_kept_rate | expected-**unknown** rules the answer still calls unknown | the core promise: unknown facts are never guessed |
| fail_recall | expected-**failing** rules the answer reports | a missed failure is the costly error |
| date_errors | dates not taken from facts or rules (Gate 1 `UNSUPPORTED_DATE`) | hallucinated deadlines |
| invented | rule ids that do not exist (`CLAIM_UNKNOWN_RULE`, `INVENTED_RULE_ID`) | hallucinated requirements |
| valid_answer_rate | answers that pass Gate 1 | can it be shown to a user |
| pass^k | scenarios where **all** k runs succeed | can the user rely on it |
| pass@k **(A)** | chance that at least one of k runs succeeds (unbiased estimator) | can it solve the case at all |
| avg_llm_calls, avg_seconds, p95_seconds **(A)** | model calls and time per run; p95 = nearest rank | cost of autonomy, worst-case wait |
| avg_tokens, cost_usd **(A)** | tokens per run, total estimated cost | what the cloud model costs |

Agent behaviour (D): valid_tool_call_rate (Pydantic-valid tool calls), avg_steps, loops_detected,
fallback_rate, human_review_rate, gate1_first_rejections, gate1_issues_caught **(A)** (problems Gate 1 stopped before
the user saw them) and reflection_recovery_rate **(A)** (refused answers the model then repaired). Procedure
identification (D): the pack chosen from a goal alone, including "ambiguous → ask". Memory: facts, runs and
lessons survive closing and reopening the Twin.

## 2. Retrieval (RAG) — `python -m eval.rag_compare` → `docs/eval_rag_compare.md`
Two query sets: every rule's description (find the official text behind a requirement) and hand-labelled user
questions (what the chat uses RAG for). Three retrievers on the same passages: BM25 keywords, bge-m3 embeddings,
and their fusion.

| Metric | Definition |
|---|---|
| hit@1 / hit@3 / hit@5 | a correct passage (from a labelled source and holding its phrase) in the first 1 / 3 / 5 |
| MRR@5 | mean of 1 / rank of the first correct passage (0 when none) |
| precision@5 **(A)** | share of the 5 passages that are correct |
| recall@5 **(A)** | share of the labelled sources (a question can have several) reached in the top 5 |
| nDCG@5 **(A)** | correct passages weighted by rank (1 / log2(rank + 1)), over the best possible order |
| query p50 / p95 ms, index s | speed of one search, and of indexing all packs from an empty cache |

Also reported by procedure and by question language, with the questions a retriever misses.
`python -m eval.rag_eval --models bge-m3 qwen3-embedding:0.6b` (→ `docs/eval_rag.md`) compares embedding
**models** with hit@1, hit@3, MRR and time; it chose bge-m3.

## 3. The chat assistant — `python -m eval.chat_eval` → `docs/eval_chat.md`
Eight typical messages; code checks what must happen, never the wording.

| Metric | Definition |
|---|---|
| answered | the reply passed Gate 1 (not replaced by the plan written by code) |
| first try | passed without any repair |
| expected effects | the expected fact recorded, proposal created, procedure selected, source cited, rule mentioned |
| citations valid | every source shown under the reply exists in the packs or the web corpus |
| reflection rounds | correction rounds (Gate 1 or self-review) |
| seconds, tokens, cost per turn | latency and cost of one turn |

## 4. Reading documents — `python -m eval.document_eval` → `docs/eval_documents.md`
Thirteen files with known values (text PDFs, scans, a phone photo, two documents in one PDF, public specimens).

| Metric | Definition |
|---|---|
| documents found | expected documents recognised in the file (extra ones counted apart) |
| values read correctly | of every expected value; how many the user is asked to confirm |
| values read wrong / believed wrong | wrong values, and those NOT shown for confirmation (must stay 0) |
| values not read | expected values missing |
| seconds, tokens per file, estimated cloud cost | speed and cost |

## 5. Routing, speed, memory
- `python -m eval.routing_eval` → `docs/eval_routing.md`: 38 goals (dev / held out); correct pack, wrong pack,
  false start, missed (tie → the agent asks).
- `python -m eval.speed` → `docs/eval_speed.md`: median and p95 time of each part (load packs, route, open the
  Twin, evaluate rules, search, model calls, a whole agent run).
- `python -m pytest -q tests/test_memory_sessions.py` and the memory section of `run_eval`: multi-session checks.

## 6. Where the latest numbers are
| Report | Last run | Mode |
|---|---|---|
| `docs/eval_results.md` | 2026-11-28 | offline (scripted policy: pipeline, not model quality) |
| `docs/eval_rag_compare.md` | 2026-11-28 | live, bge-m3 |
| `docs/eval_live_tc_kimlik_first.md`, `docs/eval_live_tc_kimlik_renewal.md` | 2026-11-24 (Neda) | live, qwen3:4b |
| `docs/eval_chat.md` | 2026-11-20 | live, qwen2.5:3b (written before the citation, reflection and cost columns) |
| `docs/eval_documents.md` | 2026-11-20 | live, local models (written before the token and cost columns) |
| `docs/eval_rag.md`, `docs/eval_routing.md`, `docs/eval_speed.md` | 2026-11-19 / 24 | live / offline |

Still to run with a key: `python -m eval.run_eval --live --provider openai --k 1`,
`python -m eval.chat_eval --provider openai`, `python -m eval.document_eval --provider openai --label gpt-6.1-terra`.
