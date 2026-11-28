# Migration plan: prototype to ReAct Orchestrator-Workers agent

Phase A (audit) for the v2 work described in `CLAUDE.md`. Written 2026-11-18. Every later rewrite or deletion
must match a row in this file, or update this file first.

## 0. Decisions approved on 2026-11-18

| Question | Decision |
|---|---|
| Branch | Work on **`dani-version-two`** (instead of a new branch). It was fast-forwarded to `origin/feature/marco-milestone1` and then merged with `origin/main` (commit `2c58f76`). Never pushed to `main`. |
| Model | `LLM_MODEL = "qwen3:4b"` by default (4 GB GPU). `qwen3:8b` / `gemma4:e4b` are documented alternatives. `EMBED_MODEL = "bge-m3"`. `OLLAMA_HOST` is written as `127.0.0.1` (same as localhost, avoids an IPv6 lookup on Windows). |
| R015 | The `rules.json` version is canonical. Marco's code-only "Milestone-1 scoped" variant is dropped. |
| Facts from chat | `record_fact` stores `user_stated` with `confirmed=False`. Unconfirmed facts never reach the rule evaluator until the user confirms them in the app. |
| Second procedure | İkamet extension (`procedures/ikamet_extension`). Its rules are `unverified` until a person checks them against the official text. |
| Docs language | English. |
| Retired stub tests | `tests/test_scenarios.py` and `tests/test_restart.py` are deselected in `pytest.ini` (phase 0) and deleted in phase 3. |

### Progress log
- **Phase 3 done.** Deleted: `src/memory/twin_store.py` (v1 store), `src/memory/session.py`,
  `src/memory/schema.sql`, `src/evaluation/`, `scripts/run_eval.py`, `data/synthetic_samples/scenarios.json`,
  `tests/test_store.py`, `tests/test_restart.py`, `tests/test_scenarios.py`. What replaced them:
  - `src/twin/store.py` + `schema.sql` v2 (`codec.py` moved unchanged),
  - `tests/test_twin_store.py`, which ports every v1 store guarantee plus the restart-in-a-new-process check,
  - `tests/test_twin_facts.py`, which ports the provenance guarantees of `test_verification.py`,
    `test_document_lifecycle.py` and `test_milestone_documents.py`,
  - `load_rule_cases` moved into `tests/test_knowledge.py` unchanged.

  The old `test_*` files that still exercise the milestone-1 graph keep running until phase 9.
- **Phase 9 done.** The milestone-1 pipeline was deleted once its replacements were green.
  - **Source deleted:** `src/core/` (graph, state, llm_diagnostics), `src/agents/`, `src/guardrails/`,
    `src/workers/`, `src/memory/{checkpointer,reflection,state_update}.py`, `streamlit_app.py`,
    `scripts/run_demo.py`.
  - **Tests deleted:** `test_action_planner`, `test_hybrid_planning`, `test_plan_validator`, `test_reflection`,
    `test_requirement_engine`, `test_evaluation`, `test_verification`, `test_document_lifecycle`,
    `test_milestone_documents`, `test_document_types`, `test_persistence`, `test_required_documents`,
    `test_rule_loader` and `test_streamlit_helpers`. Where their guarantees went:
    - rule verdicts: `tests/data/legacy_verdicts.json` + `test_rules_oracle.py`, plus `test_rules.py`;
    - provenance: `test_twin_facts.py`;
    - Gate 1: `test_gate1.py`;
    - repairs, approval and the loop: `test_agent.py` and `test_sandbox.py`;
    - persistence: `test_twin_store.py`;
    - UI helpers: `test_app.py`;
    - scenarios: `procedures/ikamet_first/eval/scenarios.json` + `test_eval.py`.
  - **Ported in place, same names and guarantees:**
    `test_processor.py::test_hybrid_observations_and_correction_notes_survive_sqlite` and
    `::test_correction_notes_alone_do_not_block_gate_1`, and
    `test_extractor.py::test_ocr_failure_leaves_document_rejected_not_analyzed`.
  - **Interface changes:**
    - `evaluate_rules` now returns a compact grouped observation (10.9k → 2.2k characters, so it is not cut
      for the model);
    - `search_procedures` returns the selected procedure's sections;
    - the Ollama backend sets `num_ctx` (`LLM_NUM_CTX`);
    - `langgraph-checkpoint-sqlite` was removed from the requirements.
  - **Historical scripts.** `scripts/convert_knowledge.py` and `scripts/snapshot_legacy_verdicts.py` are now
    historical one-offs (they need commit `486917b`).
- **Phase 10 done.** `procedures/ikamet_extension` was added with the add-procedure skill, with **zero
  changes in `src/`**.
  - **Rules.** It has 8 rules, all `unverified`, with a review list in the pack's README.md. The 4 sources are
    copied from `knowledge/sources.json`.
  - **Coverage.** Its residence permit card has no typed schema, so it uses the generic extraction path and a
    `document_complete` derived field. It has 6 scenarios.
  - **Generalization test.** `tests/test_generalization.py` fails if `src/` mentions any rule id (or
    procedure id) of any pack.
  - **One fix outside `src/`.** In `eval/run_eval.py`, goal scenarios now run only when their expected pack is
    loaded (needed for `--procedure`).
- **Live smoke test (2026-11-18, qwen3:4b on the GTX 1650, scenario `02-passport-buffer`).**
  - **First run.** Two invalid arguments (`section_id: ""` and `fields` sent as a string) plus a repeated call
    triggered the deterministic fallback. The answer was correct and passed Gate 1.
  - **Fix.** Both tools now accept those two unambiguous shapes, with tests.
  - **Second run.** status `answered`, no fallback, 9 model calls, 217 s, 2 Gate 1 repairs, correct claims.
  - **Open issue.** The accepted summary contained a wrong sentence (it called confirmed dates "not
    confirmed"). Gate 1 cannot judge the meaning of free prose (agent module doc, known limits). The full
    `python -m eval.run_eval --live` measurement is still to be run.
- **Clean-up after phase 10 (2026-11-18).** Everything below was unused by the code and tests, or duplicated
  elsewhere; git history keeps it.
  - `knowledge/` (8 files) and `tests/test_knowledge.py`. The pack replaced the data: the 15 test cases live on
    as `knowledge-<id>` contexts in `tests/data/legacy_verdicts.json`, volatility is in `rules.json`, the
    change-monitoring policy is summarised in `docs/modules/pack.md`, and `tests/test_pack.py` runs the same
    consistency checks (unique ids, known sources, known dependencies, no cycle) on every pack.
  - `scripts/convert_knowledge.py`, `scripts/snapshot_legacy_verdicts.py`: one-off scripts that could no longer
    run (their inputs are gone). `scripts/make_synthetic_form.py` stays.
  - `docs/INTEGRATION_REPORT.md`, `docs/memory_and_evaluation.md`: they described deleted milestone-1 code.
  - `README_KIT.md` (one-time kit setup), `docs/modules/.gitkeep`, `data/synthetic_samples/example.pdf` (a 2 MB
    public handbook nothing used). The `.txt` samples stay as demo uploads for the app.
  - Legacy names in `config.py` (`KNOWLEDGE_DIR`, `OLLAMA_MODEL`, `MAX_REFLECTION_ATTEMPTS`, `PLANNER_*`) and
    dead code (`schemas.ExtractedDocument`, `tools.common.Role`, `codec.FACT_TYPES`, `TwinStore.close_fact`).
- **Full manual test in the browser (2026-11-18, qwen3:4b + bge-m3).** Every app step worked (upload read in
  9 s, answers, checklist approval, sandbox, outcome, Twin deletion). Two agent runs fell back to the
  deterministic workflow: qwen3:4b sent `section_id: "null"` and then repeated `evaluate_rules`. The fallback
  also kept the active procedure for the goal "I want to extend my İkamet". Fixes, with tests:
  `"null"`/`"none"`/`"all"` mean all sections; a repeated call is answered with the next section to delegate;
  the prompt says to search again when the goal names another procedure; the fallback follows a goal that
  `search_procedures` matches unambiguously.
- **App simplified (2026-11-18).** `app.py` was rebuilt around the presentation's three questions: a goal box
  and the tabs *Where am I?*, *What is missing?*, *What next?* (see `docs/modules/app.md`). Same functions as
  before; the seven stacked panels, raw field names and the stage-id sandbox panel are gone.
- **Integration of Neda's `main` (2026-11-19, not committed).** `origin/main` gained three commits by Neda
  (`fa12d19` memory & evaluation, `d04e590` simpler UI + public test documents, `0f3e628` one run per profile).
  They are built on the milestone-1 pipeline that phase 9 deleted, so there was **no `git merge`**: it would bring
  back about 60 deleted files and a second store. Each idea was ported instead:

  | What (on `main`) | Decision | Where it is now |
  |---|---|---|
  | `streamlit_app.py` layout: status bar, decision first, five views, plain wording | adapt | `app.py` rebuilt on our engine (`docs/modules/app.md`); replaces the three tabs |
  | one run per profile (`0f3e628`) and `tests/test_ui_guard.py` | adapt | `profile_lock` / `run_locked` / `wait_for_run`, the 3 tests in `tests/test_app.py` |
  | twin record mirroring the LangGraph checkpoint (`twin_store.py`, `twin_sync.py`) | replace (not needed) | our Twin is the single source of truth (D2): there is no checkpoint to mirror |
  | `run_requirements` (rule results of each run) | adapt | table `run_results`, schema v3, upgraded in place from v2; Technical view and memory checks |
  | validated reflection lessons | already ported (phase 8) | `src/memory/episodes.py`; our orchestrator already reads them (her "not yet done") |
  | harness multi-session scenarios M1-M3 | adapt | `eval/memory_eval.py`, `tests/test_memory_sessions.py`, a section of the eval report |
  | `data/reference_samples/` (6 public PDFs + `SOURCES.md`) | keep | copied unchanged; `SOURCES.md` maps them to our document types |
  | `knowledge/`, `docs/INTEGRATION_REPORT.md`, `docs/memory_and_evaluation.md`, `example.pdf`, `prova.txt` | not brought back | they describe milestone-1 code removed after phase 10 |

  When this branch goes to `main`, the pull request replaces the milestone-1 prototype there; Neda's three commits
  stay in history.
- **Knowledge base for rule extraction (2026-11-19).** New: `src/rag/ingest.py` (page/PDF → text, standard
  library + httpx + PyMuPDF), `knowledge.py` (`fetch`, `add`, `check` with SHA-256 change detection and the rules
  to review), `drafts.py` (RAG rule drafts whose quotes are checked against the saved text, written to
  `rule_drafts.json`, never to `rules.json`) and the CLI `python -m src.rag`. `Source` gained `content_sha256`.
  - All sources of the four packs were downloaded on 2026-11-19 (the e-İkamet form also from Neda's sample PDF).
    The real GOV_FAQ text confirms R006 (60 days) and adds an earliest extension date, now in the
    `ikamet_extension` review list (no rule changed).
  - **Tests.** `test_rag.py` keeps its 10 query/source pairs unchanged; its input texts are now pinned in
    `tests/data/placeholder_sources/` (the old placeholders) so a website cannot change them.
    `test_vectors_are_cached_in_the_twin` was **changed on purpose**: embeddings now go in batches of 16 (one
    call for 567 real passages timed out), so the test checks "every passage once, then only the question"
    instead of "one call".
  - **Turkish text.** Quote checks fold I/ı/İ like `evidence.normalized`; plain `casefold` missed "ALTI" vs "altı".
- **Embedding model kept: `bge-m3` (2026-11-19, measured).** `eval/rag_eval.py` asks one question per rule (its
  English description) and expects the rule's source, and its quoted passage when the rule has a quote:

  | model | hit@1 | hit@3 | MRR | hit@3 on the Turkish NVİ pages | indexing (once) | VRAM in use |
  |---|---|---|---|---|---|---|
  | `bge-m3` | 0.561 | **0.758** | **0.644** | **0.60 / 0.64** | 392 s | 689 MB |
  | `qwen3-embedding:0.6b` (with its query instruction) | 0.561 | 0.667 | 0.606 | 0.33 / 0.36 | 175 s | 1.2 GB |

  `qwen3-embedding:0.6b` was pulled only for this comparison (`ollama rm qwen3-embedding:0.6b` frees 639 MB).
- **Two Procedure Packs for the Turkish identity card (2026-11-19).** `tc_kimlik_first` (15 rules) and
  `tc_kimlik_renewal` (14 rules), from four NVİ sources saved with the new CLI, all rules `unverified` with their
  quote in `notes`, 6 scenarios each (hand-computed verdicts, all matching the engine). No change in `src/`
  was needed for the packs. Two NVİ pages disagree on appointments; the rule is advisory with a `CONFLICT` note.
  Three goal scenarios (G05-G07) check that the right pack is chosen.
- **Live RAG rule drafts (qwen3:4b, `tc_kimlik_renewal`, 3 topics).** 10 drafts, every quote verified in the
  saved text, 3 linked to existing rules (TCR003, TCR010, TCR011), one new lead (e-Devlet re-application for a
  lost card, to review), one wrong field. A first run copied the topic as description and listed six fields:
  the prompt now asks for 1-3 fields and the code flags both cases. The drafts are in
  `procedures/tc_kimlik_renewal/rule_drafts.json` as an example; `rules.json` was not touched.
- **UX fix found in the browser.** A field used by several sections was asked in each of them (the TC kimlik
  reason appeared twice); it is now asked once, in the first section (test added).
- **Live check of the ID-card packs (qwen3:4b, 2026-11-19).** Goal: "I lost my Turkish ID card (TC kimlik) last
  week. What do I need to get a new one?". Found and fixed:
  - **Wrong pack.** The generic words ("tc kimlik", "turkish id card") were aliases of `tc_kimlik_first` only, so
    it won. Both packs now share the generic aliases and the pack-specific words decide (lost / expired / renew /
    kayıp / yenileme for the renewal; first / birth / citizenship / nüfus cüzdanı for the first card). A goal
    with neither ("I want to get a Turkish ID card") is a tie, so the agent asks. Goal scenarios G05-G09 cover
    these cases; the first-card scenarios' goal now says "my first Turkish ID card" (their verdicts unchanged).
  - **Repeated question in the fallback.** Three rules waiting for the same reason produced the same question
    three times, and a derived value was asked as "Please provide: applicant age months". The fallback now asks
    each field once, and `question_for` asks a derived value through its inputs (birth date, application date).
  - **After the fix** the same goal chose `tc_kimlik_renewal`, and the fallback asked each question once (413 s).
  - **Still slow.** Two orchestrator calls hit the 120 s timeout once the observations of five worker runs were in
    the prompt, so the deterministic fallback answered after 482 s (18 model calls). This is the known 4 GB GPU
    limit (see the `num_ctx` measurement still to do), not a correctness problem: the fallback answer passed Gate 1.
- **Documents, chat assistant, web corpus, vision model (2026-11-20).** Asked by the team: read documents better
  (also several documents in one PDF), keep RAG only where it is useful, try web scraping for other procedures, and
  an assistant that explains the state step by step and answers questions instead of only static points.

  | Module | Decision | Why |
  |---|---|---|
  | `src/document_processing/segment.py` | **new** | One upload → several documents: a page opens a document by MRZ or title, every other page continues it; keywords name an untitled document; the model only breaks ties. |
  | `src/document_processing/mrz.py` | **new** | ICAO TD3 (passports) and TD1 (ID cards, the back of the T.C. card) read by code with check digits. |
  | `src/document_processing/generic.py` | **adapt** | All pages in batches (before: 3,400 characters), only missing fields asked again, MRZ values from code. |
  | `src/document_processing/vision.py` | **new** | Vision model for pictures; values compared with the OCR text; what code cannot check waits for confirmation. |
  | `src/document_processing/extractor.py` | **adapt** | Photos (JPG/PNG); OCR also for scans with only a watermark; `TextPage.ocr`. |
  | `src/tools/extract_document.py` | **adapt** | Segments, then typed or generic reading per document, a second pass for typed documents (values to confirm), the vision step, `pages` to read one part again. The seven typed kinds keep the milestone-1 processor. |
  | `src/models.py` `DocumentSpec` | **adapt** | `titles`, `keywords`, `mrz` (pack data, so a new procedure needs no Python). |
  | `src/agent/briefing.py` | **new** | The plan computed by code; the app's shared helpers (questions, document status) moved here from `app.py`. |
  | `src/agent/chat.py` | **new** | The chat assistant: the orchestrator graph with the chat's prompt, tools, Gate 1 subset and plan fallback. |
  | `src/twin` schema v4 | **adapt** | `chat_messages`; v2 and v3 files are upgraded on open. |
  | `src/rag/crawl.py`, `index.py` | **new / adapt** | Polite crawler (robots.txt, content links only); the index searches a pack or the web corpus. |
  | `src/llm.py` | **adapt** | `OllamaBackend.see` and `ask_vision` (same contract as `ask_structured`). |
  | `app.py` "Next steps" | **replace** | By the Assistant tab (chat + documents + tools). The goal box and the agent's question card are gone: the conversation replaces them. The orchestrator with its workers stays one click away ("Run a full check"). |

  - **Measured: reading documents** (`python -m eval.document_eval`, 13 files, `docs/eval_documents.md`):

    | | documents found | values read correctly | of which to confirm | values believed wrong |
    |---|---|---|---|---|
    | before (committed code) | 8/17 | 46% | 0 | 0 |
    | after (segments, pack classification, all pages, MRZ) | 16/17 | 75% | 0 | 0 |
    | + second pass and vision model | 17/17 | 86% | 6 | 1 (a partial name read from a photo) |
    | + the two vision fixes below (rerun of the two pictures) | 17/17 | 89% | 8 | 0 (the partial name is shown for confirmation) |

    Before, every T.C. kimlik document was unknown ("detect automatically" knew only the seven İkamet kinds), the
    second document of a file was lost, photos were refused and border stamps were never recognised.
  - **Measured: speed of one reply** (1,250-token prompt, 4 GB GPU): `qwen3:4b` 42 s at `num_ctx` 8192 (45% CPU),
    37 s at 4096 (34% CPU); `qwen2.5:3b` 16 s at 8192 (100% GPU). The chat runs on `qwen2.5:3b` (`CHAT_MODEL`); the
    agent team and documents keep `qwen3:4b`; `num_ctx` stays 8192 (4096 gains little). `docs/eval_chat.md`
    compares the two models on 8 typical messages.
  - **Vision model `qwen3-vl:2b-instruct`.** The default `qwen3-vl:2b` tag always thinks first: with a JSON format
    it spent 3,000 tokens and returned nothing (132 s). The instruct variant named both stamp specimens correctly and
    read their dates (31-37 s), but a full-page photo takes 2-3 minutes, so it runs only on pictures, only when
    something is missing. Asked to name a page and read 29 fields at once it returned all fields empty: naming and
    reading are now two calls. It read a partial name ("DENİZ") that the OCR text also contained: only dates and
    numbers found in the OCR text count without confirmation now.
  - **Web scraping.** First crawl followed the site menu and filled its page budget with programme and news pages;
    links in navigation, headers, footers and sidebars are now ignored. A seed reached from another seed was not
    expanded; fixed. Corpus: 25 pages of Sabancı IRO guides (tax ID, bank account, SIM, IMEI, health insurance,
    address registration, e-Devlet, residence permit and extension, official documents). NVİ and Göç disallow
    `/kurumlar` in robots.txt; the crawler skips it. (The pack source `NVI_DAMAGED_CARD_2020`, fetched once on
    2026-11-19 as a hand-picked source, is under that path; it is not re-crawled.)
  - **Live chat debugging (qwen2.5:3b).** Asked "Explain where I am", it recorded facts the user never wrote
    ("exchange student: yes", a stay end date) and built a checklist instead of answering; told "I lost my ID card",
    it also recorded "loss reported: yes" and called `evaluate_rules` with invented section names. Fixes in code:
    facts must be in the user's words (`src/agent/stated.py`), `evaluate_rules` is not a chat tool, artifact tools
    are offered only when asked, `final_answer` is the first option, the state sent to the model was cut from
    11,000 to 3,600 characters (the first call took a minute). It also wrote 3,900-character answers, statuses as
    "not met" and whole plan steps as refs: summaries may now be 4,000 characters (`SUMMARY_MAX_CHARS`) and code
    tidies status words and refs that contain exactly one valid ref; Gate 1 still checks the result.
    Later runs showed the same models recording facts for questions, inventing fields and claiming rule ids when no
    rule was failing: code now offers `record_fact` only when the message tells something (a date, a first-person
    sentence), offers `search_procedures` only without a procedure or when the message names another one (two alias
    words), gives the model every rule id with its status, drops claims about rule ids that do not exist (never shown;
    the text is still checked), and links the web pages under a reply that has no procedure.
  - **Measured: chat model** (`python -m eval.chat_eval`, 8 typical messages, `docs/eval_chat.md`): `qwen2.5:3b`
    passed Gate 1 on 6/8 (5 at the first try) in about 50 s per turn; `qwen3:4b` on 4/8 in about 210 s. The other
    replies were replaced by the plan written by code. `CHAT_MODEL` stays `qwen2.5:3b`. With the final code
    `qwen2.5:3b` passed 6/8, all at the first try, 69 s per turn on average.
  - **Live browser tour** (Edge, scratch Twin): a first message without procedure chose `tc_kimlik_renewal` and
    recorded "lost" (unconfirmed); a PDF holding the card and the fee receipt was split into two documents
    (47 s); a suggestion and a tax-number question without procedure were answered and checked; 0 exceptions,
    0 console errors. Found and fixed: a passage cut at 600 characters gave a broken URL (passages now end at a word
    boundary), the plan asked a question whose value already waited for confirmation, a raw extraction warning,
    "1 values".
  - **RAG kept where useful** (`docs/modules/rag.md`): not for deciding or explaining rules (rules carry their
    quote), yes for questions outside the rules, for procedures without a pack (web corpus), for drafts and for
    change detection.
  - **Tests changed on purpose.**
    - `tests/test_app.py`: the goal-box and agent-question tests were replaced by chat tests, because that UI was
      replaced (chat message, suggestion, confirming a fact from the chat, full check posted in the chat, reading
      summary). No expected verdict changed.
    - `tests/test_twin_store.py`: the v2 upgrade test now also leaves out `chat_messages` (a v2 file never had it),
      expects version 4, and a v3 case was added.
    - New: `test_chat.py`, `test_segment.py`, `test_vision_and_crawl.py`.

- **2026-11-24: cloud model option, reflection made visible, address documents, plain questions, new UI.**
  User feedback: documents still read badly (the home address only partly, no utility bill), the chat assistant
  often misses the point, reflection never shows up, the chat grows downwards forever, the checks are confusing and
  some are pointless, and it is not clear where the plan comes from and when the tools (e-mail, form, calendar)
  are used. Decisions, each recorded before the change:

  | Module | Decision | Why |
  |---|---|---|
  | `src/llm.py` | **adapt**: `AnthropicBackend` (Claude, official `anthropic` SDK) and `OpenAIBackend` (GPT, official `openai` SDK) next to Ollama; `agent_backend()` / `local_backend()` | User approved both SDKs (2026-11-24) and chose "documents stay local": the cloud model runs the chat, the orchestrator and its workers and sees the state, never document pages. Schemas Claude cannot constrain (free-form `args`) are asked for in words; a refused schema is retried without it; errors become `LLMError`, so the deterministic fallback still works. |
  | `src/agent/chat.py` | **adapt**: self-review (reflection) after Gate 1, `CHAT_MAX_REPAIRS` 1 → 2, every round recorded (`reflections`), all chat tools offered with a cloud model, section ids accepted as refs | Reflection existed (Gate 1 repair) but code tidied most mistakes first and the single repair usually ended in the fallback, so it was never seen. The self-review catches what Gate 1 cannot judge (prose that contradicts the state); code keeps only criticisms that quote words of the reply. |
  | `src/document_processing/generic.py` | **adapt**: one re-reading of misquoted values (reflection), quotes over several lines, `lines` for multi-line values, choice fields must map to a known value | The address was read as its first line only, and the printed title "KİRA SÖZLEŞMESİ" was stored instead of `rental_agreement` (seen live). |
  | `src/models.py`, `src/pack.py`, `src/twin/facts.py` | **adapt**: `FieldSpec.ask` (now / later / never) and `choices`, `Section.later`, `DocumentSpec.optional`, `ExtractSpec.lines`, derived `missing_documents` / `documents_complete` | Generic engine features; no procedure is named in code. |
  | `src/agent/briefing.py`, `src/agent/gate1.py` | **adapt**: no questions for office data or computed values, yes/no about a missing document becomes an upload, later steps kept apart, Gate 1 does not require later-step failures in today's answer | "Iro window end?", "Missing required documents count?", "Application channel?" were asked as questions. |
  | `src/agent/actions.py` | **new**: what the agent can prepare now (checklist, filled form, reminders for known dates, e-mail drafts) | Makes the twin → real tools explicit and one click away; they still only create proposals. |
  | `procedures/ikamet_first` | **adapt**: plain questions, choices, ask flags, `turkey_address` / `turkey_phone` / `contact_email` as text, address proof document and optional utility bill, R030 (title deed needs a utility bill, **unverified**, source METU_RESIDENCE_2026), `iro_submission` marked later | The pack had no document for the address proof at all. Sabancı's list (dorm certificate, rental agreement, notarized commitment) has no utility bill; METU's list asks for one only with a title deed. |
  | `app.py` | **replace layout** (same functions underneath): Assistant (fixed-size scrolling chat + plan + quick actions), To do (confirm or correct, fix, upload, questions, later steps, all requirements), Documents, Actions, Technical (tabs: how it works, last answer, run history, Twin, knowledge base with a search box) | The old Your details / Checks tabs were the confusing part. |

  - **Measured live (local models, synthetic rental agreement over two lines):** before, `turkey_address` was the
    first line only and `address_proof_type` the printed title (R017 then failed); after, the whole address and
    `rental_agreement`. A chat turn on `qwen2.5:3b` with the self-review on passed Gate 1 and was reviewed and
    rewritten; the small model criticised correct sentences too, so `SELF_REVIEW=auto` keeps it on only for a cloud
    model. The cloud backends are tested with fake clients only (no API key in this environment).
  - **Tests changed on purpose.**
    - `tests/test_rules_oracle.py`: the oracle compares only the rules of the legacy snapshot (R030 did not exist
      in the old engine); no legacy verdict changed. `procedures/ikamet_first/eval/scenarios.json`: R030 expected
      statuses were added (pass where the proof is a dorm certificate or Airbnb, unknown where it is not given).
    - `tests/test_pack.py`, `tests/test_tools.py`: 29 → 30 rules; `tests/test_rules.py`: a pass/fail case for R030.
    - `tests/test_chat.py`: the repair test follows `CHAT_MAX_REPAIRS` (now 2). `tests/test_extract_document.py`:
      a misquote now gets one second reading, so the scripted model gets a reply for it (still no fact accepted).
    - `tests/test_app.py`: views renamed (details/checks → todo), status bar labels, the checklist button label, a
      choice field stored as `short_term` instead of the typed text, the open-item count = plan steps, optional
      documents not counted.
    - New: `test_llm_cloud.py`, `test_reflection.py`, `test_pack_address.py`, `test_actions.py`.

- **2026-11-24 (later): the app simplified for the presentation.** User feedback: "too confusing, a thousand things
  to tick, To do repeats Documents, too many strange rules, the plan should connect to Gmail / Google Calendar".
  Decisions: `app.py` **replace layout** again: three views (Assistant, Documents, Details); the plan is shown one
  step at a time in a *Now* box with the control the step needs (one click answers a yes/no; *Skip for now*); the
  journey bar shows the pack's official steps; the requirements list moved to Details. `src/agent/links.py`
  **new**: an approved e-mail opens in Gmail and an approved reminder in Google Calendar through their public
  "compose" / "add event" addresses, already filled in; the user presses Send or Save, so the hard rule "nothing is
  sent or booked" holds and no OAuth, key or new library is needed (sending through the Gmail API would break that
  rule). Packs `ikamet_first` and `ikamet_extension`: the IRO contact gets `irolegal@sabanciuniv.edu`, the address
  the official page SAB_EXCHANGE_CURRENT gives. Tests changed on purpose: `tests/test_app.py` (views, the Now box,
  Gmail link after approval), `tests/test_sandbox.py` (the unknown-address case now uses Migration Management, which
  has no address in the sources; a new test checks the IRO address). New: `tests/test_links.py`.

- **2026-11-25: documents can be read by the cloud model (opt-in).** User decision: "integrate GPT as the model that
  reads the documents and extracts the information, so GPT's OCR is used instead of ours" (`gpt-6.1-terra`: text and
  image input, Chat Completions and structured outputs supported, per the OpenAI model page). It reverses the
  2026-11-24 choice "documents stay local" **only when asked**: `DOCUMENT_PROVIDER=cloud` with
  `LLM_PROVIDER=openai|anthropic`; the default stays `local`. The app says which model reads the documents and that
  the pages are sent to the provider.

  | Module | Decision | Why |
  |---|---|---|
  | `src/llm.py` | **adapt**: `see()` on `OpenAIBackend` (image as a `data:` URL, detail high) and `AnthropicBackend` (base64 image block); `document_backend()`, `documents_in_cloud()`, `is_cloud_backend()`, `document_model_label()` | One switch decides who reads documents; embeddings stay local. |
  | `src/document_processing/transcribe.py` | **new**: the cloud model transcribes picture pages (scans, photos, scans with a scanner-app text layer) from a JPEG of the page; if it fails, RapidOCR reads the page | RapidOCR misreads small print, Turkish letters and stamps. The transcription becomes the page text, so every later check is unchanged. |
  | `src/document_processing/extractor.py` | **adapt**: `extract_pages(path, transcribe=None)`, `is_picture_page`, `page_png` / `page_jpeg` | Default behaviour unchanged (no `transcribe` means RapidOCR as before). JPEG for the cloud: a photographed page is ~1 MB instead of 6 MB as PNG (providers cap image size). |
  | `src/document_processing/processor.py` | **adapt**: `process_document(..., backend=None)` sends the same prompts to the cloud model; `select_chunks` takes `max_context_chars // CHUNK_CHARS` chunks (still 2 at the local 3,400 budget); `process_passport_pages(..., pages=, **options)` reuses the pages already read | The 4 GB GPU limit (3,400 characters, 2 chunks) cut long documents; a cloud model reads the whole document (`CLOUD_DOCUMENT_MAX_INPUT_CHARS` 40,000). |
  | `src/document_processing/generic.py`, `vision.py`, `src/tools/extract_document.py` | **adapt**: whole-document batches and sharper JPEG images for a cloud backend; the transcriber is used by `extract_file`; a note names the pages the cloud could not read | Same checks by code: quotes on the page, MRZ check digits, confirmation of what code cannot check. |
  | `app.py`, `eval/document_eval.py` | **adapt**: tools read with `document_backend()`; the sidebar says who reads documents and what leaves the computer; the document evaluation runs on the cloud model with the same switch | The user sees the privacy trade-off; the before/after can be measured with `python -m eval.document_eval`. |

  - **Limit, stated honestly:** with the cloud OCR, a value's quote is checked against the model's own
    transcription, so a transcription error is caught only where code has a second check (MRZ check digits, date
    and choice parsing, conflicts between documents). Locally the reference text came from RapidOCR, an independent
    but weaker reader.
  - **Verified offline:** the real `openai` 3.58 and `anthropic` 2.13 SDKs serialise the image requests (HTTP
    intercepted with a mock transport, no network). Not yet run against the real API (no key in this environment).
  - **Test changed on purpose:** `tests/test_llm_cloud.py::test_the_cloud_backends_cannot_see_images` encoded the
    old decision; it is replaced by `test_documents_go_to_the_cloud_only_when_asked` (local by default, cloud only
    with `DOCUMENT_PROVIDER=cloud` and a cloud provider). New: `tests/test_documents_cloud.py`.

## 1. What was audited

There is no single integrated branch. The repository has four separate lines of work that all start from the
same Workstream A stub graph (`8a7ee37`):

| Branch (commit) | Author | Content | Offline tests (run on 2026-11-18) |
|---|---|---|---|
| `dani-version-two` = local `main` (`8a7ee37`) | Daniele | Stub graph only (`src/core/{state,graph,stubs}.py`) + `knowledge/` | none |
| `origin/main` (`b71ac92`) | Neda | Stub graph + **SQLite Twin tables** (`TwinStore`), codec, run log, lessons, eval harness, 5 stub scenarios | **104 passed** |
| `origin/feature/marco-milestone1` (`225aad7`) | Marco | **Most integrated**: real graph, Gate 0/1, 29-rule engine, fact verification, LLM planner + repair, document processing with OCR, LangGraph SQLite checkpoints, Streamlit app | **489 passed** |
| `origin/test/integration-abc` (`2ca7d6e`) | Andrea | Earlier A+B+C integration (R014 only, root-level modules). Marco's branch already adopts its evidence logic | 95 passed + 79 subtests |
| `origin/feature/document-processing` (`c9121fb`) | Andrea | Ancestor of `integration-abc` | (contained in the above) |

How the tests were run: each branch was exported with `git archive` to a scratch folder and run with the project
`.venv` (Python 3.13.2). pytest is **not installed** in `.venv`, so a scratch copy was used and the venv was not
modified. Marco's suite needs its temp folder **inside the project root**, because Gate 0 rejects paths outside it.
Run it as `python -m pytest -q --basetemp=.test-temp/x`. With pytest's default temp folder, 73 tests fail.

**Merge check.** `git merge-tree` of Marco's branch with `origin/main` gives 6 conflicts: `.gitignore`, `config.py`,
`requirements.txt`, `pytest.ini`, `tests/conftest.py` (trivial unions) and `src/memory/checkpointer.py`. The last
is a real add/add conflict: Marco's LangGraph `SqliteSaver` wrapper and Neda's `TwinStore` have the same filename.
A simulated merge kept Marco's file and moved `TwinStore` to `src/memory/twin_store.py`. Result: **584 passed,
9 failed**. All 9 failures are Neda's graph-level scenario/restart tests. They were written against the stub
graph (stub Gate 1 with an "appointment" date conflict, no checkpointer `thread_id`). See section 5.

**Recommendation:** start `feature/react-agent-v2` from `origin/feature/marco-milestone1` and merge `origin/main` into it
(Phase 0). This needs confirmation (question 1).

## 2. Target layout (after phase 10)

```
config.py                      all settings (LLM_MODEL, EMBED_MODEL, OLLAMA_HOST, MAX_STEPS, WORKER_STEPS, MAX_REPAIRS, DATABASE_PATH, ...)
app.py                         Streamlit (adapted from streamlit_app.py)
procedures/<id>/               procedure.json, rules.json, sources.json, sources_text/, eval/scenarios.json
src/models.py                  pack models (Procedure, Field, Rule, Condition, Source) and RuleResult
src/pack.py                    load + validate a pack; CLI `python -m src.pack validate <id>`
src/rules.py                   three-valued evaluator (no rule ids)
src/llm.py                     every Ollama chat/embedding call (schema output, Pydantic, 1 retry, fallback)
src/document_processing/       existing extraction (kept), plus a generic pack-driven extraction path
src/twin/                      SQLite Twin: schema.sql, codec.py, store.py, facts.py (provenance, valid_from/to)
src/tools/                     registry.py (ToolSpec) + one file per tool
src/agent/                     decision.py, worker.py, orchestrator.py, gate1.py, fallback.py, graph.py
src/sandbox/                   mock world + simulate_submission + promotion gate
src/rag/                       chunk, embed, cosine search, vector cache in SQLite
src/memory/                    episodes (validated repairs)
eval/                          run_eval.py, baselines A/B/D
tests/
```

## 3. Module table

Decision values: **keep** (no behaviour change, may move), **adapt** (reuse the logic and change the interface),
**replace** (new module, the old one is deleted when its replacement is green), **delete**.
"Depended on by" lists importers in the merged tree (tests are counted by file).

### 3.1 Core, orchestration, guardrails

| Module | Decision | Reason | Depended on by | Phase |
|---|---|---|---|---|
| `config.py` (Marco + Neda) | adapt | Merge the two. Add the kickoff settings. Keep `DATABASE_PATH = data/digital_twin.db` (used by the app and the local DB). Neda's `DB_PATH`/`PBA_DB_PATH` is folded into it. Document timeouts and chunk limits stay. `PLANNER_*` settings go when the planner goes | everything | 0 |
| `src/core/state.py` | adapt | Keep `UserEvent`, `Document`, `ValidationIssue` and the reducers. Make `AgentState` procedure-independent. `ResidencePermitContext` (70 hard-coded İkamet fields) is replaced by pack-declared facts. `Requirement` is replaced by `RuleResult`. `PlanStep`/`ActionPlan` are replaced by `FinalAnswer`/`ProposedAction`. The 13 event kinds are reduced: chat goal plus upload go to the agent, and document lifecycle events become direct Twin calls (decision D8) | 33 files | 5 |
| `src/core/graph.py` | adapt | Keep the LangGraph `StateGraph`, bounded-repair routing and `respond`. The graph becomes `orchestrator_decide -> act -> decide`, with `final -> gate_1 -> repair/accept`. The rule-driven `supervisor` is removed | app, demo, 9 test files | 5 |
| `src/core/llm_diagnostics.py` | adapt | Its content-free logging (`logged_chat`) moves into `src/llm.py`. The `R\d{3}` regexes are removed (no rule-id format in `src/`) | planner, processor, gate_1, reflection | 2 |
| `src/core/stubs.py` (only on `dani-version-two`) | delete | Already deleted on Marco's branch, replaced by real nodes | none | 0 |
| `src/guardrails/gate_0.py` | adapt | Keep the input/path checks (uploads must be existing PDF/TXT inside the project). The hard-coded Sabancı scope test moves to `procedure.json` `scope_conditions`, evaluated by `src/rules.py` (D6) | graph, 2 test files | 4 |
| `src/guardrails/gate_1.py` | adapt | Becomes `src/agent/gate1.py` on `FinalAnswer`. Keep: completion/approval/unknown-as-failure phrase checks, unknown rule ids and fields, deadline-key matching, duplicate ids. Add: every structured claim equals the latest `evaluate_rules` result; every date comes from facts or rules; every action cites a rule or finding; unverified rules labelled. Remove the hard-coded prerequisite sets (official_application → R001–R006, etc.), which move to `procedure.json` `stages[].requires` (D7) | graph, 6 test files | 5 |
| `src/guardrails/human_approval.py` | replace | LangGraph `interrupt()` approval of a whole plan becomes per-action approval of PROPOSED actions stored in the Twin (D2). This survives restarts with no checkpointer | graph | 6 |
| `src/agents/planner.py` | replace | Replaced by orchestrator + workers + `FinalAnswer`. **Reused idea:** the model may only pick ids from a closed enum (`planner_output_schema`), and Python rebuilds the metadata. The same technique constrains `Decision.action` and `next_actions[].ref`. The repair bookkeeping (locked actions, slot reuse, ~300 lines) is not carried over | graph, gate_1, app, 5 test files | 5 |
| `src/agents/planning_policy.py` | replace | The stage logic and "immediate/future/outside_scope" categories hard-code 23 lines of rule ids. They become per-section `stage` metadata in `procedure.json`. `fallback_plan` becomes the deterministic workflow fallback (evaluate all sections, build the checklist) | planner, app, 4 test files | 5 |
| `src/memory/reflection.py` | replace | Repair = Gate 1 structured errors fed back to the model, max `MAX_REPAIRS`, then a "needs human review" flag. Validated repairs are stored as episodes (phase 8) | graph, planner, 2 test files | 5, 8 |
| `src/memory/state_update.py` | delete | Approved-plan bookkeeping becomes Twin proposal-status updates | graph | 6 |
| `src/memory/checkpointer.py` (Marco, `SqliteSaver`) | delete (after phase 6) | Only needed for `interrupt()`. Once approval lives in Twin proposals, the Twin DB is the only store. Kept until then so the old app keeps working | app, demo, 6 test files | 6 |

### 3.2 Rules and knowledge

| Module | Decision | Reason | Depended on by | Phase |
|---|---|---|---|---|
| `src/workers/rules_engine.py` | replace (by `src/rules.py`) | 29 hand-written lambdas (32 lines mention rule ids) plus a code-only R015 variant. The new JSON-tree evaluator must reproduce its verdicts first (section 5). It is deleted only after the oracle test is green and the verdict snapshot is committed | graph, 10 test files | 1 |
| `src/workers/rule_loader.py` | replace (by `src/pack.py`) | Hard-codes the set `R001..R029`. Its dependency union (inline + `dependencies.json`) is kept as data in `rules.json` `depends_on` | rules_engine, gate_1, planning_policy | 1 |
| `knowledge/rules.json` | adapt → `procedures/<ikamet>/rules.json` | Data conversion of all 29 rules into condition trees. Ids, versions, severity, sources and descriptions are unchanged (notes in section 4) | rule_loader, test_knowledge | 1 |
| `knowledge/sources.json` | adapt → pack `sources.json` | Rename `authority_level` → `authority`. Keep URLs and dates as they are. `GOV_RESIDENCE_INFO` has a truncated URL (`https://en.goc.gov.tr/residence-`): flag it, do not invent a fix | test_knowledge | 1 |
| `knowledge/dependencies.json` | adapt → merged into `depends_on` | Disagrees with inline `dependencies` (example: edge R001→R002 exists only here). Marco's engine uses the union, so the pack uses the union | rule_loader | 1 |
| `knowledge/scope.json` | adapt → `procedure.json` `scope` + `scope_conditions` | Same content, also machine-checkable | Gate 0 | 1 |
| `knowledge/test_cases.json` | keep (as oracle) → captured in `tests/data/legacy_verdicts.json`; deleted after phase 10 | 15 verdict cases, part of the regression oracle | tests | 1 |
| `knowledge/change_monitoring.json` | keep (docs) → summarised in `docs/modules/pack.md`; deleted after phase 10 | Policy for rule updates (real → twin path). Informs phase 7 and the rule-update flow; not executed | none | 7 |
| `knowledge/rule_catalog.md`, `knowledge/README.md` | keep (docs) → deleted after phase 10 (the pack is the catalogue) | Human-readable catalogue. Regenerate from the pack later so the two cannot drift | none | 1 |

### 3.3 Facts, documents, extraction

| Module | Decision | Reason | Depended on by | Phase |
|---|---|---|---|---|
| `src/workers/verification.py` | adapt (split) | This is the real → twin logic: only source-grounded values are accepted, explicit user confirmation is kept separate, derived values are labelled, name conflicts need resolution, and the document lifecycle is handled. All of that is kept. The İkamet tables (`MAPPINGS`, `IMPORTANT_FIELDS`, `ATTESTATIONS`, `CONTEXT_QUESTIONS`, `MILESTONE_DOCUMENTS` with rule ids) move into `procedure.json` `documents[].extracts`/`fields[].question`. The generic code moves to `src/twin/facts.py`. `verification_updates` is 168 lines and must be split | rules_engine, gate_0, planner, policy, app, 5 test files | 3, 4 |
| `src/workers/document_analyzer.py` | adapt → `extract_document` tool | Same adapter (pending document → typed observation, passport bundle split). Returns a tool observation and writes facts with `provenance=document` | graph, 2 test files | 2 |
| `src/document_processing/extractor.py` | keep | PDF/TXT text with page numbers, local RapidOCR only for scanned pages. Generic, pinned by tests | processor, 5 test files | 2 |
| `src/document_processing/processor.py` | keep + adapt calls | Classification → typed extraction → grounding. Only the Ollama calls change, to go through `src/llm.py`. Keyword tables stay | analyzer, 4 test files | 2 |
| `src/document_processing/evidence.py` | keep | 724 lines of deterministic evidence (MRZ check digits, labelled dates, insurance rows, snippet audit). Pinned by ~71 pure tests. No refactor unless time allows (it breaks the style rules but works) | processor, verification | 2 |
| `src/document_processing/schemas.py` | keep + extend | 7 typed document kinds stay as reusable kinds (a passport is a passport in any procedure). Add one generic `FieldObservation{field, value, page, snippet}` path for new document types declared in a pack, with no dynamic model classes (D4) | state, processor, verification, app, 8 test files | 2 |

### 3.4 Memory, Twin DB, evaluation (Neda)

| Module | Decision | Reason | Depended on by | Phase |
|---|---|---|---|---|
| `src/memory/checkpointer.py` (Neda, `TwinStore`) | adapt → `src/twin/store.py` | Clean, transactional, readable. Kept as the base of the Twin. Schema v2 adds `facts` with `provenance` (document / user_stated / tool / derived), `source_ref`, `confirmed`, `valid_from/valid_to`; `events`; `proposed_actions`; `simulations`; `episodes` (= today's `lessons`). The `CHECK` lists on `documents.status` and `runs.event_kind` must be widened (Marco has 5 statuses and 13 event kinds) | session, harness, 3 test files | 3 |
| `src/memory/schema.sql` | adapt | As above. Keep the date `CHECK (date(x) IS x)` guards | TwinStore | 3 |
| `src/memory/codec.py` | keep | Strict date/timestamp/fact encoding, 100% reusable | TwinStore, harness, 2 test files | 3 |
| `src/memory/session.py` | replace | `run_event()` (load twin → stub graph → save). The agent entry point takes over this job | harness, test_restart | 5 |
| `src/evaluation/harness.py` + `scripts/run_eval.py` | adapt → `eval/run_eval.py` | Keep the scenario runner with simulated restarts and the report format. Swap the stub graph for the agent. Add baselines A/B/D and the `run-evals` metrics | 2 test files | 9 |
| `data/synthetic_samples/scenarios.json` (S01–S05) | replace | Its expectations encode stub behaviour (an "appointment" date conflict that the real rules do not have). Ideas kept: S04 "unreadable document remembered" and S05 "Gate 0 rejection logged" are rewritten as pack scenarios | test_scenarios | 9 |

### 3.5 App, scripts, data, docs

| Module | Decision | Reason | Depended on by | Phase |
|---|---|---|---|---|
| `streamlit_app.py` | adapt → `app.py` | Reuse the upload saving (content-hashed names), field review/confirm forms, identity-conflict form, lifecycle buttons and error messages. Add the goal box, agent status, and proposed actions with approve/deny. `render_state` (141 lines) is split | 3 test files | 9 |
| `scripts/run_demo.py` | adapt | Offline demo with mocked transport. Re-pointed to the agent | none | 9 |
| `pytest.ini`, `requirements*.txt` | adapt | Merge the two. Add `numpy` (phase 7). pytest stays in `requirements-dev.txt`. Pin LangGraph `>=1.2.20` (venv has 1.2.22; Neda pinned 1.2.18) | — | 0 |
| `.gitignore` | adapt | Union of both. **Must ignore `data/uploads/` and `data/digital_twin.db` before any `git add`** (see risk 3). Also ignore `procedures/*/outbox/` (artifacts) | — | 0 |
| `data/synthetic_samples/*.txt` | keep | Synthetic fixtures; after phase 9 they are demo uploads for the app | tests | — |
| 5 fixtures only on `integration-abc` (`enrollment_ambiguous/invalid_date`, `insurance_multiline_layout/numeric_dates/turkish`) | keep (salvage) | Extra extraction coverage. Bring them over with their tests if they pass on the kept processor | — | 2 |
| `data/synthetic_samples/prova.txt` | delete | Empty file | none | 0 |
| `data/synthetic_samples/example.pdf` | keep (review) → deleted after phase 10 | A public university orientation handbook (58 pages), not personal data and not synthetic. No test references it. Could be removed to slim the repo | none | — |
| `integration-abc` root modules (`document_evidence.py`, `processor.py`, `extractor.py`, `schemas.py`) and its `src/workers/*` | delete (do not merge) | Superseded: Marco's `src/document_processing/` already adopts them | none | — |
| `docs/INTEGRATION_REPORT.md`, `docs/memory_and_evaluation.md` | keep (history) → deleted after phase 10 (described deleted code) | Accurate history of milestone 1 | none | — |
| `docs/architecture_diagram.png`, `docs/*pres1.pdf`, `other/` (git-ignored) | keep | Course material | none | — |
| `README.md` | adapt | Rewrite after phase 9 | — | 9 |

## 4. Converting the 29 rules: issues found

1. **R015 has two versions.** `rules.json` says `insurance_start <= requested_start AND insurance_end >= requested_end`.
   Marco's engine swaps in a code-only "Milestone-1 scoped" variant whenever `planning_scope` is set (always, in the
   app): `insurance_end >= requested_permit_end OR program_end`, with the start date ignored. The variant has no
   source and is not in `rules.json`. With the three-valued `any`, it also gives UNKNOWN where the old code gives
   FAIL (one target present, the other missing). It cannot be reproduced exactly with the kickoff operators (question 3).
2. **R009** in `rules.json` is `photo_age_months <= 6`. The engine checks `0 <= v <= 6`. Proposal: encode
   `all[gte 0, lte 6, ...]`, which reproduces the engine and adds no new meaning.
3. **R006** = `gte passport_expiry_date other_field requested_permit_end_date offset_days 60`. Boundary tests exist.
4. **R024** says "within current window". The engine uses the fields `iro_window_start`/`iro_window_end`, so the
   rule becomes `all[gte current_date other_field iro_window_start, lte current_date other_field iro_window_end]`.
5. **R027** `approval_status != 'guaranteed'` = `not(eq ...)`. **R025** = `eq` with `other_field`.
6. **Dependencies:** `depends_on` = union of inline `dependencies` and `dependencies.json`, which is what the old
   engine uses. Verdicts stay independent of dependencies (as today). Dependencies only drive "blocked by" and the
   stage prerequisites.
7. **Derived fields.** `planned_stay_months` (whole calendar months between program dates),
   `exchange_program_evidence_present`, `student_certificate_present` and `application_form_generated` are computed
   in Python today. The pack declares them under `derived` with a small fixed set of deterministic operations
   (`months_between`, `document_complete`). Dates are computed only by code, never by the LLM (D5).
8. **Sections (new structure, needs your OK in phase 1).** Proposed from `category` and the 7-document checklist,
   kept to about 6 because `MAX_STEPS = 8` limits how many sections can be delegated:
   eligibility (R001–R004), passport_photo (R006, R008, R009), application_fees (R005, R007, R010–R012),
   student_insurance (R013–R016), address_declaration (R017–R019), iro_submission (R020–R026). Guardrails
   (R027–R029) are always evaluated deterministically and never delegated.
9. **Fields.** The 70 `ResidencePermitContext` fields become `procedure.json` `fields` (bool/int/str/date). The 11
   user questions and the scope aliases (for example "sabanci university") become data too.
10. **`sources_text/` does not exist yet.** No official page has been saved as text. RAG (phase 7) is blocked until
    someone saves them. Nothing is fetched or invented.

## 5. Tests and the regression oracle

Counts are collected test items in Marco's branch, plus Neda's.

| Test file | Items | Role | Why |
|---|---|---|---|
| `test_requirement_engine.py` | 83 | **Oracle, phase 1** | 15 `test_cases.json` cases, 29 all-unknown cases, PASS+FAIL branch for every rule, 60-day and deadline boundaries, conditional alternatives, missing-field names |
| `test_evaluation.py` | 10 | **Oracle, phase 1** (verdict part); port, phase 5 (graph part) | 10 end-to-end scenarios, each pinning the **full 29-rule status vector** and the blockers |
| `knowledge/test_cases.json` | 15 | **Oracle, phase 1** | Original rule-team cases |
| `test_processor.py` (73 functions, 71 pure) | 118 | **Oracle, phase 2**: must stay green unchanged | Extraction and grounding behaviour. 2 graph-coupled tests are ported |
| `test_extractor.py` | 19 | **Oracle, phase 2** | PDF/TXT/OCR. The autouse fixture in `conftest.py` already fails on any real Ollama/OCR call; it is extended to embeddings |
| `test_store.py`, `test_codec.py` (Neda) | 22 + 36 | **Oracle, phase 3** | Twin storage, atomicity, date guards, lesson rules |
| `test_knowledge.py` (Neda) | 34 | adapt → pack validation tests, phase 1 | Sources exist, ids unique, no dependency cycle |
| `test_persistence.py` | 10 | port, phase 3 | Behaviour to keep: state survives a new process, threads/users isolated, upserts survive reopen |
| `test_verification.py`, `test_document_lifecycle.py`, `test_milestone_documents.py`, `test_document_types.py` | 109 | port, phases 3–4 (behaviour specs) | Provenance guarantees: unconfirmed stays UNKNOWN, confirmation never overwrites the observation, derived values labelled, conflicts need explicit resolution, deletion needs confirmation, entry date only after confirmation. The assertions move to the Twin/facts API; the guarantees do not change |
| `test_plan_validator.py` | 25 | port, phase 5 | Gate 1 checks (completion claims, unknown-as-failure, approval waiver, deadlines, dependency order) re-expressed on `FinalAnswer` |
| `test_reflection.py` | 9 | port, phase 5 | Bounded repairs, stop after 3, denial blocks write-back, no self-approval |
| `test_action_planner.py`, `test_hybrid_planning.py` | 77 | retire with the planner, phase 5 | They test planner internals (locked actions, candidate grouping, slot ids). The safety ideas (no invented rule ids, closed-enum selection, unchanged invalid output stops repair) are rewritten as agent tests with the scripted fake LLM |
| `test_rule_loader.py`, `test_required_documents.py` | 5 | retire / port, phase 1 | Replaced by pack tests; the 2 readiness cases become rule tests |
| `test_streamlit_helpers.py` | 24 | port, phase 9 | Pure display helpers |
| `test_scenarios.py`, `test_restart.py` (Neda) | 9 + 3 | retire, phase 3/9 | Stub-graph behaviour; 9 of these 12 are the failures after merging. Replaced by pack scenarios and a real restart test |

**How phase 1 proves the new evaluator matches.** One test runs old `evaluate_requirements` and new `src/rules.py`
on the same contexts and asserts equal status and equal `missing_fields` for all 29 rules. The contexts are:
the 15 knowledge cases, the 58 branch contexts, the empty context, the boundaries, the 10 scenario contexts, and
~500 deterministic perturbations of the "ready" context (seeded: drop/flip fields). Then the old engine's output
is saved as a golden snapshot (`tests/data/legacy_verdicts.json`, generated, never hand-edited), the old engine is
deleted, and the test checks the snapshot from then on. Known divergence: only the scoped R015 variant (section 4.1).

No expected value is changed anywhere. A test is retired only with a row above.

## 6. Design decisions proposed (deviations or details not fixed by the kickoff)

- **D1 Base branch.** Marco's branch plus a merge of `origin/main`. `TwinStore` moves out of the clashing filename.
- **D2 Approval lives in the Twin.** Artifact tools write `proposed_actions`. Approve/deny updates the row, and only
  then is the artifact written to an outbox folder. No LangGraph `interrupt()`, so the `SqliteSaver` can go and
  the Twin DB is the single source of truth (which is Neda's design reason). LangGraph keeps the in-run cycle.
- **D3 Fact provenance.** Marco's `source_grounded / user_confirmed / derived` maps to `document / user_stated
  (confirmed=True) / derived`. `record_fact` from chat writes `user_stated, confirmed=False` (question 4).
- **D4 Extraction.** Keep the 7 typed extractors as reusable document kinds. New pack document types use a generic
  path: the LLM proposes `{field, value, page, snippet}`, and code keeps only values whose snippet is really in the
  page text (reusing `audit_snippets`/`snippet_in_source`) and that parse as the declared type.
- **D5 Derived fields.** Declared in the pack and computed by a fixed set of Python operations.
- **D6 Scope gate.** `scope_conditions` in the pack, evaluated by the same three-valued evaluator: FALSE = out of
  scope, UNKNOWN = ask.
- **D7 Stage prerequisites.** `procedure.json` `stages: [{id, requires: [rule ids]}]` replaces the rule ids
  hard-coded in Gate 1 and in the planning policy. The sandbox promotion gate uses the same data.
- **D8 Document lifecycle stays deterministic.** Confirm, delete, replace, reprocess and resolve-conflict are direct
  Twin operations from the UI (real → twin). They are not LLM tools: no model in the loop where the user is
  already explicit.
- **D9 Style.** 16 of 148 functions are over 40 lines and many lines are packed one-liners. Every module marked
  *adapt* is rewritten in small functions with its tests pinned first. `evidence.py` is left alone (see 3.3).
- **D10 LLM call budget.** The orchestrator delegates only sections that are not all-PASS and belong to the
  current stage. The worker's first observation already contains its section's `evaluate_rules` result, so a
  typical worker finishes in 1–2 calls.

## 7. Main risks

1. **Hardware and latency.** This laptop: GTX 1650 **4 GB VRAM**, 16 GB RAM, i5-9300H. Pulled models: `qwen3:4b`,
   `qwen2.5:3b`. `qwen3:8b` (~5 GB) does not fit in VRAM, so it would run partly on CPU. A goal can take 10–40
   model calls, which could mean several minutes per answer. `bge-m3` is not pulled. All tests mock the LLM, so
   development is not blocked; live evaluation and the demo are.
2. **Small-model reliability.** 4B–8B models with schema-constrained `Decision` output fail tool arguments often,
   and the 2-failure fallback may trigger frequently. Mitigations: tool names as an enum in the schema, short
   prompts, `think=False`, temperature 0. The fallback rate is measured in phase 9.
3. **Private data in the working tree.** On the current branch `data/uploads/` (7 PDFs) and `data/digital_twin.db`
   (LangGraph checkpoints, 2 threads) are untracked and **not ignored**. A careless `git add .` would commit them.
   I did not open the PDFs. Phase 0 fixes `.gitignore` before anything else. They were never committed (checked
   with `git log --all`).
4. **Team divergence.** Four lines of work, local `main` behind `origin/main`, and the kit copied onto the least
   integrated branch. If teammates keep pushing to their branches, v2 will need repeated merges.
5. **R015 behaviour change** for the app if the `rules.json` version becomes canonical (section 4.1).
6. **Size.** About 8k lines and 489 tests are being restructured. Behaviour encoded only in tests could be lost.
   Mitigation: port behaviour tests before deleting a module (section 5).
7. **RAG and the second procedure need real sources.** `sources_text/` is empty, one URL is truncated, and
   phase 10 needs official texts for a new procedure. Nothing will be invented; unverifiable rules are marked
   `unverified`.
8. **Gate 1 text checks are English regex heuristics.** Structured `claims` reduce reliance on them, but free-text
   `summary` checks remain best-effort (Marco's report already says this).
9. **Python version.** `CLAUDE.md` says 3.11; the venv is 3.13.2. Code stays 3.11-compatible and is tested on 3.13.

## 8. Phase 0 preview (after approval)

1. Add the `.gitignore` entries for `data/uploads/` and `data/digital_twin.db` first.
2. `git switch -c feature/react-agent-v2 origin/feature/marco-milestone1`, then `git merge origin/main`. Resolve the
   6 conflicts as in section 1. Move `TwinStore` to `src/memory/twin_store.py`, to be moved to `src/twin/` in phase 3.
3. Bring over `CLAUDE.md`, `.claude/skills/`, `README_KIT.md` and this file.
4. Add the kickoff settings to `config.py`. Delete `prova.txt`. Create empty `procedures/`, `eval/`, `src/tools/`,
   `src/agent/`.
5. Mark the 9 stub-coupled tests as retired (`pytest.ini` deselect with a comment pointing here) so the suite is green.
6. Put pytest in the venv from `requirements-dev.txt`. Run `python -m pytest -q`. Commit "Phase 0".

## 9. Milestone integration (branch `integration/milestone`, 2026-11-24)

Order: the branch starts from `main` (PR #2), then merges this branch (`feature/dani-version-two`, `b6885ff`)
unchanged, then adds our changes as separate commits. Daniele's branch was chosen over
`feature/marco-milestone1` because it already has the Procedure Packs (including `tc_kimlik_first` and
`tc_kimlik_renewal`), embedding RAG and the Orchestrator-Workers agent. Nothing reaches `main` until the team
approves, and nothing is ever pushed to `feature/dani-version-two`.

### 9.1 Merge of `feature/dani-version-two` into `main`
The merge commit's tree is identical to `b6885ff`: every file PR #2 touched was already replaced or ported on
Daniele's branch. Each conflict:

| File from `main` | Decision | Why / where its guarantee lives now |
|---|---|---|
| `src/memory/{checkpointer,schema.sql,twin_store,twin_sync}.py` | delete | replaced by `src/twin/` (phase 3); store guarantees in `tests/test_twin_store.py` |
| `src/evaluation/{harness,scenarios}.py`, `scripts/run_eval.py`, `tests/test_evaluation.py` | delete | replaced by `eval/` (phase 9) and `procedures/*/eval/` |
| `streamlit_app.py`, `scripts/run_demo.py` | delete | replaced by `app.py` (phase 9) |
| `tests/test_ui_guard.py` | delete | same three tests ported to `tests/test_app.py` (`profile_lock`) |
| `tests/test_store.py`, `tests/test_knowledge.py` | delete | `tests/test_twin_store.py`, `tests/test_pack.py` |
| `docs/memory_and_evaluation.md` | delete | describes the v1 design; v2 memory is in `docs/modules/` |
| `.gitignore`, `config.py`, `tests/conftest.py`, `data/reference_samples/SOURCES.md` | Daniele's version | the `main`-only lines belong to the v1 pipeline (`TWIN_RECORD_PATH`, `DATABASE_PATH`, planner settings, v1 fixtures) |

### 9.2 Marco's later work (`feature/marco-milestone1`, `b0ee6c1`)
Not merged with git: that would bring back the milestone-1 pipeline deleted in phase 9. Features this branch
lacks are ported by hand, one commit each, crediting Marco (see the progress below).

Progress (9.2):
- **MCP server** ported to `src/integrations/mcp/server.py`. Same pinned protocol (2025-11-25) and stdio
  JSON-RPC as Marco's, now over all packs: tools `find_procedure` (the deterministic ranking of
  `search_procedures`, without its Twin writes) and `search_official_sources` (embedding RAG); one resource
  `procedure://<id>` per pack. Read-only: the only write is the shared embedding cache. Tests:
  `tests/test_mcp_server.py` (14, including a real stdio subprocess).
- **Open-in-app links** ported to `src/tools/open_links.py` (Marco's Gmail / mailto / Google Calendar link
  builders). Approved email drafts and calendar reminders get buttons next to Download in the app; each only
  opens the user's own editor pre-filled, nothing is sent or booked. `create_ics` now also stores `title` and
  `description` in its payload (additive). Tests: `tests/test_open_links.py` (5).
- **Not ported, with reasons:**
  - *Document benchmark* (`scripts/benchmark_documents.py`): `eval/document_eval.py` already scores 13
    labelled files (including Kimlik documents and `data/reference_samples/`) with a "wrong values" count and
    seconds per file; Marco's script reads private files from `data/local_samples/`, which a shared
    evaluation should not depend on.
  - *×3 repeat runs*: already in `eval/run_eval.py` (`pass_k`, k = 3).
  - *5-step İkamet stage view*: stages here come from each pack's `procedure.json`, so the app follows any
    procedure (Kimlik included); a hard-coded İkamet flow would contradict the pack design.
  - *e-İkamet field-by-field sandbox*: `src/sandbox/mock_world.py` + `fill_pdf` + `simulate_submission`
    cover the same idea generically.

## 10. Final release (branch `release/final`, 2026-11-28)

Team decision (2026-11-28): one final branch for the presentation, started from `integration/milestone` (Neda),
then `feature/dani-version-two` merged in (`332d5ea`, `0ddf6c8`: cloud models, reflection in the chat, the
simplified app, address documents). From `feature/marco-integration-review` (Marco, Andrea) only ideas that fit
this engine are ported by hand, crediting them; its architecture (deterministic router, candidate planner, one
hard-coded İkamet procedure) and its deterministic pieces are not taken, the user asked for the most agentic
solution. Its Streamlit UI is not taken either: `src/ui/` imports that engine (`src.core.procedure`,
`src.core.graph`, `AgentState`, `legal_stay`, `address_proof`) in dozens of places, so it would be a rewrite.

### 10.1 Merge of `feature/dani-version-two`
| File | Decision | Why |
|---|---|---|
| `app.py` | Daniele's layout (Journey, Now box, Documents, Details) | The only textual conflict: Neda's side added link buttons to the old `done_list`, which the new layout moved. |
| `src/agent/links.py`, `tests/test_links.py` | **delete**, use `src/tools/open_links.py` (Neda's port of Marco's links) | Two modules built the same Gmail / Google Calendar links. Neda's also offers the default e-mail app (mailto), gives no e-mail link when the sources have no address, and uses the reminder title that `create_ics` now stores. Its tests (`tests/test_open_links.py`) cover the same cases; `tests/test_app.py` still checks the Gmail link after approval. |
| `tests/test_rag_compare.py` | 66 → 67 rule queries | One query per rule; `ikamet_first` gained R030 (section 2026-11-24 above), so 30 + 8 + 15 + 14 = 67. No retrieval result changed. |

### 10.2 Choosing Ollama or OpenAI in the app; usage, cap and cost
| Module | Decision | Why |
|---|---|---|
| `src/llm.py` | **adapt**: a `provider` argument on `agent_backend`, `is_cloud`, `document_backend`, the labels; `missing_key`; usage log `USAGE` filled by `timed_call`; `CLOUD_MAX_REQUESTS` cap; `estimated_cost` / `usage_summary` | The user asked to choose between Ollama and OpenAI. The cap, the token count and the cost estimate are Andrea's ideas (`src/core/model_provider.py`, `src/evaluation/metrics.py`), written for this backend contract. |
| `src/agent/chat.py` | **adapt**: self-review and the full tool set follow the backend in use (`is_cloud_backend`) | With a run-time choice, `LLM_PROVIDER` no longer says which model answers. |
| `app.py` | **adapt**: sidebar "AI model" (Local / OpenAI, Claude when its key is set), switch "the cloud model also reads my documents", model usage in Details → How it works | The privacy switch replaces the global `DOCUMENT_PROVIDER` for the session (Andrea asked consent per file; one switch per session keeps the demo simple). |
| `config.py` | `OPENAI_MODEL` default `gpt-5.5` → `gpt-6.1-terra`; `PROVIDER_LABELS`, `PROVIDER_KEYS`, `CLOUD_MAX_REQUESTS`, `CLOUD_PRICES` | The team's model (also in Andrea's branch). |

New tests: `tests/test_usage.py`; `tests/test_app.py` (OpenAI without a key warns and keeps the local model; with a
key the cloud switch appears). No expected value changed.

### 10.3 Gate 0
| Module | Decision | Why |
|---|---|---|
| `src/agent/gate0.py` | **new**: `check_message`, `check_upload`, `check_profile` → `list[Issue]` | The kickoff design (slides 4-5) has Gate 0 (basic input checks) before the agent and Gate 1 after it. The input checks existed but were scattered (`save_upload` in `app.py`); now they are one module with codes like Gate 1's, plus new ones: message length, file size, bytes that match the extension, PDF page count. Not ported from Marco's `gate_0.py`: that one validates his `AgentState` events. |
| `src/agent/chat.py` | **adapt**: `run_chat` calls Gate 0 first; a refused message returns `status="invalid_input"` without a model call or a Twin write | No wasted model call on an empty or oversized message. |
| `app.py` | **adapt**: `save_upload` and the profile name use Gate 0; `UPLOAD_TYPES` comes from `config.py`; "How it works" explains Gate 0 | One place for the limits. |

New tests: `tests/test_gate0.py`. No expected value changed (`tests/test_app.py::test_save_upload_rejects_bad_files`
still expects a `ValueError` for `a.exe` and an empty `a.txt`).

### 10.4 Metrics
| Module | Decision | Why |
|---|---|---|
| `eval/metrics.py` | **adapt**: pass@k next to pass^k, p95 seconds, tokens and cost per run, human-review rate, issues caught by Gate 1, reflection recovery rate | Andrea's framework (`src/evaluation/metrics.py`) has many metrics; these are the ones that measure *this* engine and make sense for the course (reliability, cost, reflection). Not taken: action precision/recall/F1 against labelled candidate ids, tied to his candidate planner. |
| `eval/run_eval.py` | **adapt**: `--provider ollama|openai|anthropic` for live runs; per-run usage from `USAGE` | The team wants to compare the local model with GPT on the same scenarios. |
| `eval/*.py` | **adapt**: reports printed as UTF-8 (`eval.utf8_console`) | `python -m eval.routing_eval` crashed on the Windows console (cp1252 cannot print "ğ"). |

Report column renamed `pass_k` → `pass^k` (same number); `docs/eval_results.md` regenerated offline. New tests in
`tests/test_eval.py`. No expected value changed.
