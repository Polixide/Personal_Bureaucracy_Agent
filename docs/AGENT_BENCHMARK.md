# Router and Orchestrator evaluation

Use [benchmark.v1.json](../evaluation/benchmark.v1.json) for the shared cases. This adapter targets the Milestone-1 preparation path on `feature/marco-milestone1`. It runs the existing graph with a fresh in-memory checkpoint and a disposable, per-case SQLite file for Reflection lessons. It stops at the first human interrupt. Accepted document fixtures are typed synthetic observations and confirmations. The PDF/TXT case runs Gate 0, the local Ollama document backend, evidence projection, and the Rules Engine. The runner disables the document cache and cloud fallback for those cases.

## Differences from the earlier prototype export

| Compared area | Target branch and integration |
| --- | --- |
| Planner and canonical candidates | The Git branch uses opaque `candidate_NNN` selectors with host-side canonical references, exact tool targets, distinct candidate identities, and stage completion. The integration uses these contracts and the branch's existing `PLANNER_MODEL` override. |
| Gate 1 and Reflection | The Git branch has typed tool checks, stage-redundancy checks, and SQLite-backed repair lessons. The new duplicate check runs alongside those checks. Repair feedback uses the offered opaque IDs and preserves the lesson context. |
| Insurance and rule assessment | The active rules snapshot is version 2.0.0 and includes its own insurance, category, alias, and procedure-readiness policies. The integration adds explanatory text to the existing recommendation and plan display paths. |
| Graph, state, Streamlit | The Git branch has a staged procedure, Copilot, tool approvals, and UI workspaces. The benchmark uses the published Milestone-1 event shape. The scope rejection is driven by the branch's existing Gate 0 context. It adds no new route, UI element, or state schema. |
| Document analysis | The Git branch has provider selection, image handling, cached observations, and extraction diagnostics. Metrics observe Ollama requests through the current provider and processor. The benchmark selects local-only extraction. |
| Retrieval and evaluation | The Git branch has a lexical retriever, an existing RAG evaluator, and existing trajectory metrics. The new benchmark shares the metrics module and preserves those functions. Public exports distinguish feature availability from whether a run exercised retrieval. |
| Profile and test fixtures | The export has a separate profile resolver. The branch uses its own `ResidencePermitContext` and native fixtures. The adapter projects the common synthetic profile to that context. Scope-layer metrics are unavailable in this branch's requirement schema. |

The earlier export's generated results use its own branch and environment. Keep those measurements associated with that version.

## Planning behavior

Python projects the applicable requirements and accepted evidence into candidates. The LLM selects candidate IDs, orders them, and supplies titles and reasons. Python binds selected IDs to document targets, fields, rule IDs, action kinds, and approval requirements before Gate 1. The model receives opaque selectors such as `candidate_001`. Host metadata keeps canonical references that can identify distinct tasks attached to one rule.

The exact recorded five-action draft contained one passport upload, one entry-stamp upload, one declaration upload, and two secondary reviews of the same evidence. The JSON shape and offered IDs were valid. The earlier grounding path rejected the cross-row repeats before producing an ActionPlan. An identical retry matched the prior raw hash and stopped with UNCHANGED_INVALID_PLAN. This reproduction used mocked responses. It does not identify the cause of a different Streamlit session without that session's rejection codes.

The pre-grounding pass consolidates either of the two known upload/review pairs when exactly two singleton rows refer to the same offered selector, the canonical target matches, and the review specifies the same purpose. The upload survives in its original relative slot and the review purpose goes into its reason. The independent declaration request retains its wording. The target branch continues rejecting raw rule selectors, repeats within a selector list, identical duplicate rows, grouped overlaps, and repeated tool proposals. Gate 1 checks canonical grounding IDs for repeats in the final plan.

Repair receives the prior structured proposal, exact issue, affected selectors, and the offered candidates. An exact repeat stops. A duplicate layout that still fails after rephrasing or reordering also stops. Repair of a rejected duplicate draft must retain the known selectors it already addressed. Changed invalid responses remain within the existing three-attempt repair budget. Transport failures use the existing explicitly labeled deterministic fallback and still go through Gate 1.

The scoped insurance discrepancy is a project preparation advisory. It appears in the existing recommendations text and in the reason of a proposed R015 correction when coverage ends before the accepted dates used in the check. R015 remains failed and in the blocker list. Users can continue preparing other documents. The branch's existing prerequisites determine progression to later stages.

## Cases and modes

| Case | Decisive label | Modes |
| --- | --- | --- |
| A | Empty package. First three in-budget evidence tasks | Offline, live |
| B | Six accepted documents. Only the declaration request remains | Offline, live |
| C | Identity and passport-expiry conflicts remain unconfirmed. R006 stays UNKNOWN | Offline, live |
| D | Educational coverage advisory and preparation proposal. R015 stays FAIL | Offline, live |
| E | Preparation complete with zero model calls. Later workflow readiness remains UNKNOWN | Offline, live |
| F | Exact five-row duplicate draft yields three actions in one call | Offline |
| F-repair | Ambiguous duplicate draft is repaired by a changed model proposal | Offline |
| F-stop | Same invalid candidate selection reaches human review | Offline |
| G | Degree student receives unsupported-profile status with no rules applied | Offline, live |
| H | Mocked transport timeout uses a Gate 1-validated deterministic fallback | Offline |
| I | Synthetic insurance PDF or TXT. Five labeled fields with page provenance. R015 remains UNKNOWN | Document-offline, document-live |

The case file contains the complete profile, shared defaults, accepted document references, per-case overrides, labeled rule outcomes, source IDs, scope layers, accepted action sets, safety constraints, and budgets. Upper bounds apply independently to final validated actions and total chat invocations, including replans. A Gate 1 PASS and a benchmark pass have separate meanings.

## Row and aggregate definitions

JSON contains benchmark_version, metadata, runs, summary, by_scenario, and by_model_task. Each runs entry is one case execution. The CSV selects scalar fields from those rows. JSON null denotes an unavailable, unmeasured, or inapplicable value. CSV represents null with an empty cell and booleans as True or False. Zero denotes a measured or known empty count. Operational tallies include intentionally rejected scenarios. Treat the scripted results as functional controls.

### Model and proposal

| Metric | Definition and source |
| --- | --- |
| model, task, retry, request_success, timed_out, fallback_used | One Ollama chat invocation through logged_chat. Retry is zero for a first request and the graph repair number for a replan. Timeout identifies Python or HTTPX timeout exceptions. Fallback is attributed to the failed request that triggered the deterministic Planner. Scripted calls carry the model label scripted. |
| prompt_tokens, completion_tokens, total_tokens | Ollama prompt_eval_count, eval_count, and their sum when both are present. A total across invocations requires every constituent value. LLM calls with missing usage leave sums null. A case with zero calls has zero token cost. Summary mean_total_tokens requires all case totals. mean_known_case_total_tokens reports the available-only mean with token_total_samples. |
| wall_seconds, llm_latency_seconds | Monotonic elapsed seconds around chat, including failure and timeout waits. Case LLM latency is the sum of request wall times. The end-to-end timer is independent. |
| server_total_seconds, load_seconds, prompt_eval_seconds, generation_seconds | Native Ollama duration fields divided by 1,000,000,000. Missing metadata is null. |
| generation_tokens_per_second, ttft_seconds | Completion tokens divided by positive generation seconds. TTFT is null because requests are non-streaming. |
| Schema validity rate | Raw outputs satisfying the typed request shape divided by checked responses received. Report total received and checked responses separately. Failed requests are excluded. Planner selectors must belong to the advertised enum and be unique within each row. Document outputs are validated against the typed schema before evidence reconciliation, using only advertised fields. The Planner's per-case schema_valid is true when all checked responses pass, false when any fail, and null when none were checked. |
| Candidate contract validity rate | Raw planning/replanning outputs satisfying shape, selector membership, distinct references, compatible action kinds, and supported literal rule mentions in action text, divided by received Planner/Replan outputs checked. The duplicate contract can fail even when the JSON schema passes. |
| Duplicate and unknown references | Raw selector occurrences beyond the first for each ID and raw selector occurrences outside the offered set, respectively. Counts span all inspectable rows in planning and replanning. The final plan's duplicate count is measured separately. |
| Hallucinated action rate | Inspectable candidate rows with an unknown selector, incompatible candidate kinds, or an unsupplied literal R### mention divided by all entries in inspectable raw actions arrays. Count each affected row once. This operational metric covers observable unsupported selections. Semantic judgments about arbitrary prose are unavailable. |
| Rule-reference accuracy | Supplied rule references divided by all inspectable proposed bindings and literal rule mentions. Known selectors contribute their canonical rule bindings, unknown selectors count as one unsupported binding, and literal R### mentions are deduplicated within each action. |
| Gate 1 plan validation rate | Accepted plans divided by materialized plans submitted to Gate 1. Gate attempt acceptance also includes visits with parse or grounding errors and no new ActionPlan. Gates and accepted/rejected counts come from named graph updates. Case plan_valid is the last Gate 1 decision, with null for cases that stop earlier. |

### Workflow, Reflection, and grounding

| Metric | Definition and source |
| --- | --- |
| Task success rate | Cases satisfying every labeled assertion divided by all evaluated cases. Assertions cover stopping condition, in-scope rule results, safety, budgets, fallback, action set and action type, and additional case labels. |
| Action precision, recall, F1 | Intersection of exact final validated candidate identities with an allowed reference set divided by predicted and reference sizes. Select the alternative with highest F1, breaking ties by fewer missing and then fewer extra IDs. Both empty sets score 1. With missing predictions and nonempty references, precision and recall are 0. With extras against an empty reference, precision/F1 are 0 and recall is 1. Unlabeled cases are null. The fallback adapter binds a rule only to a unique matching canonical candidate. |
| Missing required actions, coverage, average actions, remaining blockers | Reference minus validated candidate keys, reference recall, mean number of final validated action rows per accepted plan, and count of Rules Engine blocking results still unresolved. Grouped actions can cover several keys. Remaining blockers count the rule inventory, including later phases, even for successful preparation cases. |
| Fallback rate and validated_plan_origins | Preparation cases that ever entered deterministic fallback divided by all requested preparation cases, including those stopped by profile checks and completion. Validation counts by origin track Ollama, deterministic fallback, and deterministic completion separately. |
| Reflection recovery rate and mean attempts | Cases entering Reflection with an Ollama-origin validation divided by cases entering Reflection. Mean attempts uses that same denominator. Recovered by fallback has a separate count. In offline mode Ollama-origin refers to the scripted response path. |
| Initial rejections, invalid replans, retry exhaustion | First Gate 1 rejection per case. Subsequent explicit UNCHANGED_INVALID_PLAN outcomes are unchanged. Other rejected replans are changed only when both available hashes differ, and are unclassified when the comparison is unavailable. Exhaustion requires human review at the retry limit without an unchanged terminal code. |
| Reflection overhead | Replan request count, summed native tokens, and summed request wall time. Same missing-data rules as other model usage. |
| Human review, human approval, failures | Human review is the terminal HUMAN_REVIEW_REQUIRED state. Approval reached requires a real pending human_approval interrupt. One primary terminal validation code per case feeds the reason histogram. The repair histogram uses cases that entered Reflection. Expected unsupported profiles and safe human-review stops can count as successful benchmark outcomes. |
| Dependencies, premature submission/completion, unsupported legal claims, UNKNOWN wording | Count rejected gate visits with MISSING_PREREQUISITE, PREMATURE_SUBMISSION, COMPLETION_CLAIM, MISSTATED_RULE_SCOPE or UNSUPPORTED_APPROVAL_CLAIM, and UNKNOWN_AS_FAILURE respectively. Keep the code list per rejected visit. Coverage is limited to the existing typed rules and wording checks. |
| Unsafe actions and safety boundary | Distinct action rows per rejected gate visit with a detected premature action, completion or approval claim, scope misstatement, or missing approval requirement. Summary-only issues use the reason counters. A separate assertion checks that an invalid proposal never reaches approval, that the run makes no approval decision or state update, and that actions at the pause are preparation/information requiring approval. |
| Source attribution and scope accuracy | Labeled Rules Engine rows with exact source-ID set matches divided by applicable labeled rows. Scope-layer accuracy is null in this branch because requirement rows lack that field. A model claim-level grounded or unsupported claim rate is null because the system provides no labeled claims or citation stage. |
| Contradiction and UNKNOWN preservation | Conflicting labeled context fields incorrectly accepted divided by labeled conflicting fields. Labeled UNKNOWN rules still reported UNKNOWN divided by labeled UNKNOWN rules. Each metric is null without its labels. |
| Extraction and document provenance | Exact type match against the synthetic fixture. Field precision is exact typed field/value matches over all populated observation fields. Recall uses all labeled fields. Document source accuracy counts exact labeled values with correct page and accepted source grounding divided by labeled fields. It measures attribution to fixture text. |
| End-to-end latency and execution counts | Monotonic seconds from prepared synthetic graph input to terminal snapshot. File fixture generation, export, and scoring are excluded. Router graph steps count named update events, including routing and gates. Worker calls count document analysis, rule assessment, planning, and Reflection node runs. Tool calls refer to explicit agent tool invocations. Tool counts use the branch's existing trajectory metrics for non-replayed observations. Milestone preparation and document-ingestion cases have no tool dispatch. |
| Summary statistics | Arithmetic means where applicable, median/p50, and nearest-rank p95 at 20+ measured samples. Event rates pool numerators and denominators. Per-case action and labeled accuracy metrics are macro-averaged over available cases. Always interpret latency and sample counts alongside execution mode and hardware. |

### Retrieval

Copilot can call `search_official_sources` over local allow-listed rule excerpts and guidance snapshots. The default returns up to three results. Preparation and document cases have zero source searches, so retrieval quality and latency are null. The `rag-offline` and `rag-live` modes exercise Copilot with labeled retrieval references. The shared collector records both local and cloud requests.

An adapter with retrieval and labeled relevant chunks can supply precision@k (unique relevant hits / k), recall@k (unique relevant hits / labeled relevant count), hit rate@k (fraction of queries with a hit), and MRR@k (mean reciprocal rank of first relevant hit, zero for misses). Retrieval latency, chunk count, and labeled source coverage are means over observations where those values exist. Use separate judgments for factual and citation accuracy.

## Reproduce on Windows

Run from the Git checkout root with its existing virtual environment. Each export label is single-use, and exports go to `evaluation/results`. Set the tracing variables before running tests or measurement. The first test invocation uses explicit offline unit fixtures outside the repository-wide pytest fixtures that import the graph. It uses the installed native dependencies and Windows policy. Full graph execution requires a working LangGraph import.

The target virtual environment was checked with Python 3.14.7, LangGraph 1.2.21, and xxHash 4.1.0. Windows Application Control blocked the first native `_xxhash` and LangGraph imports. Later checks loaded the installed binary and ran 432 focused tests and all 12 synthetic graph runs successfully. Windows settings and packages stayed in place. Graph integration cases report a skip when the native dependency is unavailable.

~~~powershell
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$python = '.\.venv\Scripts\python.exe'

& $python -B -m pytest --noconftest -q -rs -p no:cacheprovider --basetemp .test-temp/benchmark-native tests/test_agent_benchmark.py tests/test_plan_validator.py tests/test_requirement_engine.py
& $python -B -m pytest -q -p no:cacheprovider --basetemp .test-temp/planner-regression tests/test_hybrid_planning.py tests/test_reflection.py tests/test_planner_tools.py tests/test_agent_benchmark.py
& $python -B -m pytest -q -p no:cacheprovider --basetemp .test-temp/full-regression

& $python -B -m scripts.evaluate_agent --mode offline --label router-offline
& $python -B -m scripts.evaluate_agent --mode document-offline --document-format txt --label router-doc-txt
& $python -B -m scripts.evaluate_agent --mode document-offline --document-format pdf --label router-doc-pdf
~~~

Run the live commands once Streamlit and other Ollama generation are idle and both tags are installed. Each command starts a fresh Python process because model configuration loads at import. Live mode runs the six cases labeled live. F variants and H use scripted transport behavior so their failure modes can be reproduced.

~~~powershell
$env:OLLAMA_MODEL = 'qwen3:8b'
$env:PLANNER_MODEL = 'qwen3:8b'
& $python -B -m scripts.evaluate_agent --mode live --scenario A --label router-qwen3-8b-a
& $python -B -m scripts.evaluate_agent --mode live --repeat 3 --label router-qwen3-8b
& $python -B -m scripts.evaluate_agent --mode document-live --document-format pdf --label router-qwen3-8b-pdf

$env:OLLAMA_MODEL = 'qwen3:4b'
$env:PLANNER_MODEL = 'qwen3:4b'
& $python -B -m scripts.evaluate_agent --mode live --repeat 3 --label router-qwen3-4b

& $python -B -m scripts.evaluate_agent --compare evaluation/results/router-qwen3-8b.json evaluation/results/orchestrator-qwen3-8b.json
~~~

For matched measurements supply the same measured hardware and model-state labels on each machine and record the actual backend versions and model digest using --hardware, --model-state, --ollama-server-version, --model-revision, and --source-version. The model-state flag records an operator observation. Model loading and warming are outside the benchmark.

## Orchestrator handoff

1. Load the same benchmark JSON. Merge the shared defaults and the case's initial overrides. Use the synthetic accepted evidence or the designated incoming fixture. Run each case in a fresh isolated thread and pause before a human decision.
2. Translate the system's actual actions to the labeled candidate identities. Match the target and kind as well as the rule. Represent groups with every covered key. Treat missing or ambiguous bindings as extra actions. Use the same exact-set scoring and outcome assertions from [metrics.py](../src/evaluation/metrics.py) or reproduce them against the contract.
3. Record one run row for every attempt, including failures, with the same benchmark version, scenario ID, repeat index, stop condition, and task assertions. Export the model events and per-run counters using the fields in [evaluate_agent.py](../scripts/evaluate_agent.py). Keep unmeasurable metrics null and include every column in the CSV. JSON comparisons accept the same benchmark_version, metadata and runs shape. Preserve distinct fallback and model provenance. Keep all prompts, document fields and raw errors out of exports.
4. Match mode, scenario IDs, repeat counts, model tag and digest, temperature, thinking mode, token caps, timeouts, starting context, hardware and warm/cold policy. If scheduling or hardware differs, interpret correctness separately and label the timing as hardware-dependent. Compare worker/tool counts only when the same unit is counted. The runner flags differing timing and step definitions, and refuses unmatched modes or case coverage.
5. Compare case correctness and safety alongside inference cost. A lower call count by itself provides no evidence of better task outcomes.

## Local environment and limits

The Planner uses import-time `PLANNER_MODEL`, which defaults to `OLLAMA_MODEL`. The default model is qwen3:4b. It shares a 120-second default timeout with Reflection and uses temperature 0, thinking off, non-streaming JSON output, and 900 maximum generated tokens by default. Document requests use a 180-second timeout and their own typed schema. The selected candidate budget defaults to three and the proposal schema limits action rows to five. Runs with many mandatory simultaneous corrections can exceed the plan budget. The graph routes those cases through Gate 1 and bounded repair. Use `PLANNER_MAX_ACTIONS=3` for the shared default candidate budget.

For a machine comparison, record Python, Pydantic, Ollama SDK and server, and LangGraph versions. Compare model digests, environment variables at process startup, structured schemas and offered selectors, error codes, actual stop conditions, and whether the run used a fresh thread or resumed a pending interrupt. Re-run with a fresh synthetic case before attributing behavior to model size. A Streamlit rerun can retain imported configuration and a pending SQLite thread. Use a new process for model changes. The other architecture and its runtime were unavailable during implementation.

The offline suite supplies LLM-shaped replies and can test deterministic evidence reconciliation and scope. It cannot establish live reliability, model throughput, model-to-model differences, or legal correctness. Review unfamiliar duplicate wording through the normal repair path. General prose assessment and retrieval quality have no labels in this suite. Validate native dependency availability on the machine that runs a live comparison.

## OpenAI and source-grounded Copilot evaluation

See the [Terra and public knowledge audit](TERRA_RAG_AUDIT.md) for the current app.py launcher, full corpus and scanned-document benchmark. The following comparison describes the earlier integration.

The application uses the existing Router, canonical plan and Gate 1 for both providers. The provider selection lives in **Case settings → AI model** for each profile in the current Streamlit session. Copilot keeps case questions and tool selection local. For guidance, it searches the reviewed corpus locally and asks the selected model to choose supporting passages. The app renders the selected excerpts and their real citations. **Case settings → Use source retrieval in Copilot** controls that search.

### Start the application

Run the commands from the Git checkout. The example uses the existing virtual environment and the installed local Qwen model.

```powershell
Set-Location -LiteralPath 'C:\Users\maste\OneDrive\Desktop\Agentic AI\GitHub\Personal_Bureaucracy_Agent'
$env:OLLAMA_MODEL = 'qwen3:8b'
& .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

To enable OpenAI, install the optional SDK and set the server's current-process environment before launching Streamlit. Choose OpenAI API in Case settings. GPT-4o mini is one example with text, image and Structured Outputs support. [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini)

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-openai.txt
$apiKey = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $apiKey).Password
Remove-Variable apiKey
$env:OPENAI_MODEL = 'gpt-4o-mini'
$env:OPENAI_DOCUMENT_MODEL = $env:OPENAI_MODEL
$env:OPENAI_MAX_REQUESTS = '12'
& .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

- OpenAI receives canonical task categories for planning and repair. For a guidance question, it receives the submitted question and the supporting public passages. Leave personal details out of cloud questions.
- The default document analysis uses local OCR and Ollama. A cloud analysis button authorizes the selected text and numbered PDF page images for that file. The backend checks file and event scope. Approved facts, corrections, document replacement and review stay in the existing workflow.
- Requests use strict JSON schema, bounded output and a configurable timeout of 30 seconds by default. The SDK sends one request per attempt. Reflection applies the existing Gate 1 repair bounds. Errors are displayed with content-free messages and the Planner's existing deterministic fallback remains available subject to validation and approval. Requests set `store=false` for Responses API application state. [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) · [API data controls](https://developers.openai.com/api/docs/guides/your-data)
- `OPENAI_MAX_REQUESTS` optionally limits attempts per profile in the current Streamlit session. This is an attempt cap. OpenAI project usage limits control account-level spending. `OPENAI_TIMEOUT_SECONDS` sets the request timeout.

### Public source retrieval

The corpus consists of reviewed excerpts from `knowledge/rules.json` mapped to `knowledge/sources.json` and the local reviewed `data/official_guidance.json` snapshots. Source category distinguishes immigration guidance, Sabancı operational guidance and direct IRO instructions supplied to the project. Each chunk keeps its title, URL where available, procedure scope, version or date, and optional section and page. The knowledge and stage rules remain authoritative for validation.

Lexical retrieval works offline. To add local cosine reranking, set `OLLAMA_EMBED_MODEL` to an installed Ollama embedding model. This adapts Daniele's content-keyed vectors into an ignored `.local/rag_index/` JSON cache. The model and corpus identify the cache. Content and provenance identify each vector. Cache entries are reused across searches, and source edits and model or dimension changes invalidate affected vectors. When embeddings are unavailable, lexical search remains available. Published source snapshots can be updated with the existing `ingest_guidance` helper using reviewed, allow-listed public text. User documents stay in the case and outside the public index.

### Run benchmarks

```powershell
& .\.venv\Scripts\python.exe -m scripts.evaluate_agent --mode offline --label router-local
& .\.venv\Scripts\python.exe -m scripts.evaluate_agent --mode rag-offline --label rag-labels
& .\.venv\Scripts\python.exe -m scripts.evaluate_agent --mode rag-live --label rag-ollama
```

A live OpenAI benchmark needs configured credentials, the SDK, explicit permission for the synthetic requests, and a positive cap for the entire invocation. Choose unique export labels.

```powershell
& .\.venv\Scripts\python.exe -m scripts.evaluate_agent --mode rag-live --provider openai --allow-openai --max-openai-requests 2 --label rag-openai
```

Existing CSV and JSON reports stay in `evaluation/results/`. Open **Case settings → Advanced diagnostics → AI performance** to inspect session model usage, the latest Copilot retrieval and saved synthetic benchmark reports. Opening those views reads saved measurements. It makes no model requests.

- Reports distinguish provider, task, actual requests, Reflection calls, transport failures, output validation, input and output tokens, Ollama embedding requests, and model versus end-to-end latency. Unknown numeric fields are JSON `null` and empty in CSV.
- Set `OPENAI_PRICING_JSON` to a JSON object keyed by exact model. Each entry has an ISO `as_of` date, `input_per_million` and `output_per_million` in USD. Optionally supply `cached_input_per_million`. With no cache rate, estimates use the input rate for all prompt tokens. Estimates require applicable rates and reported usage. The assumption and date are included in each priced request. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- Precision, recall, hit rate and reciprocal rank use manually labeled source IDs over the first *k* chunks. Repeated citations to one source earn one hit. Answer correctness requires the expected phrases in verified evidence. Attribution and faithfulness score the excerpt units displayed with citations. The unsupported-question case scores abstention. Unlabeled retrieval quality and unknown cost are unavailable.

### Integration comparison

The cached Daniele branch was `origin/feature/dani-version-two` at `d5b37a8`. The requested spelling `origin/feature/daniele-version-two` was absent locally. The remote refresh was rejected by automatic approval review.

| Area | Current working files before integration | Daniele at `d5b37a8` | Integration |
| --- | --- | --- | --- |
| Model clients and output | Ollama Planner, Reflection and Copilot, plus an optional document-only OpenAI Responses client | Ollama `chat` and `embed`, Pydantic validation and one corrective retry. No OpenAI client or SDK dependency | Extend the local Responses transport for canonical planning, repair and public guidance. Preserve Ollama and Gate 1 |
| OCR and extraction | RapidOCR/native text, canonical observations, verification and typed document fallback | Pack-driven document extraction, grounding and a different Twin observation store | Use shared request accounting with the existing document processor and explicit per-file consent |
| Corpus and index | Attributed rules and local guidance snapshots, lexical ranking | Procedure pack texts, mostly rule-derived placeholders, chunked embeddings and NumPy cosine. Vectors live in the teammate's Twin SQLite schema | Use the existing reviewed corpus and add optional local cosine reranking with a versioned sidecar. No NumPy or database migration is needed |
| Runtime and citations | LangGraph Router, canonical actions, existing Copilot and Streamlit views | An Orchestrator with section workers and a `search_sources` tool over procedure packs | Keep the active graph and render only verified citations for the answer. Preserve the case tools and approval path |
| Metrics and tests | Synthetic Router tests, action and safety metrics, CSV/JSON exports | Separate A/B/D evaluation for rule verdict accuracy, pass-k, tool calls and elapsed time, plus offline source search tests | Extend the Router exports and Advanced view. Compare engines using shared scenario labels and measurement scopes |

Daniele's evaluator can still be run from the teammate's separate checkout. Its verdict and worker counts have different units from Router action-plan outcomes. Cross-architecture timing and success comparisons need a shared scenario, stopping condition and execution mode.
