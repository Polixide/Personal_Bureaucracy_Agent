# Terra and public knowledge integration audit

## Checkout and source

Target `feature/marco-milestone1` at `b0ee6c1`, with the existing mutable working tree. The configured GitHub remote advertised `refs/heads/feature/dani-version-two` at `0ddf6c889f7aa2062d761b39e5478eb6d2780161`. The commit was already cached under the matching remote-tracking ref. The `daniele-version-two` spelling was absent from both the local ref list and the remote response.

The target previously had `streamlit_app.py` only. The new `app.py` launches that same Router, state model, navigation, and approval flow. The source app's seven Details tabs use its own Twin and Orchestrator. This port uses its Knowledge base tester inside **Your details → Knowledge base** and the target's existing checks, simulation, plan repairs, and saved run views.

## Source-to-target inventory

| Source at `0ddf6c8` | Target path or adapter | Use and verification |
| --- | --- | --- |
| `knowledge/corpus.json`, `knowledge/corpus/*.txt` | Same paths | 25 manifest entries and 25 public Sabancı snapshots. All are loaded and checksum-verified when search initializes |
| `knowledge/seeds.json` | Same path | Two original crawler seeds retained for provenance and later reviewed refreshes. Search reads the committed snapshots |
| `procedures/ikamet_first/sources.json`, `sources_text/*.txt` | Same paths | 9 checksum-verified procedure sources. The case scope includes first-application government and Sabancı sources |
| `procedures/ikamet_extension/sources.json`, `sources_text/*.txt` | Same paths | 4 checksum-verified extension sources for the full public archive |
| `procedures/tc_kimlik_first/sources.json`, `sources_text/*.txt` | Same paths | 3 checksum-verified identity procedure reference sources for the full archive |
| `procedures/tc_kimlik_renewal/sources.json`, `sources_text/*.txt` | Same paths | 4 checksum-verified identity renewal reference sources for the full archive |
| `src/rag/ingest.py`, `crawl.py`, `knowledge.py`, `index.py` | `src/rag/public_sources.py`, `retriever.py`, `semantic.py` | Offline snapshot ingestion and verification, paragraph chunks, lexical matching, local embedding batches of 16, persistent cosine vectors. The loaders resolve paths from the repository and report missing or changed references |
| Twin `rag_vectors` table | `.local/rag_index/` | Generated, ignored JSON vector cache, keyed by model and chunk content. Existing case databases retain their schema and ownership |
| `src/llm.py`, `src/document_processing/transcribe.py`, `vision.py` | `src/core/model_provider.py` and the existing document processor | Adopt the source environment defaults and numbered images. Use the target's Responses Structured Outputs adapter, real usage fields, and per-file consent |
| `app.py` `knowledge_box` and `view_details` | `app.py`, `src/ui/knowledge.py`, `src/ui/technical.py` | Lookup, excerpts, separate answer generation, profile-session metrics and saved benchmark reports |
| `eval/rag_eval.py`, `eval/chat_eval.py`, `eval/metrics.py` | `src/evaluation/`, `evaluation/benchmark.v1.json`, `scripts/evaluate_agent.py` | Reuse Router reports and labeled retrieval measures. Daniele's broader rule-verdict evaluation runs in its own architecture |
| Pack `procedure.json`, `rules.json`, form assets and workflow scenarios | Existing `knowledge/rules.json`, `sources.json`, `test_cases.json` and Router | Source pack rules, forms, fixtures and databases are outside the public retrieval import |

The import adds 51 source blobs without editing the target rule JSON or the source snapshots. The first-application form snapshot uses CRLF in the original blob. The loader checks its text with universal-newline decoding, matching the upstream checksum. The full archive includes cross-check material with its original authority label. Returning a source from another procedure in the public tester does not make that procedure available in the workflow.

The web crawler, remote refresh, and experimental rule-draft generator remain source-branch tools. The target indexes verified local snapshots. It does not contact source websites on startup or when answering.

## Launch in PowerShell

```powershell
Set-Location -LiteralPath 'C:\Users\maste\OneDrive\Desktop\Agentic AI\GitHub\Personal_Bureaucracy_Agent'
$python = '.\.venv\Scripts\python.exe'
$env:OLLAMA_MODEL = 'qwen3:8b'
$env:OLLAMA_EMBED_MODEL = 'bge-m3'
& $python -m src.rag status --scope public
& $python -m streamlit run app.py --browser.gatherUsageStats false
```

Use `python -m src.rag index --scope public` to generate missing vectors from the local corpus. Use `--rebuild` on that command after replacing an embedding model under the same tag. An incremental case-scope build retains compatible vectors for the rest of the public archive. `python -m src.rag search "What address proof do I need if I live in a dormitory?" --scope case` shows the evidence, ranking method, and retrieval measurements. The application retains lexical search when no compatible vector index is configured and reports the actual method used.

## Configure OpenAI

Use only a replacement key. The application reads the environment first and can read an `OPENAI_API_KEY` setting from the ignored local `.streamlit/secrets.toml` on the server. It never puts the value in browser state or evaluation exports.

```powershell
& $python -m pip install -r requirements-openai.txt
$key = Read-Host 'Rotated OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password
Remove-Variable key
$env:LLM_PROVIDER = 'openai'
$env:OPENAI_MODEL = 'gpt-6.1-terra'
$env:OPENAI_REASONING_EFFORT = 'low'
$env:DOCUMENT_PROVIDER = 'cloud'
$env:OPENAI_DOCUMENT_MODEL = $env:OPENAI_MODEL
$env:OPENAI_MAX_REQUESTS = '8'
& $python -m streamlit run app.py --browser.gatherUsageStats false
```

`LLM_PROVIDER` selects the general provider for Planner, Reflection, guidance generation and the procedure recommender. The supervisor, case facts and Gates remain deterministic. `DOCUMENT_PROVIDER` independently controls whether the per-file OpenAI option is offered. Choose **Upload and analyze with OpenAI** or the equivalent reprocess action for the file. Local upload and review remain available. On this cloud path, readable PDF pages provide native text, while unreadable pages are rendered and sent as numbered high-detail `input_image` parts. Classification, normalization and source grounding use the recovered pages. The page selection, manual review, and consent scope still apply. [Terra capabilities](https://developers.openai.com/api/docs/models/gpt-6.1-terra) · [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)

In **Your details → Knowledge base**, select the case scope or full public archive, enter a public question and click **Retrieve evidence**. Inspect the passages, stage reference and source link before optionally clicking **Generate answer**. The selected model chooses passages from this result. The app checks the selected indices and renders attributed excerpts. The metrics area distinguishes the current interaction, totals in this profile session, and saved offline/scripted or live benchmark runs. Similarity describes retrieval ranking and accuracy needs labeled evidence.

## Verification commands

```powershell
& $python -m pytest -q tests/test_terra_audit.py tests/test_ai_integration.py tests/test_streamlit_ai_provider.py tests/test_assistant_rag.py
& $python -m pytest -q --continue-on-collection-errors
& $python -m scripts.evaluate_agent --mode rag-offline
& $python -m scripts.evaluate_agent --mode rag-live --provider ollama --scenario RAG-dormitory
```

With the replacement key loaded and a spending cap agreed, the following commands issue small synthetic requests. The request flags bound attempts per invocation and use the existing JSON/CSV benchmark exports. The account probe returns the accessible model ID, verifies the tiny structured response and reports available usage.

```powershell
& $python -m scripts.verify_openai --confirm-rotated-key --allow-openai --max-openai-requests 2
& $python -m scripts.evaluate_agent --mode document-live --document-format scan --provider openai --allow-openai --max-openai-requests 3
& $python -m scripts.evaluate_agent --mode live --scenario A --provider openai --allow-openai --max-openai-requests 2
& $python -m scripts.evaluate_agent --mode rag-live --scenario RAG-dormitory --provider openai --allow-openai --max-openai-requests 1
```

The shared collector records requested and returned model IDs, reasoning effort and usage where reported, number of image inputs, elapsed model time, transport outcomes, schema validity and optional estimated cost. Retrieval records contain actual index status, coverage for labeled sources, chunk count and elapsed search time. Pricing metadata comes from `OPENAI_PRICING_JSON` as documented in the benchmark guide. Unavailable measurements remain empty in CSV or null in JSON. [Benchmark guide](AGENT_BENCHMARK.md)
