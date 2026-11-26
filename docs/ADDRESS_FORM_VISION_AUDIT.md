# Address Form extraction audit

## Saved-case findings

The checkpoint matching the reported surname spelling recorded one page, zero native text characters, RapidOCR and two local `qwen3:8b` calls for classification and extraction. It recorded a cache miss and no cloud fallback. Neither local model request included an image. The recorded OCR and total extraction times were 5.030 s and 47.898 s. The page had 1,005 OCR characters and no recorded rotation. Classification submitted 1,014 text characters including the page header, and extraction submitted 1,107 including its prompt context. Page 1 was the only evidence page. Native request token counts were unavailable for that saved result.

The surname spelling matched the saved OCR source snippet. The save and display steps were carrying that reading forward. The saved observation contained the first name, surname and phone only. The document card selects key fields and `View all information` reads the typed observations and confirmed values. The other fields were absent from that saved extraction. The approximately 5,000 OpenAI tokens reported separately could not be tied to this analysis from the saved metadata.

## Current route and review

- Upload registers the file with Gate 0. The document analyzer opens it through the existing scoped cache and processor. The local path reads PDF text and uses RapidOCR on pages with insufficient or malformed native text and an image layer.
- A per-file OpenAI action authorizes that specific file and invocation. OpenAI receives numbered PNG page images when the Vision path runs. Known Address Forms use bounded image transcription after explicit authorization. A scan with no identified type can use Vision on poor native pages before classification. Readable native pages and filled text widgets remain available to extraction even when their page image is outside the request.
- The application renders the visible PDF page rectangle in RGB at the existing default of 200 DPI, capped at a 3,000-pixel longest side and two page images per selection. Responses uses `detail=high`. The application sends page PNGs. The original PDF stays on the local server. Advanced diagnostics show the image inputs and dimensions for each attempted request alongside native text coverage.
- The Address Form parser recognizes whole labels and bilingual equivalents. It reads wrapped address lines and textual form widgets. It preserves the original spelling and page, discards conflicting typed values, and retains their raw candidates for review. Critical OCR and Vision observations require explicit confirmation. A drawn signature or blank signature label leaves the signature question open.
- The latest request metadata is stored on the document checkpoint and round-trips through the Digital Twin's existing observation JSON column. It excludes the file name, document contents, credentials and identifiers. Old checkpoints without this metadata report N/A. The file view shows the latest extraction, the preceding typed readings after reanalysis and any applicable user-confirmed corrections. Reopening the case and ordinary Streamlit reruns read the saved result.

## Synthetic before and after

The fixture combines rasterized identity text, a labeled PDF text widget and native contact fields. Its drawn mark deliberately leaves signature status unresolved. Eight typed fields have expected values.

| Route | Exact fields before → after | Name and ID exact before → after | Precision | Recall | Missing rate | Incorrect rate | Wall time before → after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local PDF and real RapidOCR, with a stubbed local model if called | 4/8 → 8/8 | 1/3 → 3/3 | 100% → 100% | 50% → 100% | 50% → 0% | 0% → 0% | 2.249 s → 2.535 s |
| Installed OpenAI SDK with an offline HTTP mock | 4/8 → 8/8 | 1/3 → 3/3 | 100% → 100% | 50% → 100% | 50% → 0% | 0% → 0% | 0.231 s → 0.197 s |

Both routes returned a typed Address Form. Before the fix, the mock's classification and transcription validated and its extraction response failed the strict response check. After the fix, the single transcription response validated. The mock transport recorded three requests before the fix and one numbered-image transcription afterward. The mock returned 1,620 input and 570 output tokens in total before, and 540 input and 190 output tokens afterward. Those counts exercise response metadata and accounting. Live OpenAI tokens, account access, accuracy, latency and estimated billing cost are N/A. The test shell had no configured model or key for live evaluation. The local path required no model calls after the fix on this fixture.

## Reproduce

From the active checkout in PowerShell, run the offline benchmark and the focused tests.

```powershell
.\.venv\Scripts\python.exe -B -m scripts.benchmark_address_vision
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp .test-temp/address-vision-repeat tests/test_address_vision.py tests/test_terra_audit.py
```

To create the synthetic upload and launch the standard Streamlit entry point, run these commands.

```powershell
New-Item -ItemType Directory -Path '.test-temp' -Force | Out-Null
.\.venv\Scripts\python.exe -B -c "from pathlib import Path; from scripts.benchmark_address_vision import make_pdf; make_pdf(Path('.test-temp/synthetic-address.pdf'))"
.\.venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats false
```

1. Open a separate synthetic case and go to **Documents**. Upload `.test-temp/synthetic-address.pdf` using **Upload and analyze**. Choose the saved Address Form and open **View all information**. Check the identity, contact, address and date rows and the open signature question.
2. To repeat against the live service, configure the approved replacement credential on the Streamlit server, set the authorized document model and reasoning effort, and enable **Case settings → Document processing → Offer OpenAI vision for each file**. Changing this setting leaves the existing saved result available.
3. Select the synthetic file, open **Document options**, choose **Address Form** and use **Analyze this document with OpenAI Vision** once. This is the authorization to send the selected synthetic page images. The saved analysis label shows the actual route.
4. Reopen **View all information**. After a reading changes, **Readings changed by reanalysis** shows the previous and latest readings. A user-confirmed value for the same document type appears separately.
5. Open **Case settings → Advanced diagnostics → Document diagnostics** and select the Address Form. Check the original page numbers, image dimensions, responding model, outcome and native usage. Refresh or revisit the page to see the same saved diagnostics.

The automated Streamlit regression uses the actual interface code, the SDK mock and a temporary SQLite checkpoint. The standalone benchmark uses only synthetic data. The manual command above uses the usual local database and the separate synthetic case; the test suite never accesses the production checkpoint for writes.

## Implementation inventory

| File | Changed entry points |
| --- | --- |
| `src/document_processing/extractor.py` | `TextPage`, `extract_pages` |
| `src/document_processing/evidence.py` | `_address_form_field`, `_address_form_rows`, `read_address_form`, `address_candidates` |
| `src/document_processing/processor.py` | `request_json`, `checked_classification`, `_process_pages`, `_finish`, `_vision_transcription`, `process_document` |
| `src/document_processing/providers.py` | `OllamaProvider.generate_json`, `OpenAIProvider.generate_json`, `_create_openai_client` |
| `src/document_processing/analysis_diagnostics.py` | `summarize_document_analysis` |
| `src/core/llm_diagnostics.py` | `ModelCall`, `collect_model_calls` |
| `src/core/state.py` | `Document` |
| `src/workers/document_analyzer.py` | `document_analyzer` |
| `src/workers/verification.py` | `grounded_values`, `document_review` |
| `src/memory/twin_store.py` | `_encode_document_info`, `_decode_document_info`, `TwinStore.load_twin`, `TwinStore._write_twin` |
| `src/ui/components.py` | `invoke` |
| `src/ui/documents.py` | `render_analysis_status`, `render_document`, `render_document_options`, `render_document_values`, `render_full_document_information` |
| `src/ui/model_settings.py` | `DOCUMENT_CONSENT` |
| `src/ui/technical.py` | `view_technical`, `render_document_diagnostics` |
| `scripts/benchmark_address_vision.py` | Synthetic PDF generator, mocked SDK wire inspection and precision/recall benchmark |
| `tests/test_address_vision.py` | Eight regressions for extraction, consent, conflict handling, cache, Streamlit reruns and SQLite round trips |
| `docs/ADDRESS_FORM_VISION_AUDIT.md` | Audit evidence, measured benchmark and reproduction commands |

## Verification results

The first focused extraction and lifecycle run passed 283 tests. The second pass exercised the installed SDK transport, the Streamlit document action, the SQLite checkpoint and Digital Twin round-trip, and the existing provider and RAG paths. The focused run passed 38 tests. The full regression run passed 1,208 tests with zero test failures. Two existing modules failed collection because they import the missing root module `document_evidence`. These were `tests/test_classification_identity.py` and `tests/test_pdf_pipeline_recovery.py`.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short --basetemp .test-temp/vision-restart tests/test_address_vision.py tests/test_neda_evaluation.py tests/test_terra_audit.py tests/test_streamlit_ai_provider.py
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short --basetemp .test-temp/vision-final-full --continue-on-collection-errors
```

The active branch is `feature/marco-milestone1` at `b0ee6c1`. The final working tree contains 49 modified tracked files and 46 untracked entries, including the pre-existing changes. This task edited 13 files and added four. The index checksum matches the starting checksum and all changes are local and unstaged.
