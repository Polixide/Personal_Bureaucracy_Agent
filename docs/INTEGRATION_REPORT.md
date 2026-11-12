# Local A+B+C integration report

Completed on 2026-11-11. Workstream A's folder structure and `src/core/state.py` AgentState remain canonical. There is no second runtime State model. No git push, merge or commit was executed. No remote repository was modified. All reference implementations under `integration_sources/` and all root knowledge files are retained unchanged.

## Integrated behavior

START ? Gate 0 ? Supervisor ? selected Document Analyzer / Requirement Engine / Planner. Analyzer and Requirement Engine return to Supervisor. Planner always reaches Gate 1. A failed candidate goes to Reflection, which records structured errors and the previous candidate, increments the per-run repair count, and returns to Supervisor. Supervisor selects Planner to revise the plan. Gate 1 validates every revision. The initial candidate plus at most three repair candidates can be evaluated. Repeated invalid or malformed output stops with HUMAN_REVIEW_REQUIRED. No model approves its own output.

Gate 1 PASS reaches a real LangGraph approval interrupt. Only an explicit boolean True resume approves the proposal; other responses deny it. Approved plans pass through deterministic state bookkeeping and final output. Approval never performs official actions or marks proposed steps completed. All proposals require approval, including information-gathering plans.

Documents merge by a path-derived ID. Extracted facts are namespaced by that ID and retain C's quality flags. Canonical document aliases reconcile A/B/C names. Extraction supports text PDF and UTF-8 TXT with PyMuPDF. Unknown documents, extraction errors and ambiguous dates do not become proof of legal validity.

## Knowledge migration and compromises

B's knowledge copy matches root knowledge, but its old Python engine and tests use obsolete meanings for R001?R008. The engine was reimplemented with explicit R001?R029 checks, without eval or condition-string execution. PASS, FAIL and missing_information remain separate. Three-valued AND/OR logic permits known conditional alternatives without requiring irrelevant inputs. Missing-information results name the absent fields.

The dependency policy uses the union of inline root rule dependencies and root dependencies.json, preserving edges present in only one declaration. Root data itself was not rewritten. This policy is intentionally conservative for official actions.

The unified PlanStep retains A's identifiers, document dependencies, due dates and approval flag while adding B's priority, reason and rule grounding. Deadline keys associate proposed dates with explicitly supplied dates. Gate 1 checks references, schema, unique ordering/IDs, dependency order and dates, extracted-document quality/dates, blocker coverage and official-action prerequisite closure. B's English phrase checks remain supplementary heuristics; they cannot establish full semantic correctness of arbitrary prose. The prototype executes no external actions.

The three-repair limit moved from B's internal loop to graph orchestration. Reflection is deterministic feedback bookkeeping; the planner performs the repair model call. Planner transport/schema failures also consume the same bounded repair budget. This differs from B's immediate exit on an initial planner failure.

A LangGraph in-memory checkpointer is necessary for real interrupt/resume approval. It introduces no database. The same graph instance and thread ID retain the Digital Twin across new_run calls; restarting the process loses state. SQLite, Streamlit, MCP and RAG were not added.

Legal residence context is supplied explicitly as a typed component of AgentState. The document adapter stores observations rather than automatically asserting legal readiness, authenticity, insurance coverage start, or exemption status. Consequently an upload alone can leave legal requirements UNKNOWN. A future reviewed fact-confirmation flow can bridge those observations; the present prototype asks for missing information. It does not parse arbitrary chat into verified context facts.

Fees, form-window IDs and IRO submission dates are caller-supplied current inputs. The prototype neither refreshes nor invents them. Date comparisons are deterministic and current_date is an explicit reproducible input. Missing scope facts trigger information gathering; explicitly incompatible scope is rejected at Gate 0.

Only config.py defines the model name, host, timeout and maximum repair attempts. Production model calls use the configured model without environment or CLI model overrides. C's alternative-model acceptance test was migrated to host/timeout configuration coverage.

## Verification

The complete canonical automated suite passed: **163 tests, plus 4 unittest subtests**. It runs without a live Ollama server and has an autouse guard against unmocked model calls. Coverage includes all 15 root synthetic rule cases, PASS/FAIL branches for all 29 rules, all-unknown rules, exact date boundaries, conditional alternatives, extraction and document quality, structured planner output, Gate 1 checks, approval/resume and denial, graph repair feedback, malformed candidates, the three-repair limit, and document/fact continuity across events.

The initial run had 108 passing tests and six temporary-directory permission failures/errors on the Windows sandbox. These were filesystem setup failures, not failed behavioral assertions. Tests now direct temporary artifacts into .test-temp inside the project; the verified suite was run with the filesystem permission override required by this environment. No tests were skipped or weakened to hide those errors.

Final test command used:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.test-temp/complete
```

There are 13 third-party LangSmith deprecation warnings under Python 3.14 concerning asyncio.iscoroutinefunction. No integration test failures remain. Installed runtime versions used for verification: LangGraph 1.2.20, Pydantic 2.14.3, Ollama 0.6.3, HTTPX 0.28.1 and PyMuPDF 1.28.2. pytest is a development dependency only.

The synthetic CLI was executed with approval and denial; final statuses were APPROVED and APPROVAL_DENIED. The default demo mocks only model transport while exercising real extraction, Pydantic schemas, adapters, requirements, graph gates and approval. Use `python -m scripts.run_demo`; use `--live` only for an explicit local Ollama smoke test. No live Ollama call was made during this integration, so model extraction/planning quality is not claimed as verified. Python source compilation also passed.

## File inventory

No reference files were physically moved: implementations and fixtures were copied/adapted into the canonical structure. B's manual demonstration files are retained as references; their obsolete rule assumptions are not part of the executable prototype. src/core/stubs.py now contains compatibility imports only, without mock workers or automatic approval.

Modified existing project files:

- `.gitignore`
- `README.md`
- `config.py`
- `requirements.txt`
- `src/core/state.py`
- `src/core/graph.py`
- `src/core/stubs.py`

Created canonical source, test, configuration, documentation and data files:

- `data/synthetic_samples/enrollment_missing.txt`
- `data/synthetic_samples/example.pdf`
- `data/synthetic_samples/health_insurance.txt`
- `data/synthetic_samples/insurance_ambiguous.txt`
- `data/synthetic_samples/insurance_inconsistent.txt`
- `data/synthetic_samples/insurance_invalid_date.txt`
- `data/synthetic_samples/student_enrollment.txt`
- `docs/INTEGRATION_REPORT.md`
- `pytest.ini`
- `requirements-dev.txt`
- `scripts/.gitkeep`
- `scripts/__init__.py`
- `scripts/run_demo.py`
- `src/.gitkeep`
- `src/__init__.py`
- `src/agents/__init__.py`
- `src/agents/planner.py`
- `src/core/__init__.py`
- `src/document_processing/__init__.py`
- `src/document_processing/extractor.py`
- `src/document_processing/processor.py`
- `src/document_processing/schemas.py`
- `src/guardrails/__init__.py`
- `src/guardrails/gate_0.py`
- `src/guardrails/gate_1.py`
- `src/guardrails/human_approval.py`
- `src/memory/__init__.py`
- `src/memory/reflection.py`
- `src/memory/state_update.py`
- `src/workers/__init__.py`
- `src/workers/document_analyzer.py`
- `src/workers/rule_loader.py`
- `src/workers/rules_engine.py`
- `tests/.gitkeep`
- `tests/conftest.py`
- `tests/test_action_planner.py`
- `tests/test_document_types.py`
- `tests/test_extractor.py`
- `tests/test_plan_validator.py`
- `tests/test_processor.py`
- `tests/test_reflection.py`
- `tests/test_required_documents.py`
- `tests/test_requirement_engine.py`
- `tests/test_rule_loader.py`

The seven files under data/synthetic_samples are byte-for-byte copies of C's fixtures. Empty __init__.py files establish the listed canonical packages. requirements-dev.txt installs the direct runtime requirements plus pytest. pytest.ini limits normal discovery to canonical tests, so outdated reference tests are preserved without importing conflicting source namespaces.

Generated local environment/artifacts: .venv/ (installed dependencies), .test-temp/ (test temporary files), Python __pycache__/ files and pytest cache/staging directories. These are ignored by .gitignore and are not application source. No reference directories were deleted.
