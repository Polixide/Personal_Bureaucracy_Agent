# Final Knowledge & Rules Layer — İKAMET

## Implemented demo: R014 only

`src/workers/rules_engine.py` loads R014 from `rules.json` and resolves its
source metadata from `sources.json`, using paths relative to the repository.
It uses explicit Python checks; JSON condition strings are never executed.
All other rules and dependency traversal remain outside the implemented demo.

The existing graph now runs Document Processing -> state -> R014 -> final result.
`relevant_requirements` contains one requirement, with `rule_id`, `satisfied`,
`evaluation_result`, `rule_version`, `source_id`, and `source_metadata`.

- `supplied_and_accepted`: an analyzed student certificate has complete,
  unambiguous extracted facts and no quality issues.
- `missing`: no student certificate evidence is available.
- `manual_review_required`: an uploaded document could not be identified or
  accepted, or certificate facts are absent, malformed, incomplete, or ambiguous.

Missing or review outcomes are unsatisfied and create structured graph issues.
An analyzed label alone does not establish verified facts. Original quality flags
are preserved and the extracted facts are validated again before acceptance.
Acceptance here is limited to extraction quality and presence; authenticity,
recency, and legal validity are not established. No certificate recency threshold
is implemented. The planner and approval nodes remain MOCKED and the final
response explicitly labels them. There is no real approval action.

Run all offline tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

`tests/test_rules_integration.py` covers the A/B/C flow with only the Ollama
response mocked, plus provenance and checks against unverified state.

Version: 1.0.0
Retrieval date: 2026-11-05

Scope: Sabancı University incoming exchange student, 18+, staying 3 months or longer, first short-term residence-permit application in Türkiye.

Authority hierarchy:
1. Government/e-İkamet — legal/general truth.
2. Sabancı IRO — Sabancı-specific operational workflow.
3. Other universities — cross-check only.

Design choices:
- Dynamic fees and monthly submission windows are not permanent hard-coded facts.
- Unknown data is not treated as failure.
- Complete documentation does not mean guaranteed approval.
- RAG may retrieve current official evidence, but rule changes require human approval, versioning, and regression tests.
