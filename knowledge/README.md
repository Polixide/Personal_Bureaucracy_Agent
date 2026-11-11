# Final Knowledge & Rules Layer — İKAMET
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
