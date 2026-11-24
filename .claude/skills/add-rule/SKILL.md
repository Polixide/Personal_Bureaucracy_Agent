---
name: add-rule
description: Use when adding or editing a single rule in a Procedure Pack rules.json. Explains the condition format, three-valued logic and the required tests.
---

# Add or edit a rule

Rule format (one object in `procedures/<id>/rules.json`):

```json
{
  "rule_id": "R015",
  "section": "insurance",
  "description": "Health insurance must cover the whole permit period.",
  "severity": "blocking",
  "condition": {"op": "gte", "field": "insurance_end_date", "other_field": "permit_end_date"},
  "depends_on": ["R016"],
  "source_ids": ["SAB_EXCHANGE_OVERVIEW"],
  "version": "1.0",
  "status": "verified"
}
```

Condition operators (JSON tree, evaluated by `src/rules.py`):
- `all` / `any` with `args` (list of conditions), `not` with `arg`.
- `eq`, `in`, `lt`, `lte`, `gt`, `gte`: left side is `field`; right side is `value` OR `other_field`,
  with optional `offset_days` for dates (example: passport must outlive the permit by 60 days).
- `is_true` with `field`.

Three-valued logic: a missing field makes the comparison UNKNOWN (None).
`all`: False if any arg is False, else None if any is None, else True.
`any`: True if any arg is True, else None if any is None, else False.
UNKNOWN is never reported as FAIL; the agent asks for the missing field instead.

Checklist:
1. Use only field names declared in `procedure.json`.
2. Every `source_ids` entry exists in `sources.json`. If you cannot cite a source, set `"status": "unverified"`.
3. `depends_on` must not create a cycle.
4. Write three tests: passes, fails, missing information (see `tests/test_rules.py` for the pattern).
5. Run `python -m src.pack validate <id>` and `python -m pytest -q`.
