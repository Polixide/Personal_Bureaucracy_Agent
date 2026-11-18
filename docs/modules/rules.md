# `src/rules.py` — the rule evaluator

## What it does
It decides, for every rule of a Procedure Pack, whether the user's current facts make it **pass**, **fail** or
**unknown**. It is plain Python with no model calls. This is the "code verifies" half of the project principle.

## Where it sits in the flow
`evaluate_rules(pack, facts, section=None)` is called by:
- the `evaluate_rules` tool (agent workers and the orchestrator),
- Gate 1, which recomputes the verdicts to check the answer's claims,
- the sandbox, which accepts a stage only when its rules pass.

It returns a list of `RuleResult`. `required_rules(pack, ids)` gives a rule plus all its prerequisites; the
sandbox uses it for stage checks.

## Inputs and outputs
- **Input:** a `Pack` (see `pack.md`) and a fact map `{field name: value}`. Values are `bool`, `int`, `str` or
  `date`. A missing key or `None` means *unknown*.
- **Output:** `RuleResult`:
  ```json
  {"rule_id": "R006", "section": "passport_photo", "status": "unknown",
   "missing_fields": ["passport_expiry_date"], "severity": "blocking",
   "source_ids": ["SAB_EXCHANGE_OVERVIEW", "GOV_FAQ"], "rule_status": "verified", "blocked_by": []}
  ```
- **A rule condition** is a JSON tree. For example, R006 (passport valid 60 days beyond the permit) is
  `{"op": "gte", "field": "passport_expiry_date", "other_field": "requested_permit_end_date", "offset_days": 60}`.
- **Operators:** `all`, `any`, `not`, `eq`, `in`, `lt`, `lte`, `gt`, `gte`, `is_true`.

## Why it is built this way
- **Three-valued logic.** A missing fact is never a failure. `all` is false if any part is false, otherwise
  unknown if any part is unknown. `any` works the other way round. That is why the agent *asks* instead of
  saying "you fail".
- **No short-circuit.** Every branch is evaluated so that `missing_fields` lists *everything* still needed.
  The milestone-1 engine did the same, and the regression oracle depends on it.
- **No rule ids in the code.** Every rule comes from `procedures/<id>/rules.json`, so a new procedure needs no
  engine change.
- **Dependencies do not change verdicts.** They are reported in `blocked_by` only.

## How to test it
```
python -m pytest -q tests/test_rules.py tests/test_rules_oracle.py
```
- `test_rules.py` has truth tables, operator cases, and pass/fail/unknown for each of the 29 İkamet rules.
- `test_rules_oracle.py` checks 337 contexts against `tests/data/legacy_verdicts.json`, the verdicts of the
  milestone-1 engine, generated once by a script that was deleted with that engine (see commit 486917b).

## Known limits
- `offset_days` is fixed per rule; there are no month or year offsets.
- No arithmetic between fields (for example "end − start > 90 days"). Use a derived field instead (see
  `procedure.json` `derived`).
- Marco's code-only "Milestone-1 scoped" R015 variant was dropped on purpose (migration plan, section 0).
