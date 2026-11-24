# `src/memory/episodes.py` — episodic memory (lessons from validated repairs)

## What it does
When Gate 1 rejects an answer and the model's repaired answer **passes**, each rejection becomes a stored
lesson. A lesson records the failure type, the rule involved, and a fixed sentence describing the correction.
Later runs see the lessons for their procedure, and a repair prompt also gets the lessons for the same
failure types.

## Where it sits in the flow
- **Storing.** The orchestrator's verify node calls `record_repairs(...)` when an answer passes Gate 1 after at
  least one repair.
- **First prompt.** `run_agent` puts `lessons_for(store, procedure_id)` into the first message, under "LESSONS
  FROM PAST REPAIRS".
- **Repair prompt.** The repair message gets `lessons_for(store, procedure_id, [failure codes])`.
- **Retiring.** `twin.service.record_rule_update` retires the lessons tied to a rule that changed.

## Inputs and outputs
A lesson stored in the Twin's `episodes` table:
```json
{"failure_type": "CLAIM_MISMATCH", "affected_item": "R006", "rule_version": "1.0",
 "corrective_action": "Copy each claim's status from the latest evaluate_rules result; never guess it.",
 "outcome": "accepted by Gate 1 after 1 repair(s)", "correction_validated": true}
```
What the model sees: `[{"failure_type": "CLAIM_MISMATCH", "lesson": "...", "rule": "R006"}]`.

## Why it is built this way
- **Only validated repairs.** A correction that never passed Gate 1 is not a lesson. The database's
  `CHECK (correction_validated = 1)` makes storing one impossible; this idea comes from Neda's milestone-1
  design.
- **Generic text only.** Issue items can contain user text, such as a sentence with a date or name. The
  lesson therefore keeps only a rule id and a fixed sentence per failure type, so nothing personal is reused
  across users.
- **Retrieval by procedure and failure type, not by user.** The mistakes belong to the agent, not the person.

## How to test it
```
python -m pytest -q tests/test_episodes.py
```

## Known limits
- **Fixed sentences.** The lessons are fixed per failure type. They make the model more careful but cannot
  teach procedure-specific knowledge; that belongs in the pack.
- **No measurement yet.** Whether lessons reduce rejections must be measured with live runs
  (`python -m eval.run_eval --live`).
