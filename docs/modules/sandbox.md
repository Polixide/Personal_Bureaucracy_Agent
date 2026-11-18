# `src/sandbox/` + artifact tools — mock world, proposals, approval

## What it does
- **Artifact tools only propose.** `create_ics`, `draft_email`, `build_checklist` and `fill_pdf` prepare an
  artifact and store it as a PROPOSED action in the Twin.
- **The sandbox tries stages first.** `simulate_submission` tries an official stage in a **mock portal** that
  applies the pack's own rules.
- **Approval is the only exit.** `approve_action` is the only path from the Twin to the real world. It runs the
  **promotion gate**, then writes the artifact to the user's local outbox. Nothing is ever sent.

## Where it sits in the flow
```
agent (orchestrator) ── simulate_submission ──▶ mock_world.simulate ──▶ simulations table
                    └─ create_ics / draft_email / fill_pdf / build_checklist ──▶ proposed_actions (status "proposed")
app (user clicks Approve) ──▶ approval.approve_action ──▶ promotion gate ──▶ outbox file + twin_to_real event
user reports a result ──▶ twin.service.record_outcome ──▶ confirmed facts + real_to_twin event
```

## Inputs and outputs
- `create_ics(title, date_field, description, rule_ids)`: the model names a date **field**, and code reads the
  date from the facts. The iCalendar text is written by hand: CRLF line ends, escaping, 75-octet folding.
- `draft_email(to_role, subject, body, rule_ids, stage_id)`: the recipient comes from `procedure.json`
  `contacts`. An unknown address stays a placeholder. The text must pass Gate 1's text checks (no "submitted",
  no invented dates).
- `fill_pdf(form_id)`: the pack's `forms` maps PDF fields to facts. Unknown facts stay empty and are listed.
- `build_checklist(section_ids)`: deterministic Markdown of every rule result and what is still needed.
- **Every artifact tool returns** `{"action_id": "A-1f2e3d4c", "status": "proposed", "requires_stage": ...}`.
  The action id can be used as a next-action ref.
- **Simulation:**
  `{"stage_id": "official_application", "accepted": false, "unknown": {"R005": ["application_channel"]}, "message": "... nothing was submitted ..."}`

## Why it is built this way
- **Promotion gate.** A proposal tied to a stage (for example an e-mail that starts the official application)
  can be approved only if the **latest simulation of that stage was accepted on exactly the current facts**.
  The simulation stores a facts fingerprint, so a fact change after the simulation blocks approval until the
  user simulates again.
- **The mock world reuses the rule engine.** The stages and their required rules come from `procedure.json`
  `stages`, and prerequisites are added automatically. There is no second source of truth.
- **"Executed" means a file was written for the user** (`.ics`, `.eml` draft, `.md`, filled `.pdf`). The agent
  never sends, books, pays or submits.

## How to test it
```
python -m pytest -q tests/test_sandbox.py tests/test_two_way_flow.py
```
`test_two_way_flow.py` runs the whole loop: upload → extraction → agent proposes and simulates (refused) →
calendar approved, e-mail blocked → user reports an outcome → simulation accepted → e-mail approved.

## Known limits
- **The PDF is synthetic.** The only form is a synthetic preparation sheet (`scripts/make_synthetic_form.py`);
  real official forms are not bundled.
- **The mock portal checks rules only.** It does not model office hours, capacity or human discretion.
