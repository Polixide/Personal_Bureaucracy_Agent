# `src/agent/` — Orchestrator-Workers with ReAct, Gate 1, fallback

## What it does
The **orchestrator** (an LLM in a ReAct loop of at most `MAX_STEPS` decisions) does four things:
- **decomposes** the goal into the pack's sections,
- **delegates** each section to a **worker** (a smaller ReAct loop of at most `WORKER_STEPS`, with fewer tools)
  through `delegate_section`,
- **synthesizes** one `FinalAnswer`,
- runs the answer through **Gate 1**, a deterministic verifier, before it is shown.

## Where it sits in the flow
`run_agent(goal, store, packs, user_id, today, backend)` is called by the app and by evaluation. It returns an
`AgentResult` and logs the run in the Twin. The LangGraph graph in `orchestrator.py` is:

```
START → decide → act → decide …                          (ReAct)
               act → verify → END                         (Gate 1 passed)
                     verify → decide                      (errors back to the model, ≤ MAX_REPAIRS)
                     verify → END                         (still wrong: needs_human_review flag)
decide/act → fallback → END                               (2 invalid steps in a row, or out of steps)
               act → END                                  (ask_user ended the turn)
```

## Inputs and outputs
**Decision**, one per step and schema-constrained (`action` is limited to the tool names):
```json
{"thought": "Check the passport section.", "action": "delegate_section", "args": {"section_id": "passport_photo"}}
```

**Finding**, from a worker. Its `rule_results` always come from `evaluate_rules`:
`{finding_id: "F-passport_photo", rule_results: [...], missing_items: [{field, question, rule_ids}], suspected_mistakes: [...]}`

**FinalAnswer**:
```json
{"summary": "Your passport validity is too short for the requested permit (R006).",
 "claims": [{"rule_id": "R006", "status": "fail"}],
 "next_actions": [{"title": "Renew your passport or ask IRO about a shorter period", "ref": "R006"}]}
```

## Why it is built this way
- **The LLM proposes, the code verifies.** Claims are compared with a *fresh* `evaluate_rules`. Every date must
  be a fact or a rule-derived date. Every action must cite a rule, finding or proposal. "Submitted / paid /
  approved" language is refused, and unverified rules must be labelled. The model's own reasoning text is
  never trusted.
- **Failure is normal for a 4B model.** Invalid decisions and arguments become error observations. Two in a
  row, a loop, or running out of steps switch to the **deterministic workflow** (`fallback.py`). That workflow
  evaluates every section, builds findings and a checklist answer, and passes Gate 1 by construction (tested
  on every pack scenario). It first runs the deterministic `search_procedures` on the goal, so a goal that
  clearly names another procedure is answered for that procedure; a tie or no match keeps the active one.
- **A repeated call gets a way out.** The error names the next section that still needs a worker
  (`delegate_section` with its id) or says to give the final answer. A 4B model tends to repeat a call
  rather than move on.
- **Workers have fewer tools.** They only read: `read_twin`, `evaluate_rules`, `extract_document`,
  `search_sources`. Writing statements, asking the user, simulating and proposing artifacts belong to the
  orchestrator.
- **Nothing is procedure-specific.** Sections, questions and rule ids all come from the pack.

## How to test it
```
python -m pytest -q tests/test_agent.py tests/test_gate1.py
```
These tests use `tests/fakes.py:ScriptedBackend`, a fake LLM that returns a fixed sequence of decisions. They
cover the happy path, invalid arguments, loops, a wrong claim caught and repaired, repair exhaustion,
`ask_user`, "no procedure", the step budget, and worker behaviour.

## Known limits
- **Gate 1 text checks are English patterns.** Structured claims carry most of the weight, but a cleverly
  phrased false sentence can still pass the text checks.
- **Some date formats are not checked.** Only ISO and DD.MM.YYYY / DD/MM/YYYY dates are checked; "15 March
  2027" is not detected.
- **Loop detection is simple.** It catches immediate repeats and a third identical call, not slower cycles.
