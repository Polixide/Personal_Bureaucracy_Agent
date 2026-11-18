"""The agent with a scripted fake LLM that returns a fixed sequence of Decisions.

Covers: happy path (decompose, delegate, synthesize, verify), invalid tool arguments, loops,
consecutive failures -> deterministic fallback, a wrong claim caught by Gate 1 and repaired,
repairs exhausted -> human review flag, ask_user, no matching procedure, step budget.
"""
from agent_helpers import GOAL, PASSPORT_TOO_SHORT, USER, correct_final, d, run, seed


def test_happy_path_decomposes_delegates_and_synthesizes(store):
    seed(store, PASSPORT_TOO_SHORT, active=False)
    replies = [d("search_procedures", {"query": GOAL}), d("read_twin"),
               d("delegate_section", {"section_id": "passport_photo", "focus": "passport validity"}),
               d("finish", {"summary": "The passport ends too early.", "suspected_mistakes": []}),
               correct_final([{"title": "Answer the open passport questions", "ref": "F-passport_photo"}])]
    result, backend = run(store, replies)
    assert result.status == "answered" and result.repairs == 0 and not result.gate_issues
    assert result.procedure_id == "ikamet_first"
    assert [s.action for s in result.steps] == ["search_procedures", "read_twin", "delegate_section", "final_answer"]
    finding = result.findings[0]
    assert finding.finding_id == "F-passport_photo" and not finding.fallback_used
    assert {r.rule_id: r.status for r in finding.rule_results}["R006"] == "fail"
    assert result.llm_calls == 5 and [c["component"] for c in backend.calls].count("worker") == 1
    assert store.list_runs(USER)[0].status == "answered"


def test_invalid_tool_arguments_are_returned_to_the_model(store):
    seed(store, PASSPORT_TOO_SHORT)
    replies = [d("record_fact", {"field": "latest_entry_date"}),
               d("record_fact", {"field": "latest_entry_date", "value": "2026-09-15"}), correct_final()]
    result, backend = run(store, replies)
    assert result.status == "answered" and result.invalid_calls == 1
    assert "Invalid arguments" in backend.calls[1]["messages"][-1]["content"]
    assert [s.ok for s in result.steps] == [False, True, True]


def test_two_consecutive_invalid_decisions_fall_back_to_the_workflow(store):
    seed(store, PASSPORT_TOO_SHORT)
    result, backend = run(store, ["not json", "still not json", "{}", "[]"])
    assert result.status == "fallback" and result.fallback_used and not result.needs_human_review
    assert result.llm_calls == 4 and not result.gate_issues
    assert {c.rule_id: c.status for c in result.answer.claims}["R006"] == "fail"
    assert result.findings and all(f.fallback_used for f in result.findings)


def test_repeated_calls_are_detected_as_a_loop(store):
    seed(store, PASSPORT_TOO_SHORT)
    result, backend = run(store, [d("read_twin"), d("read_twin"), d("read_twin")])
    assert result.loops_detected == 2 and result.status == "fallback"
    assert "already called read_twin" in backend.calls[2]["messages"][-1]["content"]


def test_a_repeated_call_is_answered_with_the_next_section_to_delegate(store):
    seed(store, PASSPORT_TOO_SHORT)
    result, backend = run(store, [d("evaluate_rules"), d("evaluate_rules"), correct_final()])
    hint = backend.calls[2]["messages"][-1]["content"]
    assert 'delegate_section with section_id "eligibility"' in hint and result.loops_detected == 1


def test_fallback_follows_a_goal_that_names_another_procedure(store):
    seed(store, {}, procedure_id="ikamet_first")
    result, _ = run(store, [d("no_such_tool"), d("no_such_tool")], goal="I want to extend my İkamet")
    assert result.status == "fallback" and result.procedure_id == "ikamet_extension"
    assert "unverified" in result.answer.summary
    assert store.active_procedure(USER) == "ikamet_extension"


def test_fallback_keeps_the_active_procedure_when_the_goal_names_none(store):
    seed(store, {}, procedure_id="ikamet_first")
    result, _ = run(store, [d("no_such_tool"), d("no_such_tool")], goal="What should I do now?")
    assert result.status == "fallback" and result.procedure_id == "ikamet_first"


def test_wrong_claim_is_caught_by_gate_1_and_repaired(store):
    seed(store, PASSPORT_TOO_SHORT)
    wrong = d("final_answer", {"summary": "Your passport is fine.", "claims": [{"rule_id": "R006", "status": "pass"}],
                               "next_actions": [{"title": "Upload your insurance", "ref": "R015"}]})
    result, backend = run(store, [wrong, correct_final()])
    assert result.status == "answered" and result.repairs == 1 and result.first_answer_rejected
    repair = backend.calls[1]["messages"][-1]["content"]
    assert "GATE 1 REJECTED" in repair and "CLAIM_MISMATCH" in repair
    assert result.answer.claims[0].status == "fail"


def test_answer_still_wrong_after_three_repairs_is_flagged_for_human_review(store):
    seed(store, PASSPORT_TOO_SHORT)
    wrong = d("final_answer", {"summary": "Your application was submitted.", "claims": [{"rule_id": "R006", "status": "pass"}]})
    result, _ = run(store, [wrong] * 4)
    assert result.status == "needs_human_review" and result.needs_human_review and result.repairs == 3
    codes = {issue.code for issue in result.gate_issues}
    assert {"CLAIM_MISMATCH", "COMPLETION_CLAIM"} <= codes
    assert result.answer is not None


def test_ask_user_ends_the_turn_with_a_question(store):
    result, _ = run(store, [d("search_procedures", {"query": GOAL}),
                            d("ask_user", {"question": "When did you enter Türkiye?", "fields": ["latest_entry_date"]})])
    assert result.status == "needs_user_input" and result.question == "When did you enter Türkiye?"
    assert result.answer is None


def test_no_matching_procedure_is_said_clearly(store):
    replies = [d("search_procedures", {"query": "open a bank account"}),
               d("final_answer", {"summary": "No verified procedure is available for opening a bank account."})]
    result, _ = run(store, replies, goal="I want to open a bank account")
    assert result.status == "answered" and result.procedure_id is None and result.answer.claims == []


def test_running_out_of_steps_falls_back(store):
    seed(store, PASSPORT_TOO_SHORT)
    replies = [d("read_twin", {"fields": [f"field_{i}"]}) for i in range(8)]
    result, _ = run(store, replies)
    assert result.status == "fallback" and len(result.steps) == 8


def test_final_answer_with_invalid_shape_is_an_invalid_step(store):
    seed(store, PASSPORT_TOO_SHORT)
    too_many = [{"title": f"Action number {i}", "ref": "R006"} for i in range(6)]
    result, backend = run(store, [d("final_answer", {"summary": "x", "claims": [], "next_actions": too_many}), correct_final()])
    assert result.status == "answered" and result.invalid_calls == 1
    assert "final_answer args invalid" in backend.calls[1]["messages"][-1]["content"]


def test_action_refs_must_point_to_rules_or_findings_of_this_run(store):
    seed(store, PASSPORT_TOO_SHORT)
    invented = correct_final([{"title": "Read the finding", "ref": "F-passport_photo"}])
    result, backend = run(store, [invented, correct_final()])
    assert result.repairs == 1 and "UNGROUNDED_ACTION" in backend.calls[1]["messages"][-1]["content"]


# ---------------------------------------------------------------- workers --
def test_worker_cannot_change_verdicts(store):
    seed(store, PASSPORT_TOO_SHORT)
    replies = [d("delegate_section", {"section_id": "passport_photo"}),
               d("finish", {"summary": "R006 passes.", "suspected_mistakes": ["Photo may be old"]}), correct_final()]
    result, _ = run(store, replies)
    finding = result.findings[0]
    assert {r.rule_id: r.status for r in finding.rule_results}["R006"] == "fail"
    assert "Photo may be old" in finding.suspected_mistakes
    assert any(m.startswith("R006 fails") for m in finding.suspected_mistakes)
    assert {item.field for item in finding.missing_items} >= {"photo_age_months", "passport_id_page_present"}


def test_worker_that_fails_twice_returns_a_deterministic_finding(store):
    seed(store, PASSPORT_TOO_SHORT)
    replies = [d("delegate_section", {"section_id": "passport_photo"}), "bad", "bad", "bad", "bad", correct_final()]
    result, _ = run(store, replies)
    finding = result.findings[0]
    assert finding.fallback_used and finding.llm_calls == 4 and result.status == "answered"


def test_worker_uses_tools_then_finishes(store):
    seed(store, PASSPORT_TOO_SHORT)
    replies = [d("delegate_section", {"section_id": "passport_photo"}), d("read_twin"), d("record_fact", {"field": "age", "value": 3}),
               d("finish", {"summary": "Checked.", "suspected_mistakes": []}), correct_final()]
    result, backend = run(store, replies)
    worker_messages = backend.calls[3]["messages"]
    assert "Unknown tool 'record_fact'" in worker_messages[-1]["content"]  # workers only see their own tools
    assert result.findings[0].steps == 3 and not result.findings[0].fallback_used


def test_unknown_section_is_an_error_observation(store):
    seed(store, PASSPORT_TOO_SHORT)
    result, backend = run(store, [d("delegate_section", {"section_id": "visa"}), correct_final()])
    assert "Unknown section" in backend.calls[1]["messages"][-1]["content"] and result.status == "answered"
