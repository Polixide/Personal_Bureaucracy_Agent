"""Episodic memory: validated repairs are stored, retrieved by procedure + failure type, shown to the agent."""
from agent_helpers import PASSPORT_TOO_SHORT, USER, correct_final, d, run, seed
from src.memory.episodes import lessons_for
from src.twin import service

WRONG = d("final_answer", {"summary": "Your passport is fine.", "claims": [{"rule_id": "R006", "status": "pass"}],
                           "next_actions": [{"title": "Renew your passport", "ref": "R006"}]})


def test_a_repaired_answer_stores_a_generic_validated_lesson(store):
    seed(store, {**PASSPORT_TOO_SHORT, "holder_name": "Alex Example"})
    result, _ = run(store, [WRONG, correct_final()])
    assert result.status == "answered" and result.repairs == 1
    episodes = store.find_episodes("ikamet_first")
    assert [(e.failure_type, e.affected_item) for e in episodes] == [("CLAIM_MISMATCH", "R006")]
    episode = episodes[0]
    assert episode.correction_validated and episode.rule_version == "1.0" and episode.trace_id == result.trace_id
    stored_text = episode.validator_error + episode.corrective_action + episode.outcome
    assert "Alex" not in stored_text and "2027" not in stored_text


def test_no_lesson_without_a_validated_repair(store):
    seed(store, PASSPORT_TOO_SHORT)
    run(store, [correct_final()])
    run(store, [WRONG] * 4)
    assert store.find_episodes("ikamet_first") == []


def test_lessons_are_shown_in_the_next_run_and_during_repair(store):
    seed(store, PASSPORT_TOO_SHORT)
    run(store, [WRONG, correct_final()])
    _, backend = run(store, [WRONG, correct_final()])
    first_prompt = backend.calls[0]["messages"][1]["content"]
    assert "LESSONS FROM PAST REPAIRS" in first_prompt and "CLAIM_MISMATCH" in first_prompt
    repair_prompt = backend.calls[1]["messages"][-1]["content"]
    assert "LESSONS FROM PAST REPAIRS OF THE SAME PROBLEMS" in repair_prompt


def test_lessons_are_filtered_by_failure_type_and_deduplicated(store):
    seed(store, PASSPORT_TOO_SHORT)
    run(store, [WRONG, correct_final()])
    run(store, [WRONG, correct_final()])
    assert len(lessons_for(store, "ikamet_first")) == 1
    assert lessons_for(store, "ikamet_first", ["UNSUPPORTED_DATE"]) == []
    assert lessons_for(store, None) == []


def test_a_rule_update_retires_its_lessons(store):
    seed(store, PASSPORT_TOO_SHORT)
    run(store, [WRONG, correct_final()])
    service.record_rule_update(store, USER, "ikamet_first", "R006", "1.1", "buffer changed")
    assert lessons_for(store, "ikamet_first") == []
