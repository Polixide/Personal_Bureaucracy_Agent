"""app.py: pure helpers, and headless runs of the real page with Streamlit's AppTest.

The agent is replaced by a fake (no model call); everything else (Twin, rules, sandbox, approval) is real.
"""
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

import app
import config
import src.agent.orchestrator as orchestrator
from src.agent.answer import AgentResult, FinalAnswer, NextAction
from src.pack import load_pack
from src.rules import evaluate_rules
from src.sandbox.mock_world import simulate
from src.twin import service
from src.twin.facts import build_view
from src.twin.store import TwinStore

IKAMET = load_pack("ikamet_first")
APP_FILE = str(config.PROJECT_ROOT / "app.py")


# ---------------------------------------------------------- pure helpers --
def test_save_upload_uses_content_names_inside_the_project():
    folder = config.PROJECT_ROOT / ".test-temp" / "uploads"
    path = app.save_upload("../../evil name.PDF", b"%PDF-1.4 synthetic", "u", folder)
    assert path.parent == folder.resolve() and path.suffix == ".pdf" and ".." not in path.name
    assert path == app.save_upload("other.pdf", b"%PDF-1.4 synthetic", "u", folder)


@pytest.mark.parametrize("name,content,user", [("a.exe", b"x", "u"), ("a.txt", b"", "u")])
def test_save_upload_rejects_bad_files(name, content, user):
    with pytest.raises(ValueError):
        app.save_upload(name, content, user, config.PROJECT_ROOT / ".test-temp" / "uploads")


def test_save_upload_refuses_folders_outside_the_project():
    outside = config.PROJECT_ROOT.parent / "outside_uploads"
    with pytest.raises(ValueError):
        app.save_upload("a.txt", b"x", "u", outside)


@pytest.mark.parametrize("message,expected", [
    ("RapidOCR failed on page 1", "scanned pages"), ("Local Ollama timed out during extraction", "timed out"),
    ("Ollama unavailable: connection refused", "not running"), ("PDF contains no extractable text", "No readable text"),
    ("something else", "could not finish")])
def test_friendly_errors(message, expected):
    assert expected in app.friendly_error(message)


@pytest.mark.parametrize("issue,expected", [
    ("coverage_end_date: value absent from reliable source candidates; discarded", "Could not read the coverage end date"),
    ("holder_name: conflicting values; manual review needed", "holder name is unclear"),
    ("Invalid source_pages discarded", "Invalid source_pages discarded")])
def test_extraction_issues_are_shown_in_plain_words(issue, expected):
    assert expected in app.friendly_issue(issue)


def test_follow_up_goal_keeps_the_original_goal_the_question_and_the_answer():
    text = app.follow_up_goal("I want a permit", "Which permit?", "the first one")
    assert text.splitlines() == ["I want a permit", "You asked: Which permit?", "My answer: the first one"]


def test_fact_rows_say_where_each_value_came_from():
    with TwinStore(":memory:") as store:
        service.record_app_value(store, IKAMET, "u", "age", 22)
        view = build_view(IKAMET, store.current_facts("u", IKAMET.id), [], date(2026, 11, 18))
    rows = {row["fact"]: row for row in app.fact_rows(view)}
    assert rows["age"]["from"] == "you" and rows["current date"]["from"] == "computed"


def test_rule_line_shows_status_missing_information_and_unverified_label():
    passport = app.in_section(evaluate_rules(IKAMET, {}), "passport_photo")
    assert [r.rule_id for r in passport] == ["R006", "R008", "R009"]
    line = app.rule_line(passport[0])
    assert line.startswith("❔") and "passport expiry date" in line
    unverified = passport[0].model_copy(update={"rule_status": "unverified"})
    assert "(unverified rule)" in app.rule_line(unverified)


def test_guardrail_sections_are_not_shown_as_user_progress():
    ids = [section.id for section in app.user_sections(IKAMET)]
    assert "guardrails" not in ids and ids[0] == "eligibility" and len(ids) == 6


def test_derived_fields_are_asked_through_their_inputs():
    asked = app.askable_fields(IKAMET, ["planned_stay_months", "student_certificate_present", "age"])
    assert asked == ["program_start_date", "program_end_date", "age"]


def test_documents_that_would_answer_missing_fields_are_suggested():
    suggested = app.documents_for(IKAMET, ["student_certificate_present", "insurance_end_date"])
    assert any("Student certificate" in text for text in suggested)
    assert "Health insurance policy or certificate" in suggested


def test_field_question_falls_back_to_a_short_question():
    assert app.field_question(IKAMET, "is_in_turkiye") == "Are you currently in Türkiye?"
    assert app.field_question(IKAMET, "nationality") == "Nationality"
    assert app.field_question(IKAMET, "entry_stamp_present") == "Entry stamp present?"


def test_open_item_count_adds_problems_questions_and_values_to_check():
    results = evaluate_rules(IKAMET, {})
    with TwinStore(":memory:") as store:
        empty = build_view(IKAMET, [], [], date(2026, 11, 18))
        service.record_statement(store, IKAMET, "u", "age", 22, confirmed=False, source_ref="chat")
        waiting = build_view(IKAMET, store.current_facts("u", IKAMET.id), [], date(2026, 11, 18))
    questions = len(app.askable_fields(IKAMET, app.missing_fields(app.user_results(IKAMET, results))))
    assert app.open_item_count(IKAMET, results, empty) == questions
    assert app.open_item_count(IKAMET, results, waiting) == questions + 1


def test_sandbox_refusal_is_explained_in_plain_words():
    outcome = simulate(IKAMET, {"current_date": date(2026, 11, 18)}, "official_application")
    lines = app.sandbox_lines(IKAMET, outcome)
    assert lines and all(line.startswith(("Problem:", "Missing for")) for line in lines)
    assert not any("R0" in line for line in lines)


# ------------------------------------------------------- headless page runs --
@pytest.fixture()
def page_env(tmp_path, monkeypatch):
    """A fresh Twin file and outbox for each page run."""
    database = tmp_path / "twin.sqlite"
    monkeypatch.setattr(config, "DATABASE_PATH", database)
    monkeypatch.setattr(config, "OUTBOX_DIR", tmp_path / "outbox")
    return database


def run_page() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=60).run()


def widget(widgets, label):
    """The first widget with this label."""
    return next(w for w in widgets if w.label == label)


def seed(database, **facts) -> None:
    with TwinStore(database) as store:
        store.set_active_procedure("demo-user", "ikamet_first")
        for field, value in facts.items():
            service.record_app_value(store, IKAMET, "demo-user", field, value)


def test_empty_twin_asks_for_a_goal_or_a_procedure(page_env):
    page = run_page()
    assert not page.exception
    assert "What do you want to do?" in [t.value for t in page.title]
    assert "Ask the agent" in page.info[0].value


def test_seeded_twin_shows_the_three_questions_as_tabs(page_env):
    seed(page_env, passport_expiry_date="2027-07-15")
    page = run_page()
    assert not page.exception
    assert [tab.label for tab in page.tabs] == app.TAB_NAMES
    assert any(e.label.endswith("Eligibility and legal stay · 0/4") for e in page.expander)


def test_answers_typed_in_the_app_become_confirmed_facts(page_env):
    seed(page_env)
    page = run_page()
    widget(page.text_input, "Permit type").input("short-term residence permit")
    widget(page.button, "Save answers").click()
    page.run()
    assert not page.exception
    with TwinStore(page_env) as store:
        facts = {f.field: f for f in store.current_facts("demo-user", IKAMET.id)}
    assert facts["permit_type"].value == "short-term residence permit" and facts["permit_type"].confirmed


def test_checklist_is_proposed_then_written_only_after_approval(page_env):
    seed(page_env)
    page = run_page()
    widget(page.button, "☑️ Prepare my checklist").click()
    page.run()
    with TwinStore(page_env) as store:
        [proposal] = store.list_proposals("demo-user")
    assert proposal.status == "proposed" and not (config.OUTBOX_DIR / "demo-user").exists()
    widget(page.button, "Approve").click()
    page.run()
    assert not page.exception
    with TwinStore(page_env) as store:
        [proposal] = store.list_proposals("demo-user")
    assert proposal.status == "executed" and list((config.OUTBOX_DIR / "demo-user").iterdir())


def test_sandbox_button_stores_a_simulation_and_explains_the_refusal(page_env):
    seed(page_env)
    page = run_page()
    widget(page.button, "Try this step").click()
    page.run()
    assert not page.exception
    assert any("would refuse" in warning.value for warning in page.warning)
    with TwinStore(page_env) as store:
        assert store.list_simulations("demo-user", IKAMET.id)[-1].accepted is False


def test_agent_question_is_answered_with_the_original_goal(page_env, monkeypatch):
    goals = []

    def fake_run_agent(goal, store, packs, user_id, today, backend, outbox_dir):
        """Stands in for the real agent: asks once, then answers."""
        goals.append(goal)
        if len(goals) == 1:
            return AgentResult(trace_id="t1", status="needs_user_input", question="Which procedure?")
        answer = FinalAnswer(summary="Here is where you are.",
                             next_actions=[NextAction(title="Upload passport", ref="R006")])
        return AgentResult(trace_id="t2", status="answered", answer=answer)

    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    page = run_page()
    widget(page.button, "Ask the agent").click()
    page.run()
    assert "Which procedure?" in page.info[0].value
    widget(page.text_input, "Your answer").input("the first permit")
    widget(page.button, "Send").click()
    page.run()
    assert not page.exception
    assert goals[1] == app.follow_up_goal("I want to apply for İkamet", "Which procedure?", "the first permit")
    assert "Here is where you are." in [m.value for m in page.markdown]
