"""app.py: pure helpers, and headless runs of the real page with Streamlit's AppTest.

The agent is replaced by a fake (no model call); everything else (Twin, rules, sandbox, approval) is real.
"""
from datetime import date, datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

import app
import config
import src.agent.chat as chat
import src.agent.orchestrator as orchestrator
from src.agent.briefing import build_briefing
from src.agent.chat import ChatReply
from src.agent.answer import AgentResult, FinalAnswer, NextAction
from src.pack import load_all_packs, load_pack
from src.rules import evaluate_rules
from src.sandbox.mock_world import simulate
from src.twin import service
from src.twin.facts import build_view
from src.twin.models import ChatMessage, RunRecord, RunRuleResult, TwinDocument
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


def test_step_rows_hide_an_empty_thought_column():
    steps = [{"step": 1, "thought": "", "action": "read_twin", "ok": True, "error": None},
             {"step": 2, "thought": "", "action": "evaluate_rules", "ok": False, "error": "repeated call"}]
    rows = app.step_rows({"steps": steps})
    assert list(rows[0]) == ["step", "tool", "ok"] and rows[1]["ok"] == "❌ repeated call"
    steps[0]["thought"] = "Look at the Twin."
    assert list(app.step_rows({"steps": steps})[0]) == ["step", "thought", "tool", "ok"]


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


def test_open_item_count_is_the_number_of_plan_steps_and_grows_with_a_value_to_check():
    # The status bar counts what the To do view lists: the steps of the plan (it used to count every askable field,
    # also those a document upload answers, so the number did not match the list).
    results = evaluate_rules(IKAMET, {})
    with TwinStore(":memory:") as store:
        empty = build_view(IKAMET, [], [], date(2026, 11, 18))
        service.record_statement(store, IKAMET, "u", "age", 22, confirmed=False, source_ref="chat")
        waiting = build_view(IKAMET, store.current_facts("u", IKAMET.id), [], date(2026, 11, 18))
    first = build_briefing(IKAMET, [], [], empty, results)
    second = build_briefing(IKAMET, [], [], waiting, results)
    assert app.open_item_count(first) == len(first.steps)
    assert app.open_item_count(second) == app.open_item_count(first) + 1


def test_sandbox_refusal_is_explained_in_plain_words():
    outcome = simulate(IKAMET, {"current_date": date(2026, 11, 18)}, "official_application")
    lines = app.sandbox_lines(IKAMET, outcome)
    assert lines and all(line.startswith(("Problem:", "Missing for")) for line in lines)
    assert not any("R0" in line for line in lines)



def test_document_checklist_follows_the_twin():
    with TwinStore(":memory:") as store:
        empty = build_view(IKAMET, [], [], date(2026, 11, 18))
        rows = {row["doc_type"]: row["status"] for row in app.checklist_rows(IKAMET, [], empty)}
        assert set(rows.values()) == {"missing"} and len(rows) == len(IKAMET.procedure.documents)
        service.register_upload(store, "u", "doc_1", "x.pdf", IKAMET.id)
        documents = store.list_documents("u")
    pending = app.document_status(IKAMET, [d.model_copy(update={"doc_type": "passport_identity_page"}) for d in documents],
                                  empty, "passport_identity_page")
    assert pending == "ambiguous"


def test_run_labels_are_plain_words():
    assert app.run_label(None) == "Not started"
    assert app.run_label("fallback") == "Answer built from the rules alone"


# ----------------------------------------------------- one run per profile --
def test_second_action_is_refused_while_a_run_holds_the_lock():
    calls = []
    lock = app.profile_lock("guard-busy")
    assert lock.acquire(blocking=False)                 # another tab is mid-run
    try:
        assert app.run_locked("guard-busy", "working", lambda: calls.append(1)) is False
        assert calls == []
    finally:
        lock.release()


def test_lock_is_released_after_success_and_after_failure():
    assert app.run_locked("guard-free", "working", lambda: None) is True
    assert not app.profile_lock("guard-free").locked()

    def failing():
        raise RuntimeError("model down")

    assert app.run_locked("guard-free", "working", failing) is False
    assert not app.profile_lock("guard-free").locked()


def test_profiles_do_not_block_each_other():
    lock = app.profile_lock("guard-a")
    assert lock.acquire(blocking=False)
    try:
        assert app.profile_lock("guard-a") is lock      # same profile, same lock
        assert not app.profile_lock("guard-b").locked()
    finally:
        lock.release()


# ------------------------------------------------------- headless page runs --
@pytest.fixture()
def page_env(tmp_path, monkeypatch):
    """A fresh Twin file and outbox for each page run."""
    database = tmp_path / "twin.sqlite"
    monkeypatch.setattr(config, "DATABASE_PATH", database)
    monkeypatch.setattr(config, "OUTBOX_DIR", tmp_path / "outbox")
    return database


def run_page(view: str | None = None) -> AppTest:
    page = AppTest.from_file(APP_FILE, default_timeout=60)
    if view:
        page.session_state["view"] = view
    return page.run()


def widget(widgets, label):
    """The first widget with this label."""
    return next(w for w in widgets if w.label == label)


def seed(database, procedure_id: str = "ikamet_first", **facts) -> None:
    pack = load_pack(procedure_id)
    with TwinStore(database) as store:
        store.set_active_procedure("demo-user", procedure_id)
        for field, value in facts.items():
            service.record_app_value(store, pack, "demo-user", field, value)


def run_with_provider(provider: str) -> AppTest:
    page = AppTest.from_file(APP_FILE, default_timeout=60)
    page.session_state["provider"] = provider
    return page.run()


def test_openai_without_its_key_warns_and_the_local_model_answers(page_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    page = run_with_provider("openai")
    assert not page.exception
    assert any("OPENAI_API_KEY" in warning.value for warning in page.warning)
    assert page.session_state["provider_in_use"] == "ollama"


def test_openai_with_its_key_runs_the_agent_and_offers_cloud_document_reading(page_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    page = run_with_provider("openai")
    assert not page.exception
    assert page.session_state["provider_in_use"] == "openai"
    assert any(toggle.label == "The cloud model also reads my documents" for toggle in page.toggle)
    assert any("OpenAI, cloud" in caption.value for caption in page.caption)


def test_empty_twin_opens_the_chat_with_a_welcome(page_env):
    page = run_page()
    assert not page.exception
    assert "What do you want to do?" in [t.value for t in page.title]
    assert any("Tell me what you need" in m.value for m in page.markdown)
    assert len(page.chat_input) == 1


def test_the_journey_shows_where_you_are_and_details_list_every_requirement(page_env):
    seed(page_env, passport_expiry_date="2027-07-15")
    page = run_page()
    assert not page.exception
    first_stage = IKAMET.procedure.stages[0].title
    assert any(first_stage in m.value and "you are here" in m.value for m in page.markdown)
    details = run_page("details")
    assert not details.exception
    assert any("Eligibility and legal stay · 0/4" in m.value for m in details.markdown)
    assert any("Official sources" in m.value for m in details.markdown)


@pytest.mark.parametrize("procedure_id", sorted(load_all_packs()))
def test_every_view_of_every_procedure_renders(page_env, procedure_id):
    seed(page_env, procedure_id)
    for view in app.VIEWS:
        page = run_page(view)
        assert not page.exception, (procedure_id, view, page.exception)


SKIP_LABELS = ("I don't have it yet", "I don't know yet", "Skip for now")


def test_the_now_box_asks_one_thing_at_a_time_and_a_click_answers_it(page_env):
    seed(page_env)
    page = run_page()
    for _ in range(12):        # the documents come first: skip them until the first question
        if any(button.label == "Yes" for button in page.button):
            break
        next(button for button in page.button if button.label in SKIP_LABELS).click()
        page.run()
    assert any("Did you enter Türkiye with your passport?" in m.value for m in page.markdown)
    widget(page.button, "Yes").click()
    page.run()
    assert not page.exception
    with TwinStore(page_env) as store:
        facts = {f.field: f for f in store.current_facts("demo-user", IKAMET.id)}
    assert facts["entered_with_passport"].value is True and facts["entered_with_passport"].confirmed


def test_an_approved_e_mail_opens_in_gmail_already_filled_in(page_env):
    seed(page_env)
    with TwinStore(page_env) as store:
        ctx = app.tool_context(store, load_all_packs(), "demo-user", date(2026, 11, 18))
        assert app.run_tool(app.build_registry(), "draft_email", {
            "to_role": "IRO", "subject": "My residence permit file",
            "body": "Dear IRO, could you tell me which documents are still missing?"}, ctx, "orchestrator").ok
    page = run_page()
    assert any("E-mail draft to" in m.value for m in page.markdown)          # the Now box asks for approval
    widget(page.button, "Approve").click()
    page.run()
    assert not page.exception
    links = page.get("link_button")
    assert links and "mail.google.com" in links[0].proto.url and "irolegal%40sabanciuniv.edu" in links[0].proto.url


def test_checklist_is_proposed_then_written_only_after_approval(page_env):
    seed(page_env)
    page = run_page()
    widget(page.button, "Checklist of my file").click()      # in "The agent can prepare for you"
    page.run()
    with TwinStore(page_env) as store:
        [proposal] = store.list_proposals("demo-user")
    assert proposal.status == "proposed" and not (config.OUTBOX_DIR / "demo-user").exists()
    assert any("Waiting for your decision" in m.value for m in page.markdown)
    assert any("# Checklist:" in m.value for m in page.markdown)      # the preview shows what approving writes
    widget(page.button, "Approve").click()
    page.run()
    assert not page.exception
    with TwinStore(page_env) as store:
        [proposal] = store.list_proposals("demo-user")
    assert proposal.status == "executed" and list((config.OUTBOX_DIR / "demo-user").iterdir())


def test_sandbox_button_stores_a_simulation_and_explains_the_refusal(page_env):
    seed(page_env)
    page = run_page("details")
    widget(page.button, "Try this step").click()
    page.run()
    assert not page.exception
    assert any("would refuse" in warning.value for warning in page.warning)
    with TwinStore(page_env) as store:
        assert store.list_simulations("demo-user", IKAMET.id)[-1].accepted is False


def fake_chat(calls):
    """Stands in for the chat assistant: stores the turn like the real one, without a model."""
    def run(message, store, packs, user_id, today, backend, outbox_dir=None, embedder=None, progress=None, **kwargs):
        calls.append(message)
        if progress:
            progress("Writing the answer")
        store.add_chat_message(ChatMessage(user_id=user_id, role="user", content=message))
        reply = ChatReply(text=f"Answer to: {message}", status="answered", trace_id=f"t{len(calls)}",
                          sources=[{"source_id": "GOV_FAQ", "title": "Official FAQ", "url": "https://example.org"}])
        store.add_chat_message(ChatMessage(user_id=user_id, role="assistant", content=reply.text,
                                           details=reply.model_dump(exclude={"text"})))
        return reply
    return run


def test_the_plan_opens_the_chat_and_a_message_gets_an_answer(page_env, monkeypatch):
    calls = []
    monkeypatch.setattr(chat, "run_chat", fake_chat(calls))
    seed(page_env, passport_expiry_date="2027-07-15")
    page = run_page()
    assert any("Now" in m.value for m in page.markdown)
    page.chat_input[0].set_value("When does my passport expire?").run()
    assert not page.exception and calls == ["When does my passport expire?"]
    assert "Answer to: When does my passport expire?" in [m.value for m in page.markdown]
    assert any("Official FAQ" in c.value for c in page.caption)
    assert not app.profile_lock("demo-user").locked()


def test_a_suggestion_is_sent_with_one_click(page_env, monkeypatch):
    calls = []
    monkeypatch.setattr(chat, "run_chat", fake_chat(calls))
    seed(page_env)
    page = run_page()
    widget(page.button, app.SUGGESTIONS[0]).click()
    page.run()
    assert not page.exception and calls == [app.SUGGESTIONS[0]]


def test_a_fact_from_the_chat_is_confirmed_with_its_button(page_env):
    seed(page_env)
    pack = load_pack("ikamet_first")
    with TwinStore(page_env) as store:
        fact = service.record_statement(store, pack, "demo-user", "permit_type", "short-term residence permit",
                                        confirmed=False)
        store.add_chat_message(ChatMessage(user_id="demo-user", role="assistant", content="Noted.",
                                           details={"status": "answered", "facts": [
                                               {"fact_id": fact.fact_id, "field": "permit_type", "value": fact.value}]}))
    page = run_page()
    widget(page.button, "Confirm").click()
    page.run()
    assert not page.exception
    with TwinStore(page_env) as store:
        facts = {f.field: f for f in store.current_facts("demo-user", pack.id)}
    assert facts["permit_type"].confirmed


def test_full_check_by_the_agent_team_is_posted_in_the_chat(page_env, monkeypatch):
    goals = []

    def fake_run_agent(goal, store, packs, user_id, today, backend, outbox_dir, **kwargs):
        goals.append(goal)
        answer = FinalAnswer(summary="Here is where you are.", next_actions=[NextAction(title="Upload passport", ref="R006")])
        return AgentResult(trace_id="t2", status="answered", answer=answer)

    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    seed(page_env)
    page = run_page()
    widget(page.button, "Run a full check").click()
    page.run()
    assert not page.exception and goals == [app.FULL_CHECK_GOAL]
    assert any("Here is where you are." in m.value and "1. Upload passport" in m.value for m in page.markdown)


def test_reading_summary_says_what_each_document_of_an_upload_gave():
    pack = load_pack("tc_kimlik_renewal")
    card = TwinDocument(doc_id="d", file_path="x.pdf", status="analyzed", doc_type="tc_kimlik_karti",
                        observation={"pages": [1, 2], "found_by": "mrz", "facts": [
                            {"field": "birth_date", "requires_confirmation": False},
                            {"field": "holder_name", "requires_confirmation": True}]})
    unknown = TwinDocument(doc_id="d_part2", file_path="x.pdf", status="analyzed", doc_type="unknown",
                           observation={"pages": [3]})
    text = app.reading_summary(pack, "scan.pdf", [card, unknown])
    assert "found 2 documents" in text and "pages 1–2" in text and "machine-readable zone" in text
    assert "2 values read, 1 to confirm" in text and "page 3: I could not tell" in text


def test_technical_view_shows_the_run_log_with_its_rule_results(page_env):
    seed(page_env)
    results = [RunRuleResult(rule_id=r.rule_id, status=r.status, severity=r.severity)
               for r in evaluate_rules(IKAMET, {})]
    with TwinStore(page_env) as store:
        store.log_run(RunRecord(trace_id="t9", user_id="demo-user", procedure_id=IKAMET.id, goal="g",
                                status="answered", started_at=datetime.now(timezone.utc),
                                finished_at=datetime.now(timezone.utc), results=results))
    page = run_page("details")
    assert not page.exception
    assert any("Run log" in m.value for m in page.markdown)
    assert app.run_passed(RunRecord(trace_id="x", user_id="u", procedure_id=None, goal="g", status="answered",
                                    started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
                                    results=results)) == f"0/{len(results)}"


def test_a_question_used_by_several_sections_is_asked_once():
    pack = load_pack("tc_kimlik_renewal")
    view = build_view(pack, [], [], date(2026, 11, 18))
    briefing = build_briefing(pack, [], [], view, evaluate_rules(pack, view.fact_map()))
    refs = [step.ref for step in briefing.steps if step.kind == "answer"]
    assert "renewal_reason" in refs and len(refs) == len(set(refs))
