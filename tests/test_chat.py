"""The chat assistant (src/agent/chat.py) and the briefing it starts from (src/agent/briefing.py).

A scripted model plays the assistant; the checks are the code's: Gate 1 on every reply, facts stated in the
chat stay unconfirmed, proposals wait for approval, the conversation is kept, the plan is the fallback.
"""
from datetime import date

from agent_helpers import PASSPORT_TOO_SHORT, USER, d, seed
from fakes import ScriptedBackend
from src.agent.briefing import build_briefing, briefing_markdown
from src.agent.chat import UNVERIFIED_NOTE, run_chat
from src.pack import load_all_packs
from src.rules import evaluate_rules
from src.twin import service

TODAY = date(2026, 11, 18)
PACKS = load_all_packs()


def chat(store, replies, message="What should I do next?", tmp_path=None):
    backend = ScriptedBackend(replies)
    extra = {"outbox_dir": tmp_path / "outbox"} if tmp_path is not None else {}
    return run_chat(message, store, PACKS, USER, TODAY, backend, **extra), backend


def briefing_for(store, procedure_id="ikamet_first"):
    pack = PACKS[procedure_id]
    view = service.twin_view(store, pack, USER, TODAY)
    return build_briefing(pack, store.list_documents(USER), store.list_proposals(USER), view,
                          evaluate_rules(pack, view.fact_map()))


# ------------------------------------------------------------ briefing --
def test_briefing_puts_blocking_problems_before_questions(store):
    seed(store, PASSPORT_TOO_SHORT)
    briefing = briefing_for(store)
    kinds = [step.kind for step in briefing.steps]
    assert briefing.where == "Some requirements are not met yet"
    assert kinds[0] == "fix" and briefing.steps[0].ref == "R006"
    assert kinds.index("fix") < kinds.index("upload") < kinds.index("answer")
    assert len({step.ref for step in briefing.steps if step.kind == "answer"}) == kinds.count("answer")


def test_briefing_upload_step_replaces_the_questions_the_document_answers(store):
    seed(store, {}, procedure_id="tc_kimlik_renewal")
    briefing = briefing_for(store, "tc_kimlik_renewal")
    uploads = [step for step in briefing.steps if step.kind == "upload"]
    answers = {step.ref for step in briefing.steps if step.kind == "answer"}
    assert any(step.ref == "tc_kimlik_karti" for step in uploads)
    assert "birth_date" not in answers            # the card answers it
    assert "renewal_reason" in answers


def test_briefing_markdown_says_unverified_and_lists_the_plan(store):
    seed(store, {}, procedure_id="tc_kimlik_renewal")
    text = briefing_markdown(briefing_for(store, "tc_kimlik_renewal"))
    assert "What to do next" in text and "unverified" in text and "1." in text


# ---------------------------------------------------------------- chat --
def test_a_question_is_answered_in_one_call_and_kept_in_the_conversation(store):
    seed(store, PASSPORT_TOO_SHORT)
    reply, backend = chat(store, [d("final_answer", {
        "summary": "Your passport ends too early for the permit you asked for, so this blocks the application.",
        "claims": [{"rule_id": "R006", "status": "fail"}],
        "next_actions": [{"title": "Renew your passport, or ask for a shorter permit", "ref": "R006"}]})])
    assert reply.status == "answered" and len(backend.calls) == 1
    intro = backend.calls[0]["messages"][1]["content"]
    assert "STATE:" in intro and "R006" in intro and "MESSAGE: What should I do next?" in intro
    assert "1. Renew your passport" in reply.text
    assert [m.role for m in store.list_chat(USER)] == ["user", "assistant"]
    assert store.list_runs(USER)[-1].status == "answered"


def test_a_reply_with_a_wrong_status_is_repaired_then_replaced_by_the_plan(store):
    seed(store, PASSPORT_TOO_SHORT)
    wrong = d("final_answer", {"summary": "Everything is fine.", "claims": [{"rule_id": "R006", "status": "pass"}]})
    reply, backend = chat(store, [wrong, wrong])
    assert len(backend.calls) == 2 and "GATE 1 REJECTED" in backend.calls[1]["messages"][-1]["content"]
    assert reply.status == "fallback" and "What to do next" in reply.text
    assert "Everything is fine" not in reply.text


def test_unverified_label_is_added_by_code(store):
    seed(store, {}, procedure_id="tc_kimlik_renewal")
    reply, backend = chat(store, [d("final_answer", {"summary": "First tell me why you need a new card."})])
    assert reply.status == "answered" and len(backend.calls) == 1
    assert UNVERIFIED_NOTE in reply.text


def test_a_stated_fact_is_recorded_unconfirmed_and_shown_for_confirmation(store):
    seed(store, {}, procedure_id="tc_kimlik_renewal")
    replies = [d("record_fact", {"field": "renewal_reason", "value": "lost"}),
               d("final_answer", {"summary": "Thanks. Please confirm the reason, then report the loss."})]
    reply, _ = chat(store, replies, message="I lost my card last week")
    assert reply.status == "answered"
    assert [(fact["field"], fact["value"]) for fact in reply.facts] == [("renewal_reason", "lost")]
    view = service.twin_view(store, PACKS["tc_kimlik_renewal"], USER, TODAY)
    assert "renewal_reason" not in view.values       # unconfirmed: it does not count yet


def test_asking_for_a_checklist_creates_a_proposal_not_a_file(store, tmp_path):
    seed(store, PASSPORT_TOO_SHORT)
    replies = [d("build_checklist"),
               d("final_answer", {"summary": "I prepared a checklist. Approve it above to download it.",
                                  "claims": [{"rule_id": "R006", "status": "fail"}]})]
    reply, _ = chat(store, replies, message="Can you make me a checklist?", tmp_path=tmp_path)
    assert reply.status == "answered" and len(reply.proposals) == 1
    [proposal] = store.list_proposals(USER)
    assert proposal.status == "proposed" and not (tmp_path / "outbox").exists()


def test_without_the_model_the_reply_is_the_plan(store):
    seed(store, PASSPORT_TOO_SHORT)
    reply, _ = chat(store, ["not json", "still not json", "{}", "[]"])
    assert reply.status == "fallback" and "What to do next" in reply.text and not reply.gate_issues


def test_a_goal_without_active_procedure_selects_one(store):
    reply, _ = chat(store, [d("final_answer", {"summary": "Let us start with the reason for the new card."})],
                    message="I lost my Turkish ID card (TC kimlik)")
    assert reply.procedure_id == "tc_kimlik_renewal" and store.active_procedure(USER) == "tc_kimlik_renewal"


def test_previous_messages_reach_the_model(store):
    seed(store, PASSPORT_TOO_SHORT)
    chat(store, [d("final_answer", {"summary": "Your passport ends too early.",
                                    "claims": [{"rule_id": "R006", "status": "fail"}]})], message="First question")
    _, backend = chat(store, [d("final_answer", {"summary": "As I said, renew the passport first."})],
                      message="Second question")
    intro = backend.calls[0]["messages"][1]["content"]
    assert "CONVERSATION SO FAR" in intro and "First question" in intro and "Your passport ends too early." in intro


def test_dates_quoted_from_a_given_passage_are_allowed(store, monkeypatch):
    from src.rag.index import Hit
    seed(store, PASSPORT_TOO_SHORT)
    hit = Hit(source_id="GOV_FAQ", title="FAQ", chunk_id="GOV_FAQ#0", text="Rules changed on 01.06.2024.", score=0.9)
    monkeypatch.setattr("src.agent.chat.search", lambda *args, **kwargs: [hit])
    reply, _ = chat(store, [d("final_answer", {"summary": "These rules apply since 01.06.2024 (source: GOV_FAQ).",
                                               "claims": [{"rule_id": "R006", "status": "fail"}]})],
                    message="Since when do these rules apply?")
    assert reply.status == "answered"
    assert any(source["source_id"] == "GOV_FAQ" for source in reply.sources)


def test_a_fact_the_user_did_not_write_is_refused(store):
    """Seen live: told "I lost my ID card last week", a small model also recorded "loss reported: yes"."""
    seed(store, {}, procedure_id="tc_kimlik_renewal")
    replies = [d("record_fact", {"field": "loss_reported", "value": True}),
               d("final_answer", {"summary": "Have you reported the loss to Alo 199?"})]
    reply, backend = chat(store, replies, message="I lost my ID card last week")
    assert reply.status == "answered" and reply.facts == []
    assert "did not write a value for 'loss_reported'" in backend.calls[1]["messages"][-1]["content"]
    assert not any(fact.field == "loss_reported" for fact in store.current_facts(USER, "tc_kimlik_renewal"))


def test_tools_that_prepare_something_are_offered_only_when_asked(store):
    seed(store, PASSPORT_TOO_SHORT)
    final = d("final_answer", {"summary": "Your passport ends too early.", "claims": [{"rule_id": "R006", "status": "fail"}]})
    _, backend = chat(store, [final], message="Explain where I am")
    actions = backend.calls[0]["schema"]["properties"]["action"]["enum"]
    assert actions[0] == "final_answer" and "build_checklist" not in actions and "evaluate_rules" not in actions
    _, backend = chat(store, [final], message="Please prepare a checklist")
    assert "build_checklist" in backend.calls[0]["schema"]["properties"]["action"]["enum"]


def test_dates_written_with_a_month_name_count_as_stated():
    from src.agent.stated import stated_in
    pack = PACKS["ikamet_first"]
    assert stated_in("I entered Turkey on 15 September 2026", pack, "latest_entry_date", "2026-09-15")
    assert stated_in("I arrived September 15, 2026", pack, "latest_entry_date", "2026-09-15")
    assert not stated_in("I entered Turkey last week", pack, "latest_entry_date", "2026-11-13")


def test_status_words_and_copied_steps_are_tidied_but_unknown_ones_are_left_for_gate_1():
    from src.agent.chat import tidy_final_args
    args = {"summary": "x", "claims": [{"rule_id": "R006", "status": "not met"}, {"rule_id": "R001", "status": "maybe"}],
            "next_actions": [{"title": "Fix it", "ref": "fix: Fix: Passport validity [ref R006] (Source: GOV_FAQ)"},
                             {"title": "Both", "ref": "R006 or R001"}]}
    tidy = tidy_final_args(args, {"R006", "R001"})
    assert [c["status"] for c in tidy["claims"]] == ["fail", "maybe"]
    assert [a["ref"] for a in tidy["next_actions"]] == ["R006", "R006 or R001"]


def test_a_question_gets_no_tool_to_record_facts_and_no_procedure_gets_only_the_search():
    from src.agent.chat import offered_tools
    renewal = "tc_kimlik_renewal"
    assert "record_fact" not in offered_tools("Do I need an appointment to apply for the new card?", PACKS, renewal)
    assert "record_fact" in offered_tools("I lost my ID card last week", PACKS, renewal)
    assert "record_fact" in offered_tools("Entry date: 15.09.2026", PACKS, renewal)
    assert offered_tools("How do I get a tax number?", PACKS, None) == ("search_procedures",)
    assert "search_procedures" not in offered_tools("What should I do first?", PACKS, renewal)   # one alias word
    assert "search_procedures" in offered_tools("Actually I want to extend my ikamet", PACKS, renewal)


def test_without_a_procedure_the_web_pages_are_linked_and_nothing_is_said_to_be_checked(store, monkeypatch):
    from src.agent.chat import INFORMATIONAL_NOTE
    from src.rag.index import Hit
    page = Hit(source_id="WEB_IRO_EN_HOW_OBTAIN_TAX_ID", title="How to Obtain a Tax ID", chunk_id="x#0",
               text="Apply online with your passport number.", score=0.74, url="https://iro.sabanciuniv.edu/en/how-obtain-tax-id")
    monkeypatch.setattr("src.agent.chat.search_corpus", lambda *args, **kwargs: [page])
    reply, _ = chat(store, [d("final_answer", {"summary": "You can apply online with your passport number."})],
                    message="How do I get a tax number?")
    assert reply.status == "answered" and reply.procedure_id is None
    assert INFORMATIONAL_NOTE in reply.text and "https://iro.sabanciuniv.edu/en/how-obtain-tax-id" in reply.text
    assert [s["source_id"] for s in reply.sources] == ["WEB_IRO_EN_HOW_OBTAIN_TAX_ID"]
