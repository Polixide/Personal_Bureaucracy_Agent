"""Reflection: answers and document values are corrected after a check, and every round is recorded.

1. Chat: Gate 1 refuses a reply -> the model rewrites it (structured errors); a reply that passes is re-read by the
   model itself against the state (self-review), and code keeps only criticisms quoting words of the reply.
2. Documents: a value whose quote is not on the page is read a second time with the reason it was refused.
A scripted model plays both roles; every check is the code's.
"""
from datetime import date

from agent_helpers import USER, d, seed
from fakes import ScriptedBackend
from src.agent.chat import CHAT_TOOLS, Review, ReviewProblem, checked_problems, offered_tools, run_chat
from src.pack import load_all_packs
from test_extract_document import card_pack  # noqa: F401  (fixture: a pack with one synthetic card document)
from src.tools.extract_document import extract_file

TODAY = date(2026, 11, 18)
PACKS = load_all_packs()
INVENTED = "You confirmed that you are an exchange student, so the next step is your passport."
FIXED = "I do not know yet whether you are an exchange student: please tell me. Then upload your passport."


def chat(store, replies, review=True, message="What should I do next?"):
    backend = ScriptedBackend(replies)
    return run_chat(message, store, PACKS, USER, TODAY, backend, review=review), backend


def problem(quote, why="STATE has no value for exchange_student."):
    return {"problems": [{"quote": quote, "why": why}]}


# ---------------------------------------------------------------- chat --
def test_the_self_review_finds_an_unsupported_sentence_and_the_reply_is_rewritten(store):
    seed(store, {})
    replies = [d("final_answer", {"summary": INVENTED}), problem("You confirmed that you are an exchange student"),
               d("final_answer", {"summary": FIXED})]
    reply, backend = chat(store, replies)
    assert reply.status == "answered" and reply.text.startswith(FIXED)
    assert [call["component"] for call in backend.calls] == ["orchestrator", "chat_self_review", "orchestrator"]
    assert "SELF-REVIEW" in backend.calls[2]["messages"][-1]["content"]
    assert reply.reflections[0]["by"] == "self-review" and len(reply.reflections[0]["issues"]) == 1


def test_a_criticism_that_quotes_words_not_in_the_reply_is_ignored(store):
    seed(store, {})
    replies = [d("final_answer", {"summary": FIXED}), problem("you already paid the card fee")]
    reply, backend = chat(store, replies)
    assert reply.status == "answered" and len(backend.calls) == 2
    assert reply.reflections == [{"by": "self-review", "issues": []}]      # reviewed: nothing to correct


def test_gate_1_reflection_comes_first_and_the_self_review_runs_once(store):
    seed(store, {"passport_expiry_date": "2027-07-15", "requested_permit_end_date": "2027-06-01"})
    wrong = d("final_answer", {"summary": "Your passport is fine.", "claims": [{"rule_id": "R006", "status": "pass"}]})
    right = d("final_answer", {"summary": "Your passport expires too early for the permit you ask.",
                               "claims": [{"rule_id": "R006", "status": "fail"}],
                               "next_actions": [{"title": "Renew your passport or ask a shorter permit", "ref": "R006"}]})
    reply, backend = chat(store, [wrong, right, {"problems": []}])
    assert reply.status == "answered"
    assert [r["by"] for r in reply.reflections] == ["Gate 1", "self-review"]
    assert "GATE 1 REJECTED" in backend.calls[1]["messages"][-1]["content"]


def test_without_self_review_one_call_is_enough(store):
    seed(store, {})
    reply, backend = chat(store, [d("final_answer", {"summary": FIXED})], review=False)
    assert reply.status == "answered" and len(backend.calls) == 1 and reply.reflections == []


def test_checked_problems_need_a_real_quote_of_some_length():
    review = Review(problems=[ReviewProblem(quote="tell me", why="too short"),
                              ReviewProblem(quote="whether you are an exchange student", why="ok")])
    assert len(checked_problems(review, FIXED)) == 1


def test_a_cloud_model_is_offered_every_chat_tool_and_a_local_one_only_what_the_message_needs():
    assert offered_tools("What next?", PACKS, "ikamet_first", everything=True) == CHAT_TOOLS
    assert "draft_email" not in offered_tools("What next?", PACKS, "ikamet_first")
    assert offered_tools("What next?", PACKS, None, everything=True) == ("search_procedures",)


# ----------------------------------------------------------- documents --
def test_a_misquoted_value_is_read_again_and_accepted_when_the_quote_is_right(card_pack):  # noqa: F811
    pack, document = card_pack
    wrong = {"fields": [{"field": "valid_until", "value": "31.07.2027", "page": 1, "quote": "Valid till 31.07.2027"}]}
    right = {"fields": [{"field": "valid_until", "value": "31.07.2027", "page": 1, "quote": "Valid until: 31.07.2027"}]}
    backend = ScriptedBackend([wrong, right, {"fields": []}])
    observation = extract_file(document, pack, doc_type="residence_permit_card", backend=backend).observations[0]
    assert {f.field: f.value for f in observation.facts}["permit_expiry_date"] == date(2027, 7, 31)
    assert "document_generic_reflection" in [call["component"] for call in backend.calls]
    assert "quote not found" in backend.calls[1]["messages"][-1]["content"]
    assert any(note.startswith("Read again") for note in observation.notes)
    assert not any(issue.startswith("valid_until") for issue in observation.issues)


def test_a_section_id_given_as_a_ref_becomes_the_first_step_of_that_section():
    from src.agent.chat import tidy_final_args
    args = {"summary": "s", "next_actions": [{"title": "Fix the address", "ref": "address_declaration"}]}
    tidy = tidy_final_args(args, {"R017"}, {"R017"}, {"address_declaration": "R017"})
    assert tidy["next_actions"][0]["ref"] == "R017"
