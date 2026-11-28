"""Gate 0 (src/agent/gate0.py): what the user sends is checked by code before any model call or Twin write."""
from datetime import date

import pymupdf
import pytest

import src.agent.gate0 as gate0
from src.agent.chat import run_chat
from src.pack import load_all_packs
from src.twin.store import TwinStore
from tests.fakes import ScriptedBackend


def pdf_bytes(pages: int) -> bytes:
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()
    data = document.tobytes()
    document.close()
    return data


def codes(issues) -> list[str]:
    return [issue.code for issue in issues]


# -------------------------------------------------------------- messages --
@pytest.mark.parametrize("text,expected", [("", ["EMPTY_MESSAGE"]), ("   \n ", ["EMPTY_MESSAGE"]),
                                           ("I want to apply for İkamet", [])])
def test_a_message_must_have_text(text, expected):
    assert codes(gate0.check_message(text)) == expected


def test_a_message_longer_than_one_turn_is_refused(monkeypatch):
    monkeypatch.setattr(gate0, "GATE0_MAX_MESSAGE_CHARS", 10)
    [issue] = gate0.check_message("x" * 11)
    assert issue.code == "MESSAGE_TOO_LONG" and "TXT or PDF" in issue.message


# --------------------------------------------------------------- uploads --
@pytest.mark.parametrize("name,content,expected", [
    ("a.exe", b"MZ", "UNSUPPORTED_FILE"), ("noextension", b"x", "UNSUPPORTED_FILE"),
    ("a.txt", b"", "EMPTY_FILE"),
    ("passport.pdf", b"just text", "WRONG_CONTENT"), ("photo.png", b"\xff\xd8 a jpeg", "WRONG_CONTENT"),
    ("photo.jpg", b"\x89PNG a png", "WRONG_CONTENT")])
def test_bad_uploads_are_refused_with_a_reason(name, content, expected):
    assert codes(gate0.check_upload(name, content)) == [expected]


@pytest.mark.parametrize("name,content", [("notes.txt", b"Policy number 123"), ("photo.JPG", b"\xff\xd8\xff rest"),
                                          ("scan.png", b"\x89PNG\r\n rest"), ("doc.pdf", pdf_bytes(2))])
def test_good_uploads_pass(name, content):
    assert gate0.check_upload(name, content) == []


def test_a_file_over_the_size_limit_is_refused(monkeypatch):
    monkeypatch.setattr(gate0, "GATE0_MAX_UPLOAD_MB", 1)
    assert codes(gate0.check_upload("big.txt", b"x" * (1024 * 1024 + 1))) == ["FILE_TOO_LARGE"]


def test_a_pdf_with_too_many_pages_is_refused(monkeypatch):
    monkeypatch.setattr(gate0, "GATE0_MAX_PDF_PAGES", 3)
    assert codes(gate0.check_upload("book.pdf", pdf_bytes(4))) == ["TOO_MANY_PAGES"]
    assert gate0.check_upload("short.pdf", pdf_bytes(3)) == []


def test_a_damaged_pdf_is_left_to_the_reader_for_a_precise_reason():
    assert gate0.check_upload("damaged.pdf", b"%PDF-1.4 not really a pdf") == []


@pytest.mark.parametrize("name,ok", [("demo-user", True), ("", False), ("   ", False), ("x" * 61, False)])
def test_profile_names(name, ok):
    assert (gate0.check_profile(name) == []) == ok


# -------------------------------------------------------------- the chat --
def test_the_chat_refuses_an_empty_message_without_a_model_call_or_a_twin_write(tmp_path):
    backend = ScriptedBackend([])                     # any model call would fail: the script is empty
    with TwinStore(tmp_path / "twin.sqlite") as store:
        reply = run_chat("   ", store, load_all_packs(), "u1", date(2026, 11, 28), backend)
        assert reply.status == "invalid_input" and reply.gate_issues[0]["code"] == "EMPTY_MESSAGE"
        assert store.list_chat("u1") == [] and store.list_runs("u1") == [] and backend.calls == []
