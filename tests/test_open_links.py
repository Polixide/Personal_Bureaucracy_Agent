"""Links that open an approved email or calendar artifact in the user's own apps."""
from datetime import date
from urllib.parse import parse_qs, urlsplit

from src.tools.draft_email import UNKNOWN_ADDRESS
from src.tools.open_links import google_calendar_link, is_address, open_links
from src.twin.models import ProposedAction


def proposal(tool, payload, title="t"):
    return ProposedAction(action_id="a1", user_id="u1", procedure_id="ikamet_first", tool=tool, title=title,
                          payload=payload, status="executed")


def query(url):
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def test_calendar_link_is_an_all_day_event_on_the_known_date():
    q = query(google_calendar_link("Fingerprint appointment", date(2026, 12, 31), "Bring passport"))
    assert q == {"action": "TEMPLATE", "text": "Fingerprint appointment", "dates": "20261231/20270101",
                 "details": "Bring passport"}


def test_email_links_carry_the_approved_text_exactly():
    body = "Dear IRO,\nmy documents are ready & attached? Ç ş ı"
    links = dict(open_links(proposal("draft_email", {"to_email": "iro@example.edu", "subject": "Ready", "body": body})))
    assert query(links["Open in Gmail"]) == {"view": "cm", "fs": "1", "to": "iro@example.edu", "su": "Ready",
                                             "body": body}
    mailto = links["Open in email app"]
    assert mailto.startswith("mailto:iro@example.edu?") and query(mailto) == {"subject": "Ready", "body": body}


def test_no_email_link_when_the_sources_give_no_address():
    assert not is_address(UNKNOWN_ADDRESS)
    assert open_links(proposal("draft_email", {"to_email": UNKNOWN_ADDRESS, "subject": "s", "body": "b"})) == []


def test_calendar_link_uses_the_payload_title_or_falls_back_to_the_proposal_title():
    [(label, url)] = open_links(proposal("create_ics", {"date": "2026-12-01", "title": "Pay the card fee"}))
    assert label == "Add to Google Calendar" and query(url)["text"] == "Pay the card fee"
    [(_, old)] = open_links(proposal("create_ics", {"date": "2026-12-01"}, title="Calendar: Fee (2026-12-01)"))
    assert query(old)["text"] == "Calendar: Fee (2026-12-01)"            # proposals saved before this change


def test_other_tools_get_no_links():
    assert open_links(proposal("fill_pdf", {"path": "x.pdf"})) == []
    assert open_links(proposal("create_ics", {})) == []
