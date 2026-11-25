"""Gmail and Google Calendar links for approved proposals (src/agent/links.py): filled in, never sent by the agent."""
from datetime import date
from urllib.parse import parse_qs, urlparse

from src.agent.links import calendar_link, gmail_link, proposal_links
from src.twin.models import ProposedAction


def query(url: str) -> dict:
    return {key: values[0] for key, values in parse_qs(urlparse(url).query).items()}


def proposal(tool: str, payload: dict, status: str = "executed", title: str = "t") -> ProposedAction:
    return ProposedAction(action_id="A-1", user_id="u", procedure_id="p", tool=tool, title=title, payload=payload,
                          status=status)


def test_gmail_opens_a_compose_window_with_everything_filled_in():
    fields = query(gmail_link("irolegal@sabanciuniv.edu", "My file", "Dear IRO,\nis anything missing?"))
    assert fields == {"view": "cm", "fs": "1", "to": "irolegal@sabanciuniv.edu", "su": "My file",
                      "body": "Dear IRO,\nis anything missing?"}


def test_an_unknown_address_is_left_empty_for_the_user():
    assert "to" not in query(gmail_link("(address not in the procedure sources: fill it in yourself)", "s", "b"))


def test_google_calendar_gets_an_all_day_event_on_the_date_from_the_file():
    fields = query(calendar_link("Legal stay ends", date(2026, 12, 20), "From your file"))
    assert fields["action"] == "TEMPLATE" and fields["dates"] == "20261220/20261221" and fields["text"] == "Legal stay ends"


def test_only_approved_e_mails_and_reminders_get_links():
    email = {"to_email": "x@example.org", "subject": "s", "body": "b"}
    assert proposal_links(proposal("draft_email", email, status="proposed")) == []
    assert proposal_links(proposal("draft_email", email))[0][0] == "Open in Gmail"
    reminder = proposal("create_ics", {"date": "2026-12-20"}, title="Calendar: Legal stay ends (2026-12-20)")
    label, url = proposal_links(reminder)[0]
    assert label == "Add to Google Calendar" and query(url)["text"] == "Legal stay ends"
    assert proposal_links(proposal("build_checklist", {"markdown": "# x"})) == []
