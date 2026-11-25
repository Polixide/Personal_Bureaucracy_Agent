"""Open an approved artifact in the user's own Gmail or Google Calendar, already filled in.

The links use the public "compose" and "add event" addresses of Gmail and Google Calendar: they open the user's
account in the browser with the recipient, subject, text or date already written. The user presses Send or Save,
so nothing is sent or booked by the agent (CLAUDE.md hard rule) and no Google login, key or library is needed.
Only approved proposals get links (src/sandbox/approval.py writes them first).
"""
from datetime import date, timedelta
from urllib.parse import urlencode

from src.twin.models import ProposedAction

GMAIL_COMPOSE = "https://mail.google.com/mail/"
CALENDAR_EVENT = "https://calendar.google.com/calendar/render"
UNKNOWN_ADDRESS_MARK = "("          # draft_email writes "(address not in the procedure sources: ...)"


def gmail_link(to: str, subject: str, body: str) -> str:
    """A Gmail compose window with the message filled in (the user sends it)."""
    recipient = "" if not to or to.startswith(UNKNOWN_ADDRESS_MARK) else to
    return GMAIL_COMPOSE + "?" + urlencode({"view": "cm", "fs": "1", "to": recipient, "su": subject, "body": body})


def calendar_link(title: str, day: date, details: str) -> str:
    """A Google Calendar 'new event' page for an all-day event (the user saves it)."""
    dates = f"{day:%Y%m%d}/{day + timedelta(days=1):%Y%m%d}"
    return CALENDAR_EVENT + "?" + urlencode({"action": "TEMPLATE", "text": title, "dates": dates, "details": details})


def proposal_links(proposal: ProposedAction) -> list[tuple[str, str]]:
    """(label, url) pairs for an approved proposal: Gmail for e-mails, Google Calendar for reminders."""
    if proposal.status != "executed":
        return []
    payload = proposal.payload or {}
    if proposal.tool == "draft_email":
        return [("Open in Gmail", gmail_link(payload.get("to_email", ""), payload.get("subject", ""),
                                             payload.get("body", "")))]
    if proposal.tool == "create_ics" and payload.get("date"):
        title = proposal.title.removeprefix("Calendar: ").rsplit(" (", 1)[0]
        return [("Add to Google Calendar", calendar_link(title, date.fromisoformat(payload["date"]),
                                                         "Prepared by the Personal Bureaucracy Agent."))]
    return []
