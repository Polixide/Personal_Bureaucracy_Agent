"""Links that open an approved artifact in the user's own apps (ported from Marco, b0ee6c1).

Each link only opens an editor pre-filled with the approved text: the user still presses Send or Save
themselves. No API, OAuth or credentials; nothing is sent or booked by the agent.
"""
from datetime import date, timedelta
from urllib.parse import quote, urlencode

from src.twin.models import ProposedAction


def google_calendar_link(title: str, day: date, details: str = "") -> str:
    """Google Calendar's 'new event' page for an all-day event on `day`."""
    dates = f"{day:%Y%m%d}/{day + timedelta(days=1):%Y%m%d}"
    query = urlencode({"action": "TEMPLATE", "text": title, "dates": dates, "details": details}, quote_via=quote)
    return "https://calendar.google.com/calendar/render?" + query


def gmail_compose_link(to: str, subject: str, body: str) -> str:
    """Gmail's web composer, pre-filled."""
    query = urlencode({"view": "cm", "fs": "1", "to": to, "su": subject, "body": body}, quote_via=quote)
    return "https://mail.google.com/mail/?" + query


def mailto_link(to: str, subject: str, body: str) -> str:
    """The default email app's composer, pre-filled."""
    return "mailto:" + quote(to, safe="@") + "?" + urlencode({"subject": subject, "body": body}, quote_via=quote)


def is_address(text: str) -> bool:
    """A plain e-mail address (the draft tool writes a hint instead when the sources give none)."""
    name, at, domain = text.partition("@")
    return bool(name) and bool(at) and "." in domain and " " not in text


def open_links(proposal: ProposedAction) -> list[tuple[str, str]]:
    """(label, url) pairs for an approved proposal; none for other tools or incomplete payloads."""
    payload = proposal.payload
    if proposal.tool == "draft_email" and is_address(payload.get("to_email", "")):
        to, subject, body = payload["to_email"], payload.get("subject", ""), payload.get("body", "")
        return [("Open in Gmail", gmail_compose_link(to, subject, body)),
                ("Open in email app", mailto_link(to, subject, body))]
    if proposal.tool == "create_ics" and payload.get("date"):
        title = payload.get("title") or proposal.title
        return [("Add to Google Calendar",
                 google_calendar_link(title, date.fromisoformat(payload["date"]), payload.get("description", "")))]
    return []
