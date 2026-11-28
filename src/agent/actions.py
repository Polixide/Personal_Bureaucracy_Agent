"""What the agent can prepare for the user right now (code, no model): the twin -> real tools made explicit.

Each suggestion names an artifact tool and its arguments, built only from the pack and the Twin:
  - build_checklist: the whole file with every requirement and what is still missing;
  - fill_pdf: each form of the pack, filled with the values the Twin knows (the rest stays empty);
  - create_ics: a reminder for each upcoming date the Twin knows (deadlines, programme start, expiries);
  - draft_email: a message to each contact of the pack. The text is written by the assistant (the model), so the
    suggestion carries the request to send in the chat instead of arguments.
Running a suggestion creates a PROPOSAL: nothing is written, sent or booked until the user approves it, and
approving writes a file to download (src/sandbox/approval.py).
"""
from datetime import date

from pydantic import BaseModel, Field

from src.agent.briefing import OPEN_PROPOSALS, readable
from src.models import Pack
from src.twin.facts import FactView
from src.twin.models import ProposedAction

MAX_REMINDERS = 4
SKIPPED_DATES = {"current_date"}


class Suggestion(BaseModel):
    """One thing the agent can prepare."""
    tool: str                       # build_checklist | fill_pdf | create_ics | draft_email
    label: str                      # button text
    why: str                        # one line: what it is for
    args: dict = Field(default_factory=dict)
    ask_assistant: str = ""         # draft_email: the request the chat sends to the assistant
    pending: bool = False           # the same proposal already waits for approval


def checklist_suggestion(pack: Pack) -> Suggestion:
    return Suggestion(tool="build_checklist", label="Checklist of my file",
                      why=f"Every requirement of '{pack.procedure.title}' with its status and what is still missing.")


def form_suggestions(pack: Pack, view: FactView) -> list[Suggestion]:
    """One per form of the pack, with how many of its fields the Twin already knows."""
    found = []
    for form in pack.procedure.forms:
        known = sum(fact in view.values for fact in form.fields.values())
        found.append(Suggestion(tool="fill_pdf", label=f"Fill: {form.title}", args={"form_id": form.form_id},
                                why=f"{known} of {len(form.fields)} fields are known; the others stay empty for you."))
    return found


def reminder_suggestions(pack: Pack, view: FactView, today: date) -> list[Suggestion]:
    """Upcoming dates the Twin knows, nearest first: a calendar file for each."""
    upcoming = sorted((value, field.name) for field in pack.procedure.fields
                      if field.type == "date" and field.name not in SKIPPED_DATES
                      for value in [view.values.get(field.name)] if isinstance(value, date) and value >= today)
    found = []
    for day, name in upcoming[:MAX_REMINDERS]:
        title = readable(name).capitalize()
        found.append(Suggestion(tool="create_ics", label=f"Calendar: {readable(name)} ({day.isoformat()})",
                                why="An all-day event for your Google Calendar (or any calendar), from the date in "
                                    "your file.",
                                args={"title": title, "date_field": name,
                                      "description": f"From your file for: {pack.procedure.title}."}))
    return found


def email_suggestions(pack: Pack) -> list[Suggestion]:
    """A draft to each contact; the assistant writes it from your state (you read it before approving)."""
    return [Suggestion(tool="draft_email", label=f"E-mail to {contact.role} (Gmail)",
                       why=f"The assistant drafts a message to {contact.name} about your next step; after you "
                           "approve it, it opens in Gmail for you to send.",
                       ask_assistant=f"Draft an e-mail to {contact.role} about my next step in this procedure.")
            for contact in pack.procedure.contacts]


def is_pending(suggestion: Suggestion, proposals: list[ProposedAction]) -> bool:
    """An open proposal of the same tool with the same key argument already exists."""
    for proposal in proposals:
        if proposal.status not in OPEN_PROPOSALS or proposal.tool != suggestion.tool:
            continue
        key = {"create_ics": "date_field", "fill_pdf": "form_id"}.get(suggestion.tool)
        if key is None or proposal.payload.get(key) == suggestion.args.get(key):
            return True
    return False


def already_in_calendar(suggestion: Suggestion, proposals: list[ProposedAction]) -> bool:
    """A reminder for this date was already approved: it is not offered again."""
    return suggestion.tool == "create_ics" and any(
        p.tool == "create_ics" and p.status == "executed" and p.payload.get("date_field") == suggestion.args.get("date_field")
        for p in proposals)


def suggestions(pack: Pack, view: FactView, today: date, proposals: list[ProposedAction] | None = None) -> list[Suggestion]:
    """Everything the agent can prepare now, checklist first, e-mails last (approved reminders are not repeated)."""
    found = ([checklist_suggestion(pack)] + form_suggestions(pack, view) + reminder_suggestions(pack, view, today)
             + email_suggestions(pack))
    found = [suggestion for suggestion in found if not already_in_calendar(suggestion, proposals or [])]
    for suggestion in found:
        if suggestion.tool != "draft_email":
            suggestion.pending = is_pending(suggestion, proposals or [])
    return found
