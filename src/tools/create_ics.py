"""create_ics tool (artifact): propose a calendar file for a date that is a FACT in the Twin.

The model names a date field; the code reads the date from the facts. The model never writes or
computes a date. The iCalendar text (RFC 5545) is written by hand: CRLF line ends, escaping, folding.
"""
from datetime import date, datetime, timedelta, timezone

from pydantic import BaseModel, Field

from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.tools.proposals import ProposalOutput, check_rule_ids, check_text, propose

FOLD_OCTETS = 75


class CreateIcsInput(BaseModel):
    title: str = Field(min_length=3, max_length=120, description="Event title, e.g. 'Legal stay ends'")
    date_field: str = Field(description="A date field of the procedure whose value is known (e.g. legal_stay_expiry_date)")
    description: str = Field(default="", max_length=500, description="Short reminder text (no other dates)")
    rule_ids: list[str] = Field(default_factory=list, description="Rules this date matters for")


def ics_escape(text: str) -> str:
    """Escape backslash, semicolon, comma and newlines (RFC 5545 TEXT)."""
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\r\n", "\\n").replace("\n", "\\n")


def fold_line(line: str) -> str:
    """Split a content line into pieces of at most 75 octets; continuations start with a space."""
    pieces, current, size = [], "", 0
    for char in line:
        width = len(char.encode("utf-8"))
        if size + width > FOLD_OCTETS:
            pieces.append(current)
            current, size = " ", 1
        current += char
        size += width
    pieces.append(current)
    return "\r\n".join(pieces)


def ics_text(uid: str, title: str, day: date, description: str, stamp: datetime) -> str:
    """A one-event, all-day VCALENDAR."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Personal Bureaucracy Agent//EN", "CALSCALE:GREGORIAN",
             "BEGIN:VEVENT", f"UID:{uid}@personal-bureaucracy-agent.local",
             f"DTSTAMP:{stamp.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}",
             f"DTSTART;VALUE=DATE:{day:%Y%m%d}", f"DTEND;VALUE=DATE:{day + timedelta(days=1):%Y%m%d}",
             f"SUMMARY:{ics_escape(title)}", f"DESCRIPTION:{ics_escape(description)}",
             "END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(fold_line(line) for line in lines) + "\r\n"


def run(args: CreateIcsInput, ctx: ToolContext) -> ProposalOutput:
    pack = require_pack(ctx)
    spec = pack.field(args.date_field)
    if spec is None or spec.type != "date":
        raise ToolError(f"{args.date_field!r} is not a date field of this procedure")
    day = ctx.view().values.get(args.date_field)
    if day is None:
        raise ToolError(f"{args.date_field} is not known yet; ask the user or extract a document first")
    check_rule_ids(pack, args.rule_ids)
    check_text(ctx, f"{args.title}\n{args.description}")
    uid = f"{ctx.user_id}-{args.date_field}-{day:%Y%m%d}"
    text = ics_text(uid, args.title, day, args.description, datetime.now(timezone.utc))
    payload = {"ics": text, "date": day.isoformat(), "date_field": args.date_field}
    return propose(ctx, pack, "create_ics", f"Calendar: {args.title} ({day.isoformat()})", payload,
                   args.rule_ids, None, f"{args.title} on {day.isoformat()}")


SPEC = ToolSpec(
    name="create_ics",
    description="Propose a calendar reminder (.ics) for a date the Twin already knows. Give the date FIELD, "
                "not a date. Written only after the user approves.",
    input_model=CreateIcsInput, output_model=ProposalOutput, fn=run, kind="artifact", roles=("orchestrator",))
