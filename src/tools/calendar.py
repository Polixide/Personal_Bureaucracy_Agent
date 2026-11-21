"""Personal reminder preview, local ICS export and optional approved adapter."""
from datetime import timedelta, timezone
import json
from hashlib import sha256
from urllib.error import HTTPError
import os
from urllib.request import Request, urlopen
from src.tools.schemas import CalendarResult


def _ics_text(value):
    return value.replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,")


def reminder_preview(args, call_id):
    end = args.start_at + timedelta(minutes=args.duration_minutes)
    stamp = lambda value: value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Ikamet local prototype//EN", "BEGIN:VEVENT",
        "UID:" + sha256(call_id.encode()).hexdigest() + "@ikamet-local", "DTSTAMP:" + stamp(args.start_at), "DTSTART:" + stamp(args.start_at),
        "DTEND:" + stamp(end), "SUMMARY:" + _ics_text(args.title), "DESCRIPTION:" + _ics_text(args.description),
        "END:VEVENT", "END:VCALENDAR", ""]
    # RFC5545 logical lines are folded by UTF-8 byte length, not character count.
    folded = []
    for line in lines:
        piece = ""
        for character in line:
            if len((piece + character).encode("utf-8")) > 74:
                folded.append(piece);piece = " "
            piece += character
        folded.append(piece)
    return CalendarResult(reminder_id=call_id, mode="LOCAL_ICS", title=args.title,
        start_at=args.start_at.isoformat(), end_at=end.isoformat(), description=args.description,
        ics="\r\n".join(folded))


class GoogleCalendarAdapter:
    """Bring an existing operator-managed access token; no mandatory OAuth setup."""
    def __init__(self, token, calendar_id="primary", *, transport=urlopen):
        self.token, self.calendar_id, self.transport = token, calendar_id, transport

    def create(self, reminder):
        from urllib.parse import quote
        payload = {"id": sha256(reminder.reminder_id.encode()).hexdigest(), "summary": reminder.title, "description": reminder.description,
            "start": {"dateTime": reminder.start_at}, "end": {"dateTime": reminder.end_at}}
        request = Request("https://www.googleapis.com/calendar/v3/calendars/" + quote(self.calendar_id, safe="") + "/events",
            data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}, method="POST")
        try:
            with self.transport(request, timeout=30) as response:
                return str(json.load(response)["id"])
        except HTTPError as error:
            if error.code != 409:
                raise
            # Provider-side ID bounds replay across crash-before-checkpoint.
            check = Request(request.full_url + "/" + payload["id"], headers={"Authorization":"Bearer " + self.token})
            with self.transport(check, timeout=30) as response:
                existing = json.load(response)
            from datetime import datetime
            if (existing.get("summary") != reminder.title or existing.get("description", "") != reminder.description
                or datetime.fromisoformat(existing["start"]["dateTime"].replace("Z", "+00:00")) != datetime.fromisoformat(reminder.start_at)
                or datetime.fromisoformat(existing["end"]["dateTime"].replace("Z", "+00:00")) != datetime.fromisoformat(reminder.end_at)):
                raise ValueError("Existing calendar event differs from the approved reminder")
            return payload["id"]


def configured_calendar_adapter():
    token = os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN")
    return GoogleCalendarAdapter(token, os.getenv("GOOGLE_CALENDAR_ID", "primary")) if token else None


def google_calendar_link(reminder):
    """User-controlled event editor; creates nothing and needs no credentials."""
    from datetime import datetime
    from urllib.parse import urlencode,quote
    data=reminder.model_dump(mode='json') if hasattr(reminder,'model_dump') else reminder
    start=datetime.fromisoformat(data['start_at']);end=datetime.fromisoformat(data['end_at'])
    if start.tzinfo is None or end.tzinfo is None or end<=start:raise ValueError('Reminder needs a valid timezone-aware interval')
    stamp=lambda d:d.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return 'https://calendar.google.com/calendar/render?'+urlencode({'action':'TEMPLATE','text':data['title'],'dates':stamp(start)+'/'+stamp(end),'details':data.get('description','')},quote_via=quote)


def declared_reminder_candidates(state):
    """Primary UX reminders use accepted dates only; no nationality/stay arithmetic."""
    from datetime import timedelta
    from src.core.procedure import _accepted_snapshot
    from config import RECOMMENDED_TARGET_BUFFER_DAYS
    c=_accepted_snapshot(state)['residence_permit_context']
    result=[]
    if c.legal_stay_expiry_date:
        result.extend([{'candidate_id':'declared_legal_stay','title':'Review my declared legal-stay expiry','date':c.legal_stay_expiry_date.isoformat(),'kind':'Your declared legal-stay expiry'},
            {'candidate_id':'declared_preparation_target','title':'Complete residence-permit preparation','date':(c.legal_stay_expiry_date-timedelta(days=RECOMMENDED_TARGET_BUFFER_DAYS)).isoformat(),'kind':'Agent recommendation - not an official deadline'}])
    if c.insurance_end_date:result.append({'candidate_id':'declared_insurance_expiry','title':'Review my health-insurance expiry','date':c.insurance_end_date.isoformat(),'kind':'Insurance expiry from your accepted information'})
    return result
