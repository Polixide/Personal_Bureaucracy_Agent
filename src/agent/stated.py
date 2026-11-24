"""Did the user really say it? Code checks every fact the chat assistant wants to record.

Seen live with qwen2.5:3b (2026-11-20): asked "Explain where I am", it recorded "exchange student: yes" and a stay
end date of 2026-12-31; told "I lost my ID card last week", it also recorded "loss reported: yes". Recorded chat
facts stay unconfirmed, but they must still come from the user's words. A fact is accepted only when the message
shows it:
  - a date: the same day written in the message (31.12.2026, 2026-12-31, 31 December 2026, December 31, 2026);
  - a number: its digits are in the message;
  - yes/no: the message uses a word of the field's name or question ("reported" for loss_reported);
  - text: a word of the value is in the message ("lost" for "lost (or stolen)").
Otherwise the call is refused and the model is told to ask the user.
"""
import re
from datetime import date

from src.document_processing.evidence import normalized
from src.document_processing.vision import dates_in, to_date
from src.models import Pack

MONTHS = {name: number for number, names in enumerate([
    ("january", "jan", "ocak"), ("february", "feb", "subat"), ("march", "mar", "mart"), ("april", "apr", "nisan"),
    ("may", "mayis"), ("june", "jun", "haziran"), ("july", "jul", "temmuz"), ("august", "aug", "agustos"),
    ("september", "sep", "sept", "eylul"), ("october", "oct", "ekim"), ("november", "nov", "kasim"),
    ("december", "dec", "aralik")], start=1) for name in names}
DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?,?\s+(\d{4})\b")
MONTH_DAY_YEAR = re.compile(r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
STOPWORDS = {"have", "has", "will", "your", "you", "with", "from", "that", "this", "when", "what", "where", "which",
             "there", "they", "their", "would", "should", "could", "does", "date", "please", "about", "into", "than"}
MIN_WORD = 4


def words(text: str) -> set[str]:
    """Lower-case words without accents, at least MIN_WORD letters, without filler words."""
    return {word for word in re.findall(r"[a-z0-9]+", normalized(text)) if len(word) >= MIN_WORD and word not in STOPWORDS}


def message_dates(message: str) -> set[date]:
    """Every date written in the message, with digits or with a month name (English or Turkish)."""
    found = {day for day, _ in dates_in(message)}
    text = normalized(message)
    for day, month, year in DAY_MONTH_YEAR.findall(text):
        if month in MONTHS:
            found.add(to_date(int(day), MONTHS[month], int(year)))
    for month, day, year in MONTH_DAY_YEAR.findall(text):
        if month in MONTHS:
            found.add(to_date(int(day), MONTHS[month], int(year)))
    return {day for day in found if day is not None}


def as_date(value) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        days = dates_in(text)
        return days[0][0] if days else None


def shares_word(message: str, text: str) -> bool:
    """A word of `text` (or a word starting with the same five letters) appears in the message."""
    said = words(message)
    for word in words(text):
        if word in said or any(len(word) >= 5 and other.startswith(word[:5]) for other in said):
            return True
    return False


STATEMENT_STARTS = ("i", "i'm", "im", "i've", "my", "we", "our", "it", "yes", "no", "evet", "hayir")


def states_something(message: str) -> bool:
    """The message tells something (a date, or a sentence in the first person, or yes/no), not only asks.

    Seen live: asked "Do I need an appointment?", a small model recorded an invented field. A question gets no
    record_fact tool; "I lost my card last week" or "My passport ends on 15.07.2027" do.
    """
    if message_dates(message):
        return True
    for sentence in re.split(r"[.!?;\n]+", normalized(message)):
        first = sentence.strip().split(" ", 1)[0].strip(",") if sentence.strip() else ""
        if first in STATEMENT_STARTS:
            return True
    return False


def stated_in(message: str, pack: Pack, field: str, value) -> bool:
    """True when the message shows this value for this field (see the module docstring)."""
    spec = pack.field(field)
    if spec is None:
        return True                      # the tool itself refuses unknown fields
    if spec.type == "date":
        day = as_date(value)
        return day is not None and day in message_dates(message)
    if spec.type == "int":
        digits = re.sub(r"\D", "", str(value))
        return bool(digits) and digits in re.sub(r"\D", " ", message).split()
    if spec.type == "bool":
        return shares_word(message, f"{field.replace('_', ' ')} {spec.question}")
    return shares_word(message, str(value))
