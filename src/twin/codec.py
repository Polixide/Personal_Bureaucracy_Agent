"""Strict conversions between Python values and SQLite TEXT columns.

SQLite has no date type, and Python's default sqlite3 date adapters are
deprecated (3.12+) and silent about mistakes. Every value is converted here,
explicitly, so a wrong type fails loudly instead of being stored wrongly.

Rules
-----
* Calendar date (`date`) -> 'YYYY-MM-DD'. A document's expiry date or an
  appointment day is a day, not an instant, so it carries no time zone and is
  never shifted. A `datetime` is refused where a date is expected: `datetime`
  is a subclass of `date`, and storing it would silently drop the time.
* Instant (`datetime`) -> UTC 'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'. It must be
  timezone-aware; naive datetimes are refused because their meaning depends
  on the machine that wrote them. Used for audit fields (created_at, ...).
* Fact values (the free-form `extracted_facts` dict) are stored with a type
  tag, so the string "2026-11-20" and the date 2026-11-20 stay different
  after a restart.
"""
import json
import math
import re
from datetime import date, datetime, timezone
from typing import Any

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

FACT_TYPES = ("null", "bool", "int", "float", "str", "date", "datetime", "json")


class EncodingError(ValueError):
    """A value cannot be stored (or read back) without losing meaning."""


# ------------------------------------------------------------------ dates --
def encode_date(value: date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise EncodingError(f"expected a date, got {type(value).__name__}: {value!r}")
    return value.isoformat()


def decode_date(text: str | None) -> date | None:
    if text is None:
        return None
    # fromisoformat alone also accepts '20261120' and week dates; be strict.
    if not isinstance(text, str) or not _DATE_RE.fullmatch(text):
        raise EncodingError(f"not a YYYY-MM-DD date: {text!r}")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:                      # e.g. 2026-02-30
        raise EncodingError(f"not a real calendar date: {text!r}") from exc


# ------------------------------------------------------------- timestamps --
def encode_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise EncodingError(f"expected a datetime, got {type(value).__name__}: {value!r}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise EncodingError(f"naive datetime refused, attach a time zone: {value!r}")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def decode_timestamp(text: str) -> datetime:
    try:
        value = datetime.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise EncodingError(f"not an ISO timestamp: {text!r}") from exc
    if value.tzinfo is None:
        raise EncodingError(f"stored timestamp has no time zone: {text!r}")
    return value.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ facts --
def _refuse_in_json(value: Any) -> Any:
    raise EncodingError(
        f"{type(value).__name__} inside a list/dict fact cannot be stored safely; "
        f"store it as its own fact instead: {value!r}")


def encode_fact(value: Any) -> tuple[str, str | None]:
    """Return (type_tag, text) for one extracted_facts value."""
    if value is None:
        return "null", None
    if isinstance(value, bool):                    # before int: bool is an int
        return "bool", "true" if value else "false"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EncodingError(f"non-finite float refused: {value!r}")
        return "float", repr(value)
    if isinstance(value, str):
        return "str", value
    if isinstance(value, datetime):                # before date: datetime is a date
        return "datetime", encode_timestamp(value)
    if isinstance(value, date):
        return "date", encode_date(value)
    if isinstance(value, (list, dict)):
        return "json", json.dumps(value, sort_keys=True, default=_refuse_in_json)
    raise EncodingError(f"unsupported fact type {type(value).__name__}: {value!r}")


def decode_fact(type_tag: str, text: str | None) -> Any:
    if type_tag == "null":
        return None
    if type_tag == "bool":
        return {"true": True, "false": False}[text]
    if type_tag == "int":
        return int(text)
    if type_tag == "float":
        return float(text)
    if type_tag == "str":
        return text
    if type_tag == "date":
        return decode_date(text)
    if type_tag == "datetime":
        return decode_timestamp(text)
    if type_tag == "json":
        return json.loads(text)
    raise EncodingError(f"unknown fact type tag: {type_tag!r}")
