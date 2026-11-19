"""Date and fact encoding: nothing may be stored with a silently changed meaning."""
from datetime import date, datetime, timedelta, timezone

import pytest

from src.memory.codec import (EncodingError, decode_date, decode_fact, decode_timestamp,
                              encode_date, encode_fact, encode_timestamp)

ISTANBUL = timezone(timedelta(hours=3))


@pytest.mark.parametrize("value", [date(2026, 11, 20), date(2028, 2, 29), date(1, 1, 1),
                                   date(9999, 12, 31)])
def test_date_round_trip(value):
    assert decode_date(encode_date(value)) == value


def test_datetime_is_refused_where_a_date_is_expected():
    # datetime is a subclass of date; storing it as a date would drop the time
    with pytest.raises(EncodingError):
        encode_date(datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc))


@pytest.mark.parametrize("text", ["2026-02-30", "2027-02-29", "2026-13-01", "20261120",
                                  "2026/11/20", "20-11-2026", "2026-11-20T00:00", "", " 2026-11-20"])
def test_bad_date_text_is_refused(text):
    with pytest.raises(EncodingError):
        decode_date(text)


def test_naive_datetime_is_refused():
    with pytest.raises(EncodingError):
        encode_timestamp(datetime(2026, 11, 20, 9, 30))


def test_timestamp_is_stored_in_utc_and_keeps_the_instant():
    local = datetime(2026, 11, 20, 1, 30, tzinfo=ISTANBUL)       # 22:30 UTC the day before
    text = encode_timestamp(local)
    assert text == "2026-11-19T22:30:00.000000+00:00"
    assert decode_timestamp(text) == local                       # same instant


@pytest.mark.parametrize("value", [None, True, False, 0, 1, -7, 2.5, "", "hello",
                                   "2026-11-20",                 # a string that looks like a date
                                   date(2026, 11, 20),
                                   datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc),
                                   ["a", 1, None], {"b": [1, 2], "a": "x"}])
def test_fact_round_trip_keeps_type(value):
    restored = decode_fact(*encode_fact(value))
    assert restored == value
    assert type(restored) is type(value)


def test_bool_is_not_stored_as_int():
    assert encode_fact(True)[0] == "bool"
    assert encode_fact(1)[0] == "int"


def test_date_nested_in_a_list_fact_is_refused():
    with pytest.raises(EncodingError):
        encode_fact({"appointments": [date(2026, 11, 20)]})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {1, 2}])
def test_unsupported_fact_values_are_refused(value):
    with pytest.raises(EncodingError):
        encode_fact(value)
