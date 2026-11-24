"""eval/speed.py offline part: every part is timed and the report renders (no model)."""
from eval.speed import Timing, offline_timings, report, summarize


def test_offline_parts_are_timed():
    timings = offline_timings(repeat=2)
    assert [t.part for t in timings][:2] == ["load all packs", "route one goal"]
    assert all(t.samples > 0 and 0 <= t.median_ms <= t.p95_ms for t in timings)
    text = report(timings, None)
    assert "Offline parts (no model)" in text and "Not run (use --live" in text


def test_summary_percentiles():
    t = summarize("x", [5.0, 1.0, 3.0, 2.0, 4.0])
    assert t == Timing(part="x", samples=5, median_ms=3.0, p95_ms=5.0)
