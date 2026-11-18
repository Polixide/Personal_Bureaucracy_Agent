from datetime import datetime, timedelta, timezone

import pytest

from src.memory.checkpointer import TwinStore


class FakeClock:
    """Deterministic clock: starts at a fixed instant, advances 1 s per call."""

    def __init__(self, start=datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)):
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "twin.sqlite"


@pytest.fixture
def store(db_path):
    with TwinStore(db_path, clock=FakeClock()) as s:
        yield s
