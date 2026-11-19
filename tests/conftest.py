from pathlib import Path
import tempfile
from config import PROJECT_ROOT
import pytest

@pytest.fixture(autouse=True)
def prohibit_live_ollama(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Automated tests must mock Ollama")
    monkeypatch.setattr("ollama.Client.chat", denied)
    monkeypatch.setattr("ollama.chat", denied)
    def denied_ocr(*args, **kwargs):
        raise AssertionError("Automated tests must mock OCR")
    from src.document_processing.extractor import get_ocr_engine
    get_ocr_engine.cache_clear()
    monkeypatch.setattr("src.document_processing.extractor._create_ocr_engine", denied_ocr)
    yield
    get_ocr_engine.cache_clear()


def pytest_configure(config):
    # Keep every test artifact local to the project, including migrated unittest fixtures.
    root = PROJECT_ROOT / ".test-temp"
    root.mkdir(exist_ok=True)
    tempfile.tempdir = str(root)
    if config.option.basetemp is None:
        config.option.basetemp = str(root / "pytest")


# ------------------------------------------------- twin record fixtures --
from datetime import datetime, timedelta, timezone


class FakeClock:
    """Deterministic clock: starts at a fixed instant, advances 1 s per call."""

    def __init__(self, start=datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)):
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "twin_record.db"


@pytest.fixture
def store(db_path):
    from src.memory.twin_store import TwinStore
    with TwinStore(db_path, clock=FakeClock()) as s:
        yield s
