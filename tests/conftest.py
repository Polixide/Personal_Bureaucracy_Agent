from pathlib import Path
import tempfile
from config import PROJECT_ROOT
import pytest

@pytest.fixture(autouse=True)
def prohibit_live_ollama(monkeypatch):
    monkeypatch.setattr("config.DOCUMENT_CACHE_ENABLED",False)
    def denied(*args, **kwargs):
        raise AssertionError("Automated tests must mock Ollama")
    monkeypatch.setattr("ollama.Client.chat", denied)
    monkeypatch.setattr("ollama.chat", denied)
    def denied_openai(*args, **kwargs):
        raise AssertionError("Automated tests must mock OpenAI")
    monkeypatch.setattr("src.document_processing.providers._create_openai_client",denied_openai)
    monkeypatch.setattr("src.document_processing.processor.DOCUMENT_AI_PROVIDER","legacy")
    monkeypatch.setattr("src.document_processing.processor.LLM_PROVIDER","ollama")
    monkeypatch.setattr("src.document_processing.processor.DOCUMENT_FALLBACK_PROVIDER","none")
    monkeypatch.setattr("src.tools.registry.configured_calendar_adapter",lambda:None)
    monkeypatch.setattr("src.tools.registry.configured_email_adapter",lambda:None)
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


@pytest.fixture(autouse=True)
def isolate_neda_record(monkeypatch,tmp_path):
    monkeypatch.setattr("src.ui.components.TWIN_RECORD_PATH",tmp_path/"record.db")
    monkeypatch.setattr("config.TWIN_RECORD_PATH",tmp_path/"record.db")



def wire_planner_response(response,payload):
    """Migrate a scripted legacy fixture to the actual opaque wire selectors.

    Only exact known candidates map. Invalid references, malformed fields and
    duplicate selections stay invalid; this helper never touches production.
    """
    import json
    from copy import deepcopy
    from types import SimpleNamespace
    if isinstance(response,dict):raw=deepcopy(response)
    else:
        try:raw=json.loads(response.message.content)
        except (ValueError,TypeError,AttributeError):return response
    if not isinstance(raw,dict) or not isinstance(raw.get('actions'),list):return response
    for row in raw['actions']:
        if not isinstance(row,dict):continue
        keys=row.get('candidate_ids')
        if keys is None:
            keys=[row.get('related_rule_id',row.get('rule_id'))]
        if not isinstance(keys,list):continue
        converted=[]
        for key in keys:
            matches=[c['candidate_id'] for c in payload['canonical_candidates'] if key in c.get('rule_ids',[]) or key==c.get('canonical_ref') or key==c['candidate_id']]
            converted.append(matches[0] if len(matches)==1 else key or 'unknown_candidate')
        row['candidate_ids']=converted
    result=deepcopy(response) if not isinstance(response,dict) else SimpleNamespace()
    result.message=SimpleNamespace(content=json.dumps(raw))
    return result


def wire_planner_sequence(responses,*,repeat=False):
    """Contextual mocked transport: preserve the real response/error sequence."""
    import json
    values=list(responses);position=[0]
    def reply(**kwargs):
        index=position[0];position[0]+=1
        raw=values[min(index,len(values)-1)] if repeat else values[index]
        if isinstance(raw,Exception):raise raw
        return wire_planner_response(raw,json.loads(kwargs['messages'][1]['content']))
    return reply
