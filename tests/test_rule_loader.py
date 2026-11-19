import pytest
from pathlib import Path
from src.workers.rule_loader import load_rules, load_dependencies

def test_load_from_another_directory(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_rules()) == 29
    assert load_rules()[0]["name"] == "exchange_stay_requires_ikamet"

def test_missing_file(monkeypatch):
    def missing(*a,**k): raise FileNotFoundError("missing")
    monkeypatch.setattr(Path,"read_text",missing)
    with pytest.raises(FileNotFoundError): load_rules()

def test_dependency_union():
    deps=load_dependencies()
    assert "R006" in deps["R020"]
    assert {"R024","R025","R026"} <= deps["R023"]
