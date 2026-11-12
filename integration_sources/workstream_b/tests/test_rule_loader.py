"""Check loading independently of the current working directory."""

from pathlib import Path

import pytest

from src.rule_loader import load_rules


def test_load_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rules = load_rules()
    assert len(rules) == 8
    assert rules[0]["rule_id"] == "R001"
    assert isinstance(rules[0]["deterministic_condition"], str)


def test_missing_file(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(Path, "open", missing)
    with pytest.raises(FileNotFoundError, match="Rules file not found:"):
        load_rules()
