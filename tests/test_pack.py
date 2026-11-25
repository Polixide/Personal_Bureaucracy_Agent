"""src/pack.py: loading, validation errors, fact typing and the CLI."""
import json
import shutil
from datetime import date

import pytest

from config import PROCEDURES_DIR
from src.pack import coerce_fact, list_pack_ids, load_pack, main, validate_pack_dir


@pytest.fixture
def pack_copy(tmp_path):
    """A writable copy of the İkamet pack under a temporary procedures root."""
    shutil.copytree(PROCEDURES_DIR / "ikamet_first", tmp_path / "ikamet_first")
    return tmp_path


def edit(root, filename, change):
    path = root / "ikamet_first" / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def codes(root):
    return {issue.code for issue in validate_pack_dir("ikamet_first", root)}


@pytest.mark.parametrize("pack_id", list_pack_ids())
def test_every_committed_pack_is_valid(pack_id):
    assert validate_pack_dir(pack_id) == []


def test_unknown_field_in_condition(pack_copy):
    edit(pack_copy, "rules.json", lambda rules: rules[0]["condition"]["args"][0].update(field="no_such_field"))
    assert "UNKNOWN_FIELD" in codes(pack_copy)


def test_unknown_source(pack_copy):
    edit(pack_copy, "rules.json", lambda rules: rules[0]["source_ids"].append("NOT_A_SOURCE"))
    assert "UNKNOWN_SOURCE" in codes(pack_copy)


def test_dependency_cycle(pack_copy):
    def make_cycle(rules):
        by_id = {rule["rule_id"]: rule for rule in rules}
        by_id["R001"]["depends_on"] = ["R002"]
        by_id["R002"]["depends_on"] = ["R001"]
    edit(pack_copy, "rules.json", make_cycle)
    assert "DEPENDENCY_CYCLE" in codes(pack_copy)


def test_empty_section(pack_copy):
    edit(pack_copy, "procedure.json", lambda p: p["sections"].append({"id": "empty", "title": "Nothing here"}))
    assert "EMPTY_SECTION" in codes(pack_copy)


def test_value_of_wrong_type(pack_copy):
    edit(pack_copy, "rules.json", lambda rules: rules[0]["condition"]["args"][1].update(value="three"))
    assert "TYPE_MISMATCH" in codes(pack_copy)


def test_unknown_dependency(pack_copy):
    edit(pack_copy, "rules.json", lambda rules: rules[0]["depends_on"].append("R999"))
    assert "UNKNOWN_RULE" in codes(pack_copy)


def test_verified_rule_without_source(pack_copy):
    edit(pack_copy, "rules.json", lambda rules: rules[0].update(source_ids=[]))
    assert "NO_SOURCE" in codes(pack_copy)


def test_unknown_key_is_a_schema_error(pack_copy):
    edit(pack_copy, "rules.json", lambda rules: rules[0].update(sevrity="blocking"))
    assert codes(pack_copy) == {"SCHEMA_ERROR"}


def test_missing_file_is_reported(pack_copy):
    (pack_copy / "ikamet_first" / "sources.json").unlink()
    assert codes(pack_copy) == {"UNREADABLE_PACK"}


def test_too_few_scenarios(pack_copy):
    path = pack_copy / "ikamet_first" / "eval" / "scenarios.json"
    path.write_text("[]", encoding="utf-8")
    assert "FEW_SCENARIOS" in codes(pack_copy)


def test_every_rule_cites_a_known_source_and_has_a_section():
    pack = load_pack("ikamet_first")
    assert len(pack.rules) == 30          # 29 from milestone 1 + R030 (own property: utility bill)
    for rule in pack.rules:
        assert rule.source_ids and all(pack.source(s) for s in rule.source_ids)
        assert pack.section(rule.section)


@pytest.mark.parametrize("field,raw,expected", [
    ("is_in_turkiye", "yes", True), ("is_in_turkiye", "No", False), ("age", "22", 22),
    ("passport_expiry_date", "2027-08-01", date(2027, 8, 1)),
    ("institution", "  sabanci   UNIVERSITY ", "Sabancı University"), ("permit_type", "Short term", "short_term")])
def test_coerce_fact_types_and_aliases(field, raw, expected):
    assert coerce_fact(load_pack("ikamet_first"), field, raw) == expected


@pytest.mark.parametrize("field,raw", [
    ("is_in_turkiye", "maybe"), ("age", "twenty"), ("age", True), ("passport_expiry_date", "01/08/2027"),
    ("passport_expiry_date", "3027-01-01"), ("no_such_field", "x"), ("holder_name", "  ")])
def test_coerce_fact_rejects_bad_values(field, raw):
    with pytest.raises(ValueError):
        coerce_fact(load_pack("ikamet_first"), field, raw)


def test_cli_ok_and_failure(capsys, pack_copy, monkeypatch):
    assert main(["validate", "ikamet_first"]) == 0
    assert "OK: ikamet_first" in capsys.readouterr().out
    assert main(["validate", "does_not_exist"]) == 1
    assert main(["bad"]) == 2
