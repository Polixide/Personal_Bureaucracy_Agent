"""The engine is procedure-independent: src/ must not mention any rule id of any Procedure Pack."""
import re
from pathlib import Path

from config import PROJECT_ROOT
from src.pack import load_all_packs

SRC = PROJECT_ROOT / "src"


def rule_id_mentions(text: str, rule_ids: set[str]) -> set[str]:
    """Rule ids written as whole words in `text`."""
    return {rule_id for rule_id in rule_ids if re.search(rf"\b{re.escape(rule_id)}\b", text)}


def all_rule_ids() -> set[str]:
    return {rule_id for pack in load_all_packs().values() for rule_id in pack.rule_ids()}


def test_there_are_at_least_two_procedures():
    assert len(load_all_packs()) >= 2


def test_src_mentions_no_rule_id_of_any_procedure():
    rule_ids = all_rule_ids()
    offenders = {}
    for path in sorted(SRC.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".sql", ".json", ".txt", ".md"}:
            found = rule_id_mentions(path.read_text(encoding="utf-8"), rule_ids)
            if found:
                offenders[str(path.relative_to(PROJECT_ROOT))] = sorted(found)
    assert offenders == {}


def test_the_check_would_catch_a_hard_coded_rule_id():
    assert rule_id_mentions('if rule_id == "R015": ...', all_rule_ids()) == {"R015"}
    assert rule_id_mentions("ER0150 and XR015X are not ids", all_rule_ids()) == set()


def test_src_mentions_no_procedure_id_either():
    pack_ids = set(load_all_packs())
    offenders = {str(p.relative_to(PROJECT_ROOT)) for p in SRC.rglob("*.py")
                 if any(re.search(rf"\b{pack_id}\b", p.read_text(encoding="utf-8")) for pack_id in pack_ids)}
    assert offenders == set()
