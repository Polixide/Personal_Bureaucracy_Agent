"""Load and validate Procedure Packs (procedures/<id>/). CLI: python -m src.pack validate <id>.

Validation catches what would otherwise surface as wrong answers: unknown fields or sources,
dependency cycles, empty sections, values of the wrong type, missing scenarios.
"""
import json
import sys
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from config import PROCEDURES_DIR
from src.models import (COMPARISON_OPS, Condition, FactValue, FieldSpec, Issue, Pack,
                        Procedure, Rule, Source)

MIN_SCENARIOS = 5
TRUE_WORDS = {"true", "yes", "y", "1", "evet"}
FALSE_WORDS = {"false", "no", "n", "0", "hayir", "hayır"}


class PackError(ValueError):
    """The pack files cannot be read or parsed."""


# --------------------------------------------------------------- loading --
def pack_dir(procedure_id: str, root: Path = PROCEDURES_DIR) -> Path:
    """Folder of one pack."""
    return Path(root) / procedure_id


def list_pack_ids(root: Path = PROCEDURES_DIR) -> list[str]:
    """Ids of every folder under `root` that has a procedure.json, sorted."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "procedure.json").is_file())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackError(f"Missing file: {path.name}") from error
    except json.JSONDecodeError as error:
        raise PackError(f"{path.name} is not valid JSON: {error}") from error


def load_pack(procedure_id: str, root: Path = PROCEDURES_DIR) -> Pack:
    """Read and parse one pack. Raises PackError (bad files) or ValidationError (bad shape)."""
    folder = pack_dir(procedure_id, root)
    procedure = Procedure.model_validate(_read_json(folder / "procedure.json"))
    rules = [Rule.model_validate(item) for item in _read_json(folder / "rules.json")]
    sources = [Source.model_validate(item) for item in _read_json(folder / "sources.json")]
    return Pack(procedure=procedure, rules=rules, sources=sources, root=folder)


def load_all_packs(root: Path = PROCEDURES_DIR) -> dict[str, Pack]:
    """Every pack under `root`, keyed by id."""
    return {pack_id: load_pack(pack_id, root) for pack_id in list_pack_ids(root)}


def load_scenarios(pack: Pack) -> list[dict]:
    """The pack's eval/scenarios.json (a list), or [] when the file is missing."""
    path = pack.root / "eval" / "scenarios.json"
    if not path.is_file():
        return []
    return _read_json(path)


# ----------------------------------------------------------- fact typing --
def normalize_text(text: str) -> str:
    """Comparison form of free text: NFKC, casefold, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def coerce_value(spec: FieldSpec, raw: Any) -> FactValue:
    """Turn user or document input into the field's type. Raises ValueError if impossible."""
    if spec.type == "bool":
        return _to_bool(raw)
    if spec.type == "int":
        return _to_int(raw)
    if spec.type == "date":
        return _to_date(raw)
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{spec.name}: empty text")
    key = normalize_text(text)
    chosen = {normalize_text(label): value for label, value in spec.choices.items()}
    if key in spec.aliases or key in chosen or not spec.choices:
        return spec.aliases.get(key, chosen.get(key, text))
    return choice_in_text(spec, text)


def folded(text: str) -> str:
    """Comparison form without accents or Turkish dotted/dotless i: 'KİRA SÖZLEŞMESİ' -> 'kira sozlesmesi'."""
    text = unicodedata.normalize("NFKD", text.casefold().replace("\u0131", "i"))
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).split())


def choice_in_text(spec: FieldSpec, text: str) -> str:
    """A field with choices: the stored value whose alias, label or code appears in the text, else the text.

    Documents print a title ("KİRA SÖZLEŞMESİ / RENTAL AGREEMENT"), not the code the rules compare.
    """
    found = folded(text)
    phrases = {folded(alias): value for alias, value in spec.aliases.items()}
    phrases.update({folded(value.replace("_", " ")): value for value in spec.choices.values()})
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase and phrase in found:
            return phrases[phrase]
    return text


def _to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    word = normalize_text(str(raw))
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"expected yes/no, got {raw!r}")


def _to_int(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("expected a whole number, got yes/no")
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if not text.lstrip("-").isdigit():
        raise ValueError(f"expected a whole number, got {raw!r}")
    return int(text)


def _to_date(raw: Any) -> date:
    if isinstance(raw, date):
        return raw
    try:
        parsed = date.fromisoformat(str(raw).strip())
    except ValueError as error:
        raise ValueError(f"expected a date YYYY-MM-DD, got {raw!r}") from error
    if not 1900 <= parsed.year <= 2100:
        raise ValueError(f"implausible year in {raw!r}")
    return parsed


def coerce_fact(pack: Pack, field_name: str, raw: Any) -> FactValue:
    """Type a value for a pack field. Raises ValueError for unknown fields or bad values."""
    spec = pack.field(field_name)
    if spec is None:
        raise ValueError(f"Unknown field {field_name!r} for procedure {pack.id}")
    return coerce_value(spec, raw)


# ------------------------------------------------------------ validation --
def validate_pack_dir(procedure_id: str, root: Path = PROCEDURES_DIR) -> list[Issue]:
    """Load + validate a pack. Returns issues; an empty list means the pack is valid."""
    try:
        pack = load_pack(procedure_id, root)
    except PackError as error:
        return [Issue(code="UNREADABLE_PACK", message=str(error), item=procedure_id)]
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        return [Issue(code="SCHEMA_ERROR", message=f"{where}: {first['msg']}", item=procedure_id)]
    return validate_pack(pack)


def validate_pack(pack: Pack) -> list[Issue]:
    """All semantic checks on a parsed pack."""
    issues: list[Issue] = []
    issues += _check_unique(pack)
    issues += _check_rules(pack)
    issues += _check_sections(pack)
    issues += _check_cycles(pack)
    issues += _check_procedure_refs(pack)
    issues += _check_sources_text(pack)
    issues += _check_scenarios(pack)
    return issues


def _duplicates(names: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated = []
    for name in names:
        if name in seen:
            repeated.append(name)
        seen.add(name)
    return repeated


def _check_unique(pack: Pack) -> list[Issue]:
    groups = {
        "DUPLICATE_FIELD": [f.name for f in pack.procedure.fields],
        "DUPLICATE_RULE": pack.rule_ids(),
        "DUPLICATE_SECTION": [s.id for s in pack.procedure.sections],
        "DUPLICATE_SOURCE": [s.source_id for s in pack.sources],
    }
    return [Issue(code=code, message=f"{name} is declared twice", item=name)
            for code, names in groups.items() for name in _duplicates(names)]


def _check_rules(pack: Pack) -> list[Issue]:
    issues: list[Issue] = []
    known_rules = set(pack.rule_ids())
    for rule in pack.rules:
        if pack.section(rule.section) is None:
            issues.append(Issue(code="UNKNOWN_SECTION", message=f"section {rule.section!r}", item=rule.rule_id))
        for source_id in rule.source_ids:
            if pack.source(source_id) is None:
                issues.append(Issue(code="UNKNOWN_SOURCE", message=f"source {source_id!r}", item=rule.rule_id))
        if rule.status == "verified" and not rule.source_ids:
            issues.append(Issue(code="NO_SOURCE", message="a verified rule needs a source", item=rule.rule_id))
        for dep in rule.depends_on:
            if dep not in known_rules:
                issues.append(Issue(code="UNKNOWN_RULE", message=f"depends_on {dep!r}", item=rule.rule_id))
        issues += check_condition(pack, rule.condition, rule.rule_id)
    return issues


def check_condition(pack: Pack, condition: Condition, item: str) -> list[Issue]:
    """Field names exist and constant values fit the field type."""
    issues: list[Issue] = []
    for name in condition.leaf_fields():
        if pack.field(name) is None:
            issues.append(Issue(code="UNKNOWN_FIELD", message=f"field {name!r}", item=item))
    for node in _walk(condition):
        issues += _check_leaf_types(pack, node, item)
    return issues


def _walk(condition: Condition) -> list[Condition]:
    nodes = [condition]
    for child in list(condition.args) + ([condition.arg] if condition.arg else []):
        nodes.extend(_walk(child))
    return nodes


def _check_leaf_types(pack: Pack, node: Condition, item: str) -> list[Issue]:
    spec = pack.field(node.field) if node.field else None
    if spec is None:
        return []
    if node.op == "is_true" and spec.type != "bool":
        return [Issue(code="TYPE_MISMATCH", message=f"is_true on {spec.type} field {spec.name}", item=item)]
    if node.offset_days and spec.type != "date":
        return [Issue(code="TYPE_MISMATCH", message=f"offset_days on {spec.type} field {spec.name}", item=item)]
    if node.op not in COMPARISON_OPS:
        return []
    if node.other_field is not None:
        other = pack.field(node.other_field)
        if other is not None and other.type != spec.type:
            return [Issue(code="TYPE_MISMATCH", message=f"{spec.name} vs {other.name}", item=item)]
        return []
    values = node.value if node.op == "in" else [node.value]
    for value in values:
        try:
            coerce_value(spec, value)
        except ValueError as error:
            return [Issue(code="TYPE_MISMATCH", message=f"{spec.name}: {error}", item=item)]
    return []


def _check_sections(pack: Pack) -> list[Issue]:
    return [Issue(code="EMPTY_SECTION", message="section has no rules", item=section.id)
            for section in pack.procedure.sections if not pack.rules_in_section(section.id)]


def _check_cycles(pack: Pack) -> list[Issue]:
    """Depth-first search over depends_on; report the first cycle found."""
    graph = {rule.rule_id: list(rule.depends_on) for rule in pack.rules}
    state: dict[str, str] = {}
    for start in graph:
        cycle = _find_cycle(start, graph, state, [])
        if cycle:
            return [Issue(code="DEPENDENCY_CYCLE", message=" -> ".join(cycle), item=cycle[0])]
    return []


def _find_cycle(node: str, graph: dict[str, list[str]], state: dict[str, str], trail: list[str]) -> list[str]:
    if state.get(node) == "done":
        return []
    if state.get(node) == "visiting":
        return trail[trail.index(node):] + [node]
    state[node] = "visiting"
    for nxt in graph.get(node, []):
        cycle = _find_cycle(nxt, graph, state, trail + [node])
        if cycle:
            return cycle
    state[node] = "done"
    return []


def _check_procedure_refs(pack: Pack) -> list[Issue]:
    issues: list[Issue] = []
    procedure = pack.procedure
    for stage in procedure.stages:
        for rule_id in stage.requires:
            if pack.rule(rule_id) is None:
                issues.append(Issue(code="UNKNOWN_RULE", message=f"stage requires {rule_id!r}", item=stage.id))
    for document in procedure.documents:
        for extract in document.extracts:
            if pack.field(extract.field) is None:
                issues.append(Issue(code="UNKNOWN_FIELD", message=f"extracts {extract.field!r}", item=document.doc_type))
    for derived in procedure.derived:
        for name in [derived.field] + derived.args:
            if pack.field(name) is None:
                issues.append(Issue(code="UNKNOWN_FIELD", message=f"derived uses {name!r}", item=derived.field))
        if derived.op == "document_complete" and pack.document(derived.doc_type) is None:
            issues.append(Issue(code="UNKNOWN_DOCUMENT", message=f"{derived.doc_type!r}", item=derived.field))
        for doc_type in derived.doc_types:
            if pack.document(doc_type) is None:
                issues.append(Issue(code="UNKNOWN_DOCUMENT", message=f"{doc_type!r}", item=derived.field))
    for index, condition in enumerate(procedure.scope_conditions):
        issues += check_condition(pack, condition, f"scope_conditions[{index}]")
    for form in procedure.forms:
        issues += _check_form(pack, form.form_id, form.template, list(form.fields.values()), form.stage_id)
    return issues


def _check_form(pack: Pack, form_id: str, template: str, fields: list[str], stage_id: str) -> list[Issue]:
    issues = []
    if not (pack.root / template).is_file():
        issues.append(Issue(code="MISSING_TEMPLATE", message=template, item=form_id))
    for name in fields:
        if pack.field(name) is None:
            issues.append(Issue(code="UNKNOWN_FIELD", message=f"form uses {name!r}", item=form_id))
    if stage_id and pack.stage(stage_id) is None:
        issues.append(Issue(code="UNKNOWN_STAGE", message=stage_id, item=form_id))
    return issues


def _check_sources_text(pack: Pack) -> list[Issue]:
    folder = pack.root / "sources_text"
    if not folder.is_dir():
        return []
    return [Issue(code="UNKNOWN_SOURCE_TEXT", message=f"{path.name} matches no source_id", item=path.name)
            for path in sorted(folder.glob("*.txt")) if pack.source(path.stem) is None]


def _check_scenarios(pack: Pack) -> list[Issue]:
    try:
        scenarios = load_scenarios(pack)
    except PackError as error:
        return [Issue(code="BAD_SCENARIOS", message=str(error), item="eval/scenarios.json")]
    if len(scenarios) < MIN_SCENARIOS:
        return [Issue(code="FEW_SCENARIOS", message=f"{len(scenarios)} < {MIN_SCENARIOS}", item="eval/scenarios.json")]
    issues = []
    for scenario in scenarios:
        for name in scenario.get("facts", {}):
            if pack.field(name) is None:
                issues.append(Issue(code="UNKNOWN_FIELD", message=f"scenario fact {name!r}", item=scenario.get("scenario_id")))
        for rule_id in scenario.get("expected", {}).get("rule_status", {}):
            if pack.rule(rule_id) is None:
                issues.append(Issue(code="UNKNOWN_RULE", message=f"scenario expects {rule_id!r}", item=scenario.get("scenario_id")))
    return issues


# ------------------------------------------------------------------- CLI --
def main(argv: list[str]) -> int:
    """`validate <id>` prints OK or the issues; exit code 1 when invalid."""
    if len(argv) != 2 or argv[0] != "validate":
        print("usage: python -m src.pack validate <procedure_id>")
        return 2
    procedure_id = argv[1]
    issues = validate_pack_dir(procedure_id)
    if issues:
        for issue in issues:
            print(f"[{issue.code}] {issue.item}: {issue.message}")
        return 1
    pack = load_pack(procedure_id)
    print(f"OK: {procedure_id} ({len(pack.rules)} rules, {len(pack.procedure.sections)} sections, "
          f"{len(pack.procedure.fields)} fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
