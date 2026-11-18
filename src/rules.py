"""Three-valued rule evaluator. Rules are JSON condition trees; facts are typed values.

A missing fact makes a comparison UNKNOWN (None), never FAIL. Groups combine like this:
  all: False if any arg is False, else None if any is None, else True
  any: True if any arg is True, else None if any is None, else False
Every argument is evaluated (no short-circuit) so the list of missing fields is complete.
This module never mentions a specific rule: everything comes from the pack.
"""
from datetime import date, timedelta
from typing import Any

from src.models import Condition, FactMap, Pack, Rule, RuleResult

Truth = bool | None


def all_of(values: list[Truth]) -> Truth:
    """Three-valued AND."""
    if False in values:
        return False
    if None in values:
        return None
    return True


def any_of(values: list[Truth]) -> Truth:
    """Three-valued OR."""
    if True in values:
        return True
    if None in values:
        return None
    return False


def _lookup(name: str, facts: FactMap, missing: set[str]) -> Any:
    """Read one fact; remember its name when it is unknown."""
    value = facts.get(name)
    if value is None:
        missing.add(name)
    return value


def _as_date(value: Any) -> Any:
    """Turn an ISO date string into a date; leave other values alone."""
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return value
    return value


def _right_side(condition: Condition, facts: FactMap, missing: set[str]) -> Any:
    """The comparison target: a constant `value` or `other_field`, plus `offset_days` for dates."""
    if condition.other_field is not None:
        value = _lookup(condition.other_field, facts, missing)
    else:
        value = condition.value
    if value is None or not condition.offset_days:
        return value
    value = _as_date(value)
    if not isinstance(value, date):
        raise ValueError(f"offset_days needs a date, got {value!r}")
    return value + timedelta(days=condition.offset_days)


def _compare(op: str, left: Any, right: Any) -> bool:
    """Apply one comparison operator to two known values."""
    if isinstance(left, date):
        right = [_as_date(item) for item in right] if op == "in" else _as_date(right)
    if op == "eq":
        return left == right
    if op == "in":
        return left in right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    raise ValueError(f"Unknown comparison operator {op!r}")


def evaluate_condition(condition: Condition, facts: FactMap, missing: set[str]) -> Truth:
    """Evaluate one condition tree. Unknown fact names are added to `missing`."""
    if condition.op == "all":
        return all_of([evaluate_condition(arg, facts, missing) for arg in condition.args])
    if condition.op == "any":
        return any_of([evaluate_condition(arg, facts, missing) for arg in condition.args])
    if condition.op == "not":
        inner = evaluate_condition(condition.arg, facts, missing)
        return None if inner is None else not inner
    left = _lookup(condition.field, facts, missing)
    if condition.op == "is_true":
        return None if left is None else left is True
    right = _right_side(condition, facts, missing)
    if left is None or right is None:
        return None
    return _compare(condition.op, left, right)


def _status(truth: Truth) -> str:
    """Map three-valued truth to a rule status."""
    if truth is None:
        return "unknown"
    return "pass" if truth else "fail"


def evaluate_rule(rule: Rule, facts: FactMap) -> RuleResult:
    """Verdict for one rule. `blocked_by` is filled by evaluate_rules (it needs all results)."""
    missing: set[str] = set()
    truth = evaluate_condition(rule.condition, facts, missing)
    return RuleResult(
        rule_id=rule.rule_id, section=rule.section, description=rule.description,
        status=_status(truth), severity=rule.severity,
        missing_fields=sorted(missing) if truth is None else [],
        source_ids=list(rule.source_ids), rule_status=rule.status)


def evaluate_rules(pack: Pack, facts: FactMap, section: str | None = None) -> list[RuleResult]:
    """Evaluate every rule of the pack (dependencies need all of them), then filter by section.

    `blocked_by` lists prerequisites (`depends_on`) that do not pass yet. Verdicts themselves
    never depend on prerequisites.
    """
    results = [evaluate_rule(rule, facts) for rule in pack.rules]
    by_id = {result.rule_id: result for result in results}
    for rule, result in zip(pack.rules, results):
        result.blocked_by = [dep for dep in rule.depends_on
                             if dep in by_id and by_id[dep].status != "pass"]
    if section is None:
        return results
    return [result for result in results if result.section == section]


def required_rules(pack: Pack, rule_ids: list[str]) -> list[str]:
    """`rule_ids` plus all their transitive prerequisites, in pack order."""
    needed: set[str] = set()
    pending = list(rule_ids)
    while pending:
        rule_id = pending.pop()
        if rule_id in needed:
            continue
        needed.add(rule_id)
        rule = pack.rule(rule_id)
        if rule is not None:
            pending.extend(rule.depends_on)
    return [rule_id for rule_id in pack.rule_ids() if rule_id in needed]
