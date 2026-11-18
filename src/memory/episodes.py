"""Episodic memory: lessons from repairs that Gate 1 finally accepted.

Stored only when a repaired answer PASSED Gate 1 (validated). Each lesson keeps the failure type, the
rule it was about (if any), and a fixed, generic description of the correction: no user values,
no model text, no chain-of-thought. Retrieval is by procedure and failure type; lessons tied to a
rule are retired when that rule changes (twin.service.record_rule_update).
"""
from src.models import Pack
from src.twin.models import Episode
from src.twin.store import TwinStore

CORRECTIONS = {
    "CLAIM_MISMATCH": "Copy each claim's status from the latest evaluate_rules result; never guess it.",
    "CLAIM_UNKNOWN_RULE": "Claim only rules that exist in the procedure.",
    "MISSING_FAILED_RULE": "Add a claim or a next action for every failing blocking rule.",
    "COMPLETION_CLAIM": "Never say something was submitted, paid, booked or sent; the agent only prepares.",
    "APPROVAL_CLAIM": "Never promise approval; only the authority decides.",
    "UNKNOWN_AS_FAILURE": "Describe unknown rules as missing information, not as failures.",
    "UNSUPPORTED_DATE": "Write only dates that are facts in the Twin or derived from a rule.",
    "UNGROUNDED_ACTION": "Give every next action a ref that is a rule id, finding id or proposal id of this run.",
    "INVENTED_RULE_ID": "Mention only rule ids returned by evaluate_rules.",
    "UNVERIFIED_NOT_LABELLED": "Write 'unverified' in the summary when unverified rules are used.",
}
GENERIC_CORRECTION = "Fix exactly the problem Gate 1 reported, then answer again."
LESSONS_PER_TYPE = 2


def _rule_of(pack: Pack, item: str | None) -> str | None:
    """Keep an issue's item only if it is a rule id (other items may contain user text)."""
    return item if item and pack.rule(item) is not None else None


def record_repairs(store: TwinStore, pack: Pack, user_id: str, rejected: list[list[dict]], trace_id: str) -> int:
    """Store one validated lesson per distinct (failure type, rule) of the rejected answers. Returns the count."""
    seen: set[tuple[str, str | None]] = set()
    rounds = len(rejected)
    for batch in rejected:
        for issue in batch:
            rule_id = _rule_of(pack, issue.get("item"))
            key = (issue["code"], rule_id)
            if key in seen:
                continue
            seen.add(key)
            rule = pack.rule(rule_id) if rule_id else None
            store.add_episode(Episode(
                user_id=user_id, procedure_id=pack.id, failure_type=issue["code"], affected_item=rule_id,
                validator_error=f"Gate 1 rejected the answer with {issue['code']}",
                corrective_action=CORRECTIONS.get(issue["code"], GENERIC_CORRECTION),
                outcome=f"accepted by Gate 1 after {rounds} repair(s)", correction_validated=True,
                rule_or_source_id=rule_id, rule_version=rule.version if rule else None, trace_id=trace_id))
    return len(seen)


def lessons_for(store: TwinStore, procedure_id: str | None, failure_types: list[str] | None = None,
                limit: int = 5) -> list[dict]:
    """Recent distinct lessons for a procedure, optionally only for some failure types."""
    if not procedure_id:
        return []
    if failure_types:
        episodes = [e for code in failure_types for e in store.find_episodes(procedure_id, code, LESSONS_PER_TYPE)]
    else:
        episodes = store.find_episodes(procedure_id, limit=limit * 3)
    lessons, seen = [], set()
    for episode in episodes:
        key = (episode.failure_type, episode.corrective_action)
        if key in seen:
            continue
        seen.add(key)
        lessons.append({"failure_type": episode.failure_type, "lesson": episode.corrective_action,
                        "rule": episode.rule_or_source_id})
    return lessons[:limit]
