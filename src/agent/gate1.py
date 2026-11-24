"""Gate 1: deterministic verification of the final answer. No model calls, no state changes.

Checks (each failure is an Issue with a code the model gets back for repair):
  CLAIM_UNKNOWN_RULE    a claim names a rule that does not exist
  CLAIM_MISMATCH        a claim's status differs from the latest evaluate_rules result
  MISSING_FAILED_RULE   a failing blocking rule is neither claimed nor addressed by an action
  COMPLETION_CLAIM      text says something was submitted / paid / booked / sent / approved
  APPROVAL_CLAIM        text promises or announces approval
  UNKNOWN_AS_FAILURE    text calls an unknown rule failed
  UNSUPPORTED_DATE      a date that is not a fact and not derived from a rule
  UNGROUNDED_ACTION     a next action whose ref is not a rule, finding or proposal of this run
  INVENTED_RULE_ID      text cites a rule id the procedure does not have
  UNVERIFIED_NOT_LABELLED  unverified rules (or procedure) used without saying "unverified"
Generic: nothing here names a procedure, a rule or a field.
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.agent.answer import FinalAnswer
from src.models import FactMap, Issue, Pack, RuleResult

COMPLETION_CLAIM = re.compile(
    r"\b(?:already|successfully)\s+(?:been\s+)?(?:submitted|paid|booked|renewed|sent|approved|filed)\b"
    r"|\b(?:has|have)\s+(?:been\s+)?(?:submitted|paid|booked|renewed|sent|approved|filed)\b"
    r"|\b(?:was|were|is|are)\s+(?:been\s+)?(?:submitted|paid|booked|renewed|sent|approved|filed)\b"
    r"|\b(?:i|we)\s+(?:have\s+)?(?:submitted|paid|booked|sent|filed)\b", re.IGNORECASE)
APPROVAL_CLAIM = re.compile(
    r"\b(?:permit|application|request)\b.{0,60}\b(?:officially approved|will be approved|approval is guaranteed)\b"
    r"|\bguaranteed?\b.{0,30}\b(?:approval|acceptance|permit)\b", re.IGNORECASE)
REPORTED_BY_USER = re.compile(r"\byou (?:reported|said|told|stated|mentioned)\b|\baccording to you\b", re.IGNORECASE)
FAILURE_WORDS = re.compile(
    r"\b(?:fail(?:s|ed|ing)?|not met|not satisfied|insufficient|invalid|(?:does|do|did)(?: not|n't) meet|non-?compliant)\b",
    re.IGNORECASE)
ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
EUROPEAN_DATE = re.compile(r"(?<!\d)(\d{2})[./](\d{2})[./](\d{4})(?!\d)")
RULE_ID_LIKE = re.compile(r"\b[A-Z]{1,4}\d{3}\b")
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")


@dataclass
class GateContext:
    """What Gate 1 compares the answer with (computed fresh from the Twin)."""
    pack: Pack | None
    results: list[RuleResult] = field(default_factory=list)
    values: FactMap = field(default_factory=dict)
    refs: set[str] = field(default_factory=set)


def sentences(text: str) -> list[str]:
    return [part for part in SENTENCE_END.split(text) if part.strip()]


def dates_in(text: str) -> list[date]:
    """ISO and DD.MM.YYYY / DD/MM/YYYY dates written in the text (impossible dates are skipped)."""
    found = []
    for year, month, day in ISO_DATE.findall(text):
        found.append((int(year), int(month), int(day)))
    for day, month, year in EUROPEAN_DATE.findall(text):
        found.append((int(year), int(month), int(day)))
    output = []
    for year, month, day in found:
        try:
            output.append(date(year, month, day))
        except ValueError:
            continue
    return output


def allowed_dates(pack: Pack | None, values: FactMap) -> set[date]:
    """Dates that are facts, or a fact shifted by a rule's offset_days, or a rule constant."""
    allowed = {value for value in values.values() if isinstance(value, date)}
    if pack is None:
        return allowed
    for rule in pack.rules:
        for node in rule.condition.nodes():
            allowed |= _rule_dates(node, values)
    return allowed


def _rule_dates(node, values: FactMap) -> set[date]:
    dates: set[date] = set()
    shift = timedelta(days=node.offset_days)
    other = values.get(node.other_field) if node.other_field else None
    left = values.get(node.field) if node.field else None
    if node.offset_days and isinstance(other, date):
        dates.add(other + shift)
    if node.offset_days and isinstance(left, date):
        dates.add(left - shift)
    if isinstance(node.value, str):
        try:
            dates.add(date.fromisoformat(node.value))
        except ValueError:
            pass
    return dates


def answer_text(answer: FinalAnswer) -> str:
    return "\n".join([answer.summary] + [action.title for action in answer.next_actions])


# ------------------------------------------------------------- checks --
def check_claims(answer: FinalAnswer, gate: GateContext) -> list[Issue]:
    by_id = {result.rule_id: result for result in gate.results}
    issues = []
    for claim in answer.claims:
        result = by_id.get(claim.rule_id)
        if result is None:
            issues.append(Issue(code="CLAIM_UNKNOWN_RULE", message=f"{claim.rule_id} is not a rule of this procedure", item=claim.rule_id))
        elif result.status != claim.status:
            issues.append(Issue(code="CLAIM_MISMATCH", item=claim.rule_id,
                                message=f"{claim.rule_id} is '{result.status}' in evaluate_rules, not '{claim.status}'"))
    return issues


def check_failed_rules_addressed(answer: FinalAnswer, gate: GateContext) -> list[Issue]:
    mentioned = {claim.rule_id for claim in answer.claims} | {action.ref for action in answer.next_actions}
    return [Issue(code="MISSING_FAILED_RULE", item=r.rule_id, message=f"{r.rule_id} fails; claim it or add an action for it")
            for r in gate.results if r.status == "fail" and r.severity == "blocking" and r.rule_id not in mentioned]


def check_text_claims(text: str) -> list[Issue]:
    """Completion / approval language, except sentences that quote what the user reported."""
    issues = []
    for sentence in sentences(text):
        if REPORTED_BY_USER.search(sentence):
            continue
        if COMPLETION_CLAIM.search(sentence):
            issues.append(Issue(code="COMPLETION_CLAIM", item=sentence[:80],
                                message="Do not say anything was submitted, paid, booked, sent or approved; you only prepare"))
        if APPROVAL_CLAIM.search(sentence):
            issues.append(Issue(code="APPROVAL_CLAIM", item=sentence[:80], message="Only the authority decides; never promise approval"))
    return issues


def check_unknown_as_failure(text: str, gate: GateContext) -> list[Issue]:
    unknown = [r.rule_id for r in gate.results if r.status == "unknown"]
    issues = []
    for sentence in sentences(text):
        for rule_id in unknown:
            if re.search(rf"\b{re.escape(rule_id)}\b", sentence) and FAILURE_WORDS.search(sentence):
                issues.append(Issue(code="UNKNOWN_AS_FAILURE", item=rule_id,
                                    message=f"{rule_id} is unknown (information missing), not failed; ask for the information"))
    return issues


def check_dates(text: str, gate: GateContext) -> list[Issue]:
    allowed = allowed_dates(gate.pack, gate.values)
    return [Issue(code="UNSUPPORTED_DATE", item=day.isoformat(),
                  message=f"{day.isoformat()} is not a known fact or a rule-derived date; remove it or use read_twin")
            for day in dates_in(text) if day not in allowed]


def check_action_refs(answer: FinalAnswer, gate: GateContext) -> list[Issue]:
    rule_ids = {r.rule_id for r in gate.results}
    return [Issue(code="UNGROUNDED_ACTION", item=action.ref,
                  message=f"next action ref {action.ref!r} is not a rule id, finding id or proposal id from this run")
            for action in answer.next_actions if action.ref not in rule_ids | gate.refs]


def check_rule_ids_in_text(text: str, gate: GateContext) -> list[Issue]:
    rule_ids = {r.rule_id for r in gate.results}
    found = sorted(set(RULE_ID_LIKE.findall(text)) - rule_ids - gate.refs)
    return [Issue(code="INVENTED_RULE_ID", item=token, message=f"{token} is not a rule of this procedure") for token in found]


def check_unverified_label(answer: FinalAnswer, gate: GateContext) -> list[Issue]:
    if gate.pack is None:
        return []
    unverified = {r.rule_id for r in gate.results if r.rule_status == "unverified"}
    used = ({claim.rule_id for claim in answer.claims} | {action.ref for action in answer.next_actions}) & unverified
    needs_label = bool(used) or gate.pack.procedure.status == "unverified"
    if needs_label and "unverified" not in answer.summary.casefold():
        item = ", ".join(sorted(used)) or gate.pack.id
        return [Issue(code="UNVERIFIED_NOT_LABELLED", item=item,
                      message="The answer uses unverified rules: say 'unverified' in the summary")]
    return []


def verify(answer: FinalAnswer, gate: GateContext) -> list[Issue]:
    """All Gate 1 checks. An empty list means the answer may be shown."""
    text = answer_text(answer)
    issues: list[Issue] = []
    issues += check_claims(answer, gate)
    issues += check_failed_rules_addressed(answer, gate)
    issues += check_text_claims(text)
    issues += check_unknown_as_failure(text, gate)
    issues += check_dates(text, gate)
    issues += check_action_refs(answer, gate)
    issues += check_rule_ids_in_text(text, gate)
    issues += check_unverified_label(answer, gate)
    return issues


def verify_text(text: str, gate: GateContext) -> list[Issue]:
    """Gate 1 text checks alone (used for e-mail drafts before they are proposed)."""
    return check_text_claims(text) + check_unknown_as_failure(text, gate) + check_dates(text, gate) + check_rule_ids_in_text(text, gate)


def build_gate_context(ctx) -> GateContext:
    """Fresh verdicts and facts from the Twin, plus the ids this run may reference.

    `ctx` is a src.tools.common.ToolContext (imported lazily to keep this module free of tool code).
    """
    from src.tools.evaluate_rules import current_results
    refs = set(ctx.findings) | set(ctx.created_proposals)
    pack = ctx.pack()
    if pack is None:
        return GateContext(pack=None, refs=refs)
    return GateContext(pack=pack, results=current_results(ctx), values=ctx.view().fact_map(), refs=refs)
