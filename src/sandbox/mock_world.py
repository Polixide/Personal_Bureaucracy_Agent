"""Mock world: a pretend portal that applies the pack's rules to one stage. Nothing real is touched.

A stage (procedure.json `stages`) is accepted only if every required rule and every prerequisite of
those rules passes. The result is stored with a fingerprint of the facts it saw, so an approval
later can check that the facts did not change since the successful simulation.
"""
from datetime import date

from pydantic import BaseModel, Field

from src.models import FactMap, Pack
from src.rules import evaluate_rules, required_rules
from src.twin.facts import facts_hash
from src.twin.models import Event, Simulation
from src.twin.service import twin_view
from src.twin.store import TwinStore

SIMULATION_NOTE = "This is a simulation in the agent's sandbox; nothing was submitted to any real service."


class SimulationOutcome(BaseModel):
    """What the mock portal answered."""
    stage_id: str
    accepted: bool
    required: list[str]
    failing: list[str] = Field(default_factory=list)
    unknown: dict[str, list[str]] = Field(default_factory=dict)
    message: str
    facts_hash: str


def simulate(pack: Pack, values: FactMap, stage_id: str) -> SimulationOutcome:
    """Apply the stage's required rules (with prerequisites) to the facts."""
    stage = pack.stage(stage_id)
    if stage is None:
        raise ValueError(f"Unknown stage {stage_id!r}. Stages: {', '.join(s.id for s in pack.procedure.stages)}")
    required = required_rules(pack, stage.requires)
    results = {r.rule_id: r for r in evaluate_rules(pack, values) if r.rule_id in required}
    failing = [rule_id for rule_id in required if results[rule_id].status == "fail"]
    unknown = {rule_id: results[rule_id].missing_fields for rule_id in required if results[rule_id].status == "unknown"}
    accepted = not failing and not unknown
    if accepted:
        message = f"The mock portal would accept '{stage.title}'. {SIMULATION_NOTE}"
    else:
        message = (f"The mock portal would refuse '{stage.title}': failing {failing or 'none'}, "
                   f"missing information for {sorted(unknown) or 'none'}. {SIMULATION_NOTE}")
    return SimulationOutcome(stage_id=stage_id, accepted=accepted, required=required, failing=failing,
                             unknown=unknown, message=message, facts_hash=facts_hash(values))


def run_simulation(store: TwinStore, pack: Pack, user_id: str, today: date, stage_id: str) -> tuple[SimulationOutcome, Simulation]:
    """Simulate on the Twin's current facts and store the attempt."""
    outcome = simulate(pack, twin_view(store, pack, user_id, today).fact_map(), stage_id)
    with store.transaction():
        stored = store.add_simulation(Simulation(user_id=user_id, procedure_id=pack.id, stage_id=stage_id,
                                                 accepted=outcome.accepted, facts_hash=outcome.facts_hash,
                                                 result=outcome.model_dump(mode="json")))
        store.add_event(Event(user_id=user_id, procedure_id=pack.id, kind="simulation", direction="internal",
                              payload={"stage_id": stage_id, "accepted": outcome.accepted}))
    return outcome, stored


def promotion_check(store: TwinStore, pack: Pack, user_id: str, today: date, stage_id: str | None) -> tuple[bool, str]:
    """Promotion gate: an action tied to a stage needs an accepted simulation on the current facts."""
    if not stage_id:
        return True, "no stage requirement"
    latest = store.latest_simulation(user_id, pack.id, stage_id)
    if latest is None:
        return False, f"Run simulate_submission for stage '{stage_id}' first."
    if not latest.accepted:
        return False, f"The last simulation of '{stage_id}' was refused; fix the problems and simulate again."
    current = facts_hash(twin_view(store, pack, user_id, today).fact_map())
    if latest.facts_hash != current:
        return False, f"Facts changed since the last simulation of '{stage_id}'; simulate again."
    return True, "accepted simulation on the current facts"
