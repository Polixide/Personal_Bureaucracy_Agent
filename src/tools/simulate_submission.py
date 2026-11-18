"""simulate_submission tool (sandbox): try a stage in the mock world first."""
from pydantic import BaseModel, Field

from src.sandbox.mock_world import run_simulation
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack


class SimulateInput(BaseModel):
    stage_id: str = Field(description="A stage id of the procedure (see the procedure's stages)")


class SimulateOutput(BaseModel):
    simulation_id: int
    stage_id: str
    accepted: bool
    failing: list[str]
    unknown: dict[str, list[str]]
    message: str


def run(args: SimulateInput, ctx: ToolContext) -> SimulateOutput:
    pack = require_pack(ctx)
    try:
        outcome, stored = run_simulation(ctx.store, pack, ctx.user_id, ctx.today, args.stage_id)
    except ValueError as error:
        raise ToolError(str(error)) from error
    return SimulateOutput(simulation_id=stored.simulation_id, stage_id=outcome.stage_id, accepted=outcome.accepted,
                          failing=outcome.failing, unknown=outcome.unknown, message=outcome.message)


SPEC = ToolSpec(
    name="simulate_submission",
    description="Try an official stage in the sandbox (mock portal) using the current facts. Nothing is sent. "
                "Artifacts tied to a stage can only be approved after an accepted simulation.",
    input_model=SimulateInput, output_model=SimulateOutput, fn=run, kind="sandbox", roles=("orchestrator",))
