"""record_fact tool: store something the user said, as an UNCONFIRMED statement.

It never counts for the rules until the user confirms it in the app (approved design decision).
"""
from pydantic import BaseModel, Field

from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.twin import service


class RecordFactInput(BaseModel):
    field: str = Field(description="A procedure field name")
    value: str | int | bool = Field(description="Value as stated (dates YYYY-MM-DD)")
    note: str = Field(default="", max_length=200, description="Context of the statement")


class RecordFactOutput(BaseModel):
    fact_id: int
    field: str
    value: str
    confirmed: bool
    message: str


def run(args: RecordFactInput, ctx: ToolContext) -> RecordFactOutput:
    pack = require_pack(ctx)
    try:
        fact = service.record_statement(ctx.store, pack, ctx.user_id, args.field, args.value,
                                        confirmed=False, note=args.note)
    except ValueError as error:
        raise ToolError(str(error)) from error
    return RecordFactOutput(fact_id=fact.fact_id, field=fact.field, value=str(fact.value), confirmed=False,
                            message="Stored as the user's statement. It counts only after the user confirms it in the app.")


SPEC = ToolSpec(
    name="record_fact",
    description="Store a fact the user stated in the chat; it stays unconfirmed until the user confirms it.",
    input_model=RecordFactInput, output_model=RecordFactOutput, fn=run, kind="read", roles=("orchestrator",))
