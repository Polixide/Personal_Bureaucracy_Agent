"""ask_user tool: end the turn with one question to the user (kind 'ask')."""
from pydantic import BaseModel, Field

from src.tools.common import ToolContext, ToolError, ToolSpec
from src.twin.models import Event


class AskUserInput(BaseModel):
    question: str = Field(min_length=5, max_length=400, description="One clear question for the user")
    fields: list[str] = Field(default_factory=list, description="Procedure fields the answer will fill, if any")


class AskUserOutput(BaseModel):
    question: str
    fields: list[str]


def run(args: AskUserInput, ctx: ToolContext) -> AskUserOutput:
    pack = ctx.pack()
    unknown = [name for name in args.fields if pack is None or pack.field(name) is None]
    if unknown:
        raise ToolError(f"Unknown fields: {', '.join(unknown)}")
    ctx.question = args.question
    ctx.store.add_event(Event(user_id=ctx.user_id, procedure_id=ctx.procedure_id, kind="question_asked",
                              direction="internal", payload={"fields": args.fields}))
    return AskUserOutput(question=args.question, fields=args.fields)


SPEC = ToolSpec(
    name="ask_user",
    description="Ask the user one question and end this turn (for example when the goal is ambiguous or "
                "information only the user knows is missing).",
    input_model=AskUserInput, output_model=AskUserOutput, fn=run, kind="ask", roles=("orchestrator",), ends_turn=True)
