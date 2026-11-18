"""search_sources tool (read): find the official text behind a requirement, to cite and explain it."""
from pydantic import BaseModel, Field

from src.llm import LLMError
from src.rag.index import Hit, search
from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack

NOTE = "Sources explain and cite. They never decide a rule result: use evaluate_rules for that."


class SearchSourcesInput(BaseModel):
    query: str = Field(min_length=3, max_length=300, description="What you want to find in the official sources")
    k: int = Field(default=3, ge=1, le=5, description="How many passages")


class SearchSourcesOutput(BaseModel):
    hits: list[Hit]
    note: str = NOTE


def run(args: SearchSourcesInput, ctx: ToolContext) -> SearchSourcesOutput:
    pack = require_pack(ctx)
    if ctx.backend is None:
        raise ToolError("No embedding model available")
    try:
        hits = search(ctx.store, pack, ctx.backend, args.query, args.k)
    except LLMError as error:
        raise ToolError(f"Source search unavailable: {error}") from error
    return SearchSourcesOutput(hits=hits)


SPEC = ToolSpec(
    name="search_sources",
    description="Search the procedure's official source texts and return passages with their source_id, to cite "
                "or explain a requirement.",
    input_model=SearchSourcesInput, output_model=SearchSourcesOutput, fn=run, kind="read")
