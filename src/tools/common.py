"""The tool contract shared by every tool (see the add-tool skill).

A tool is a plain function `fn(args: InputModel, ctx: ToolContext) -> OutputModel`, described by a
ToolSpec. Kinds: read (reads or records Twin state), sandbox (mock world only), artifact (creates a
PROPOSED action, executed only after user approval), ask (asks the user), orchestrator (delegation).
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from config import OUTBOX_DIR
from src.llm import Backend
from src.models import Pack
from src.twin.facts import FactView
from src.twin.service import twin_view
from src.twin.store import TwinStore

ToolKind = Literal["read", "sandbox", "artifact", "ask", "orchestrator"]
Role = Literal["orchestrator", "worker"]
BOTH_ROLES: tuple[str, ...] = ("orchestrator", "worker")


class ToolError(Exception):
    """An expected tool failure. Its message becomes the error observation the model sees."""


@dataclass
class ToolContext:
    """Everything a tool may use during one agent run. Not sent to the model."""
    store: TwinStore
    packs: dict[str, Pack]
    user_id: str
    today: date
    backend: Backend | None = None
    procedure_id: str | None = None
    outbox_dir: Path = OUTBOX_DIR
    trace_id: str = ""
    run_worker: Callable[..., Any] | None = None
    created_proposals: list[str] = field(default_factory=list)
    findings: dict[str, Any] = field(default_factory=dict)
    question: str | None = None

    def pack(self) -> Pack | None:
        """The active procedure's pack, if one is selected."""
        return self.packs.get(self.procedure_id) if self.procedure_id else None

    def view(self) -> FactView:
        """Current fact view of the active procedure."""
        return twin_view(self.store, require_pack(self), self.user_id, self.today)


def require_pack(ctx: ToolContext) -> Pack:
    """The active pack, or a ToolError telling the model what to do first."""
    pack = ctx.pack()
    if pack is None:
        raise ToolError("No active procedure. Call search_procedures with the user's goal first.")
    return pack


class ToolSpec(BaseModel):
    """Registry entry of one tool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], BaseModel]
    kind: ToolKind
    roles: tuple[str, ...] = BOTH_ROLES
    ends_turn: bool = False


class Observation(BaseModel):
    """What the model sees after a tool call: the output, or a short error (the call did not run)."""
    tool: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None
