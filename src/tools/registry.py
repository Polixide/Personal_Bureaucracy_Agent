"""Tool registry: which tools exist, who may call them, and the safe way to run one.

`run_tool` is the only place a tool is executed:
  1. unknown tool or wrong role       -> error observation, nothing runs;
  2. arguments fail Pydantic validation -> error observation, nothing runs;
  3. the tool raises                   -> error observation (the agent loop never crashes);
  4. otherwise                         -> the validated output as a dict.
"""
from typing import Any

from pydantic import BaseModel, ValidationError

from src.llm import short_validation_error
from src.tools import (ask_user, evaluate_rules, extract_document, read_twin, record_fact,
                       search_procedures)
from src.tools.common import Observation, ToolContext, ToolError, ToolSpec


def base_tools() -> list[ToolSpec]:
    """Tools defined in src/tools (delegate_section is added by the agent)."""
    tools = [read_twin.SPEC, search_procedures.SPEC, evaluate_rules.SPEC, extract_document.SPEC,
             record_fact.SPEC, ask_user.SPEC]
    tools += _optional_tools()
    return tools


def _optional_tools() -> list[ToolSpec]:
    """Sandbox, artifact and retrieval tools (added in phases 6 and 7)."""
    return []


def build_registry(extra: list[ToolSpec] | None = None) -> dict[str, ToolSpec]:
    """Name -> spec for every tool, plus `extra` (for example delegate_section)."""
    registry = {spec.name: spec for spec in base_tools()}
    for spec in extra or []:
        registry[spec.name] = spec
    return registry


def tools_for(registry: dict[str, ToolSpec], role: str) -> dict[str, ToolSpec]:
    """The subset a role may call (workers get a restricted set)."""
    return {name: spec for name, spec in registry.items() if role in spec.roles}


def run_tool(registry: dict[str, ToolSpec], name: str, raw_args: dict[str, Any] | None,
             ctx: ToolContext, role: str) -> Observation:
    """Validate and run one tool call. Never raises."""
    spec = registry.get(name)
    if spec is None:
        return Observation(tool=name, ok=False, error=f"Unknown tool {name!r}. Use one of: {', '.join(sorted(registry))}")
    if role not in spec.roles:
        return Observation(tool=name, ok=False, error=f"{name} is not available to the {role}")
    try:
        args = spec.input_model.model_validate(raw_args or {})
    except ValidationError as error:
        return Observation(tool=name, ok=False, error=f"Invalid arguments: {short_validation_error(error)}")
    try:
        output = spec.fn(args, ctx)
    except ToolError as error:
        return Observation(tool=name, ok=False, error=str(error))
    except Exception as error:  # the agent loop must survive any tool bug
        return Observation(tool=name, ok=False, error=f"{name} failed: {type(error).__name__}: {error}")
    if not isinstance(output, spec.output_model):
        return Observation(tool=name, ok=False, error=f"{name} returned an unexpected result")
    return Observation(tool=name, ok=True, output=output.model_dump(mode="json"))


def _arg_summary(model: type[BaseModel]) -> dict[str, str]:
    """{argument: 'type - description'} for a prompt; small on purpose."""
    schema = model.model_json_schema()
    summary = {}
    for name, prop in schema.get("properties", {}).items():
        kind = prop.get("type") or "/".join(item.get("type", "?") for item in prop.get("anyOf", [])) or "value"
        required = name in schema.get("required", [])
        summary[name] = f"{kind}{'' if required else ' (optional)'} - {prop.get('description', '')}".rstrip(" -")
    return summary


def describe_tools(registry: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    """Compact tool list for the model: name, kind, what it does, its arguments."""
    return [{"name": spec.name, "kind": spec.kind, "description": spec.description,
             "args": _arg_summary(spec.input_model)} for spec in registry.values()]
