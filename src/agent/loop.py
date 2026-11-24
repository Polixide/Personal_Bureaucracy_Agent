"""Small helpers shared by the orchestrator and worker ReAct loops."""
import json

from config import OBSERVATION_MAX_CHARS
from src.agent.answer import Decision
from src.llm import Backend, StructuredResult, ask_structured, to_json
from src.tools.common import Observation


def call_key(decision: Decision) -> str:
    """Identity of a tool call, used to detect the model repeating itself."""
    return decision.action + ":" + json.dumps(decision.args, sort_keys=True, default=str)


def assistant_message(decision: Decision) -> dict:
    return {"role": "assistant", "content": decision.model_dump_json()}


def error_message(text: str) -> dict:
    """An error observation: the call did not run, the model must correct itself."""
    return {"role": "user", "content": f"ERROR: {text}"}


def observation_message(observation: Observation) -> dict:
    """A tool observation, cut to OBSERVATION_MAX_CHARS."""
    if observation.ok:
        text = to_json(observation.output)
    else:
        text = to_json({"error": observation.error})
    if len(text) > OBSERVATION_MAX_CHARS:
        text = text[:OBSERVATION_MAX_CHARS] + "...(cut)"
    return {"role": "user", "content": f"OBSERVATION from {observation.tool}: {text}"}


def ask_decision(backend: Backend, messages: list[dict], schema: dict, component: str) -> StructuredResult:
    """One schema-constrained Decision (with the llm module's single retry)."""
    return ask_structured(backend, messages, Decision, component, schema)
