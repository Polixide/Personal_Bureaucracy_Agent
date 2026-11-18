"""Run one event with memory: load the Digital Twin, run the graph, persist.

    with TwinStore("data/digital_twin.sqlite") as store:
        final = run_event(build_graph(), store, "student_001",
                          UserEvent(kind="upload", file_path="insurance_valid.pdf"))

Why a wrapper and not a LangGraph checkpointer: the graph stays unchanged and
every fact lives in its own table (see checkpointer.py). The state written back
is the run's final state, so:
  * PASS runs persist new documents, facts, and the blockers / completed steps
    set by `state_update` (the only node that writes them);
  * FAIL runs still persist what was observed (e.g. an analyzed document that
    expires too early, so it is not asked for again) but no blockers or
    completed steps change, because `state_update` never ran;
  * every run, PASS or FAIL, is logged in `runs` / `run_issues`.
Twin and run log are committed in one transaction.
"""
from datetime import datetime

from src.core.state import AgentState, UserEvent, new_run
from src.memory.checkpointer import TwinStore
from src.memory.codec import utc_now


def start_state(store: TwinStore, user_id: str, event: UserEvent,
                procedure: str = "ikamet") -> dict:
    """Graph input: the stored Digital Twin plus fresh run memory."""
    twin = store.load_twin(user_id)
    return {**twin, **new_run(event, user_id=user_id,
                              procedure=twin.get("active_procedure", procedure))}


def run_event(graph, store: TwinStore, user_id: str, event: UserEvent,
              procedure: str = "ikamet", started_at: datetime | None = None,
              path: list[str] | None = None) -> AgentState:
    """Run one event and persist the outcome. If `path` is given, the names of
    the nodes that ran are appended to it (used by the evaluation harness)."""
    started_at = started_at or utc_now()
    final: AgentState = {}
    for mode, chunk in graph.stream(start_state(store, user_id, event, procedure),
                                    stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        elif path is not None:
            path.extend(chunk)
    store.commit_run(final, started_at)
    return final
