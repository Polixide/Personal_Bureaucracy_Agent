"""Small synchronous SQLite lifecycle for the existing LangGraph workflow."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.graph import build_graph
from src.core.state import CHECKPOINT_TYPES


def session_config(session_id: str) -> dict:
    """Each caller must supply its own nonempty session identifier."""
    if not session_id.strip():
        raise ValueError("A nonempty session ID is required.")
    return {"configurable": {"thread_id": session_id}}


@contextmanager
def persistent_graph(database: str | Path):
    """Open, initialize, and always close a local checkpoint connection.

    Use get_state(config) for inspection and invoke(None, config) to resume
    pending work. Start a fresh event with new_run(...), retaining stored facts.
    """
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        serializer = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
        saver = SqliteSaver(connection, serde=serializer)
        saver.setup()
        yield build_graph(checkpointer=saver)
    finally:
        connection.close()
