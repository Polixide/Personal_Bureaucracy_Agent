"""Local SQLite checkpoints of the canonical AgentState, with explicit lifetime."""
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from config import DATABASE_PATH, PROJECT_ROOT
from src.core.state import CHECKPOINT_TYPES


@contextmanager
def sqlite_checkpointer(db_path: str | Path = DATABASE_PATH):
    """Keep this context open while using the graph; close the connection on exit.

    SqliteSaver owns its checkpoint tables. No parallel Digital Twin schema is
    maintained. Relative paths resolve against the project, not the shell cwd.
    """
    path = Path(db_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Checkpoint database must stay inside the project")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path), check_same_thread=False)) as connection:
        yield SqliteSaver(
            connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES),
        )
