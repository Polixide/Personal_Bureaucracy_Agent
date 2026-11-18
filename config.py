"""Global configuration."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Digital Twin database (src/memory/checkpointer.py). Override with PBA_DB_PATH.
DB_PATH = Path(os.environ.get("PBA_DB_PATH", ROOT / "data" / "digital_twin.sqlite"))
