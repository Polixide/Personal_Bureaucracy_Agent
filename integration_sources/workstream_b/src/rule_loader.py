"""Load rule metadata without executing condition strings."""

import json
from pathlib import Path


def load_rules() -> list[dict]:
    """Read the project's rules independently of the working directory."""
    path = Path(__file__).resolve().parent.parent / "knowledge" / "rules.json"
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Rules file not found: {path}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise ValueError(f"Rules file must contain a 'rules' list: {path}")
    return data["rules"]
