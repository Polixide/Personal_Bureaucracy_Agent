"""Evaluation scripts (python -m eval.<name>). Shared helper: a console that prints Turkish letters on Windows."""
import sys


def utf8_console() -> None:
    """Print reports as UTF-8: the Windows console default (cp1252) cannot print İ, ş or ğ and the script would crash."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
