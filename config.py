"""All project settings live here (CLAUDE.md rule). Other modules import, never redefine."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths --
PROJECT_ROOT = Path(__file__).resolve().parent
PROCEDURES_DIR = PROJECT_ROOT / "procedures"          # one Procedure Pack per sub-folder
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"            # legacy rule data (old engine and oracle only)
DATABASE_PATH = PROJECT_ROOT / "data" / "digital_twin.db"   # the Digital Twin (SQLite)
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"        # user documents (git-ignored)
OUTBOX_DIR = PROJECT_ROOT / "data" / "outbox"         # approved artifacts (git-ignored)

# ---------------------------------------------------------------- models --
# qwen3:8b is the course default; qwen3:4b is used because the reference laptop has a 4 GB GPU.
# Alternatives: "qwen3:8b" (more reliable, slower here), "gemma4:e4b" (8 GB machines).
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")   # alternative if slow: "qwen3-embedding:0.6b"
# 127.0.0.1 is localhost; written as an IP so Windows does not try IPv6 first.
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
LLM_TEMPERATURE = 0.0

# ------------------------------------------------------------ agent loop --
MAX_STEPS = 8            # orchestrator ReAct steps
WORKER_STEPS = 5         # worker ReAct steps per section
MAX_REPAIRS = 3          # Gate 1 repair rounds before "needs human review"
MAX_CONSECUTIVE_FAILURES = 2   # invalid decisions in a row before the deterministic fallback
MAX_NEXT_ACTIONS = 5

# ------------------------------------------------------ document processing --
DOCUMENT_OLLAMA_TIMEOUT = 180.0
DOCUMENT_MAX_CHUNKS = 2
DOCUMENT_CHUNK_CHARS = 1600
DOCUMENT_MAX_INPUT_CHARS = 3400

# ------------------------------------------------------------------- RAG --
RAG_CHUNK_CHARS = 800
RAG_TOP_K = 3

# ------------------------------------------- legacy names (milestone-1 code) --
# The milestone-1 planner/processor import these names. They are removed with that code (phase 9).
OLLAMA_MODEL = LLM_MODEL
MAX_REFLECTION_ATTEMPTS = MAX_REPAIRS
PLANNER_MAX_OUTPUT_TOKENS = int(os.getenv("PLANNER_MAX_OUTPUT_TOKENS", "900"))
PLANNER_MAX_ACTIONS = int(os.getenv("PLANNER_MAX_ACTIONS", "3"))
