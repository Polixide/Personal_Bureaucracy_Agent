import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DATABASE_PATH = PROJECT_ROOT / "data" / "digital_twin.db"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", OLLAMA_MODEL)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", OLLAMA_MODEL)
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
PLANNER_MAX_OUTPUT_TOKENS = int(os.getenv("PLANNER_MAX_OUTPUT_TOKENS", "900"))
DOCUMENT_OLLAMA_TIMEOUT = 180.0
DOCUMENT_MAX_CHUNKS = 2
DOCUMENT_CHUNK_CHARS = 1600
DOCUMENT_MAX_INPUT_CHARS = 3400
MAX_REFLECTION_ATTEMPTS = 3

PLANNER_MAX_ACTIONS = int(os.getenv("PLANNER_MAX_ACTIONS", "3"))

# These are independent server defaults. Cloud document analysis also requires
# an explicit action authorizing the selected file.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
DOCUMENT_PROVIDER = os.getenv("DOCUMENT_PROVIDER", "local").strip().lower()
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
DOCUMENT_FALLBACK_PROVIDER = os.getenv("DOCUMENT_FALLBACK_PROVIDER", "none").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
DOCUMENT_OCR_DPI = int(os.getenv("DOCUMENT_OCR_DPI", "200"))
DOCUMENT_MAX_IMAGE_SIDE = int(os.getenv("DOCUMENT_MAX_IMAGE_SIDE", "3000"))
DOCUMENT_FALLBACK_MAX_PAGES = int(os.getenv("DOCUMENT_FALLBACK_MAX_PAGES", "2"))
DOCUMENT_MIN_TEXT_CHARS = 30
DOCUMENT_OCR_MIN_CONFIDENCE = 0.6

# Cloud document analysis needs explicit authorization for the current file.
DOCUMENT_AI_PROVIDER = os.getenv("DOCUMENT_AI_MODE", os.getenv("DOCUMENT_AI_PROVIDER", "local-only")).strip().lower()
OPENAI_DOCUMENT_MODEL = os.getenv("OPENAI_DOCUMENT_MODEL", OPENAI_MODEL).strip()

DOCUMENT_CLASSIFICATION_MAX_CHARS = int(os.getenv("DOCUMENT_CLASSIFICATION_MAX_CHARS", "1200"))
DOCUMENT_CACHE_ENABLED = os.getenv("DOCUMENT_CACHE_ENABLED", "1") == "1"

# Neda queryable record shares the checkpoint file; checkpoints remain authoritative.
TWIN_RECORD_PATH = DATABASE_PATH

LOCAL_TIMEZONE = "Europe/Istanbul"
RECOMMENDED_TARGET_BUFFER_DAYS = 30

LOCAL_BENCHMARK_MODELS = ("qwen3:4b", "qwen3:8b")
