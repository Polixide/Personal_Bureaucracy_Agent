import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DATABASE_PATH = PROJECT_ROOT / "data" / "digital_twin.db"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
PLANNER_MAX_OUTPUT_TOKENS = int(os.getenv("PLANNER_MAX_OUTPUT_TOKENS", "900"))
DOCUMENT_OLLAMA_TIMEOUT = 180.0
DOCUMENT_MAX_CHUNKS = 2
DOCUMENT_CHUNK_CHARS = 1600
DOCUMENT_MAX_INPUT_CHARS = 3400
MAX_REFLECTION_ATTEMPTS = 3

PLANNER_MAX_ACTIONS = int(os.getenv("PLANNER_MAX_ACTIONS", "3"))

# Document providers only; Planner/Reflection continue using their existing Ollama path.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
DOCUMENT_FALLBACK_PROVIDER = os.getenv("DOCUMENT_FALLBACK_PROVIDER", "none").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
DOCUMENT_OCR_DPI = int(os.getenv("DOCUMENT_OCR_DPI", "200"))
DOCUMENT_MAX_IMAGE_SIDE = int(os.getenv("DOCUMENT_MAX_IMAGE_SIDE", "3000"))
DOCUMENT_FALLBACK_MAX_PAGES = int(os.getenv("DOCUMENT_FALLBACK_MAX_PAGES", "2"))
DOCUMENT_MIN_TEXT_CHARS = 30
DOCUMENT_OCR_MIN_CONFIDENCE = 0.6
