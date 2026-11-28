"""All project settings live here (CLAUDE.md rule). Other modules import, never redefine."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths --
PROJECT_ROOT = Path(__file__).resolve().parent
PROCEDURES_DIR = PROJECT_ROOT / "procedures"          # one Procedure Pack per sub-folder
# The Digital Twin (SQLite). TWIN_DATABASE points the app at another file, e.g. a demo database.
DATABASE_PATH = Path(os.getenv("TWIN_DATABASE", PROJECT_ROOT / "data" / "digital_twin.db"))
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"        # user documents (git-ignored)
OUTBOX_DIR = PROJECT_ROOT / "data" / "outbox"         # approved artifacts (git-ignored)

# ---------------------------------------------------------------- models --
# qwen3:8b is the course default; qwen3:4b is used because the reference laptop has a 4 GB GPU.
# Alternatives: "qwen3:8b" (more reliable, slower here), "gemma4:e4b" (8 GB machines).
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
# bge-m3 was chosen by `python -m eval.rag_eval` (docs/eval_rag.md): hit@3 0.76 vs 0.67 for qwen3-embedding:0.6b,
# and 0.60-0.64 vs 0.33-0.36 on the Turkish NVİ pages (English questions, Turkish text). It also uses less VRAM.
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
# Models trained with an instruction before each query (never before passages), by model-name prefix.
EMBED_QUERY_INSTRUCTIONS = {
    "qwen3-embedding": "Instruct: Given a question, retrieve passages of official texts that answer it\nQuery: ",
}
# 127.0.0.1 is localhost; written as an IP so Windows does not try IPv6 first.
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
LLM_TEMPERATURE = 0.0
# Ollama's default context window is small; 8192 tokens fits qwen3:4b plus its cache in 4 GB of VRAM.
LLM_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# ------------------------------------------------------------ agent loop --
MAX_STEPS = 8            # orchestrator ReAct steps
WORKER_STEPS = 5         # worker ReAct steps per section
MAX_REPAIRS = 3          # Gate 1 repair rounds before "needs human review"
MAX_CONSECUTIVE_FAILURES = 2   # invalid decisions in a row before the deterministic fallback
MAX_NEXT_ACTIONS = 5
OBSERVATION_MAX_CHARS = 6000   # tool observations are cut to this size before the model sees them
AGENT_RECURSION_LIMIT = 100    # LangGraph node visits per run (well above MAX_STEPS + repairs)

# ------------------------------------------------------------------ chat --
# The chat assistant (src/agent/chat.py) starts from the state computed by code, so most turns need one call.
# It runs on its own model: measured on the 4 GB GPU (one reply, 1250-token prompt), qwen3:4b took 42 s
# (45% of it on the CPU) and qwen2.5:3b 16 s (all on the GPU). See docs/modules/chat.md.
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
CHAT_MAX_STEPS = 4       # decisions per chat turn (record a fact, recheck, answer)
CHAT_MAX_REPAIRS = 2     # reflection rounds (Gate 1 or self-review), then the reply is the plan written by code
CHAT_HISTORY_TURNS = 4   # previous exchanges the model sees
# Self-review (reflection): after Gate 1 passes, the model reads its own reply against the state and quotes the
# sentences the state does not support; code keeps only quotes that really are in the reply, and the model rewrites.
# "auto": on with a cloud model (one cheap extra call), off with Ollama (about a minute more per turn on 4 GB).
SELF_REVIEW = os.getenv("SELF_REVIEW", "auto").strip().lower()

# ---------------------------------------------------- cloud models (optional) --
# LLM_PROVIDER=anthropic or openai runs the chat assistant, the orchestrator and its workers on that provider.
# They receive the state computed by code (rule results, values already read, the plan), never the document pages:
# reading documents (text, OCR, vision) stays on the local Ollama server unless DOCUMENT_PROVIDER=cloud (below),
# and embeddings always stay local.
# Keys come from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY) and are never stored in the project.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()      # ollama | anthropic | openai
CLOUD_PROVIDERS = ("anthropic", "openai")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")      # or claude-sonnet-5-5 / claude-haiku-5-5
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
CLOUD_EFFORT = os.getenv("CLOUD_EFFORT", "low")          # Claude effort: low | medium | high ("" sends none)
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "")   # "" sends none (model default)
CLOUD_MAX_TOKENS = 16000  # per reply, thinking included
CLOUD_TIMEOUT = 120.0

# ------------------------------------------------------ document processing --
DOCUMENT_OLLAMA_TIMEOUT = 180.0
DOCUMENT_MAX_CHUNKS = 2
DOCUMENT_CHUNK_CHARS = 1600
DOCUMENT_MAX_INPUT_CHARS = 3400
# Vision model for pages that are pictures (scans, photos): it names documents the text cannot (a border stamp
# has no title) and reads values the OCR text did not give. Code checks its values against the OCR text; values not
# found there wait for the user's confirmation. Measured on the 4 GB GPU: about 35 s for a stamp, more for a full
# page (src/document_processing/vision.py). The "-instruct" variant: the default qwen3-vl tag always thinks first.
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct")
VISION_ENABLED = os.getenv("VISION_ENABLED", "1") == "1"
VISION_TIMEOUT = 300.0
VISION_MAX_SIDE = 1024    # pixels: a smaller image is fewer tokens, so a faster answer
VISION_NUM_CTX = 8192

# Documents read by the cloud model (optional). DOCUMENT_PROVIDER=cloud, together with LLM_PROVIDER=openai or
# anthropic, makes that model read the documents: it transcribes scans and photos from the page image instead of
# RapidOCR, reads whole documents instead of DOCUMENT_MAX_INPUT_CHARS, and is the vision model. Code checks its
# values exactly as before (quotes on the page, MRZ check digits, the user confirms what code cannot check).
# The page images and their text are sent to the provider, so this is an explicit choice; "local" keeps them here.
DOCUMENT_PROVIDER = os.getenv("DOCUMENT_PROVIDER", "local").strip().lower()    # local | cloud
CLOUD_DOCUMENT_MAX_INPUT_CHARS = 40000   # characters per request for a cloud model: a whole document
CLOUD_IMAGE_MAX_SIDE = 2048              # pixels of a page image sent to a cloud model: small print stays legible
CLOUD_IMAGE_JPEG_QUALITY = 90            # JPEG: a photographed page is ~1 MB (6 MB as PNG; providers cap images at a few MB)

# ------------------------------------------------------------------- RAG --
RAG_CHUNK_CHARS = 800
RAG_TOP_K = 3
RAG_EMBED_BATCH = 16      # passages per embedding call (one big call times out on a laptop)
RAG_FETCH_TIMEOUT = 30.0
# Sent when downloading official pages: a generic client name, no project or personal details.
RAG_USER_AGENT = "Mozilla/5.0 (compatible; source-text-fetcher/1.0)"
# Web corpus for procedures without a pack (python -m src.rag crawl): seeds.json, corpus.json, corpus/<WEB_ID>.txt.
CORPUS_DIR = PROJECT_ROOT / "knowledge"
CRAWL_DELAY = 1.0         # seconds between two requests (be polite to public servers)
CORPUS_TOP_K = 2          # web passages added to a chat turn
CORPUS_MIN_SCORE = 0.45   # cosine similarity below this: the passage is not about the question
