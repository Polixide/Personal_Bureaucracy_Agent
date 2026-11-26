"""Optional local cosine reranking, with a versioned public-corpus vector cache."""
import hashlib
import json
from math import hypot, isfinite
from pathlib import Path
from collections import Counter
from functools import lru_cache
import re
from tempfile import NamedTemporaryFile
from threading import Lock

import httpx
from ollama import Client, ResponseError

from config import OLLAMA_HOST, PROJECT_ROOT
from src.core.llm_diagnostics import logged_request

INDEX_DIR = PROJECT_ROOT / ".local" / "rag_index"
INDEX_VERSION = 1
PUBLIC_CORPUS = "ikamet_first_public"
EMBED_BATCH = 16
SEMANTIC_THRESHOLD = 0.45  # Daniele's BGE-M3 threshold for the web corpus.
_index_lock = Lock()


def _normalized(vector):
    if not isinstance(vector, (list, tuple)) or not 1 <= len(vector) <= 8192:
        raise ValueError("Invalid embedding dimensions")
    if not all(type(value) in (int, float) and isfinite(value) for value in vector):
        raise ValueError("Invalid embedding values")
    length = hypot(*vector)
    if not isfinite(length):
        raise ValueError("Invalid embedding length")
    return [value / length for value in vector] if length else list(vector)


def _index_file(model, directory):
    key = hashlib.sha256(f"{PUBLIC_CORPUS}:{model}".encode("utf-8")).hexdigest()[:24]
    return directory / f"{key}.json"


@lru_cache(maxsize=3)
def _load_vectors(path, model, mtime_ns, size):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("version"), payload.get("corpus"), payload.get("model")) != (INDEX_VERSION, PUBLIC_CORPUS, model):
            return {}
        return {key: _normalized(value) for key, value in payload["vectors"].items()}
    except (OSError, ValueError, AttributeError, KeyError, OverflowError):
        return {}


def _read_index(path, model, corpus_ids):
    try:
        stat = path.stat()
    except OSError:
        return {}
    return {key: value for key, value in _load_vectors(path, model, stat.st_mtime_ns, stat.st_size).items()
            if key in corpus_ids}


def _write_index(path, model, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".index-", delete=False) as stream:
        temp = Path(stream.name)
    try:
        temp.write_text(json.dumps({"version": INDEX_VERSION, "corpus": PUBLIC_CORPUS,
                                   "model": model, "vectors": vectors}, allow_nan=False), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _model_tag(model):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,119}", model) or "://" in model:
        raise ValueError("Use a local Ollama embedding model tag")
    return model


def local_embedder(model):
    """One local client per operation. Only the public corpus reaches its index."""
    client = Client(host=OLLAMA_HOST, timeout=30.0)

    def embed(texts, task):
        response = logged_request(client.embed, provider="ollama", model=model, task=task,
                                  model_kwargs={"model": model, "input": texts, "truncate": False})
        return response.embeddings

    return embed


def _write_batches(path, model, cache, rows, embed, *, dimensions=None):
    missing = [row for row in rows if row.chunk_id not in cache]
    for offset in range(0, len(missing), EMBED_BATCH):
        batch = missing[offset:offset + EMBED_BATCH]
        vectors = embed([row.content for row in batch], "rag_embedding_index")
        if len(vectors) != len(batch):
            raise ValueError("Unexpected chunk embedding count")
        checked = {}
        for row, vector in zip(batch, vectors, strict=True):
            normalized = _normalized(vector)
            dimensions = dimensions or len(normalized)
            if len(normalized) != dimensions:
                raise ValueError("Incompatible embedding dimensions")
            checked[row.chunk_id] = normalized
        cache.update(checked)
        _write_index(path, model, cache)


def _compatible(vectors):
    dimensions = Counter(len(value) for value in vectors.values()).most_common(1)
    return {key: value for key, value in vectors.items() if len(value) == dimensions[0][0]} if dimensions else {}


def index_status(rows, model, *, directory=None):
    if not model:
        return {"vector_status": "disabled", "indexed_vectors": None, "index_file": None}
    path = _index_file(_model_tag(model), INDEX_DIR if directory is None else directory)
    keys = {row.chunk_id for row in rows}
    vectors = _compatible(_read_index(path, model, keys))
    status = "ready" if keys and len(vectors) == len(keys) else ("partial" if vectors else "missing")
    return {"vector_status": status, "indexed_vectors": len(vectors), "index_file": path.name}


def build_index(rows, model, *, directory=None, embed=None, rebuild=False, retained_ids=()):
    """Index the whole supplied public corpus in resumable, content-keyed batches."""
    rows = list({row.chunk_id: row for row in rows}.values())
    path = _index_file(_model_tag(model), INDEX_DIR if directory is None else directory)
    embed = embed or local_embedder(model)
    known_ids = {row.chunk_id for row in rows}.union(retained_ids)
    with _index_lock:
        cache = {} if rebuild else _compatible(_read_index(path, model, known_ids))
        dimensions = len(next(iter(cache.values()))) if cache else None
        _write_batches(path, model, cache, rows, embed, dimensions=dimensions)
    return index_status(rows, model, directory=directory)


def rerank(query, ranked, corpus_ids, model, *, directory=None, embed=None, available=(),
           allow_lazy=True):
    """Use complete cached vectors for cosine scores. Optionally fill lexical candidates."""
    path = _index_file(_model_tag(model), INDEX_DIR if directory is None else directory)
    cache = _read_index(path, model, corpus_ids)
    if not allow_lazy and not cache:
        raise ValueError("Build the local embedding index before semantic search")
    embed = embed or local_embedder(model)
    query_text = "Instruct: Given a question, retrieve passages that answer it\nQuery: " + query if model.startswith("qwen3-embedding") else query
    query_rows = embed([query_text], "rag_embedding_query")
    if len(query_rows) != 1:
        raise ValueError("Unexpected query embedding count")
    query_vector = _normalized(query_rows[0])
    size = len(query_vector)
    with _index_lock:
        cache = {key: vec for key, vec in _read_index(path, model, corpus_ids).items() if len(vec) == size}
        if allow_lazy:
            _write_batches(path, model, cache, ranked, embed, dimensions=size)
    candidates = {row.chunk_id: row for row in ranked}
    complete = bool(available) and all(row.chunk_id in cache for row in available)
    if complete and model.split(":", 1)[0] == "bge-m3":
        candidates.update((row.chunk_id, row) for row in available if row.chunk_id not in candidates)
    if not candidates or not any(key in cache for key in candidates):
        raise ValueError("No compatible embedding vectors")
    best = max((row.score for row in ranked), default=1.0)
    result = []
    for key, row in candidates.items():
        vector = cache.get(key)
        cosine = sum(a*b for a, b in zip(vector, query_vector, strict=True)) if vector is not None else None
        if row.score <= 0 and (cosine is None or cosine < SEMANTIC_THRESHOLD):
            continue
        lexical = row.score or 0.0
        score = 0.7 * lexical / best + 0.3 * max(0.0, cosine or 0.0)
        result.append(row.model_copy(update={"score": round(score, 6), "lexical_score": lexical,
                                             "cosine_score": round(cosine, 6) if cosine is not None else None}))
    return sorted(result, key=lambda row: (-row.score, row.chunk_id))


EMBEDDING_ERRORS = (OSError, ConnectionError, httpx.HTTPError, ResponseError, ValueError, TypeError, OverflowError)
