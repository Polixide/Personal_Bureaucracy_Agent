"""Retrieval over a pack's sources_text/: chunk, embed (EMBED_MODEL), cosine search with numpy.

Vectors are cached in the Twin database (rag_vectors) per pack, chunk and model, and recomputed
only when the chunk text changes. RAG cites and explains; it never decides a rule result.
"""
import hashlib

import numpy as np
from pydantic import BaseModel

from config import EMBED_MODEL, RAG_CHUNK_CHARS, RAG_TOP_K
from src.llm import Backend
from src.models import Pack
from src.twin.store import TwinStore


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str


class Hit(BaseModel):
    source_id: str
    title: str
    chunk_id: str
    text: str
    score: float


def split_long(paragraph: str, max_chars: int) -> list[str]:
    """Cut a paragraph at line ends (or hard, if one line is too long) into pieces <= max_chars."""
    pieces, current = [], ""
    for line in paragraph.splitlines():
        while len(line) > max_chars:
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if current and len(current) + 1 + len(line) > max_chars:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = RAG_CHUNK_CHARS) -> list[str]:
    """Paragraph-based chunks of at most max_chars."""
    chunks, current = [], ""
    for paragraph in [p.strip() for p in text.split("\n\n") if p.strip()]:
        for piece in split_long(paragraph, max_chars):
            if current and len(current) + 2 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def pack_chunks(pack: Pack) -> list[Chunk]:
    """Chunks of every sources_text/<SOURCE_ID>.txt of the pack, in a stable order."""
    folder = pack.root / "sources_text"
    if not folder.is_dir():
        return []
    chunks = []
    for path in sorted(folder.glob("*.txt")):
        for index, text in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            chunks.append(Chunk(chunk_id=f"{path.stem}#{index}", source_id=path.stem, text=text))
    return chunks


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def model_name(backend: Backend) -> str:
    """Cache key for vectors: the embedding model of the backend."""
    return getattr(backend, "embed_model", EMBED_MODEL)


def chunk_vectors(store: TwinStore, pack: Pack, backend: Backend, chunks: list[Chunk]) -> np.ndarray:
    """One vector per chunk, from the cache or (in one batch) from the embedding model."""
    model = model_name(backend)
    vectors: list[list[float] | None] = [store.get_vector(pack.id, c.chunk_id, model, text_hash(c.text)) for c in chunks]
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        fresh = backend.embed([chunks[i].text for i in missing])
        for i, vector in zip(missing, fresh):
            store.put_vector(pack.id, chunks[i].chunk_id, model, chunks[i].source_id, text_hash(chunks[i].text), vector)
            vectors[i] = vector
    return np.array(vectors, dtype=float)


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row with the query (0 for zero vectors)."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def search(store: TwinStore, pack: Pack, backend: Backend, query: str, k: int = RAG_TOP_K) -> list[Hit]:
    """The k most similar chunks (positive scores only), best first."""
    chunks = pack_chunks(pack)
    if not chunks:
        return []
    scores = cosine_scores(chunk_vectors(store, pack, backend, chunks), np.array(backend.embed([query])[0], dtype=float))
    order = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:k]
    return [Hit(source_id=chunks[i].source_id, title=pack.source(chunks[i].source_id).title, chunk_id=chunks[i].chunk_id,
                text=chunks[i].text[:600], score=round(float(scores[i]), 4)) for i in order]
