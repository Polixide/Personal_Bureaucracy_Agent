"""Retrieval over a pack's sources_text/: chunk, embed (EMBED_MODEL), cosine search with numpy.

Vectors are cached in the Twin database (rag_vectors) per pack, chunk and model, and recomputed
only when the chunk text changes. RAG cites and explains; it never decides a rule result.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from config import (CORPUS_DIR, CORPUS_TOP_K, EMBED_MODEL, EMBED_QUERY_INSTRUCTIONS, RAG_CHUNK_CHARS,
                    RAG_EMBED_BATCH, RAG_TOP_K)
from src.llm import Backend
from src.models import Pack
from src.twin.store import TwinStore

CORPUS_SCOPE = "_web_corpus"   # rag_vectors.procedure_id of the web corpus


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
    url: str = ""


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
    return folder_chunks(pack.root / "sources_text")


def folder_chunks(folder: Path) -> list[Chunk]:
    """Chunks of every <SOURCE_ID>.txt in a folder, in a stable order."""
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


def chunk_vectors(store: TwinStore, scope: str, backend: Backend, chunks: list[Chunk]) -> np.ndarray:
    """One vector per chunk, from the cache or from the embedding model in small batches.

    Each batch is cached before the next one starts, so a timeout keeps the work already done.
    """
    model = model_name(backend)
    vectors: list[list[float] | None] = [store.get_vector(scope, c.chunk_id, model, text_hash(c.text)) for c in chunks]
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    for start in range(0, len(missing), RAG_EMBED_BATCH):
        batch = missing[start:start + RAG_EMBED_BATCH]
        for i, vector in zip(batch, backend.embed([chunks[i].text for i in batch])):
            store.put_vector(scope, chunks[i].chunk_id, model, chunks[i].source_id, text_hash(chunks[i].text), vector)
            vectors[i] = vector
    return np.array(vectors, dtype=float)


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row with the query (0 for zero vectors)."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def query_text(backend: Backend, query: str) -> str:
    """The text embedded for a question: some models expect an instruction before queries (not passages)."""
    model = model_name(backend)
    for prefix, instruction in EMBED_QUERY_INSTRUCTIONS.items():
        if model.startswith(prefix):
            return instruction + query
    return query


PASSAGE_CHARS = 600


WORD_END = re.compile(r"\s")


def passage_text(text: str, limit: int = PASSAGE_CHARS, slack: int = 200) -> str:
    """The first `limit` characters and the end of the word they cut (a URL cut in half gave a broken link, seen live)."""
    if len(text) <= limit:
        return text
    end = WORD_END.search(text, limit)
    return text[:end.start()] if end is not None and end.start() - limit <= slack else text[:limit]


def rank(store: TwinStore, scope: str, backend: Backend, chunks: list[Chunk], query: str, k: int) -> list[tuple[Chunk, float]]:
    """The k chunks most similar to the query (positive scores only), best first.

    Passages are embedded before the query, so a first search fills the cache in order.
    """
    if not chunks:
        return []
    passages = chunk_vectors(store, scope, backend, chunks)
    question = np.array(backend.embed([query_text(backend, query)])[0], dtype=float)
    scores = cosine_scores(passages, question)
    order = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:k]
    return [(chunks[i], round(float(scores[i]), 4)) for i in order]


def search(store: TwinStore, pack: Pack, backend: Backend, query: str, k: int = RAG_TOP_K) -> list[Hit]:
    """The k passages of the pack's sources most similar to the query."""
    hits = []
    for chunk, score in rank(store, pack.id, backend, pack_chunks(pack), query, k):
        source = pack.source(chunk.source_id)
        hits.append(Hit(source_id=chunk.source_id, title=source.title if source else chunk.source_id,
                        chunk_id=chunk.chunk_id, text=passage_text(chunk.text), score=score, url=source.url if source else ""))
    return hits


def search_corpus(store: TwinStore, backend: Backend, query: str, k: int = CORPUS_TOP_K,
                  folder: Path = CORPUS_DIR, min_score: float = 0.0) -> list[Hit]:
    """The k passages of the web corpus (python -m src.rag crawl) most similar to the query."""
    from src.rag.crawl import load_corpus   # crawl imports ingest; keep index light for the app
    pages = {page.source_id: page for page in load_corpus(folder)}
    chunks = [chunk for chunk in folder_chunks(folder / "corpus") if chunk.source_id in pages]
    hits = []
    for chunk, score in rank(store, CORPUS_SCOPE, backend, chunks, query, k):
        if score < min_score:
            continue
        page = pages[chunk.source_id]
        hits.append(Hit(source_id=chunk.source_id, title=page.title, chunk_id=chunk.chunk_id, text=passage_text(chunk.text),
                        score=score, url=page.url))
    return hits
