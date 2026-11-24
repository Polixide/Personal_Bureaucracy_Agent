"""Keyword retrieval (BM25) and a hybrid with the embedding search, for comparison and fallback.

Why: the embedding search needs a model on the machine and misses the right passage on the Turkish NVİ
pages about 4 times in 10 (docs/eval_rag.md). A keyword ranking needs no model and costs milliseconds, and
fusing the two can recover passages one of them misses. `eval/rag_compare.py` measures all three.

Tokens: lower case, accents and Turkish ı/İ folded (the same folding as the rest of the RAG), then cut to the
first 5 letters. Turkish words take long suffixes (kaybı, kaybettim, kaybolan); prefix stemming is a simple,
known baseline for Turkish retrieval. Known limit: consonant softening changes the stem (kayıp -> kaybı gives
"kayip" vs "kaybi"), so those forms do not match. English questions share almost no words with the Turkish
pages; that is the gap the embedding model has to close. Standard library only.
"""
import math
import re
from collections import Counter

from src.document_processing.evidence import normalized
from src.rag.index import Chunk

PREFIX = 5          # letters kept per token
K1, B = 1.5, 0.75   # standard BM25 parameters
RRF_K = 60          # reciprocal rank fusion constant (Cormack et al., 2009)
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Folded, prefix-stemmed word tokens of at least 2 characters."""
    return [word[:PREFIX] for word in TOKEN.findall(normalized(text)) if len(word) >= 2]


class Bm25Index:
    """BM25 over a fixed list of chunks (one pack's passages)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [Counter(tokens(chunk.text)) for chunk in chunks]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        frequency = Counter(term for doc in self.docs for term in doc)
        count = len(self.docs)
        self.idf = {term: math.log(1 + (count - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def score(self, index: int, query_terms: list[str]) -> float:
        """BM25 score of one chunk for the query terms."""
        doc, length, total = self.docs[index], self.lengths[index], 0.0
        for term in query_terms:
            tf = doc.get(term, 0)
            if tf:
                norm = K1 * (1 - B + B * length / self.average)
                total += self.idf[term] * tf * (K1 + 1) / (tf + norm)
        return total

    def rank(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """The k best chunks with a positive score, best first (ties keep chunk order)."""
        terms = list(dict.fromkeys(tokens(query)))
        scored = [(i, self.score(i, terms)) for i in range(len(self.chunks))]
        best = sorted((pair for pair in scored if pair[1] > 0), key=lambda pair: -pair[1])[:k]
        return [(self.chunks[i], round(score, 4)) for i, score in best]


def fuse(rankings: list[list[Chunk]], k: int, rrf_k: int = RRF_K) -> list[tuple[Chunk, float]]:
    """Reciprocal rank fusion: each list adds 1 / (rrf_k + rank) to a chunk; best k by total."""
    totals: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            totals[chunk.chunk_id] = totals.get(chunk.chunk_id, 0.0) + 1 / (rrf_k + position)
            by_id[chunk.chunk_id] = chunk
    best = sorted(totals.items(), key=lambda item: -item[1])[:k]
    return [(by_id[chunk_id], round(score, 5)) for chunk_id, score in best]
