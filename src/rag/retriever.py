"""Bounded lexical retrieval; scores choose evidence, never decide legal truth."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import math
import os
import re
from time import perf_counter
import unicodedata
from src.core.model_provider import ai_options
from src.rag.ingestion import build_corpus, GUIDANCE_CACHE
from src.rag.public_sources import load_public_sources, relevant_to_case
from src.rag.semantic import EMBEDDING_ERRORS, index_status, rerank

STOP_WORDS = {"a","an","the","of","to","is","are","in","for","with","on","and","or","do","i","my","me","you","your","why","what","how","does","need","this","that","it","be","can","should"}
GENERIC_TERMS = {"official", "guidance", "requirement", "requirements", "rule", "rules", "policy",
                 "information", "say", "about", "please", "know", "tell", "require", "requires", "required", "must"}
ALIASES = {
    "validity":"valid", "valid":"valid", "expiry":"valid", "expired":"valid", "expiration":"valid",
    "pasaport":"passport", "sigorta":"insurance", "health":"insurance", "medical":"insurance",
    "fingerprints":"fingerprint", "payments":"payment", "paid":"payment", "documents":"document",
    "fees":"fee", "completed":"complete", "completion":"complete", "eikamet":"ikamet",
    "dormitory":"dorm", "dormitories":"dorm",
}


def tokens(text):
    text = text.casefold().replace("\u0131", "i")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", text)
    return [ALIASES.get(word, word) for word in words if word not in STOP_WORDS and len(word) > 1]


def _index(rows):
    counts = [Counter(tokens(" ".join((*row.rule_ids, row.title, row.content)))) for row in rows]
    return rows, counts, Counter(word for count in counts for word in count)


def _signature(path):
    try:
        return sha256(path.read_bytes()).digest()
    except FileNotFoundError:
        return None


@lru_cache(maxsize=8)
def _scoped_index(cache_path, canonical_signature, public_fingerprint, scope):
    published = load_public_sources()
    canonical = build_corpus(cache_path)
    if scope == "case":
        rows = [*canonical, *(row for row in published.chunks if relevant_to_case(row))]
    elif scope == "public":
        rows = [*canonical, *published.chunks]
    else:
        raise ValueError("Unknown public retrieval scope")
    return _index(rows)


def _active_index(scope):
    from config import KNOWLEDGE_DIR

    published = load_public_sources()
    paths = (GUIDANCE_CACHE, KNOWLEDGE_DIR / "rules.json", KNOWLEDGE_DIR / "sources.json")
    return _scoped_index(GUIDANCE_CACHE, tuple(_signature(path) for path in paths), published.fingerprint, scope)


def configured_embedding(model=None):
    if model is not None:
        return model
    scoped = ai_options().embedding_model
    return os.getenv("OLLAMA_EMBED_MODEL", "").strip() if scoped is None else scoped


def retrieval_status(scope="case", *, embedding_model=None):
    public = load_public_sources()
    rows, _, _ = _active_index(scope)
    model = configured_embedding(embedding_model)
    try:
        vector = index_status(rows, model)
    except EMBEDDING_ERRORS:
        vector = {"vector_status": "invalid configuration", "indexed_vectors": None, "index_file": None}
    return {"scope": scope, "corpus_status": "partial" if public.issues else "ready",
            "indexed_chunks": len(rows), "fingerprint": sha256("".join(row.chunk_id for row in rows).encode()).hexdigest(),
            "public_sources_loaded": public.loaded,
            "public_sources_listed": public.sources, "issues": public.issues,
            "embedding_model": model or None, "mode": "lexical" if not model else "lexical + local embeddings",
            **vector}


def retrieval_chunks(scope="public"):
    """Return the immutable list used by the standalone public index builder."""
    return tuple(_active_index(scope)[0])


def search_official_sources(query: str, *, limit=3, corpus=None, details=None, embedding_model=None, scope="case"):
    """Search only the reviewed public corpus. Opt-in reranking uses local Ollama."""
    started = perf_counter()
    if not isinstance(query, str) or not query.strip():
        return []
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("Retrieval limit must be 1..5")
    if corpus is None:
        rows, counts, frequencies = _active_index(scope)
        model = configured_embedding(embedding_model)
        all_ids = {row.chunk_id for row in retrieval_chunks("public")} if model else set()
    else:
        rows, counts, frequencies = _index(list(corpus))
        model = embedding_model or ""
        all_ids = {row.chunk_id for row in rows}
    result = _rank(query, rows, counts, frequencies, limit, details=details,
                   embedding_model=model, all_ids=all_ids,
                   allow_lazy=corpus is not None or embedding_model is not None, prefer_rules=scope == "case")
    if details is not None and corpus is None:
        status = retrieval_status(scope, embedding_model=model)
        details.update({key: status[key] for key in ("scope", "fingerprint", "indexed_chunks", "vector_status",
                                                     "indexed_vectors", "corpus_status")})
        details["retrieved_sources"] = len({(row.procedure, row.source_id) for row in result})
    if details is not None:
        details.update(latency_seconds=perf_counter()-started, result_count=len(result), k=limit)
    return result


def _rank(query, rows, counts, frequencies, limit, *, details, embedding_model, all_ids, allow_lazy, prefer_rules):
    if details is not None:
        details.update(mode="lexical", embedding_model=embedding_model or None)
    q = tokens(query[:2000])
    # An explicit document topic must be in the evidence, not merely a shared
    # adjective such as valid/complete. Mixed-topic queries keep either topic.
    topics = tuple(dict.fromkeys(word for word in q if word in {"passport", "insurance", "address", "fingerprint", "payment", "photo"}))
    subjects = set(topics)
    if "after" in q and ("ikamet" in q or "application" in q): q += ["online", "complete", "fingerprint", "gi"]
    q = set(q) - GENERIC_TERMS
    if not q: return []
    ranked = []
    for row, count in zip(rows, counts):
        content_terms = set(tokens(row.content))
        if subjects and not subjects.intersection(content_terms):
            continue
        matched = q.intersection(count)
        if subjects and q - subjects and not (q - subjects).intersection(content_terms):
            continue
        # A single incidental word cannot support a multi-term query without
        # an identified document topic or an explicit canonical rule reference.
        if len(matched) < min(len(q), 2) and not subjects and not any(re.fullmatch(r"r\d{3}", word) for word in matched):
            continue
        score = sum((1 + math.log(count[word])) * math.log(1 + len(rows) / (1 + frequencies[word])) for word in sorted(matched))
        if score > 0:
            # A reviewed rule excerpt remains readily available for exact rule
            # questions, alongside the public passages used for explanations.
            boost = 1.7 if row.rule_ids and (len(matched) >= 2 or matched.intersection(tokens(" ".join(row.rule_ids)))) else 1
            ranked.append(row.model_copy(update={"score": round(score*boost, 6), "lexical_score": round(score, 6)}, deep=True))
    ranked.sort(key=lambda row: (-row.score, row.content_kind != "official_guidance", row.chunk_id))
    # Retain direct, reviewed requirement citations for covered case topics.
    # The rest of the result window is available for official source passages.
    pinned = []
    if prefer_rules and q - subjects:
        for topic in topics[:2]:
            match = next((row for row in ranked if row.rule_ids and topic in tokens(row.content)), None)
            if match and match.chunk_id not in pinned:
                pinned.append(match.chunk_id)
    if embedding_model and ranked:
        try:
            shortlist = ranked[:12]
            primary_ids = {row.chunk_id for row in shortlist}
            shortlist.extend(row for row in ranked if row.chunk_id in pinned and row.chunk_id not in primary_ids)
            eligible = [row for row in rows if not subjects or subjects.intersection(tokens(row.content))]
            ranked = rerank(query[:2000], shortlist, all_ids, embedding_model, available=eligible, allow_lazy=allow_lazy)
        except EMBEDDING_ERRORS:
            if details is not None:
                details["embedding_unavailable"] = True
                details["embedding_message"] = "Semantic search is unavailable. Results use lexical matching."
        else:
            if details is not None:
                details["mode"] = "lexical+ollama-cosine"
    if pinned:
        positions = {key: index for index, key in enumerate(pinned)}
        ranked.sort(key=lambda row: positions.get(row.chunk_id, len(positions)))
    result = []
    seen = set()
    for row in ranked:
        # Same rule/content should not consume the prompt window twice.
        key = tuple(row.rule_ids) or row.content
        if key in seen: continue
        seen.add(key); result.append(row)
        if len(result) == limit: break
    return result
