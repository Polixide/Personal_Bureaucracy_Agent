"""Bounded lexical retrieval; scores choose evidence, never decide legal truth."""
from collections import Counter
from functools import lru_cache
import math
import re
import unicodedata
from src.rag.ingestion import build_corpus, GUIDANCE_CACHE

STOP_WORDS = {"a","an","the","of","to","is","are","in","for","with","on","and","or","do","i","my","me","you","your","why","what","how","does","need","this","that","it","be","can","should"}
ALIASES = {
    "validity":"valid", "valid":"valid", "expiry":"valid", "expired":"valid", "expiration":"valid",
    "pasaport":"passport", "sigorta":"insurance", "health":"insurance", "medical":"insurance",
    "fingerprints":"fingerprint", "payments":"payment", "paid":"payment", "documents":"document",
    "fees":"fee", "completed":"complete", "completion":"complete", "eikamet":"ikamet",
}


def tokens(text):
    text = text.casefold().replace("\u0131", "i")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", text)
    return [ALIASES.get(word, word) for word in words if word not in STOP_WORDS and len(word) > 1]


@lru_cache(maxsize=4)
def _cached_corpus(cache_signature, rules_signature, sources_signature):
    return build_corpus()


def _signature(path):
    if not path.is_file(): return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def search_official_sources(query: str, *, limit=3, corpus=None):
    """Return source-attributed top matches, or [] when the corpus cannot support it."""
    if not isinstance(query, str) or not query.strip(): return []
    if type(limit) is not int or not 1 <= limit <= 5: raise ValueError("Retrieval limit must be 1..5")
    from config import KNOWLEDGE_DIR
    rows = corpus if corpus is not None else _cached_corpus(_signature(GUIDANCE_CACHE), _signature(KNOWLEDGE_DIR / "rules.json"), _signature(KNOWLEDGE_DIR / "sources.json"))
    q = tokens(query[:2000])
    # An explicit document topic must be in the evidence, not merely a shared
    # adjective such as valid/complete. Mixed-topic queries keep either topic.
    subjects=set(q) & {"passport","insurance","address","fingerprint","payment","photo"}
    if "after" in q and ("ikamet" in q or "application" in q): q += ["online", "complete", "fingerprint", "gi"]
    q = set(q)
    if not q: return []
    counts = [Counter(tokens(" ".join((*row.rule_ids, row.title, row.content)))) for row in rows]
    frequencies = Counter(word for counter in counts for word in counter)
    ranked = []
    for row, count in zip(rows, counts):
        if subjects and not subjects.intersection(tokens(row.content)):continue
        score = sum((1 + math.log(count[word])) * math.log(1 + len(rows) / (1 + frequencies[word])) for word in q if count[word])
        if score > 0:
            ranked.append(row.model_copy(update={"score": round(score, 6)}, deep=True))
    ranked.sort(key=lambda row: (-row.score, row.content_kind != "official_guidance", row.chunk_id))
    result = []
    seen = set()
    for row in ranked:
        # Same rule/content should not consume the prompt window twice.
        key = tuple(row.rule_ids) or row.content
        if key in seen: continue
        seen.add(key); result.append(row)
        if len(result) == limit: break
    return result
