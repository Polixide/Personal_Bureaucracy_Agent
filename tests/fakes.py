"""Test doubles for the LLM layer: a scripted backend and a deterministic embedder."""
import json
import re
import zlib

from pydantic import BaseModel

EMBED_DIMS = 512


def hash_embedding(text: str, dims: int = EMBED_DIMS) -> list[float]:
    """Bag-of-words vector: each lower-cased word adds 1 at crc32(word) % dims."""
    vector = [0.0] * dims
    for word in re.findall(r"[a-z0-9ğüşöçıİ]+", text.lower()):
        if len(word) > 2:
            vector[zlib.crc32(word.encode("utf-8")) % dims] += 1.0
    return vector


class ScriptedBackend:
    """Returns scripted replies in order: str (raw text), dict/list/BaseModel (as JSON) or an
    Exception instance (raised). Records every call for assertions."""

    def __init__(self, replies=(), embed_fn=hash_embedding):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.embed_calls: list[list[str]] = []
        self.embed_fn = embed_fn

    def chat(self, messages, schema, component):
        self.calls.append({"component": component, "messages": [dict(m) for m in messages], "schema": schema})
        if not self.replies:
            raise AssertionError(f"ScriptedBackend has no reply left for {component}")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, BaseModel):
            return item.model_dump_json()
        if isinstance(item, (dict, list)):
            return json.dumps(item, default=str)
        return item

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return [self.embed_fn(text) for text in texts]
