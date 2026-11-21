"""Source contracts distinguish project rule excerpts from fetched guidance."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    source_id: str
    title: str
    url: str | None
    authority: Literal["government_primary", "sabanci_operational", "sabanci_direct_instruction"]
    content: str = Field(min_length=1)
    content_kind: Literal["project_rule_excerpt", "official_guidance", "project_operational_policy"]
    retrieved_at: str | None = None
    verified_at: str | None = None
    rule_ids: list[str] = Field(default_factory=list)
    score: float = 0.0


class GuidanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    content: str = Field(min_length=1, max_length=500000)
    verified_at: str | None = None
