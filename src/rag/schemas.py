"""Source contracts distinguish project rule excerpts from fetched guidance."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    source_id: str
    title: str
    url: str | None
    authority: Literal["government_primary", "sabanci_operational", "sabanci_direct_instruction", "crosscheck_only"]
    content: str = Field(min_length=1)
    content_kind: Literal["project_rule_excerpt", "official_guidance", "project_operational_policy", "reference_guidance"]
    retrieved_at: str | None = None
    verified_at: str | None = None
    section: str | None = None
    page: int | None = None
    source_version: str | None = None
    procedure: Literal["ikamet_first", "ikamet_extension", "tc_kimlik_first", "tc_kimlik_renewal", "public_web"] = "ikamet_first"
    file_ref: str | None = None
    organization: str | None = None
    topic: str | None = None
    rule_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    lexical_score: float | None = None
    cosine_score: float | None = None


class GuidanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    content: str = Field(min_length=1, max_length=500000)
    verified_at: str | None = None
    section: str | None = Field(default=None, max_length=240)
    page: int | None = Field(default=None, ge=1, strict=True)
    source_version: str | None = Field(default=None, max_length=80)
