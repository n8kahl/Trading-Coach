"""Shared request/response schemas for GPT endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field, field_validator


class ScanFilters(BaseModel):
    min_rvol: float | None = Field(default=None, ge=0.0)
    exclude: list[str] | None = None
    min_liquidity_rank: int | None = Field(default=None, ge=1)


class ScanRequest(BaseModel):
    universe: str | list[str]
    style: Literal["scalp", "intraday", "swing", "leaps"]
    limit: int = Field(default=20, ge=1, le=100)
    asof_policy: Literal["live", "frozen", "live_or_lkg"] = "live_or_lkg"
    filters: ScanFilters | None = None
    cursor: str | None = None

    @field_validator("universe")
    @staticmethod
    def _ensure_universe_not_empty(value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not value:
            raise ValueError("Universe list cannot be empty.")
        if isinstance(value, str) and not value.strip():
            raise ValueError("Universe token cannot be empty.")
        return value


class ScanCandidate(BaseModel):
    symbol: str
    score: float
    reasons: list[str] = Field(default_factory=list)
    plan_id: str | None = None
    entry: float | None = None
    stop: float | None = None
    tps: list[float] = Field(default_factory=list)
    rr_t1: float | None = None
    confidence: float | None = None
    chart_url: AnyUrl | None = None


class ScanPage(BaseModel):
    as_of: str
    planning_context: Literal["live", "frozen"]
    banner: str | None = None
    meta: dict[str, Any]
    candidates: list[ScanCandidate]
    data_quality: dict[str, Any]
    next_cursor: str | None = None
