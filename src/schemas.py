"""Shared request/response schemas for GPT endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator


class ScanFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_rvol: float | None = Field(default=None, ge=0.0)
    exclude: list[str] | None = None
    min_liquidity_rank: int | None = Field(default=None, ge=1)


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    universe: str | list[str]
    style: Literal["scalp", "intraday", "swing", "leaps"]
    limit: int = Field(default=20, ge=1, le=100)
    asof_policy: Literal["live", "frozen", "live_or_lkg"] = "live_or_lkg"
    filters: ScanFilters | None = None
    cursor: str | None = None
    simulate_open: bool = False
    planning_mode: bool = False
    use_extended_hours: bool = False

    @field_validator("universe")
    @staticmethod
    def _ensure_universe_not_empty(value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not value:
            raise ValueError("Universe list cannot be empty.")
        if isinstance(value, str) and not value.strip():
            raise ValueError("Universe token cannot be empty.")
        return value


class ScanCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    rank: int
    score: float
    reasons: list[str] = Field(default_factory=list)
    plan_id: str | None = None
    entry: float | None = None
    stop: float | None = None
    tps: list[float] = Field(default_factory=list)
    rr_t1: float | None = None
    confidence: float | None = None
    chart_url: AnyUrl | None = None
    target_meta: list[Dict[str, Any]] | None = None
    targets_meta: list[Dict[str, Any]] | None = None
    tp_reasons: list[Dict[str, Any]] = Field(default_factory=list)
    structured_plan: Dict[str, Any] | None = None
    target_profile: Dict[str, Any] | None = None
    runner_policy: Dict[str, Any] | None = None
    snap_trace: list[str] | None = None
    key_levels_used: Dict[str, Any] | None = None
    entry_candidates: list[Dict[str, Any]] = Field(default_factory=list)
    expected_move: float | None = None
    remaining_atr: float | None = None
    em_used: bool | None = None
    risk_block: Dict[str, Any] | None = None
    execution_rules: Dict[str, Any] | None = None
    confluence: list[str] = Field(default_factory=list)
    accuracy_levels: list[str] = Field(default_factory=list)
    events: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None
    options_contracts: list[Dict[str, Any]] = Field(default_factory=list)
    options_note: str | None = None
    rejected_contracts: list[Dict[str, Any]] = Field(default_factory=list)
    entry_distance_pct: float | None = None
    entry_distance_atr: float | None = None
    bars_to_trigger: float | None = None
    actionable_soon: bool | None = None
    source_paths: Dict[str, str] = Field(default_factory=dict)
    planning_snapshot: Dict[str, Any] | None = None


class ScanPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    planning_context: Literal["live", "frozen"]
    use_extended_hours: bool | None = None
    banner: str | None = None
    meta: dict[str, Any]
    candidates: list[ScanCandidate]
    data_quality: dict[str, Any]
    session: dict[str, Any] | None = None
    phase: Literal["scan"] | None = None
    count_candidates: int | None = None
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)
    snap_trace: list[str] | None = None


class FinalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: int
    symbol: str
    status: Literal["finalized", "rejected", "deferred"]
    live_inputs: Dict[str, Any] = Field(default_factory=dict)
    selected_contracts: List[Dict[str, Any]] | None = None
    reject_reason: str | None = None


class FinalizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: int
    status: Literal["finalized", "rejected", "deferred"]
    updated: bool
