"""Persistence helpers for planning-mode scans."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

try:  # optional dependency
    import asyncpg  # type: ignore
except ImportError:  # pragma: no cover
    asyncpg = None

from .. import db

logger = logging.getLogger(__name__)


@dataclass
class PlanningRunRecord:
    as_of_utc: str
    universe_name: str
    universe_source: str
    tickers: Sequence[str]
    indices_context: Dict[str, Any]
    data_windows: Dict[str, Any]
    notes: Optional[str]


@dataclass
class PlanningCandidateRecord:
    scan_id: int
    symbol: str
    metrics: Dict[str, Any]
    levels: Dict[str, Any]
    readiness_score: float
    components: Dict[str, Any]
    contract_template: Dict[str, Any]
    requires_live_confirmation: bool
    missing_live_inputs: Sequence[str]


@dataclass
class FinalizationRecord:
    candidate_id: int
    status: str
    finalized_at: Optional[str]
    live_inputs: Dict[str, Any]
    selected_contracts: Dict[str, Any]
    reject_reason: Optional[str]


class PlanningPersistence:
    """Facade around asyncpg for planning-mode persistence."""

    def __init__(self) -> None:
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False

    async def ensure_schema(self) -> bool:
        if self._schema_ready:
            return True
        async with self._schema_lock:
            if self._schema_ready:
                return True
            success = await db.ensure_planning_schema()
            self._schema_ready = bool(success)
        return self._schema_ready

    async def record_universe_snapshot(self, snapshot: "UniverseSnapshot") -> Optional[int]:
        if not await self.ensure_schema():
            return None
        pool = await db.get_pool()
        if pool is None or asyncpg is None:
            return None
        payload = json.dumps(snapshot.metadata or {})
        try:
            async with pool.acquire() as conn:
                record = await conn.fetchrow(
                    """
                    INSERT INTO evening_scan_universe_snapshots (as_of_utc, universe_name, universe_source, tickers, metadata)
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                    RETURNING id
                    """,
                    snapshot.as_of_utc,
                    snapshot.name,
                    snapshot.source,
                    json.dumps(snapshot.symbols),
                    payload,
                )
            return int(record["id"]) if record else None
        except Exception as exc:  # pragma: no cover
            logger.warning("Universe snapshot persistence failed: %s", exc)
            return None

    async def create_scan_run(self, run: PlanningRunRecord) -> Optional[int]:
        if not await self.ensure_schema():
            return None
        pool = await db.get_pool()
        if pool is None or asyncpg is None:
            return None
        try:
            async with pool.acquire() as conn:
                record = await conn.fetchrow(
                    """
                    INSERT INTO evening_scan_runs (
                        as_of_utc,
                        universe_name,
                        universe_source,
                        tickers,
                        indices_context,
                        data_windows,
                        notes
                    )
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7)
                    RETURNING id
                    """,
                    run.as_of_utc,
                    run.universe_name,
                    run.universe_source,
                    json.dumps(list(run.tickers)),
                    json.dumps(run.indices_context),
                    json.dumps(run.data_windows),
                    run.notes,
                )
            return int(record["id"]) if record else None
        except Exception as exc:  # pragma: no cover
            logger.warning("Scan run persistence failed: %s", exc)
            return None

    async def store_candidate(self, candidate: PlanningCandidateRecord) -> Optional[int]:
        if not await self.ensure_schema():
            return None
        pool = await db.get_pool()
        if pool is None or asyncpg is None:
            return None
        try:
            async with pool.acquire() as conn:
                record = await conn.fetchrow(
                    """
                    INSERT INTO evening_candidates (
                        scan_id,
                        symbol,
                        metrics,
                        levels,
                        readiness_score,
                        components,
                        contract_template,
                        requires_live_confirmation,
                        missing_live_inputs
                    )
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6::jsonb, $7::jsonb, $8, $9)
                    RETURNING id
                    """,
                    candidate.scan_id,
                    candidate.symbol,
                    json.dumps(candidate.metrics),
                    json.dumps(candidate.levels),
                    float(candidate.readiness_score),
                    json.dumps(candidate.components),
                    json.dumps(candidate.contract_template),
                    bool(candidate.requires_live_confirmation),
                    list(candidate.missing_live_inputs),
                )
            return int(record["id"]) if record else None
        except Exception as exc:  # pragma: no cover
            logger.warning("Candidate persistence failed: %s", exc)
            return None

    async def upsert_finalization(self, record: FinalizationRecord) -> Optional[int]:
        if not await self.ensure_schema():
            return None
        pool = await db.get_pool()
        if pool is None or asyncpg is None:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO plan_finalizations (
                        candidate_id,
                        finalized_at,
                        live_inputs,
                        selected_contracts,
                        status,
                        reject_reason
                    )
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
                    ON CONFLICT (candidate_id) DO UPDATE SET
                        finalized_at = EXCLUDED.finalized_at,
                        live_inputs = EXCLUDED.live_inputs,
                        selected_contracts = EXCLUDED.selected_contracts,
                        status = EXCLUDED.status,
                        reject_reason = EXCLUDED.reject_reason,
                        created_at = NOW()
                    RETURNING id
                    """,
                    record.candidate_id,
                    record.finalized_at,
                    json.dumps(record.live_inputs),
                    json.dumps(record.selected_contracts),
                    record.status,
                    record.reject_reason,
                )
            return int(row["id"]) if row else None
        except Exception as exc:  # pragma: no cover
            logger.warning("Finalization persistence failed: %s", exc)
            return None


__all__ = [
    "PlanningPersistence",
    "PlanningRunRecord",
    "PlanningCandidateRecord",
    "FinalizationRecord",
]
