"""Database utilities for persistent storage of Trading Coach artifacts.

This module provides a lightweight asyncpg connection pool along with helpers
for creating schema objects and performing basic CRUD operations.  The current
surface area focuses on persisting idea/plan snapshots so that they survive
process restarts and can be shared across multiple Railway dynos.

All helpers gracefully fall back to `None` when `DB_URL` is not configured so
the application can continue to operate in in-memory mode during development.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

try:  # optional dependency during local development / testing
    import asyncpg
except ImportError:  # pragma: no cover - exercised only when dependency missing
    asyncpg = None

from .config import get_settings

logger = logging.getLogger(__name__)

_POOL: Any | None = None
_POOL_LOCK = asyncio.Lock()

_IDEA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS idea_snapshots (
    plan_id     TEXT        NOT NULL,
    version     INTEGER     NOT NULL,
    snapshot    JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (plan_id, version)
);

CREATE INDEX IF NOT EXISTS idx_idea_snapshots_latest
    ON idea_snapshots (plan_id, version DESC);
"""


async def _create_pool() -> asyncpg.Pool | None:
    if asyncpg is None:
        logger.warning("asyncpg not installed; database features disabled")
        return None
    settings = get_settings()
    db_url = settings.db_url
    if not db_url:
        logger.info("DB_URL not configured; idea snapshots will remain in-memory")
        return None
    db_url = db_url.strip()
    if not db_url:
        logger.info("DB_URL configured but empty after trimming; skipping database setup")
        return None
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://") :]
    try:
        pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=5, statement_cache_size=0)
        logger.info("database connection pool initialised")
        return pool
    except Exception as exc:  # pragma: no cover - only hit when DB misconfigured
        logger.error("failed to initialise database pool", exc_info=exc)
        return None


async def get_pool() -> Any | None:
    """Return an asyncpg pool, initialising it on first use."""

    global _POOL
    if _POOL is not None:
        return _POOL
    async with _POOL_LOCK:
        if _POOL is None:
            _POOL = await _create_pool()
    return _POOL


async def ensure_schema() -> bool:
    """Create required tables if the database is available.

    Returns:
        bool: True when the schema exists (or was created), False otherwise.
    """

    pool = await get_pool()
    if pool is None or asyncpg is None:
        return False
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(_IDEA_TABLE_DDL)
        logger.info("database schema ensured for idea snapshots")
        return True
    except Exception as exc:  # pragma: no cover - depends on external DB state
        logger.error("failed to ensure database schema", exc_info=exc)
        return False


async def store_idea_snapshot(plan_id: str, version: int, snapshot: Dict[str, Any]) -> bool:
    """Persist a plan snapshot to the database."""

    pool = await get_pool()
    if pool is None or asyncpg is None:
        return False
    payload = json.dumps(snapshot)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO idea_snapshots (plan_id, version, snapshot)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (plan_id, version)
                DO UPDATE SET snapshot = EXCLUDED.snapshot,
                              created_at = NOW();
                """,
                plan_id,
                int(version),
                payload,
            )
        return True
    except Exception as exc:  # pragma: no cover - network / DB specific failure
        logger.error("failed to store idea snapshot", exc_info=exc)
        return False


async def fetch_idea_snapshot(plan_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Fetch a plan snapshot from the database.

    Args:
        plan_id: Plan identifier.
        version: Version to retrieve. When None, the latest version is returned.
    """

    pool = await get_pool()
    if pool is None or asyncpg is None:
        return None
    sql = """
        SELECT snapshot
        FROM idea_snapshots
        WHERE plan_id = $1
        ORDER BY version DESC
        LIMIT 1
    """
    params = [plan_id]
    if version is not None:
        sql = """
            SELECT snapshot
            FROM idea_snapshots
            WHERE plan_id = $1 AND version = $2
        """
        params.append(int(version))
    try:
        async with pool.acquire() as conn:
            record = await conn.fetchrow(sql, *params)
        if record is None:
            return None
        snapshot = record["snapshot"]
        if isinstance(snapshot, dict):
            return snapshot
        if snapshot is None:
            return None
        return json.loads(snapshot)
    except Exception as exc:  # pragma: no cover
        logger.error("failed to fetch idea snapshot", exc_info=exc)
        return None


async def fetch_all_plan_versions(plan_id: str) -> Optional[list[Dict[str, Any]]]:
    """Return all stored snapshots for a plan, newest first."""

    pool = await get_pool()
    if pool is None or asyncpg is None:
        return None
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT snapshot
                FROM idea_snapshots
                WHERE plan_id = $1
                ORDER BY version ASC
                """,
                plan_id,
            )
        snapshots: list[Dict[str, Any]] = []
        for row in rows:
            snap = row["snapshot"]
            if isinstance(snap, dict):
                snapshots.append(snap)
            elif snap is not None:
                snapshots.append(json.loads(snap))
        return snapshots
    except Exception as exc:  # pragma: no cover
        logger.error("failed to fetch plan versions", exc_info=exc)
        return None
