from datetime import datetime

import pytest

from src.services.persist import PlanningPersistence, PlanningRunRecord


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self):
        self.last_params = None
        self.introspection_calls = 0

    async def fetchval(self, *_args, **_kwargs):
        self.introspection_calls += 1
        return 1  # pretend the style column exists

    async def fetchrow(self, _query, *params):
        self.last_params = params
        return {"id": 123}


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.mark.asyncio
async def test_create_scan_run_coerces_iso_timestamp(monkeypatch):
    conn = _FakeConn()
    pool = _FakePool(conn)
    persistence = PlanningPersistence()

    async def fake_get_pool():
        return pool

    async def fake_ensure_schema():
        return True

    monkeypatch.setattr("src.services.persist.db.get_pool", fake_get_pool)
    monkeypatch.setattr("src.services.persist.db.ensure_planning_schema", fake_ensure_schema)
    monkeypatch.setattr("src.services.persist.asyncpg", object())

    record = PlanningRunRecord(
        as_of_utc="2025-10-20T01:13:39.298024+00:00",
        universe_name="A+",
        universe_source="adhoc",
        tickers=["AAPL"],
        indices_context={},
        data_windows={},
        notes=None,
        style="swing",
    )

    run_id = await persistence.create_scan_run(record)

    assert run_id == 123
    assert isinstance(conn.last_params[0], datetime)
    assert conn.last_params[0].tzinfo is not None
    assert conn.last_params[0].isoformat().startswith("2025-10-20T01:13:39.298024")
