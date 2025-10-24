from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.agent_server import app
from src.config import get_settings
from src.providers.series import SeriesBundle
import src.services.scan_service as scan_service_module


def _series_bundle(symbols: Iterable[str], as_of: datetime) -> SeriesBundle:
    bundle = SeriesBundle(symbols=list(symbols), mode="live", as_of=as_of)
    for offset, symbol in enumerate(bundle.symbols):
        base_price = 90 + offset * 3
        dates = pd.date_range(end=as_of, periods=25, freq="1D", tz="UTC")
        closes = pd.Series([base_price + 0.2 * day for day in range(len(dates))], index=dates)
        frame = pd.DataFrame(
            {
                "open": closes - 0.3,
                "high": closes + 0.6,
                "low": closes - 0.7,
                "close": closes,
                "volume": 500_000,
            },
            index=dates,
        )
        bundle.frames[symbol] = {"1d": frame}
        bundle.latest_close[symbol] = float(frame["close"].iloc[-1])
    return bundle


@pytest.mark.asyncio()
async def test_blocking_banner_is_only_empty_case(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2024, 6, 12, 20, 0, 0, tzinfo=timezone.utc)
    symbols = ["AAPL", "MSFT"]
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_backend_v2_enabled", True, raising=False)

    async def fake_fetch_series(
        symbols_input: list[str],
        *,
        mode: str,
        as_of: datetime,
        extended: bool = False,
    ) -> SeriesBundle:
        return _series_bundle(symbols_input, as_of)

    async def fake_select_contracts(symbol: str, as_of_dt: datetime, plan: dict) -> dict:
        return {"options_contracts": [], "rejected_contracts": [], "options_note": "no contracts"}

    async def fake_scan(symbols_input: list[str], *, style: str, limit: int, series: SeriesBundle, geometry, route):
        return {
            "phase": "scan",
            "candidates": [],
            "data_quality": {},
            "meta": {},
            "count_candidates": 0,
            "snap_trace": ["scanner:empty"],
        }

    async def fake_resolve(universe: str | list[str], style: str) -> list[str]:
        return list(symbols)

    monkeypatch.setattr(scan_service_module, "fetch_series", fake_fetch_series)
    monkeypatch.setattr(scan_service_module, "select_contracts", fake_select_contracts)
    monkeypatch.setattr(scan_service_module, "run_scan", fake_scan)
    monkeypatch.setattr("src.agent_server.resolve_universe", fake_resolve)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/gpt/scan",
            json={"universe": "LAST_SNAPSHOT", "style": "swing", "limit": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["banner"] == "SCAN_NO_CANDIDATES"
    assert payload["candidates"] == []
    assert payload["count_candidates"] == 0
