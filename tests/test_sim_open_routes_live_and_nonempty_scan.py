from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Iterable, List

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.agent_server import app
from src.config import get_settings
from src.providers.series import SeriesBundle
import src.services.scan_service as scan_service_module


def _series_bundle(symbols: Iterable[str], as_of: datetime) -> SeriesBundle:
    bundle = SeriesBundle(symbols=list(symbols), mode="live", as_of=as_of)
    for index, symbol in enumerate(bundle.symbols):
        base_price = 100 + index * 10
        dates = pd.date_range(end=as_of, periods=40, freq="1D", tz="UTC")
        closes = pd.Series([base_price + 0.5 * day for day in range(len(dates))], index=dates)
        frame = pd.DataFrame(
            {
                "open": closes - 0.5,
                "high": closes + 1.0,
                "low": closes - 1.2,
                "close": closes,
                "volume": 1_000_000,
            },
            index=dates,
        )
        bundle.frames[symbol] = {"1d": frame}
        bundle.latest_close[symbol] = float(frame["close"].iloc[-1])
    return bundle


@pytest.mark.asyncio()
async def test_sim_open_routes_live_and_nonempty_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2024, 6, 7, 20, 0, 0, tzinfo=timezone.utc)
    symbols: List[str] = ["AAPL", "MSFT", "NVDA"]
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_backend_v2_enabled", True, raising=False)

    async def fake_fetch_series(symbols_input: List[str], *, mode: str, as_of: datetime) -> SeriesBundle:
        return _series_bundle(symbols_input, as_of)

    async def fake_select_contracts(symbol: str, as_of_dt: datetime, plan: dict) -> dict:
        direction = plan.get("direction", "long")
        option_type = "call" if direction == "long" else "put"
        strike = plan["targets"][0]
        return {
            "options_contracts": [
                {
                    "symbol": f"{symbol}-{option_type.upper()}",
                    "option_type": option_type,
                    "strike": strike,
                    "delta": 0.45 if option_type == "call" else -0.45,
                }
            ],
            "rejected_contracts": [],
            "options_note": None,
        }

    async def fake_resolve(universe: str | list[str], style: str) -> list[str]:
        return list(symbols)

    monkeypatch.setattr(scan_service_module, "fetch_series", fake_fetch_series)
    monkeypatch.setattr(scan_service_module, "select_contracts", fake_select_contracts)
    monkeypatch.setattr("src.agent_server.resolve_universe", fake_resolve)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/gpt/scan",
            json={
                "universe": "LAST_SNAPSHOT",
                "style": "swing",
                "limit": 5,
                "simulate_open": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "live"
    assert payload["meta"]["snapshot"]["symbol_count"] == len(symbols)
    assert len(payload["candidates"]) > 0
    for candidate in payload["candidates"][:3]:
        assert candidate["entry"] is not None
        assert candidate["stop"] is not None
        assert candidate["tps"], "targets must not be empty"
        if candidate.get("chart_url") is not None:
            assert candidate["chart_url"]
