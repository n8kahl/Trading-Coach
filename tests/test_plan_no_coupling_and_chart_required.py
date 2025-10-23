from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.agent_server import app
from src.config import get_settings
from src.providers.series import SeriesBundle
import src.services.plan_service as plan_service_module


def _series_bundle(symbols: Iterable[str], as_of: datetime) -> SeriesBundle:
    bundle = SeriesBundle(symbols=list(symbols), mode="live", as_of=as_of)
    for offset, symbol in enumerate(bundle.symbols):
        base_price = 120 + offset * 5
        dates = pd.date_range(end=as_of, periods=30, freq="1D", tz="UTC")
        closes = pd.Series([base_price + 0.3 * idx for idx in range(len(dates))], index=dates)
        frame = pd.DataFrame(
            {
                "open": closes - 0.4,
                "high": closes + 0.8,
                "low": closes - 0.9,
                "close": closes,
                "volume": 750_000,
            },
            index=dates,
        )
        bundle.frames[symbol] = {"1d": frame}
        bundle.latest_close[symbol] = float(frame["close"].iloc[-1])
    return bundle


@pytest.mark.asyncio()
async def test_plan_no_coupling_and_chart_required(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2024, 6, 10, 20, 0, 0, tzinfo=timezone.utc)
    symbol = "NVDA"
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_backend_v2_enabled", True, raising=False)

    async def fake_fetch_series(symbols: list[str], *, mode: str, as_of: datetime) -> SeriesBundle:
        return _series_bundle(symbols, as_of)

    async def fake_select_contracts(symbol: str, as_of_dt: datetime, plan: dict) -> dict:
        direction = plan.get("direction", "long")
        option_type = "call" if direction == "long" else "put"
        return {
            "options_contracts": [
                {
                    "symbol": f"{symbol}-{option_type.upper()}",
                    "option_type": option_type,
                    "strike": plan["targets"][0],
                    "delta": 0.42 if option_type == "call" else -0.42,
                }
            ],
            "rejected_contracts": [],
            "options_note": None,
        }

    monkeypatch.setattr(plan_service_module, "fetch_series", fake_fetch_series)
    monkeypatch.setattr(plan_service_module, "select_contracts", fake_select_contracts)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/gpt/plan",
            json={"symbol": symbol, "simulate_open": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == symbol
    assert payload["planning_context"] == "live"
    assert payload["entry"] is not None
    assert payload["stop"] is not None
    assert payload["targets"] and all(isinstance(t, float) for t in payload["targets"])
    assert payload["key_levels_used"]
    assert payload["snap_trace"]
    assert payload["data_quality"]["snapshot"]["symbol_count"] == 1
    assert payload["chart_url"]
