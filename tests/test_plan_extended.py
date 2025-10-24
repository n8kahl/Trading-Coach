from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi import FastAPI

from src.lib.data_route import DataRoute
from src.providers.geometry import build_geometry
from src.providers.series import SeriesBundle
import src.services.plan_service as plan_service


def _extended_bundle(symbols: list[str], as_of: datetime) -> SeriesBundle:
    bundle = SeriesBundle(symbols=list(symbols), mode="live", as_of=as_of, extended=True)
    for offset, symbol in enumerate(symbols):
        base = 200 + offset * 5
        daily_idx = pd.date_range(end=as_of, periods=20, freq="1D", tz="UTC")
        sixtyfive_idx = pd.date_range(end=as_of, periods=30, freq="65min", tz="UTC")
        fifteen_idx = pd.date_range(end=as_of, periods=40, freq="15min", tz="UTC")
        five_idx = pd.date_range(end=as_of, periods=144, freq="5min", tz="UTC")

        daily = pd.DataFrame(
            {
                "open": base - 0.8,
                "high": base + 1.6,
                "low": base - 1.2,
                "close": [base + 0.4 * idx for idx in range(len(daily_idx))],
                "volume": 800_000,
            },
            index=daily_idx,
        )
        sixtyfive = pd.DataFrame(
            {
                "open": base - 0.6,
                "high": base + 1.2,
                "low": base - 1.0,
                "close": [base + 0.1 * idx for idx in range(len(sixtyfive_idx))],
                "volume": 250_000,
            },
            index=sixtyfive_idx,
        )
        fifteen = pd.DataFrame(
            {
                "open": base - 0.5,
                "high": base + 0.9,
                "low": base - 0.7,
                "close": [base + 0.05 * idx for idx in range(len(fifteen_idx))],
                "volume": 120_000,
            },
            index=fifteen_idx,
        )
        five = pd.DataFrame(
            {
                "open": base - 0.4,
                "high": base + 0.6,
                "low": base - 0.6,
                "close": [base + 0.02 * idx for idx in range(len(five_idx))],
                "volume": 60_000,
            },
            index=five_idx,
        )

        bundle.frames[symbol] = {
            "1d": daily,
            "65m": sixtyfive,
            "15m": fifteen,
            "5m": five,
        }
        bundle.latest_close[symbol] = float(daily["close"].iloc[-1])
    return bundle


@pytest.mark.asyncio
async def test_generate_plan_extended_adds_session(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2024, 6, 10, 20, 0, tzinfo=timezone.utc)
    symbol = "TSLA"
    bundle = _extended_bundle([symbol], as_of)
    geometry = await build_geometry([symbol], bundle)

    async def fake_fetch_series(
        symbols: list[str],
        *,
        mode: str,
        as_of: datetime,
        extended: bool = False,
    ) -> SeriesBundle:
        assert extended is True
        return bundle

    async def fake_build_geometry(symbols: list[str], series: SeriesBundle):
        assert series.extended is True
        return geometry

    async def fake_select_contracts(symbol: str, as_of_dt: datetime, plan: dict) -> dict:
        return {"options_contracts": [], "rejected_contracts": [], "options_note": None}

    async def fake_resolve_chart(app: FastAPI, params: dict) -> str:  # type: ignore[override]
        return "https://chart.test/extended"

    monkeypatch.setattr(plan_service, "fetch_series", fake_fetch_series)
    monkeypatch.setattr(plan_service, "build_geometry", fake_build_geometry)
    monkeypatch.setattr(plan_service, "select_contracts", fake_select_contracts)
    monkeypatch.setattr(plan_service, "_resolve_chart_url", fake_resolve_chart)

    route = DataRoute(mode="live", as_of=as_of, planning_context="live", extended=True)
    app = FastAPI()
    plan = await plan_service.generate_plan(symbol, style="intraday", route=route, app=app)

    assert plan["use_extended_hours"] is True
    assert plan["charts"]["params"]["session"] == "extended"
    assert plan["charts"]["params"]["range"] == "5d"
    assert plan["entry_actionability"] == pytest.approx(1.0)
    assert "ENTRY_STALE" not in plan.get("warnings", [])
    assert plan["chart_url"] == "https://chart.test/extended"
