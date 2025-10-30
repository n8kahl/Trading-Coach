from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd
import pytest
from fastapi import FastAPI

from src.lib.data_route import DataRoute
from src.providers.geometry import build_geometry
from src.providers.series import SeriesBundle
import src.services.scan_service as scan_service


def _extended_bundle(symbols: list[str], as_of: datetime) -> SeriesBundle:
    bundle = SeriesBundle(symbols=list(symbols), mode="live", as_of=as_of, extended=True)
    for offset, symbol in enumerate(symbols):
        base = 310 + offset * 7
        daily_idx = pd.date_range(end=as_of, periods=20, freq="1D", tz="UTC")
        sixtyfive_idx = pd.date_range(end=as_of, periods=24, freq="65min", tz="UTC")
        fifteen_idx = pd.date_range(end=as_of, periods=40, freq="15min", tz="UTC")
        five_idx = pd.date_range(end=as_of, periods=144, freq="5min", tz="UTC")

        daily = pd.DataFrame(
            {
                "open": base - 1.2,
                "high": base + 2.0,
                "low": base - 1.6,
                "close": [base + 0.3 * idx for idx in range(len(daily_idx))],
                "volume": 950_000,
            },
            index=daily_idx,
        )
        sixtyfive = pd.DataFrame(
            {
                "open": base - 0.9,
                "high": base + 1.3,
                "low": base - 1.1,
                "close": [base + 0.12 * idx for idx in range(len(sixtyfive_idx))],
                "volume": 320_000,
            },
            index=sixtyfive_idx,
        )
        fifteen = pd.DataFrame(
            {
                "open": base - 0.7,
                "high": base + 1.1,
                "low": base - 0.9,
                "close": [base + 0.06 * idx for idx in range(len(fifteen_idx))],
                "volume": 150_000,
            },
            index=fifteen_idx,
        )
        five = pd.DataFrame(
            {
                "open": base - 0.5,
                "high": base + 0.8,
                "low": base - 0.7,
                "close": [base + 0.025 * idx for idx in range(len(five_idx))],
                "volume": 75_000,
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
async def test_generate_scan_extended_threads_session(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2024, 6, 10, 20, 0, tzinfo=timezone.utc)
    symbols = ["AAPL", "MSFT"]
    bundle = _extended_bundle(symbols, as_of)
    geometry = await build_geometry(symbols, bundle)

    async def fake_fetch_series(
        symbols_input: list[str],
        *,
        mode: str,
        as_of: datetime,
        extended: bool = False,
    ) -> SeriesBundle:
        assert extended is True
        return bundle

    async def fake_build_geometry(symbols_input: list[str], series: SeriesBundle):
        assert series.extended is True
        return geometry

    async def fake_select_contracts(symbol: str, as_of_dt: datetime, plan: dict) -> dict:
        return {"options_contracts": [], "rejected_contracts": [], "options_note": None}

    captured_params: list[dict] = []

    async def fake_chart_urls(app: FastAPI, payloads: list[dict]) -> list[str | None]:  # type: ignore[override]
        captured_params.extend(payloads)
        return ["https://chart.test/extended" for _ in payloads]

    monkeypatch.setattr(scan_service, "fetch_series", fake_fetch_series)
    monkeypatch.setattr(scan_service, "build_geometry", fake_build_geometry)
    monkeypatch.setattr(scan_service, "select_contracts", fake_select_contracts)
    monkeypatch.setattr(scan_service, "_chart_urls", fake_chart_urls)

    route = DataRoute(mode="live", as_of=as_of, planning_context="live", extended=True)
    app = FastAPI()
    page = await scan_service.generate_scan(symbols=symbols, style="intraday", limit=2, route=route, app=app)

    assert page["use_extended_hours"] is True
    assert page["meta"]["session"] == "extended"
    assert len(page["candidates"]) > 0
    for candidate in page["candidates"]:
        assert candidate["chart_url"] == "https://chart.test/extended"
        assert candidate["tps"]
        assert candidate["entry"] is not None
    assert any(params and params.get("session") == "extended" for params in captured_params)
    assert all("levels" not in params for params in captured_params if params)
    assert all("supportingLevels" not in params for params in captured_params if params)
    assert all("ui_state" in params for params in captured_params if params)
    for params in (payload for payload in captured_params if payload):
        parsed_state = json.loads(params["ui_state"])
        assert parsed_state["session"] == "after"
        assert parsed_state["style"] in {"intraday", "scalp", "swing"}
        assert 0.0 <= parsed_state["confidence"] <= 1.0
