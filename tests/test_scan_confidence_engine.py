import math
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi import FastAPI

from src.lib.data_route import DataRoute
from src.providers.geometry import GeometryDetail, GeometryBundle
from src.providers.series import SeriesBundle
from src.services.scan_confidence import compute_scan_confidence
from src.services.scan_service import generate_scan


def _geometry_detail(
    *,
    bias: str = "long",
    expected_move: float = 5.0,
    last_close: float = 250.0,
    ema_fast: float = 252.0,
    ema_slow: float = 246.0,
) -> GeometryDetail:
    return GeometryDetail(
        symbol="TSLA",
        expected_move=expected_move,
        remaining_atr=max(expected_move - 2.0, 0.0),
        em_used=True,
        snap_trace=["geometry:test"],
        key_levels_used={},
        bias=bias,
        last_close=last_close,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
    )


def test_compute_scan_confidence_full_features():
    detail = _geometry_detail()
    plan_obj = {
        "bias": "long",
        "options_contracts": [{"symbol": "TSLA 250C"}],
        "rejected_contracts": [],
        "options_note": None,
    }
    data_quality = {"series_present": True, "indices_present": True}
    confidence = compute_scan_confidence(
        detail=detail,
        plan_obj=plan_obj,
        data_quality=data_quality,
        planning_context="live",
        banner=None,
    )
    assert confidence is not None
    assert 0.0 <= confidence <= 1.0
    assert math.isclose(confidence, 0.79, rel_tol=0.05)


def test_compute_scan_confidence_missing_liquidity_decreases_score():
    detail = _geometry_detail()
    full_plan = {"bias": "long", "options_contracts": [{"symbol": "TSLA 250C"}]}
    sparse_plan = {"bias": "long", "options_contracts": [], "options_note": "no chain"}
    data_quality = {"series_present": True, "indices_present": True}
    full_conf = compute_scan_confidence(
        detail=detail,
        plan_obj=full_plan,
        data_quality=data_quality,
        planning_context="frozen",
        banner=None,
    )
    sparse_conf = compute_scan_confidence(
        detail=detail,
        plan_obj=sparse_plan,
        data_quality=data_quality,
        planning_context="frozen",
        banner=None,
    )
    assert full_conf is not None
    assert sparse_conf is not None
    assert sparse_conf < full_conf


def test_compute_scan_confidence_missing_emas_returns_none():
    detail = _geometry_detail(ema_fast=None, ema_slow=None)  # type: ignore[arg-type]
    plan_obj = {"bias": "long"}
    data_quality = {"series_present": True}
    confidence = compute_scan_confidence(
        detail=detail,
        plan_obj=plan_obj,
        data_quality=data_quality,
        planning_context="live",
        banner=None,
    )
    assert confidence is None


@pytest.mark.asyncio
async def test_generate_scan_wires_confidence(monkeypatch: pytest.MonkeyPatch):
    as_of = datetime(2025, 10, 20, 14, 30, tzinfo=timezone.utc)
    series_bundle = SeriesBundle(symbols=["TSLA"], mode="live", as_of=as_of)
    daily_index = pd.date_range(end=as_of, periods=40, freq="1D", tz="UTC")
    closes = [240 + idx * 0.8 for idx in range(len(daily_index))]
    daily = pd.DataFrame(
        {
            "open": [close - 0.5 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.2 for close in closes],
            "close": closes,
        },
        index=daily_index,
    )
    series_bundle.frames["TSLA"] = {"1d": daily}
    series_bundle.latest_close["TSLA"] = float(closes[-1])

    async def fake_fetch_series(symbols, mode, as_of):  # noqa: ARG001
        return series_bundle

    async def fake_select_contracts(symbol, as_of, plan):  # noqa: ARG001
        return {
            "options_contracts": [{"symbol": f"{symbol} 250C", "delta": 0.42}],
            "rejected_contracts": [],
            "options_note": None,
        }

    async def fake_run_plan(symbol, series, geometry, route):  # noqa: ARG001
        detail = geometry.get(symbol)
        return {
            "plan_id": f"{symbol}-TEST",
            "bias": detail.bias if detail else "long",
            "entry": detail.last_close - 1 if detail else 100.0,
            "stop": detail.last_close - 3 if detail else 98.0,
            "targets": [(detail.last_close or 100.0) + 2, (detail.last_close or 100.0) + 3],
            "rr_to_t1": 1.5,
            "snap_trace": ["planner:test"],
            "key_levels_used": detail.key_levels_used if detail else {},
            "data_quality": {"series_present": True, "indices_present": True},
            "warnings": [],
            "options_contracts": [{"symbol": f"{symbol} 250C"}],
            "rejected_contracts": [],
            "options_note": None,
            "charts": {"params": {"symbol": symbol, "interval": "15m", "direction": "long", "entry": 100, "stop": 98, "tp": "102"}},
        }

    async def fake_chart_urls(app, payloads):  # noqa: ARG001
        return [None for _ in payloads]

    detail = _geometry_detail()
    geometry_bundle = GeometryBundle(symbols=["TSLA"], series=series_bundle, details={"TSLA": detail})

    async def fake_build_geometry(symbols, series):  # noqa: ARG001
        return geometry_bundle

    monkeypatch.setattr("src.services.scan_service.fetch_series", fake_fetch_series)
    monkeypatch.setattr("src.services.scan_service.select_contracts", fake_select_contracts)
    monkeypatch.setattr("src.services.scan_service.run_plan", fake_run_plan)
    monkeypatch.setattr("src.services.scan_service._chart_urls", fake_chart_urls)
    monkeypatch.setattr("src.services.scan_service.build_geometry", fake_build_geometry)

    route = DataRoute(mode="live", as_of=as_of, planning_context="live")
    app = FastAPI()
    page = await generate_scan(symbols=["TSLA"], style="intraday", limit=1, route=route, app=app)
    candidate = page["candidates"][0]
    assert candidate["confidence"] is not None
    assert 0.0 <= candidate["confidence"] <= 1.0
    assert candidate.get("source_paths", {}).get("confidence") == "scan_confidence_engine"
