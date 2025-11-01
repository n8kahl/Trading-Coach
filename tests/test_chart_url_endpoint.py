from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from urllib.parse import parse_qs, urlsplit
from fastapi.testclient import TestClient

import src.agent_server as agent_server
from src.agent_server import (
    ChartParams,
    PlanRequest,
    PlanResponse,
    build_plan_layers,
    _append_query_params,
    app,
    gpt_chart_url,
)
from src.config import get_settings


def _stub_plan_components(symbol: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    entry = 430.0
    stop = 428.6
    target = 432.4
    now = datetime(2025, 10, 28, 14, 30, tzinfo=timezone.utc)
    as_of_iso = now.isoformat().replace("+00:00", "Z")
    session_state = {
        "status": "open",
        "as_of": as_of_iso,
        "next_open": None,
        "tz": "America/New_York",
    }
    plan_core = {
        "plan_id": f"{symbol}-demo",
        "symbol": symbol,
        "direction": "long",
        "entry": entry,
        "stop": stop,
        "targets": [target],
        "meta": {"entry_status": {"state": "waiting"}},
        "waiting_for": "1m close above VWAP",
        "session_state": session_state,
        "context_overlays": {"volume_profile": {"vwap": entry - 0.12}},
        "key_levels": {"PDL": entry - 1.0},
    }
    plan_layers = build_plan_layers(
        symbol=symbol,
        interval="5m",
        as_of=as_of_iso,
        planning_context="live",
        key_levels=plan_core["key_levels"],
        overlays=plan_core["context_overlays"],
        strategy_id="orb_retest",
        direction="long",
        waiting_for=plan_core["waiting_for"],
        plan=plan_core,
        last_time_s=int(now.timestamp()),
        interval_s=300,
        last_close=entry - 0.25,
    )
    plan_layers["plan_id"] = plan_core["plan_id"]
    plan_core = dict(plan_core)
    plan_core["plan_layers"] = plan_layers
    return plan_core, plan_layers, session_state


@pytest.mark.asyncio
async def test_chart_url_endpoint_returns_tv_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    get_settings.cache_clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="https://test.local"
    ) as client:
        response = await client.post(
            "/gpt/chart-url",
            json={
                "symbol": "SPY",
                "interval": "5m",
                "direction": "long",
                "entry": 430.1,
                "stop": 428.8,
                "tp": "432.1,433.55",
                "focus": "plan",
                "center_time": "latest",
            },
        )

    get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    link = payload["interactive"]
    parsed = urlsplit(link)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "test.local"
    assert parsed.path == "/chart"
    assert query["symbol"] == ["SPY"]
    assert query["interval"] == ["5m"]
    assert query["range"] == ["1d"]
    assert query["ui_state"] == ['{"style":"intraday"}']


@pytest.mark.asyncio
async def test_plan_trade_detail_contains_tv_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.setenv("GPT_BACKEND_V2_ENABLED", "0")
    get_settings.cache_clear()

    async def fake_generate_fallback_plan(
        symbol: str,
        style: str | None,
        request,
        user,
        *,
        simulate_open: bool = False,  # noqa: ARG001
        plan_request: PlanRequest | None = None,  # noqa: ARG001
    ) -> PlanResponse:
        params = ChartParams(
            symbol=symbol,
            interval="5m",
            direction="long",
            entry=430.1,
            stop=428.8,
            tp="432.1,433.55",
            focus="plan",
            center_time="latest",
            scale_plan="auto",
            range="5D",
            view="6M",
            ema="9,20,50",
        )
        links = await gpt_chart_url(params, request)
        plan_id = f"{symbol.upper()}-demo"
        plan_version = 1
        trade_url = _append_query_params(
            links.interactive,
            {"plan_id": plan_id, "plan_version": str(plan_version)},
        )
        targets: List[float] = [float(token) for token in params.tp.split(",") if token]
        return PlanResponse(
            plan_id=plan_id,
            version=plan_version,
            trade_detail=trade_url,
            planning_context="frozen",
            symbol=symbol,
            style=style or "intraday",
            entry=float(params.entry),
            stop=float(params.stop),
            targets=targets,
            charts_params=params.model_dump(),
            chart_url=trade_url,
        )

    monkeypatch.setattr(
        "src.agent_server._generate_fallback_plan", fake_generate_fallback_plan
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="https://test.local"
    ) as client:
        response = await client.post("/gpt/plan", json={"symbol": "SPY"})

    get_settings.cache_clear()

    assert response.status_code == 200
    plan_payload = response.json()
    trade_detail = plan_payload["trade_detail"]
    parsed = urlsplit(trade_detail)
    query = parse_qs(parsed.query)
    assert parsed.path == "/chart"
    assert query["interval"] == ["5m"]
    assert query.get("range") == ["5D"]
    assert query["ui_state"] == ['{"style":"intraday"}']
    assert "plan_id" in query and "plan_version" in query


@pytest.mark.asyncio
async def test_chart_layers_endpoint_returns_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()

    async def fake_ensure_snapshot(
        plan_id: str, version: int | None, request
    ) -> Dict[str, Any]:  # noqa: ARG001
        return {
            "plan": {"plan_id": plan_id, "symbol": "SPY"},
            "layers": {"levels": [], "meta": {"as_of": "2025-10-28T12:00:00Z"}},
        }

    def fake_extract_plan_layers(
        snapshot: Dict[str, Any], plan_id: str
    ) -> Dict[str, Any]:  # noqa: ARG001
        return {
            "levels": [{"label": "SESSION_HIGH", "price": 460.16}],
            "zones": [{"type": "demand", "low": 458.5, "high": 459.2}],
            "meta": {"as_of": "2025-10-28T12:00:00Z"},
        }

    monkeypatch.setattr("src.agent_server._ensure_snapshot", fake_ensure_snapshot)
    monkeypatch.setattr(
        "src.agent_server.extract_plan_layers", fake_extract_plan_layers
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="https://test.local"
    ) as client:
        response = await client.get(
            "/api/v1/gpt/chart-layers", params={"plan_id": "SPY-demo"}
        )

    get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan_id"] == "SPY-demo"
    assert payload["levels"]
    assert payload["meta"]["as_of"] == "2025-10-28T12:00:00Z"


@pytest.mark.asyncio
async def test_plan_layers_next_objective_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.setenv("GPT_BACKEND_V2_ENABLED", "1")
    monkeypatch.setenv("GPT_MARKET_ROUTING_ENABLED", "0")
    get_settings.cache_clear()

    async def fake_generate_plan_v2(
        symbol: str,
        style: str | None,
        route,  # noqa: ANN001
        app,  # noqa: ANN001
    ) -> Dict[str, Any]:
        plan_core, plan_layers, session_state = _stub_plan_components(symbol.upper())
        return {
            "plan_id": plan_core["plan_id"],
            "version": 1,
            "planning_context": "live",
            "symbol": symbol.upper(),
            "style": style or "intraday",
            "bias": plan_core["direction"],
            "entry": plan_core["entry"],
            "stop": plan_core["stop"],
            "targets": plan_core["targets"],
            "plan": plan_core,
            "plan_layers": plan_layers,
            "session_state": session_state,
            "charts": {},
            "charts_params": {},
        }

    monkeypatch.setattr("src.agent_server.generate_plan_v2", fake_generate_plan_v2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        plan_response = await client.post("/gpt/plan", json={"symbol": "SPY"})
        assert plan_response.status_code == 200
        plan_payload = plan_response.json()
        plan_id = plan_payload["plan_id"]
        assert plan_id in agent_server._IDEA_STORE
        snapshot_plan = dict(plan_payload.get("plan") or {})
        snapshot_plan["plan_id"] = plan_id
        snapshot_plan["version"] = plan_payload.get("version") or 1
        if plan_payload.get("session_state"):
            snapshot_plan.setdefault("session_state", plan_payload.get("session_state"))
        snapshot_plan["plan_layers"] = plan_payload["plan_layers"]
        await agent_server._store_idea_snapshot(
            plan_id,
            {
                "plan": snapshot_plan,
                "plan_layers": plan_payload["plan_layers"],
            },
        )

        layers_response = await client.get(
            "/api/v1/gpt/chart-layers",
            params={"plan_id": plan_id},
        )

    get_settings.cache_clear()

    assert layers_response.status_code == 200
    layers_payload = layers_response.json()
    meta_block = layers_payload["meta"]
    assert "next_objective" in meta_block
    next_objective = meta_block["next_objective"]
    assert set(next_objective.keys()) >= {"state", "why", "objective_price", "band", "timeframe", "progress"}
    assert isinstance(next_objective["progress"], float)
    assert isinstance(next_objective["band"]["low"], float)
    assert next_objective["state"] in {"arming", "ready", "cooldown", "invalid"}
    internal_meta = meta_block.get("_next_objective_internal")
    assert internal_meta is not None
    assert "entry_distance_pct" in internal_meta
    assert internal_meta.get("tick_size")


def test_coach_ws_emits_pulse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.setenv("FF_OPTIONS_ALWAYS", "1")
    monkeypatch.setenv("GPT_BACKEND_V2_ENABLED", "1")
    monkeypatch.setenv("GPT_MARKET_ROUTING_ENABLED", "0")
    monkeypatch.setenv("BACKEND_API_KEY", "supersecret")
    get_settings.cache_clear()

    async def fake_generate_plan_v2(
        symbol: str,
        style: str | None,
        route,  # noqa: ANN001
        app,  # noqa: ANN001
    ) -> Dict[str, Any]:
        plan_core, plan_layers, session_state = _stub_plan_components(symbol.upper())
        return {
            "plan_id": plan_core["plan_id"],
            "version": 1,
            "planning_context": "live",
            "symbol": symbol.upper(),
            "style": style or "intraday",
            "bias": plan_core["direction"],
            "entry": plan_core["entry"],
            "stop": plan_core["stop"],
            "targets": plan_core["targets"],
            "plan": plan_core,
            "plan_layers": plan_layers,
            "session_state": session_state,
            "charts": {},
            "charts_params": {},
        }

    monkeypatch.setattr("src.agent_server.generate_plan_v2", fake_generate_plan_v2)

    with TestClient(app) as client:
        plan_response = client.post(
            "/gpt/plan",
            json={"symbol": "SPY"},
            headers={"Authorization": "Bearer supersecret"},
        )
        assert plan_response.status_code == 200
        plan_payload = plan_response.json()
        plan_id = plan_payload["plan_id"]
        snapshot_plan = dict(plan_payload.get("plan") or {})
        snapshot_plan["plan_id"] = plan_id
        snapshot_plan["version"] = plan_payload.get("version") or 1
        if plan_payload.get("session_state"):
            snapshot_plan.setdefault("session_state", plan_payload.get("session_state"))
        snapshot_plan["plan_layers"] = plan_payload["plan_layers"]
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            agent_server._store_idea_snapshot(
                plan_id,
                {
                    "plan": snapshot_plan,
                    "plan_layers": plan_payload["plan_layers"],
                },
            )
        )
        loop.close()

        with client.websocket_connect(
            f"/ws/coach/{plan_id}",
            headers={"Authorization": "Bearer supersecret"},
        ) as ws:
            message = ws.receive_json()
            assert message["t"] == "coach_pulse"
            diff = message["diff"]
            objective_progress = diff["objective_progress"]
            assert isinstance(objective_progress["progress"], float)
            assert objective_progress["entry_distance_pct"] is not None
            session_block = message["session"]
            assert session_block["status"]
            assert session_block["tz"]

    get_settings.cache_clear()

@pytest.mark.asyncio
async def test_tv_api_bars_returns_ok_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = 5
    index = pd.date_range("2025-10-28 14:30", periods=rows, freq="5T", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [430 + i * 0.5 for i in range(rows)],
            "high": [430.6 + i * 0.5 for i in range(rows)],
            "low": [429.8 + i * 0.5 for i in range(rows)],
            "close": [430.3 + i * 0.5 for i in range(rows)],
            "volume": [100_000 + i * 1000 for i in range(rows)],
        },
        index=index,
    )

    async def fake_load_remote(symbol: str, timeframe: str) -> pd.DataFrame:  # noqa: ARG001
        return frame

    monkeypatch.setattr("src.agent_server._load_remote_ohlcv", fake_load_remote)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="https://test.local"
    ) as client:
        response = await client.get(
            "/tv-api/bars",
            params={"symbol": "SPY", "resolution": "5", "range": "5D"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["s"] == "ok"
    assert payload["symbol"] == "SPY"
    assert payload["resolution"] == "5"
    assert len(payload["t"]) == rows
    assert len(payload["o"]) == rows
    assert len(payload["h"]) == rows
    assert len(payload["l"]) == rows
    assert len(payload["c"]) == rows
    assert len(payload["v"]) == rows
    assert payload["t"][0] == int(index[0].timestamp() * 1000)
    assert payload["o"][0] == pytest.approx(float(frame["open"].iloc[0]), rel=1e-6)
    assert payload["c"][-1] == pytest.approx(float(frame["close"].iloc[-1]), rel=1e-6)
