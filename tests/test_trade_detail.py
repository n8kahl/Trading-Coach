import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from starlette.requests import Request

from urllib.parse import parse_qs, urlparse

from src import agent_server
from src.config import get_settings


@pytest.fixture(autouse=True)
def _disable_v2_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_backend_v2_enabled", False, raising=False)


async def _run_fallback_plan(
    monkeypatch,
    *,
    symbol: str = "TSLA",
    expected_move: float = 3.5,
    atr: float = 1.2,
    key_levels: dict | None = None,
    daily_levels: list | None = None,
    weekly_levels: list | None = None,
    daily_profile: dict | None = None,
    weekly_profile: dict | None = None,
    options_payload: dict | None = None,
    user_id: str = "fallback-tester",
    ema_bias: str | None = None,
    min_actionability: float | None = None,
    must_be_actionable: bool = False,
    plan_request: agent_server.PlanRequest | None = None,
):
    async def fake_scan(universe, request, user):  # noqa: ARG001
        return []

    if key_levels is None:
        key_levels = {
            "session_high": 250.0,
            "session_low": 240.0,
            "opening_range_high": 248.5,
            "opening_range_low": 242.5,
            "prev_close": 247.0,
            "prev_high": 252.0,
            "prev_low": 243.5,
        }
    base_price = float(key_levels.get("session_high", 250.0))
    index = pd.date_range("2025-10-20 14:30", periods=6, freq="5T", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [
                base_price - 1.6,
                base_price - 1.15,
                base_price - 0.9,
                base_price - 0.55,
                base_price - 0.3,
                base_price - 0.1,
            ],
            "high": [
                base_price - 1.1,
                base_price - 0.6,
                base_price - 0.2,
                base_price + 0.1,
                base_price + 0.25,
                base_price + 0.5,
            ],
            "low": [
                base_price - 2.4,
                base_price - 1.9,
                base_price - 1.3,
                base_price - 1.0,
                base_price - 0.7,
                base_price - 0.4,
            ],
            "close": [
                base_price - 1.3,
                base_price - 0.95,
                base_price - 0.6,
                base_price - 0.25,
                base_price - 0.05,
                base_price,
            ],
            "volume": [100_000, 110_000, 120_000, 130_000, 125_000, 140_000],
            "atr14": [atr] * 6,
        },
        index=index,
    )
    daily_levels = daily_levels or []
    weekly_levels = weekly_levels or []
    daily_profile = daily_profile or {}
    weekly_profile = weekly_profile or {}

    def fake_prepare_symbol_frame(raw_frame):
        prepared = raw_frame.copy()
        prepared["ema9"] = prepared["close"]
        prepared["ema20"] = prepared["close"]
        prepared["ema50"] = prepared["close"]
        bias_token = (ema_bias or "").strip().lower()
        if bias_token == "long":
            prepared["ema9"] = prepared["close"] + 0.3
            prepared["ema20"] = prepared["close"]
            prepared["ema50"] = prepared["close"] - 0.3
        elif bias_token == "short":
            prepared["ema9"] = prepared["close"] - 0.3
            prepared["ema20"] = prepared["close"]
            prepared["ema50"] = prepared["close"] + 0.3
        prepared["vwap"] = prepared["close"]
        return prepared

    def fake_build_context(prepared):
        latest = prepared.iloc[-1]
        return {
            "latest": latest,
            "ema9": float(latest.get("ema9", latest["close"])),
            "ema20": float(latest.get("ema20", latest["close"])),
            "ema50": float(latest.get("ema50", latest["close"])),
            "atr": atr,
            "expected_move_horizon": expected_move,
            "timestamp": prepared.index[-1],
            "volume_profile": {},
            "key": dict(key_levels),
            "levels_daily": list(daily_levels),
            "levels_weekly": list(weekly_levels),
            "vol_profile_daily": dict(daily_profile),
            "vol_profile_weekly": dict(weekly_profile),
        }

    def fake_build_market_snapshot(prepared, levels):  # noqa: ARG001
        return {"volatility": {"expected_move": expected_move}}

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        return {symbols[0]: frame}, {symbols[0]: "polygon"}

    async def fake_fetch_context_enrichment(symbol_token):  # noqa: ARG001
        return {}

    async def fake_gpt_contracts(request, user):  # noqa: ARG001
        return options_payload or {}

    async def fake_chart_url(params, request):  # noqa: ARG001
        return agent_server.ChartLinks(interactive="https://example.com/chart")

    async def fake_mtf_confluence(**kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "_session_payload_from_request", lambda request: {})
    monkeypatch.setattr(
        agent_server,
        "_market_snapshot_payload",
        lambda session_payload, simulate_open=False: (
            {"session_state": {"status": "closed"}},
            {"session_state": {"status": "closed"}},
            datetime(2025, 10, 20, 15, 30, tzinfo=timezone.utc),
            False,
        ),
    )
    monkeypatch.setattr(agent_server, "_collect_market_data", fake_collect_market_data)
    monkeypatch.setattr(agent_server, "_prepare_symbol_frame", fake_prepare_symbol_frame)
    monkeypatch.setattr(agent_server, "_build_context", fake_build_context)
    monkeypatch.setattr(agent_server, "_build_market_snapshot", fake_build_market_snapshot)
    monkeypatch.setattr(agent_server, "_fetch_context_enrichment", fake_fetch_context_enrichment)
    monkeypatch.setattr(agent_server, "gpt_contracts", fake_gpt_contracts)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)
    monkeypatch.setattr(agent_server, "_compute_multi_timeframe_confluence", fake_mtf_confluence)
    monkeypatch.setattr(agent_server, "_get_index_mode", lambda: None)
    monkeypatch.setattr(agent_server, "first", {}, raising=False)
    monkeypatch.setattr(agent_server, "results", [], raising=False)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id=user_id)

    plan_request = plan_request or agent_server.PlanRequest(
        symbol=symbol,
        style="intraday",
        min_actionability=min_actionability,
        must_be_actionable=must_be_actionable,
    )

    return await agent_server._generate_fallback_plan(
        symbol,
        "intraday",
        request,
        user,
        plan_request=plan_request,
    )


@pytest.mark.skip("Legacy plan pipeline superseded by backend v2")
@pytest.mark.asyncio
async def test_gpt_plan_includes_trade_detail(monkeypatch):
    async def fake_scan(universe, request, user):
        return [
            {
                "symbol": "TSLA",
                "style": "swing",
                "plan": {
                    "plan_id": "TSLA-SWING-20251010",
                    "entry": 254.0,
                    "stop": 247.8,
                    "targets": [266.0, 273.5],
                    "direction": "long",
                },
                "charts": {
                    "params": {
                        "symbol": "TSLA",
                        "interval": "1D",
                        "direction": "long",
                        "entry": "254.0",
                        "stop": "247.8",
                        "tp": "266.0,273.5",
                    }
                },
                "market_snapshot": {
                    "indicators": {"atr14": 6.2},
                    "volatility": {"expected_move_horizon": 12},
                    "trend": {"direction_hint": "long"},
                },
                "key_levels": {},
                "features": {},
            }
        ]

    async def fake_chart_url(params, request):
        return agent_server.ChartLinks(
            interactive="https://example.com/chart",
        )

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id="test-user")

    response = await agent_server.gpt_plan(agent_server.PlanRequest(symbol="TSLA", style="swing"), request, user)

    assert response.trade_detail, "trade_detail should be populated"
    assert response.plan["trade_detail"] == response.trade_detail, "embedded plan must carry trade_detail"
    parsed = urlparse(response.trade_detail)
    params = parse_qs(parsed.query)
    assert params.get("plan_id") == ["TSLA-SWING-20251010"]
    assert params.get("plan_version") == [str(response.version)]


@pytest.mark.skip("Legacy plan pipeline superseded by backend v2")
@pytest.mark.asyncio
async def test_gpt_plan_respects_strategy_id(monkeypatch):
    async def fake_scan(universe, request, user):  # noqa: ARG001
        return [
            {
                "symbol": "AAPL",
                "style": "intraday",
                "strategy_id": "orb_retest",
                "plan": {
                    "plan_id": "AAPL-ORB-TEST",
                    "version": 3,
                    "entry": 150.0,
                    "stop": 148.8,
                    "targets": [151.2, 152.6],
                    "direction": "long",
                    "confidence": 0.68,
                    "strategy": "orb_retest",
                },
                "charts": {
                    "params": {
                        "symbol": "AAPL",
                        "interval": "5m",
                        "direction": "long",
                        "entry": "150.0",
                        "stop": "148.8",
                        "tp": "151.2,152.6",
                    }
                },
                "market_snapshot": {
                    "indicators": {"atr14": 1.1},
                    "volatility": {"expected_move_horizon": 2.3},
                    "trend": {"direction_hint": "long"},
                },
                "key_levels": {"session_high": 152.0, "session_low": 147.5},
                "features": {"plan_confidence_factors": ["VWAP confirmation"]},
                "events": {"label": "earnings watch"},
                "earnings": {"next_earnings_at": "2025-11-01"},
            }
        ]

    async def fake_chart_url(params, request):  # noqa: ARG001
        return agent_server.ChartLinks(interactive="https://example.com/chart")

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)
    monkeypatch.setattr(
        agent_server,
        "get_session",
        lambda request=None: {"status": "open", "as_of": "2025-10-16T15:55:00-04:00", "tz": "America/New_York"},
    )

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id="test-user")

    response = await agent_server.gpt_plan(
        agent_server.PlanRequest(symbol="AAPL", style="intraday"),
        request,
        user,
    )

    assert response.setup == "orb_retest"
    assert response.plan["strategy"] == "orb_retest"
    assert response.events == {"label": "earnings watch"}
    assert response.earnings == {"next_earnings_at": "2025-11-01"}
    assert response.data_quality["events_present"] is True
    assert response.data_quality["earnings_present"] is True
    data_meta = response.data or {}
    assert data_meta.get("events_present") is True
    assert data_meta.get("earnings_present") is True


@pytest.mark.skip("Legacy plan pipeline superseded by backend v2")
@pytest.mark.asyncio
async def test_gpt_plan_macro_events_fallback(monkeypatch):
    async def fake_scan(universe, request, user):  # noqa: ARG001
        return [
            {
                "symbol": "SPY",
                "style": "intraday",
                "strategy_id": "vwap_avwap",
                "plan": {
                    "plan_id": "SPY-VWAP-TEST",
                    "entry": 430.0,
                    "stop": 428.5,
                    "targets": [431.2, 432.4],
                    "direction": "long",
                    "confidence": 0.62,
                    "strategy": "vwap_avwap",
                },
                "charts": {
                    "params": {
                        "symbol": "SPY",
                        "interval": "5m",
                        "direction": "long",
                        "entry": "430.0",
                        "stop": "428.5",
                        "tp": "431.2,432.4",
                    }
                },
                "market_snapshot": {
                    "indicators": {"atr14": 1.5},
                    "volatility": {"expected_move_horizon": 2.8},
                    "trend": {"direction_hint": "long"},
                },
                "key_levels": {"session_high": 432.0, "session_low": 427.5},
                "features": {"plan_confidence_factors": ["EMA alignment"]},
            }
        ]

    async def fake_chart_url(params, request):  # noqa: ARG001
        return agent_server.ChartLinks(interactive="https://example.com/chart")

    async def fake_enrichment(symbol):  # noqa: ARG001
        return {}

    def fake_event_window(as_of):  # noqa: ARG001
        return {
            "upcoming": [{"name": "FOMC decision", "minutes": 45}],
            "active": [],
            "min_minutes_to_event": 45,
        }

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)
    monkeypatch.setattr(agent_server, "_fetch_context_enrichment", fake_enrichment)
    monkeypatch.setattr(agent_server, "get_event_window", fake_event_window)
    monkeypatch.setattr(
        agent_server,
        "get_session",
        lambda request=None: {"status": "closed", "as_of": "2025-10-16T16:00:00-04:00", "tz": "America/New_York"},
    )

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id="test-user")

    response = await agent_server.gpt_plan(
        agent_server.PlanRequest(symbol="SPY", style="intraday"),
        request,
        user,
    )

    assert response.setup == "vwap_avwap"
    assert response.events is not None
    assert response.events.get("label") == "macro_window"
    assert response.events.get("next_fomc_minutes") == 45
    assert response.earnings is None
    assert response.data_quality["events_present"] is True
    assert response.data_quality["earnings_present"] is False
    data_meta = response.data or {}
    assert data_meta.get("events_present") is True
    assert data_meta.get("earnings_present") is False


@pytest.mark.asyncio
async def test_fallback_plan_handles_missing_strategy_profile(monkeypatch):
    response = await _run_fallback_plan(monkeypatch)

    assert response is not None
    profile = response.plan.get("strategy_profile") or {}
    assert profile.get("name"), "Fallback plan should expose a strategy profile name"
    badges = response.plan.get("badges") or []
    assert any(badge.get("kind") == "strategy" for badge in badges)
    assert response.plan["runner_policy"]


@pytest.mark.asyncio
async def test_fallback_plan_handles_selector_rejections(monkeypatch):
    options_payload = {
        "best": [
            {
                "symbol": "TSLA 20251025C450",
                "label": "TSLA 10/25 450C",
                "option_type": "call",
                "strike": 450.0,
                "bid": 0.45,
                "ask": 1.35,
                "open_interest": 50,
                "spread_pct": 20.0,
            }
        ],
        "rejections": [
            {"symbol": "TSLA", "reason": "WIDE_SPREAD", "message": "Spread too wide"},
            {"symbol": "TSLA", "reason": "LOW_OI"},
        ]
    }
    response = await _run_fallback_plan(
        monkeypatch,
        options_payload=options_payload,
        user_id="selector-rejections",
    )

    assert response is not None
    assert response.plan.get("plan_state") == "WAIT"
    assert response.plan["entry"] is None
    assert response.waiting_for
    assert response.plan["runner_policy"]
    assert response.plan["badges"]
    assert response.options_contracts, "Expected fallback contracts even when guardrails reject defaults"
    assert response.options_contracts[0].get("guardrail_flags"), "Contracts should surface guardrail flags"
    note_upper = (response.options_note or "").upper()
    assert "GUARDRAIL_FLAGS" in note_upper
    assert "WIDE_SPREAD" in note_upper
    assert "LOW_OI" in note_upper


@pytest.mark.asyncio
async def test_fallback_plan_em_capped_targets_spread(monkeypatch):
    key_levels = {
        "session_high": 235.6,
        "session_low": 223.5,
        "opening_range_high": 233.98,
        "opening_range_low": 228.9,
        "prev_close": 232.0,
        "prev_high": 242.28,
        "prev_low": 231.84,
    }
    response = await _run_fallback_plan(
        monkeypatch,
        symbol="AMD",
        expected_move=1.1424,
        atr=0.6720287548900553,
        key_levels=key_levels,
        user_id="em-cap-check",
    )

    assert response.symbol == "AMD"
    assert response.target_profile["em_used"] is True
    assert response.target_profile["tp_reasons"]
    assert response.plan.get("plan_state") == "WAIT"
    assert response.waiting_for


@pytest.mark.asyncio
async def test_fallback_plan_snaps_to_daily_high(monkeypatch):
    base_key_levels = {
        "session_high": 265.4,
        "session_low": 258.2,
        "opening_range_high": 264.9,
        "opening_range_low": 259.6,
        "prev_close": 263.75,
        "prev_high": 266.1,
        "prev_low": 260.8,
    }
    captured_levels = {}
    original_build = agent_server.build_plan_geometry

    def _capturing_geometry(*args, **kwargs):
        levels = kwargs.get("levels")
        if isinstance(levels, dict):
            captured_levels.clear()
            captured_levels.update(levels)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(agent_server, "build_plan_geometry", _capturing_geometry)
    daily_levels = [("DAILY_HIGH", 264.05), ("DAILY_LOW", 257.4)]
    response = await _run_fallback_plan(
        monkeypatch,
        symbol="MSFT",
        key_levels=base_key_levels,
        daily_levels=daily_levels,
        daily_profile={"VAH": 266.5},
        expected_move=0.6,
        atr=0.3,
        user_id="daily-snap",
    )

    target_meta = response.plan.get("target_meta") or []
    assert target_meta == []
    assert "daily_high" in captured_levels, "expected daily_high to be injected into geometry levels"
    assert captured_levels["daily_high"] == pytest.approx(264.05)


@pytest.mark.asyncio
async def test_fallback_plan_entry_anchors_to_intraday_structure(monkeypatch):
    key_levels = {
        "session_high": 250.0,
        "session_low": 240.0,
        "opening_range_high": 248.6,
        "opening_range_low": 242.2,
        "prev_close": 247.1,
        "prev_high": 252.4,
        "prev_low": 243.0,
    }
    response = await _run_fallback_plan(monkeypatch, key_levels=key_levels, ema_bias="long")

    assert response.plan["direction"] == "long"
    assert response.entry is None
    assert response.entry_anchor == "session_low"
    assert response.waiting_for


@pytest.mark.asyncio
async def test_fallback_plan_entry_uses_style_levels_for_short(monkeypatch):
    daily_levels = [("DAILY_HIGH", 251.4)]
    response = await _run_fallback_plan(monkeypatch, daily_levels=daily_levels, ema_bias="short")

    assert response.plan["direction"] == "short"
    assert response.plan["entry"] == pytest.approx(250.5, rel=0, abs=1e-6)
    assert response.entry_anchor == "swing_high"


@pytest.mark.asyncio
async def test_simulate_generator_serializes_timestamp(monkeypatch):
    index = pd.date_range("2024-01-01", periods=1, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [1000],
        },
        index=index,
    )
    frame["time"] = index

    monkeypatch.setattr(agent_server, "get_candles", lambda symbol, interval, lookback=30: frame)

    params = {"minutes": 5, "entry": 1.0, "stop": 0.9, "tp1": 1.1, "direction": "long"}
    generator = agent_server._simulate_generator("AAPL", params)
    chunk = await generator.__anext__()
    assert chunk.startswith("data: ")
    payload = json.loads(chunk[len("data: ") :].strip())
    assert payload["time"].endswith("+00:00")
    await generator.aclose()


@pytest.mark.skip("Legacy plan pipeline superseded by backend v2")
@pytest.mark.asyncio
async def test_gpt_plan_populates_completeness_fields(monkeypatch):
    monkeypatch.setenv("FF_OPTIONS_ALWAYS", "1")
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()

    async def fake_scan(universe, request, user):  # noqa: ARG001
        return [
            {
                "symbol": "AAPL",
                "style": "intraday",
                "plan": {
                    "plan_id": "AAPL-INTRADAY-TEST",
                    "entry": 150.0,
                    "stop": 148.5,
                    "targets": [151.8, 153.2],
                    "target_meta": [
                        {
                            "label": "TP1",
                            "source": "stats",
                            "distance": 1.8,
                            "rr": 1.2,
                            "snap_tag": "POC",
                            "prob_touch": 0.62,
                        },
                        {
                            "label": "TP2",
                            "source": "fallback",
                            "distance": 3.2,
                            "rr": 1.9,
                        },
                    ],
                    "targets_meta": [
                        {
                            "label": "TP1",
                            "prob_touch": 0.62,
                            "distance": 1.8,
                            "rr_multiple": 1.2,
                        },
                        {
                            "label": "TP2",
                            "prob_touch": 0.48,
                            "distance": 3.2,
                            "rr_multiple": 1.9,
                        },
                    ],
                    "direction": "long",
                    "runner": {"trail": "ATR"},
                    "runner_policy": {
                        "fraction": 0.22,
                        "atr_trail_mult": 0.85,
                        "atr_trail_step": 0.45,
                        "em_fraction_cap": 0.5,
                        "notes": ["Runner fraction adjusted"],
                    },
                    "confidence": 0.74,
                    "notes": "Structure aligned",
                    "snap_trace": ["tp1:151.50->151.80 via POC"],
                    "expected_move": 4.6,
                    "remaining_atr": 2.4,
                    "em_used": True,
                    "source_paths": {"entry": "geometry_engine", "runner_policy": "geometry_engine"},
                    "accuracy_levels": ["EM cap"],
                },
                "charts": {
                    "params": {
                        "symbol": "AAPL",
                        "interval": "5m",
                        "direction": "long",
                        "entry": "150.0",
                        "stop": "148.5",
                        "tp": "151.8,153.2",
                    }
                },
                "market_snapshot": {
                    "indicators": {"atr14": 1.2},
                    "volatility": {"expected_move_horizon": 2.5},
                    "trend": {"direction_hint": "long", "ema_stack": "bullish"},
                },
                "key_levels": {"session_high": 152.0, "session_low": 147.5},
                "context_overlays": {
                    "volume_profile": {"poc": 151.78},
                    "avwap": {"from_session_low": 149.2},
                },
                "features": {"plan_confidence_factors": ["EMA alignment", "VWAP confirmation"]},
                "options": {
                    "best": [
                        {
                            "symbol": "AAPL241122C00150000",
                            "label": "C150",
                            "option_type": "call",
                            "dte": 5,
                            "strike": 150.0,
                            "price": 2.35,
                            "bid": 2.3,
                            "ask": 2.4,
                            "delta": 0.55,
                            "gamma": 0.12,
                            "theta": -0.04,
                            "vega": 0.08,
                            "open_interest": 2500,
                            "volume": 980,
                            "liquidity_score": 0.82,
                            "spread_pct": 6.5,
                            "pnl": {"per_contract_cost": 235.0, "rr_to_tp1": 1.4},
                            "pl_projection": {"contracts_possible": 1, "risk_budget": 100.0},
                        }
                    ]
                },
            }
        ]

    async def fake_chart_url(params, request):  # noqa: ARG001
        return agent_server.ChartLinks(interactive="https://example.com/chart")

    async def fake_contracts(request_payload, user):  # noqa: ARG001
        return {
            "best": [
                {
                    "symbol": "AAPL241122C00150000",
                    "label": "C150",
                    "option_type": "call",
                    "dte": 5,
                    "strike": 150.0,
                    "price": 2.35,
                    "bid": 2.3,
                    "ask": 2.4,
                    "delta": 0.55,
                    "gamma": 0.12,
                    "theta": -0.04,
                    "vega": 0.08,
                    "open_interest": 2500,
                    "volume": 980,
                    "liquidity_score": 0.82,
                    "spread_pct": 6.5,
                    "pnl": {"per_contract_cost": 235.0, "rr_to_tp1": 1.4},
                    "pl_projection": {"contracts_possible": 1, "risk_budget": 100.0},
                }
            ],
            "alternatives": [],
            "filters": {},
            "relaxed_filters": False,
            "symbol": "AAPL",
        }

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)
    monkeypatch.setattr(agent_server, "gpt_contracts", fake_contracts)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id="test-user")

    response = await agent_server.gpt_plan(agent_server.PlanRequest(symbol="AAPL", style="intraday"), request, user)

    assert response.confluence_tags, "should emit confluence tags"
    assert response.confluence, "should emit multi-timeframe confluence"
    assert response.key_levels_used, "should report key levels used"
    assert response.runner_policy and response.runner_policy.get("fraction") is not None
    assert response.source_paths.get("runner_policy") == "geometry_engine"
    assert response.plan and response.plan.get("target_meta")
    assert response.target_meta
    assert response.snap_trace is None
    assert "snap_trace" not in (response.plan or {})
    assert response.targets_meta == response.target_meta
    assert response.risk_block and response.risk_block["risk_points"] > 0
    assert response.execution_rules and response.execution_rules["trigger"]
    assert any("TP1" in item["label"] for item in response.tp_reasons), "tp reasons should describe targets"


@pytest.mark.skip("Legacy plan pipeline superseded by backend v2")
@pytest.mark.asyncio
async def test_gpt_plan_simulated_open_retains_geometry(monkeypatch):
    async def fake_scan(universe, request, user):  # noqa: ARG001
        return [
            {
                "symbol": "MSFT",
                "style": "intraday",
                "plan": {
                    "plan_id": "MSFT-INTRADAY-TEST",
                    "entry": 330.0,
                    "stop": 327.5,
                    "targets": [332.4, 334.1],
                    "direction": "long",
                    "target_meta": [
                        {
                            "label": "TP1",
                            "prob_touch": 0.6,
                            "distance": 2.4,
                            "rr_multiple": 1.5,
                        }
                    ],
                    "runner_policy": {
                        "fraction": 0.18,
                        "atr_trail_mult": 0.9,
                        "atr_trail_step": 0.4,
                        "em_fraction_cap": 0.45,
                        "notes": [],
                    },
                    "expected_move": 3.4,
                    "remaining_atr": 1.7,
                    "em_used": False,
                },
                "charts": {
                    "params": {
                        "symbol": "MSFT",
                        "interval": "5m",
                        "direction": "long",
                        "entry": "330.0",
                        "stop": "327.5",
                        "tp": "332.4,334.1",
                    }
                },
                "market_snapshot": {
                    "indicators": {"atr14": 1.8},
                    "volatility": {"expected_move_horizon": 3.2},
                    "trend": {"direction_hint": "long"},
                },
                "key_levels": {"session_high": 333.0, "session_low": 326.2},
                "features": {},
            }
        ]

    async def fake_chart_url(params, request):  # noqa: ARG001
        return agent_server.ChartLinks(interactive="https://example.com/chart/msft")

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)
    monkeypatch.setattr(
        agent_server,
        "get_session",
        lambda request=None: {"status": "closed", "as_of": "2025-02-10T20:00:00-05:00", "tz": "America/New_York"},
    )

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id="planner")

    response = await agent_server.gpt_plan(
        agent_server.PlanRequest(symbol="MSFT", style="intraday", simulate_open=True),
        request,
        user,
    )

    assert response.meta and response.meta.get("simulated_open")
    assert response.targets_meta
    assert response.runner_policy
    assert response.risk_block
    assert response.execution_rules
    # Options may be empty during simulated sessions; ensure the field exists (even if empty).
    assert response.options_contracts is not None
    assert response.options_note is not None
    assert response.plan.get("tp_reasons"), "plan payload should include tp_reasons"
    assert response.plan.get("confluence_tags"), "plan payload should include confluence tags"

    get_settings.cache_clear()
