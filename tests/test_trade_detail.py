import asyncio
import json

import pandas as pd
import pytest
from starlette.requests import Request

from urllib.parse import parse_qs, urlparse

from src import agent_server
from src.config import get_settings


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
    assert response.snap_trace is not None
    assert response.targets_meta == response.target_meta
    assert response.risk_block and response.risk_block["risk_points"] > 0
    assert response.execution_rules and response.execution_rules["trigger"]
    assert any("TP1" in item["label"] for item in response.tp_reasons), "tp reasons should describe targets"


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
