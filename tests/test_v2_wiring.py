import pandas as pd
import pytest
from starlette.requests import Request

from src import polygon_options
from src.polygon_options import fetch_polygon_option_chain_asof
from src.agent_server import (
    _expand_universe_tokens,
    _screen_contracts,
    AssistantExecRequest,
    AuthedUser,
    PlanResponse,
    assistant_exec,
    symbol_diagnostics,
)


@pytest.mark.asyncio
async def test_fetch_polygon_option_chain_asof_filters_future_rows(monkeypatch):
    async def fake_fetch(symbol: str, expiration: str | None = None, *, limit: int = 400):
        return pd.DataFrame(
            [
                {"symbol": "KEEP", "last_updated": "2024-01-01T14:30:00Z"},
                {"symbol": "DROP", "last_updated": "2024-01-01T16:00:00Z"},
            ]
        )

    monkeypatch.setattr(polygon_options, "fetch_polygon_option_chain", fake_fetch)

    frame = await fetch_polygon_option_chain_asof("AAPL", "2024-01-01T15:00:00Z")
    assert list(frame["symbol"]) == ["KEEP"]


@pytest.mark.asyncio
async def test_expand_universe_tokens_resolves_top_active(monkeypatch):
    async def fake_load_universe(*, style, sector, limit):  # noqa: ARG001
        return ["AAPL", "MSFT", "NVDA"]

    monkeypatch.setattr("src.agent_server.load_universe", fake_load_universe)

    expanded = await _expand_universe_tokens(["top_active_setups", "TSLA"], style="scalp", limit=5)

    assert expanded[:3] == ["AAPL", "MSFT", "NVDA"]
    assert expanded[-1] == "TSLA"
    assert "TOP_ACTIVE_SETUPS" not in expanded


def test_screen_contracts_includes_liquidity_score():
    chain = pd.DataFrame(
        [
            {
                "symbol": "OPT1",
                "option_type": "call",
                "bid": 1.1,
                "ask": 1.3,
                "spread_pct": 0.2,
                "expiration_date": "2025-01-01",
                "dte": 5,
                "delta": 0.52,
                "open_interest": 600,
                "volume": 120,
                "iv_percentile": 0.4,
            }
        ]
    )

    filters = {"min_dte": 0, "max_dte": 10, "min_delta": 0.4, "max_delta": 0.6, "max_spread_pct": 50.0, "min_oi": 100}

    result = _screen_contracts(
        chain,
        quotes={},
        symbol="AAPL",
        style="intraday",
        side="call",
        filters=filters,
    )

    assert result
    assert result[0].get("liquidity_score") is not None
    assert result[0]["tradeability"] >= 0


@pytest.mark.asyncio
async def test_assistant_exec_builds_options_example(monkeypatch):
    async def fake_plan(plan_request, request, user):  # noqa: ARG001
        return PlanResponse(
            plan_id="T123",
            version=1,
            trade_detail="https://example.com/plan",
            symbol="AAPL",
            style="intraday",
            plan={"plan_id": "T123"},
            structured_plan={"plan_id": "T123", "targets": [102.0]},
            charts_params={"entry": "100"},
            chart_url="https://example.com/chart",
            session_state={"as_of": "2024-01-01T21:00:00Z"},
            market={"session_state": {"as_of": "2024-01-01T21:00:00Z"}},
        )

    async def fake_best_contract_example(symbol, style, as_of=None):  # noqa: ARG001
        return {"symbol": "OPTX", "type": "call", "strike": 105.0}

    monkeypatch.setattr("src.agent_server.gpt_plan", fake_plan)
    monkeypatch.setattr("src.agent_server.best_contract_example", fake_best_contract_example)

    request_payload = AssistantExecRequest(symbol="AAPL")
    request = Request({"type": "http", "method": "POST", "headers": []})

    response = await assistant_exec(request_payload, request, AuthedUser(user_id="tester"))

    assert response.plan["plan_id"] == "T123"
    assert response.options["best"][0]["symbol"] == "OPTX"
    assert response.meta["plan_id"] == "T123"


@pytest.mark.asyncio
async def test_symbol_diagnostics_uses_normalised_interval(monkeypatch):
    async def fake_context(symbol: str, interval: str, lookback: int):  # noqa: ARG001
        return {
            "key_levels": {"session_high": 100.0},
            "snapshot": {"price": {"close": 99.5}},
            "indicators": {"ema9": [1.0]},
        }

    class DummySession:
        def to_dict(self):
            return {"status": "closed", "as_of": "2024-01-01T21:00:00Z", "tz": "America/New_York", "banner": "Closed", "next_open": "2024-01-02T09:30:00-05:00"}

    monkeypatch.setattr("src.agent_server._build_interval_context", fake_context)
    monkeypatch.setattr("src.app.middleware.session.session_now", lambda: DummySession())

    request = Request({"type": "http", "method": "GET", "headers": []})

    response = await symbol_diagnostics("spy", request, interval="15m", lookback=200)

    assert response.symbol == "SPY"
    assert response.interval == "15m"
    assert "session_high" in response.key_levels
    assert response.session["status"] == "closed"
