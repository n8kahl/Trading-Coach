from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest

from src import agent_server
from src.agent_server import (
    ContractsRequest,
    _compute_dte_from_expiration,
    _screen_contracts,
    gpt_contracts,
)
from src.market_clock import MarketSessionSnapshot


def _make_quote(
    *,
    symbol: str,
    bid: float,
    ask: float,
    delta: float,
    oi: int,
    expiration: datetime,
    **extras: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "last": (bid + ask) / 2,
        "delta": delta,
        "open_interest": oi,
        "expiration_date": expiration.isoformat(),
    }
    payload.update(extras)
    return payload


def test_spread_fraction_normalized():
    chain = pd.DataFrame(
            [
                {
                    "symbol": "OPT_FRACTION",
                    "option_type": "call",
                    "bid": None,
                    "ask": None,
                    "mid": 1.01,
                    "dte": 5,
                    "delta": 0.52,
                    "open_interest": 800,
                    "strike": 100,
                    "spread_pct": 0.2,
            }
        ]
    )
    screened = _screen_contracts(
        chain,
        quotes={},
        symbol="XYZ",
        style="intraday",
        side="call",
        filters={"min_dte": 0, "max_dte": 10, "min_delta": 0.45, "max_delta": 0.6, "max_spread_pct": 20.0, "min_oi": 200},
    )
    assert screened
    assert screened[0]["spread_pct"] == pytest.approx(20.0)


def test_dte_from_expiration_parses_and_floors():
    tomorrow = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=1, hours=2)
    dte = _compute_dte_from_expiration(tomorrow.isoformat())
    assert isinstance(dte, int)
    assert dte >= 1
    assert _compute_dte_from_expiration("not-a-date") is None


@pytest.mark.asyncio
async def test_relaxation_ladder_returns_three(monkeypatch: pytest.MonkeyPatch):
    expiration = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=10)
    chain = pd.DataFrame(
        [
            {"symbol": "CALL_BASE", "option_type": "call", "bid": 1.0, "ask": 1.1, "strike": 100},
            {"symbol": "CALL_DELTA_LO", "option_type": "call", "bid": 0.4, "ask": 0.5, "strike": 102},
            {"symbol": "CALL_DTE_WIDE", "option_type": "call", "bid": 0.8, "ask": 0.95, "strike": 103},
            {"symbol": "CALL_SPREAD_WIDE", "option_type": "call", "bid": 0.3, "ask": 0.6, "strike": 104},
        ]
    )
    quotes = {
        "CALL_BASE": _make_quote(symbol="CALL_BASE", bid=1.0, ask=1.1, delta=0.52, oi=900, expiration=expiration),
        "CALL_DELTA_LO": _make_quote(symbol="CALL_DELTA_LO", bid=0.4, ask=0.5, delta=0.18, oi=800, expiration=expiration),
        "CALL_DTE_WIDE": _make_quote(
            symbol="CALL_DTE_WIDE",
            bid=0.8,
            ask=0.95,
            delta=0.5,
            oi=700,
            expiration=expiration + timedelta(days=10),
        ),
        "CALL_SPREAD_WIDE": _make_quote(
            symbol="CALL_SPREAD_WIDE",
            bid=0.3,
            ask=0.6,
            delta=0.48,
            oi=1000,
            expiration=expiration,
        ),
    }

    async def fake_chain(symbol: str, expiration: str | None = None, *, as_of: str | None = None):
        assert symbol == "XYZ"
        assert as_of is None
        return chain.copy()

    async def fake_quotes(symbols: List[str]):
        return {sym: quotes[sym] for sym in symbols}, {"mode": "test"}

    snapshot = MarketSessionSnapshot(
        status="open",
        session="RTH",
        now_et=datetime.now(timezone.utc),
        next_open_et=datetime.now(timezone.utc) + timedelta(days=1),
        next_close_et=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setattr(agent_server, "fetch_option_chain_cached", fake_chain)
    monkeypatch.setattr(agent_server, "fetch_option_quotes", fake_quotes)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "snapshot", lambda: snapshot)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "last_rth_close", lambda at=None: snapshot.now_et)

    payload = ContractsRequest(symbol="XYZ", side="call", style="intraday")
    result = await gpt_contracts(payload, agent_server.AuthedUser(user_id="tester"))
    assert len(result["best"]) == 3
    assert result["relaxed_filters"] is True
    assert set(result.get("relaxation_reasons", []))


@pytest.mark.asyncio
async def test_market_closed_uses_prev_close(monkeypatch: pytest.MonkeyPatch):
    expiration = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=5)
    chain = pd.DataFrame(
        [
            {"symbol": "CALL_PRIMARY", "option_type": "call", "bid": 1.0, "ask": 1.1, "strike": 100},
            {"symbol": "CALL_SECONDARY", "option_type": "call", "bid": 0.8, "ask": 0.9, "strike": 101},
            {"symbol": "CALL_TERTIARY", "option_type": "call", "bid": 0.6, "ask": 0.7, "strike": 102},
        ]
    )
    quotes = {
        sym: _make_quote(symbol=sym, bid=row["bid"], ask=row["ask"], delta=0.5, oi=600, expiration=expiration)
        for sym, row in chain.set_index("symbol").iterrows()
    }
    observed_as_of: Dict[str, Any] = {}

    async def fake_chain(symbol: str, expiration: str | None = None, *, as_of: str | None = None):
        observed_as_of["value"] = as_of
        return chain.copy()

    async def fake_quotes(symbols: List[str]):
        return {sym: quotes[sym] for sym in symbols}, {"mode": "test"}

    closed_snapshot = MarketSessionSnapshot(
        status="closed",
        session="CLOSED",
        now_et=datetime(2024, 5, 20, 21, tzinfo=timezone.utc),
        next_open_et=datetime(2024, 5, 21, 13, tzinfo=timezone.utc),
        next_close_et=datetime(2024, 5, 21, 20, tzinfo=timezone.utc),
    )
    last_close = datetime(2024, 5, 20, 20, tzinfo=timezone.utc)

    monkeypatch.setattr(agent_server, "fetch_option_chain_cached", fake_chain)
    monkeypatch.setattr(agent_server, "fetch_option_quotes", fake_quotes)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "snapshot", lambda: closed_snapshot)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "last_rth_close", lambda at=None: last_close)

    payload = ContractsRequest(symbol="XYZ", side="call", style="intraday")
    result = await gpt_contracts(payload, agent_server.AuthedUser(user_id="tester"))
    assert result["quote_session"] == "regular_close"
    assert result["as_of_timestamp"] == last_close.isoformat()
    assert observed_as_of.get("value") == "prev_close"


@pytest.mark.asyncio
async def test_hedge_emitted_when_mtf_weak(monkeypatch: pytest.MonkeyPatch):
    expiration = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=4)
    chain = pd.DataFrame(
        [
            {"symbol": "CALL_MAIN", "option_type": "call", "bid": 1.2, "ask": 1.35, "strike": 100},
            {"symbol": "CALL_ALT", "option_type": "call", "bid": 0.9, "ask": 1.0, "strike": 101},
            {"symbol": "PUT_HEDGE", "option_type": "put", "bid": 0.35, "ask": 0.42, "strike": 95},
            {"symbol": "PUT_HEDGE_B", "option_type": "put", "bid": 0.25, "ask": 0.32, "strike": 94},
        ]
    )
    quotes = {
        "CALL_MAIN": _make_quote(symbol="CALL_MAIN", bid=1.2, ask=1.35, delta=0.55, oi=900, expiration=expiration),
        "CALL_ALT": _make_quote(symbol="CALL_ALT", bid=0.9, ask=1.0, delta=0.5, oi=800, expiration=expiration),
        "PUT_HEDGE": _make_quote(symbol="PUT_HEDGE", bid=0.35, ask=0.42, delta=-0.22, oi=700, expiration=expiration),
        "PUT_HEDGE_B": _make_quote(symbol="PUT_HEDGE_B", bid=0.25, ask=0.32, delta=-0.18, oi=600, expiration=expiration),
    }

    async def fake_chain(symbol: str, expiration: str | None = None, *, as_of: str | None = None):
        return chain.copy()

    async def fake_quotes(symbols: List[str]):
        return {sym: quotes[sym] for sym in symbols}, {"mode": "test"}

    open_snapshot = MarketSessionSnapshot(
        status="open",
        session="RTH",
        now_et=datetime.now(timezone.utc),
        next_open_et=datetime.now(timezone.utc) + timedelta(days=1),
        next_close_et=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    monkeypatch.setattr(agent_server, "fetch_option_chain_cached", fake_chain)
    monkeypatch.setattr(agent_server, "fetch_option_quotes", fake_quotes)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "snapshot", lambda: open_snapshot)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "last_rth_close", lambda at=None: open_snapshot.now_et)

    plan_anchor = {
        "underlying_entry": 100.0,
        "stop": 98.0,
        "targets": [103.0],
        "expected_duration": {"minutes": 90},
        "mtf_bias": {"agreement": 0.3},
        "rr_to_tp1": 0.8,
    }
    payload = ContractsRequest(symbol="XYZ", side="call", style="intraday", plan_anchor=plan_anchor)
    result = await gpt_contracts(payload, agent_server.AuthedUser(user_id="tester"))
    assert len(result["best"]) == 3
    hedge = result.get("hedge")
    assert hedge is not None
    assert hedge.get("role") == "hedge"


@pytest.mark.asyncio
async def test_never_empty_top_three(monkeypatch: pytest.MonkeyPatch):
    expiration = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=2)
    chain = pd.DataFrame(
        [
            {"symbol": "CALL_ONE", "option_type": "call", "bid": 1.0, "ask": 1.1, "strike": 100},
            {"symbol": "CALL_TWO", "option_type": "call", "bid": 0.8, "ask": 0.9, "strike": 101},
            {"symbol": "CALL_WIDE_SPREAD", "option_type": "call", "bid": 0.5, "ask": 0.9, "strike": 102},
        ]
    )
    quotes = {
        "CALL_ONE": _make_quote(symbol="CALL_ONE", bid=1.0, ask=1.1, delta=0.52, oi=500, expiration=expiration),
        "CALL_TWO": _make_quote(symbol="CALL_TWO", bid=0.8, ask=0.9, delta=0.48, oi=400, expiration=expiration),
        "CALL_WIDE_SPREAD": _make_quote(
            symbol="CALL_WIDE_SPREAD",
            bid=0.5,
            ask=0.9,
            delta=0.35,
            oi=350,
            expiration=expiration + timedelta(days=7),
        ),
    }

    async def fake_chain(symbol: str, expiration: str | None = None, *, as_of: str | None = None):
        return chain.copy()

    async def fake_quotes(symbols: List[str]):
        payload = {sym: quotes[sym] for sym in symbols if sym in quotes}
        return payload, {"mode": "test"}

    open_snapshot = MarketSessionSnapshot(
        status="open",
        session="RTH",
        now_et=datetime.now(timezone.utc),
        next_open_et=datetime.now(timezone.utc) + timedelta(days=1),
        next_close_et=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    monkeypatch.setattr(agent_server, "fetch_option_chain_cached", fake_chain)
    monkeypatch.setattr(agent_server, "fetch_option_quotes", fake_quotes)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "snapshot", lambda: open_snapshot)
    monkeypatch.setattr(agent_server._MARKET_CLOCK, "last_rth_close", lambda at=None: open_snapshot.now_et)

    payload = ContractsRequest(symbol="XYZ", side="call", style="intraday")
    result = await gpt_contracts(payload, agent_server.AuthedUser(user_id="tester"))
    assert len(result["best"]) == 3
    assert len(result["best"]) == 3
