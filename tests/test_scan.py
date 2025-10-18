import asyncio
from datetime import datetime

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.agent_server import (
    AuthedUser,
    PlanRequest,
    ScanContext,
    ScanRequest,
    ScanUniverse,
    _legacy_scan,
    gpt_plan,
    gpt_scan_endpoint,
)
from src.scan_features import Metrics, Penalties
from src.schemas import ScanFilters, ScanPage
from src.scanner import Plan, Signal, _build_context, _prepare_symbol_frame


def _make_history() -> pd.DataFrame:
    index = pd.date_range("2025-10-14 15:30", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0, 101.5, 102.0],
            "high": [100.6, 101.1, 101.6, 102.1, 102.6],
            "low": [99.8, 100.2, 100.7, 101.0, 101.4],
            "close": [100.4, 100.9, 101.4, 101.9, 102.2],
            "volume": [1_000_000, 1_200_000, 1_100_000, 1_300_000, 1_250_000],
        },
        index=index,
    )


def _make_request() -> Request:
    scope = {
        "type": "http",
        "headers": [],
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "client": ("test", 0),
        "scheme": "https",
        "server": ("test", 443),
    }
    return Request(scope)


def _context_stub() -> tuple[ScanContext, None, dict[str, object]]:
    context = ScanContext(
        as_of=datetime(2025, 10, 14, 20, 0),
        label="live",
        is_open=True,
        data_timeframe="5",
        market_meta={},
        data_meta={"ok": True},
    )
    return context, None, {"ok": True}


def _metrics_for(
    symbol: str,
    sector: str | None,
    *,
    entry_quality: float = 0.7,
    rr_t1: float = 1.5,
    rr_t2: float = 2.0,
    confidence: float = 0.7,
    penalties: Penalties | None = None,
) -> Metrics:
    penalties = penalties or Penalties(0.0, 0.0, 0.0, 0.0, 0.0)
    return Metrics(
        symbol=symbol,
        sector=sector,
        entry_quality=entry_quality,
        rr_t1=rr_t1,
        rr_t2=rr_t2,
        liquidity=0.7,
        confluence_micro=0.65,
        momentum_micro=0.6,
        vol_ok=0.6,
        struct_d1=0.6,
        conf_d1=0.6,
        mom_htf=0.6,
        conf_htf=0.6,
        struct_w1=0.6,
        vol_regime=0.55,
        opt_eff=0.6,
        rr_multi=1.8,
        macro_fit=0.62,
        context_score=0.6,
        penalties=penalties,
        confidence=confidence,
        actionability=0.6,
        entry_distance_pct=0.8,
        entry_distance_atr=0.8,
        bars_to_trigger=2.5,
    )


def _make_signal(symbol: str, idx: int, sector: str) -> Signal:
    plan = Plan(
        direction="long",
        entry=100.0 + idx,
        stop=99.0 + idx,
        targets=[101.0 + idx, 102.0 + idx],
        confidence=0.6,
        risk_reward=1.5,
    )
    signal = Signal(
        symbol=symbol,
        strategy_id="baseline_auto",
        description=f"{symbol} setup",
        score=float(200 - idx),
    )
    signal.plan = plan
    signal.features = {"direction_bias": "long", "sector": sector}
    return signal


def test_build_context_uses_rth_phase_for_close_bar():
    index = pd.date_range(
        start="2025-10-16 09:30",
        end="2025-10-16 16:00",
        freq="5min",
        tz="America/New_York",
    ).tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(len(index))],
            "high": [100.2 + i * 0.1 for i in range(len(index))],
            "low": [99.8 + i * 0.1 for i in range(len(index))],
            "close": [100.1 + i * 0.1 for i in range(len(index))],
            "volume": [1_000_000 + i * 1000 for i in range(len(index))],
        },
        index=index,
    )
    prepared = _prepare_symbol_frame(frame)
    ctx = _build_context(prepared)
    assert ctx["session_phase"] in {"power_hour", "afternoon"}, "should reflect RTH context even at close"


@pytest.mark.asyncio
async def test_scan_top100_paging(monkeypatch):
    tickers = [f"SYM{idx}" for idx in range(120)]

    async def fake_expand_universe(universe, style, limit):  # noqa: ARG001
        return tickers[:limit]

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        frames = {sym: _make_history() for sym in symbols}
        sources = {sym: "polygon" for sym in symbols}
        return frames, sources

    async def fake_scan_market(symbols, market_data, **kwargs):  # noqa: ARG001
        signals: list[Signal] = []
        for idx, sym in enumerate(symbols):
            sector = f"SEC{idx % 10}"
            signals.append(_make_signal(sym, idx, sector))
        return signals

    def fake_indicator_bundle(symbol, history):  # noqa: ARG001
        return {
            "key_levels": {"session_high": 105.0},
            "snapshot": {"timestamp_utc": "2025-10-14T20:00:00Z", "indicators": {}},
            "indicators": {},
        }

    async def fake_indicator_metrics(symbol, history):  # noqa: ARG001
        return {}

    def fake_compute_metrics_fast(symbol, style, context):  # noqa: ARG001
        idx = int(symbol.replace("SYM", ""))
        sector = f"SEC{idx % 10}"
        base = max(0.5, 0.95 - idx * 0.003)
        rr1 = 1.5 + max(0.0, 0.4 - idx * 0.004)
        rr2 = 2.1 + max(0.0, 0.3 - idx * 0.003)
        confidence = max(0.4, 0.78 - idx * 0.002)
        return _metrics_for(symbol, sector, entry_quality=base, rr_t1=rr1, rr_t2=rr2, confidence=confidence)

    monkeypatch.setattr("src.agent_server.expand_universe", fake_expand_universe)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server._indicator_bundle", fake_indicator_bundle)
    monkeypatch.setattr("src.agent_server._indicator_metrics", fake_indicator_metrics)
    monkeypatch.setattr("src.agent_server.compute_metrics_fast", fake_compute_metrics_fast)
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: _context_stub())

    request_payload = ScanRequest(universe="FT-TopLiquidity", style="intraday", limit=100)
    request = _make_request()
    user = AuthedUser(user_id="tester")

    first_page: ScanPage = await gpt_scan_endpoint(request_payload, request, user)
    assert len(first_page.candidates) == 50
    assert first_page.candidates[0].symbol == "SYM0"
    assert first_page.candidates[0].score >= first_page.candidates[1].score
    assert first_page.next_cursor is not None

    second_payload = request_payload.model_copy(update={"cursor": first_page.next_cursor})
    second_page: ScanPage = await gpt_scan_endpoint(second_payload, request, user)
    assert len(second_page.candidates) == 50
    assert second_page.candidates[0].symbol == "SYM50"
    assert second_page.next_cursor is None
    combined = {cand.symbol for cand in first_page.candidates + second_page.candidates}
    assert len(combined) == 100


@pytest.mark.asyncio
async def test_scan_never_5xx_on_stale(monkeypatch):
    async def fake_expand_universe(universe, style, limit):  # noqa: ARG001
        return ["AAPL", "MSFT", "TSLA", "NVDA"]

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        if as_of is None:
            raise HTTPException(status_code=502, detail="polygon down")
        frames = {sym: _make_history() for sym in symbols}
        return frames, {sym: "polygon" for sym in symbols}

    async def fake_scan_market(symbols, market_data, **kwargs):  # noqa: ARG001
        return [_make_signal(sym, idx, "TECH") for idx, sym in enumerate(symbols)]

    def fake_indicator_bundle(symbol, history):  # noqa: ARG001
        return {"key_levels": {}, "snapshot": {"timestamp_utc": "2025-10-14T20:00:00Z", "indicators": {}}, "indicators": {}}

    async def fake_indicator_metrics(symbol, history):  # noqa: ARG001
        return {}

    def fake_compute_metrics_fast(symbol, style, context):  # noqa: ARG001
        base = 0.75 if symbol == "AAPL" else 0.7
        confidence = 0.8 if symbol == "AAPL" else 0.7
        return _metrics_for(symbol, "TECH", entry_quality=base, confidence=confidence)

    monkeypatch.setattr("src.agent_server.expand_universe", fake_expand_universe)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server._indicator_bundle", fake_indicator_bundle)
    monkeypatch.setattr("src.agent_server._indicator_metrics", fake_indicator_metrics)
    monkeypatch.setattr("src.agent_server.compute_metrics_fast", fake_compute_metrics_fast)
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: _context_stub())

    request_payload = ScanRequest(universe="FT-TopLiquidity", style="intraday", limit=20)
    request = _make_request()
    user = AuthedUser(user_id="tester")

    page = await gpt_scan_endpoint(request_payload, request, user)
    assert page.candidates
    assert page.planning_context == "frozen"
    assert page.data_quality.get("mode") == "degraded"
    assert page.banner is not None and "frozen" in page.banner.lower()


@pytest.mark.asyncio
async def test_scan_relaxes_min_rvol_when_frozen(monkeypatch):
    symbols = ["SYM0", "SYM1", "SYM2"]

    async def fake_expand_universe(universe, style, limit):  # noqa: ARG001
        return symbols[:limit]

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        frames = {sym: _make_history() for sym in symbols}
        return frames, {sym: "polygon" for sym in symbols}

    async def fake_scan_market(symbols, market_data, **kwargs):  # noqa: ARG001
        return [_make_signal(sym, idx, "TECH") for idx, sym in enumerate(symbols)]

    def fake_indicator_bundle(symbol, history):  # noqa: ARG001
        return {"key_levels": {}, "snapshot": {"timestamp_utc": "2025-10-14T20:00:00Z", "indicators": {}}, "indicators": {}}

    async def fake_indicator_metrics(symbol, history):  # noqa: ARG001
        return {}

    def fake_compute_metrics_fast(symbol, style, context):  # noqa: ARG001
        return _metrics_for(symbol, "TECH")

    def frozen_context(policy, style):  # noqa: ARG001
        context = ScanContext(
            as_of=datetime(2025, 10, 14, 20, 0),
            label="frozen",
            is_open=False,
            data_timeframe="5",
            market_meta={},
            data_meta={"ok": True},
        )
        return context, "Frozen planning context requested.", {"ok": True}

    monkeypatch.setattr("src.agent_server.expand_universe", fake_expand_universe)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server._indicator_bundle", fake_indicator_bundle)
    monkeypatch.setattr("src.agent_server._indicator_metrics", fake_indicator_metrics)
    monkeypatch.setattr("src.agent_server.compute_metrics_fast", fake_compute_metrics_fast)
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: frozen_context(policy, style))

    request_payload = ScanRequest(
        universe=symbols,
        style="intraday",
        limit=20,
        filters=ScanFilters(min_rvol=1.2),
    )
    request = _make_request()
    user = AuthedUser(user_id="tester")

    page = await gpt_scan_endpoint(request_payload, request, user)
    assert page.planning_context == "frozen"
    assert page.candidates
    assert page.data_quality.get("ok") is True


@pytest.mark.asyncio
async def test_scan_fallback_limits_and_confidence_sort(monkeypatch):
    tickers = [f"T{idx}" for idx in range(40)]

    async def fake_expand_universe(universe, style, limit):  # noqa: ARG001
        return tickers[:limit]

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        frames = {}
        for idx, sym in enumerate(symbols):
            if idx % 3 == 0:
                frames[sym] = pd.DataFrame()  # force missing data
            else:
                history = _make_history()
                history["volume"] = [200_000 + (idx * 10_000)] * len(history)
                frames[sym] = history
        return frames, {sym: "polygon" for sym in symbols}

    async def fake_scan_market(symbols, market_data, **kwargs):  # noqa: ARG001
        return []  # force fallback

    def frozen_context(policy, style):  # noqa: ARG001
        context = ScanContext(
            as_of=datetime(2025, 10, 14, 20, 0),
            label="frozen",
            is_open=False,
            data_timeframe="5",
            market_meta={},
            data_meta={"ok": True},
        )
        return context, "Frozen planning context requested.", {"ok": True}

    monkeypatch.setattr("src.agent_server.expand_universe", fake_expand_universe)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: frozen_context(policy, style))

    request_payload = ScanRequest(
        universe="FT-TopLiquidity",
        style="intraday",
        limit=100,
    )
    request = _make_request()
    user = AuthedUser(user_id="tester")

    page = await gpt_scan_endpoint(request_payload, request, user)
    assert page.candidates
    assert 1 <= len(page.candidates) <= 15
    confidences = [cand.confidence or 0.0 for cand in page.candidates]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.asyncio
async def test_universe_not_symbol(monkeypatch):
    monkeypatch.setattr("src.agent_server.expand_universe", lambda universe, style, limit: ["AAPL"])  # noqa: ARG001
    monkeypatch.setattr("src.agent_server._collect_market_data", lambda symbols, timeframe, as_of=None: ({sym: _make_history() for sym in symbols}, {sym: "polygon" for sym in symbols}))  # noqa: ARG001
    monkeypatch.setattr("src.agent_server.scan_market", lambda symbols, market_data, **kwargs: [])  # noqa: ARG001
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: _context_stub())

    request = _make_request()
    user = AuthedUser(user_id="tester")

    with pytest.raises(HTTPException) as excinfo:
        await gpt_plan(PlanRequest(symbol="FT-TopLiquidity"), request, user)
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_chart_query_guard(monkeypatch):
    async def fake_expand_universe(universe, style, limit):  # noqa: ARG001
        return ["AAPL"]

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        frames = {sym: _make_history() for sym in symbols}
        return frames, {sym: "polygon" for sym in symbols}

    async def fake_scan_market(symbols, market_data, **kwargs):  # noqa: ARG001
        signal = Signal(symbol="AAPL", strategy_id="baseline_auto", description="Test setup", score=1.0)
        signal.plan = None
        signal.features = {"direction_bias": "long"}
        return [signal]

    def fake_indicator_bundle(symbol, history):  # noqa: ARG001
        return {"key_levels": {}, "snapshot": {"timestamp_utc": "2025-10-14T20:00:00Z", "indicators": {}}, "indicators": {}}

    async def fake_indicator_metrics(symbol, history):  # noqa: ARG001
        return {}

    def fake_compute_metrics_fast(symbol, style, context):  # noqa: ARG001
        return _metrics_for(symbol, "TECH", entry_quality=0.65)

    monkeypatch.setattr("src.agent_server.expand_universe", fake_expand_universe)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server._indicator_bundle", fake_indicator_bundle)
    monkeypatch.setattr("src.agent_server._indicator_metrics", fake_indicator_metrics)
    monkeypatch.setattr("src.agent_server.compute_metrics_fast", fake_compute_metrics_fast)
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: _context_stub())

    universe = ScanUniverse(tickers=["AAPL"], style="intraday", limit=10)
    request = _make_request()
    user = AuthedUser(user_id="tester")

    results = await _legacy_scan(universe, request, user, fetch_options=False)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_sector_cap_and_min_quality(monkeypatch):
    tickers = [f"S{idx}" for idx in range(110)]

    async def fake_expand_universe(universe, style, limit):  # noqa: ARG001
        return tickers[:limit]

    async def fake_collect_market_data(symbols, timeframe, as_of=None):  # noqa: ARG001
        frames = {sym: _make_history() for sym in symbols}
        return frames, {sym: "polygon" for sym in symbols}

    async def fake_scan_market(symbols, market_data, **kwargs):  # noqa: ARG001
        return [_make_signal(sym, idx, "ONESECTOR") for idx, sym in enumerate(symbols)]

    def fake_indicator_bundle(symbol, history):  # noqa: ARG001
        return {"key_levels": {}, "snapshot": {"timestamp_utc": "2025-10-14T20:00:00Z", "indicators": {}}, "indicators": {}}

    async def fake_indicator_metrics(symbol, history):  # noqa: ARG001
        return {}

    def fake_compute_metrics_fast(symbol, style, context):  # noqa: ARG001
        idx = int(symbol[1:])
        quality = 0.3 if idx >= 80 else 0.7
        penalties = Penalties(0.0, 0.0, 0.05 if idx % 7 == 0 else 0.0, 0.0, 0.0)
        return _metrics_for(symbol, "ONE", entry_quality=quality, penalties=penalties)

    monkeypatch.setattr("src.agent_server.expand_universe", fake_expand_universe)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server._indicator_bundle", fake_indicator_bundle)
    monkeypatch.setattr("src.agent_server._indicator_metrics", fake_indicator_metrics)
    monkeypatch.setattr("src.agent_server.compute_metrics_fast", fake_compute_metrics_fast)
    monkeypatch.setattr("src.agent_server._evaluate_scan_context", lambda policy, style, session: _context_stub())

    request_payload = ScanRequest(universe="FT-TopLiquidity", style="intraday", limit=100)
    request = _make_request()
    user = AuthedUser(user_id="tester")

    page = await gpt_scan_endpoint(request_payload, request, user)
    assert len(page.candidates) == 20
    assert page.next_cursor is None
    assert {cand.symbol for cand in page.candidates}.issubset(set(tickers))
    assert all(c.reasons for c in page.candidates)
