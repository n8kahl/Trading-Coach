from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.providers.options import select_contracts
import src.providers.options as options_provider


def _row(opt_type, strike, delta, dte, spread_pct, oi, vol):
    return {
        "symbol": f"TSLA-{opt_type}-{strike}",
        "option_type": opt_type,
        "strike": float(strike),
        "delta": float(delta),
        "dte": float(dte),
        "spread_pct": float(spread_pct),
        "open_interest": int(oi),
        "volume": int(vol),
        "bid": 1.00,
        "ask": 1.10,
        "mid": 1.05,
        "expiration_date": "2025-10-31",
    }


@pytest.mark.asyncio()
async def test_closed_market_uses_prev_close_quotes_and_labels(monkeypatch):
    frame = pd.DataFrame([
        _row("put", 455, -0.45, 3, 0.06, 4000, 1100),
        _row("put", 450, -0.35, 4, 0.07, 3000, 1000),
        _row("put", 445, -0.28, 4, 0.09, 1800, 900),
    ])

    async def fake_chain_asof(symbol, as_of, *_, **__):
        return frame

    monkeypatch.setattr(options_provider, "fetch_polygon_option_chain_asof", fake_chain_asof)

    as_of = datetime(2025, 10, 27, 21, 0, 0, tzinfo=timezone.utc)
    plan = {"direction": "short", "targets": [445.6], "bias": "short", "style": "intraday", "strategy_id": "baseline_auto"}

    result = await select_contracts("TSLA", as_of, plan)
    picks = result.get("options_contracts") or []

    assert len(picks) == 3
    for contract in picks:
        assert contract.get("quote_session") in {"regular_close", "regular_open"}
        assert contract.get("as_of_timestamp"), "as_of timestamp missing"
    assert result.get("options_quote_session") in {"regular_close", "regular_open"}
    assert result.get("options_as_of")
