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
@pytest.mark.parametrize("style,expected_min,expected_max", [
    ("intraday", 3, 3),
    ("scalp", 3, 3),
    ("swing", 2, 2),
    ("leaps", 2, 2),
])
async def test_always_returns_two_to_three_with_ratings(monkeypatch, style, expected_min, expected_max):
    rows = [
        _row("put", 460, -0.55, 3, 0.05, 5000, 1200),
        _row("put", 455, -0.45, 3, 0.06, 4000, 1100),
        _row("put", 450, -0.35, 4, 0.07, 3000, 1000),
        _row("put", 445, -0.28, 4, 0.09, 1800, 900),
        _row("put", 440, -0.22, 5, 0.12, 900, 600),
    ]
    frame = pd.DataFrame(rows)

    async def fake_chain(*args, **kwargs):
        return frame

    monkeypatch.setattr(options_provider, "fetch_polygon_option_chain_asof", fake_chain)

    plan = {"direction": "short", "targets": [445.6], "bias": "short", "style": style, "strategy_id": "baseline_auto"}
    as_of = datetime(2025, 10, 27, 20, 0, 0, tzinfo=timezone.utc)

    result = await select_contracts("TSLA", as_of, plan)
    picks = result.get("options_contracts") or []

    assert expected_min <= len(picks) <= expected_max
    for contract in picks:
        assert contract.get("rating") in {"green", "yellow", "red"}
        reasons = contract.get("reasons")
        assert isinstance(reasons, list)
        assert reasons, "reasons should not be empty"
