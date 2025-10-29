from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.features.htf_levels import compute_intraday_htf_levels
from src.services.chart_levels import extract_supporting_levels


def _frame_from_rows(rows, freq: str = "60min") -> pd.DataFrame:
    index = pd.date_range(
        end=datetime.now(tz=timezone.utc),
        periods=len(rows),
        freq=freq,
        tz="UTC",
    )
    return pd.DataFrame(rows, index=index)


def test_compute_intraday_htf_levels_returns_last_completed_bar_values() -> None:
    rows_60m = [
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 101, "high": 103, "low": 100, "close": 102},
    ]
    frame_60 = _frame_from_rows(rows_60m)
    levels = compute_intraday_htf_levels(frame_60, None)
    assert levels["h1_high"] == 102
    assert levels["h1_low"] == 99
    assert "h1_pivot" in levels
    assert "h1_r1" in levels
    assert "h1_s1" in levels


def test_chart_tokenization_includes_h1_h4_and_pivots() -> None:
    plan = {
        "entry": 100.0,
        "stop": 99.0,
        "targets": [101.0],
        "key_levels": {
            "h1_high": 100.75,
            "h1_low": 99.85,
            "h1_pivot": 100.33,
            "h4_high": 101.8,
            "h4_low": 98.7,
            "h4_pivot": 100.10,
            "h1_r1": 100.90,
            "h1_s1": 99.80,
        },
        "decimals": 2,
    }
    token_string = extract_supporting_levels(plan)
    assert token_string is not None
    tokens = token_string.split(";")
    assert any("|H1H" in token for token in tokens)
    assert any("|H1L" in token for token in tokens)
    assert any("|H4H" in token for token in tokens)
    assert any("|H4L" in token for token in tokens)
    assert any("|Pivot" in token for token in tokens)
    assert any("|R1" in token for token in tokens)
    assert any("|S1" in token for token in tokens)
