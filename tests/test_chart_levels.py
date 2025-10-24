from __future__ import annotations

from src.services.chart_levels import extract_supporting_levels


def test_extract_supporting_levels_prioritizes_known_labels():
    plan = {
        "entry": 100.25,
        "stop": 99.9,
        "targets": [101.1, 101.8],
        "decimals": 2,
        "key_levels": {
            "opening_range_high": 100.75,
            "opening_range_low": 99.85,
            "session_high": 101.55,
            "prev_high": 103.2,
            "prev_low": 98.7,
            "prev_close": 100.05,
            "gap_fill": 99.6,
            "vah": 102.1,
            "poc": 101.7,
            "val": 99.1,
            "pivot": 100.8,
            "r1": 102.2,
            "s1": 99.4,
        },
        "key_levels_used": {
            "session": [
                {"label": "SESSION_LOW", "price": 99.32},
            ],
            "structural": [
                {"label": "Fib50", "price": 101.0},
                {"label": "DH", "price": 104.85},
            ],
        },
    }

    token_string = extract_supporting_levels(plan)
    assert token_string is not None

    tokens = token_string.split(";")
    assert tokens[:6] == [
        "100.75|ORH",
        "99.85|ORL",
        "101.55|SessionH",
        "99.32|SessionL",
        "103.20|PDH",
        "98.70|PDL",
    ]
    assert "100.05|PDC" in tokens
    assert "99.60|GapFill" in tokens
    assert "102.10|VAH" in tokens
    assert "101.70|POC" in tokens
    assert "99.10|VAL" in tokens
    assert "100.80|Pivot" in tokens
    assert "102.20|R1" in tokens
    assert "99.40|S1" in tokens
    assert "101.00|Fib50" in tokens
    assert "104.85|DH" in tokens


def test_extract_supporting_levels_returns_none_when_no_levels():
    assert extract_supporting_levels({}) is None
