import math

import pytest

from src.scanner import _apply_tp_logic, _base_targets_for_style, rr


def _make_tp_ctx(
    *,
    key_levels=None,
    vwap=None,
    ema9=None,
    ema20=None,
    ema50=None,
    htf_levels=None,
    expected_move=None,
    atr=None,
):
    key_levels = key_levels or {}
    return {
        "key": {k: float(v) for k, v in key_levels.items() if isinstance(v, (int, float))},
        "vol_profile": {},
        "vwap": vwap,
        "ema9": ema9,
        "ema20": ema20,
        "ema50": ema50,
        "fib_up": {},
        "fib_down": {},
        "htf_levels": [float(v) for v in (htf_levels or []) if isinstance(v, (int, float))],
        "expected_move_horizon": expected_move,
        "atr": atr,
    }


def test_apply_tp_logic_respects_expected_move_cap_with_warning():
    entry = 100.0
    stop = 98.0
    atr = 1.5
    expected_move = 1.0
    min_rr = 1.2

    base_targets = _base_targets_for_style(
        style="intraday",
        bias="long",
        entry=entry,
        stop=stop,
        atr=atr,
        expected_move=expected_move,
        min_rr=min_rr,
        prefer_em_cap=True,
    )

    ctx = _make_tp_ctx(expected_move=expected_move, atr=atr)
    targets, warnings, debug = _apply_tp_logic(
        symbol="AAPL",
        style="intraday",
        bias="long",
        entry=entry,
        stop=stop,
        base_targets=base_targets,
        ctx=ctx,
        min_rr=min_rr,
        atr=atr,
        expected_move=expected_move,
        prefer_em_cap=True,
    )

    assert len(targets) == 2
    assert targets[0] > entry
    diff = targets[0] - entry
    assert diff <= expected_move + 1e-6
    assert diff >= expected_move * 0.6  # RR nudge may pull slightly lower but should stay meaningful
    assert any("TP1 R:R" in message for message in warnings)
    assert debug["base"]


def test_apply_tp_logic_snap_allows_em_extension():
    entry = 50.0
    stop = 49.0
    atr = 1.2
    expected_move = 1.5
    min_rr = 1.2

    base_targets = _base_targets_for_style(
        style="swing",
        bias="long",
        entry=entry,
        stop=stop,
        atr=atr,
        expected_move=expected_move,
        min_rr=min_rr,
        prefer_em_cap=True,
    )

    key_level_price = 51.6
    ctx = _make_tp_ctx(
        key_levels={"prev_high": key_level_price},
        htf_levels=[key_level_price],
        expected_move=expected_move,
        atr=atr,
    )

    targets, warnings, debug = _apply_tp_logic(
        symbol="AAPL",
        style="swing",
        bias="long",
        entry=entry,
        stop=stop,
        base_targets=base_targets,
        ctx=ctx,
        min_rr=min_rr,
        atr=atr,
        expected_move=expected_move,
        prefer_em_cap=True,
    )

    assert not warnings
    assert math.isclose(targets[0], key_level_price, abs_tol=0.15)
    assert targets[1] >= targets[0]
    assert debug["snap"][0]["used_snapped"] is True
    assert targets[0] - entry <= expected_move * 1.10 + 1e-6


def test_apply_tp_logic_short_geometry_and_rr():
    entry = 100.0
    stop = 102.0
    atr = 1.5
    expected_move = 3.0
    min_rr = 1.2

    base_targets = _base_targets_for_style(
        style="intraday",
        bias="short",
        entry=entry,
        stop=stop,
        atr=atr,
        expected_move=expected_move,
        min_rr=min_rr,
        prefer_em_cap=True,
    )

    ctx = _make_tp_ctx(expected_move=expected_move, atr=atr)
    targets, warnings, debug = _apply_tp_logic(
        symbol="TSLA",
        style="intraday",
        bias="short",
        entry=entry,
        stop=stop,
        base_targets=base_targets,
        ctx=ctx,
        min_rr=min_rr,
        atr=atr,
        expected_move=expected_move,
        prefer_em_cap=True,
    )

    assert len(targets) == 2
    assert targets[0] < entry
    assert targets[0] > targets[1]
    rr_value = rr(entry, stop, targets[0], "short")
    assert rr_value >= min_rr - 1e-3 or not warnings
    assert math.isclose(debug["tick_size"], 0.05, rel_tol=0.0, abs_tol=0.05)
