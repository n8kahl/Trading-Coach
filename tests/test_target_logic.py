import math

import numpy as np

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
    atr_5m=None,
    atr_15m=None,
    atr_1d=None,
    atr_1w=None,
    vol_profile=None,
    vol_profile_daily=None,
    vol_profile_weekly=None,
    levels_daily=None,
    levels_weekly=None,
    fib_daily=None,
    fib_weekly=None,
    anchored_vwaps=None,
    target_stats=None,
):
    key_levels = key_levels or {}
    return {
        "key": {k: float(v) for k, v in key_levels.items() if isinstance(v, (int, float))},
        "vol_profile": vol_profile or {},
        "vol_profile_daily": vol_profile_daily or {},
        "vol_profile_weekly": vol_profile_weekly or {},
        "vwap": vwap,
        "ema9": ema9,
        "ema20": ema20,
        "ema50": ema50,
        "fib_up": {},
        "fib_down": {},
        "htf_levels": [float(v) for v in (htf_levels or []) if isinstance(v, (int, float))],
        "expected_move_horizon": expected_move,
        "atr": atr,
        "atr_5m": atr_5m,
        "atr_15m": atr_15m,
        "atr_1d": atr_1d,
        "atr_1w": atr_1w,
        "levels_daily": levels_daily or [],
        "levels_weekly": levels_weekly or [],
        "fib_daily": fib_daily or {"long": [], "short": []},
        "fib_weekly": fib_weekly or {"long": [], "short": []},
        "anchored_vwaps_intraday": anchored_vwaps or {},
        "target_stats": target_stats or {},
    }


def test_intraday_tp1_respects_min_distance_and_em_cap():
    entry = 100.0
    stop = 98.0
    atr = 1.4
    atr_15m = 1.6
    expected_move = 2.5
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

    ctx = _make_tp_ctx(
        expected_move=expected_move,
        atr=atr,
        atr_15m=atr_15m,
        atr_5m=1.1,
        key_levels={"session_high": 101.8, "prev_high": 102.1},
        vwap=101.2,
        ema9=101.0,
        ema20=100.7,
        vol_profile={"poc": 101.3, "vah": 101.9, "val": 99.6},
        htf_levels=[102.4],
    )

    targets, target_meta, warnings, debug = _apply_tp_logic(
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
    assert diff >= atr_15m * 0.8 - 0.05
    assert diff <= expected_move * 1.05  # EM cap applied with small allowance
    assert target_meta and len(target_meta) >= 2
    assert debug["meta"]


def test_swing_ignores_expected_move_cap():
    entry = 100.0
    stop = 89.0
    atr = 2.5
    atr_1d = 7.0
    expected_move = 3.5  # should be ignored for swing targets
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

    daily_level = 210.0
    ctx = _make_tp_ctx(
        key_levels={"prev_high": 198.0},
        htf_levels=[205.0],
        atr=atr,
        atr_1d=atr_1d,
        atr_1w=12.0,
        expected_move=expected_move,
        levels_daily=[("DAILY_HIGH", daily_level)],
        vol_profile_daily={"poc": 206.0},
        fib_daily={"long": [("FIB1.272", 215.0)], "short": []},
        target_stats={"swing": {
            "style": "swing",
            "expected_move": 15.0,
            "long": {
                "mfe": np.array([0.2, 0.25, 0.3, 0.35]),
                "quantiles": {"q50": 0.2, "q70": 0.25, "q80": 0.3, "q90": 0.35},
            },
        }},
    )

    targets, target_meta, warnings, debug = _apply_tp_logic(
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

    if warnings:
        assert warnings[0].startswith("TP1 R:R"), warnings
    assert targets[0] >= entry + max(atr_1d * 1.20, entry * 0.015) - 0.01
    assert targets[0] > entry + expected_move  # EM cap not applied
    assert target_meta and target_meta[0].get("source") == "stats"
    assert target_meta[0].get("em_fraction") and target_meta[0]["em_fraction"] > 0


def test_rr_floor_triggers_watch_warning_when_unresolved():
    entry = 50.0
    stop = 49.5
    atr = 0.4
    expected_move = 5.0
    min_rr = 1.2

    base_targets = _base_targets_for_style(
        style="scalp",
        bias="long",
        entry=entry,
        stop=stop,
        atr=atr,
        expected_move=expected_move,
        min_rr=min_rr,
        prefer_em_cap=True,
    )

    ctx = _make_tp_ctx(
        expected_move=expected_move,
        atr=atr,
        atr_5m=0.4,
        key_levels={},
        vol_profile={},
        target_stats={"scalp": {
            "style": "scalp",
            "expected_move": 1.2,
            "long": {"mfe": np.array([0.12, 0.18, 0.22]), "quantiles": {"q50": 0.12, "q70": 0.18, "q80": 0.2, "q90": 0.24}},
        }},
    )

    targets, target_meta, warnings, debug = _apply_tp_logic(
        symbol="MSFT",
        style="scalp",
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

    assert len(targets) >= 2
    assert targets[0] > entry
    assert warnings and any("watch plan" in msg.lower() for msg in warnings)
    assert target_meta and len(target_meta) >= len(targets)
    if len(target_meta) >= 3:
        assert target_meta[2].get("optional") is True
