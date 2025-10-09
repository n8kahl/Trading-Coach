"""Utilities for deriving higher timeframe zones and contextual overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

TZ_ET = "America/New_York"


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    return frame


def _latest_session(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    frame = _ensure_datetime_index(frame)
    if frame.empty:
        return frame, None
    et_index = frame.index.tz_convert(TZ_ET)
    session_dates = pd.Series(et_index.date, index=frame.index)
    if session_dates.empty:
        return frame, None

    last_date = session_dates.iloc[-1]
    session_mask = session_dates.eq(last_date).astype(bool).to_numpy()
    session_df = frame.iloc[session_mask]

    prev_df: pd.DataFrame | None = None
    if session_dates.nunique() >= 2:
        unique_dates = session_dates.unique()
        prev_candidates = [d for d in unique_dates if d != last_date]
        if prev_candidates:
            prev_date = prev_candidates[-1]
            prev_mask = session_dates.eq(prev_date).astype(bool).to_numpy()
            prev_df = frame.iloc[prev_mask]

    return session_df if not session_df.empty else frame.tail(240), prev_df


def _strength_from_volume(volume: float, baseline: float) -> str:
    if baseline <= 0 or volume is None or math.isnan(volume):
        return "unknown"
    if volume >= baseline * 1.8:
        return "strong"
    if volume >= baseline * 1.1:
        return "moderate"
    return "weak"


def _timeframe_label(interval: str) -> str:
    return interval.lower()


def _detect_zones(session: pd.DataFrame, interval: str, kind: str, window: int = 5, max_zones: int = 5) -> List[Dict[str, object]]:
    if session.empty or len(session) < window * 2 + 1:
        return []
    highs = session["high"]
    lows = session["low"]
    opens = session["open"]
    closes = session["close"]
    volumes = session.get("volume", pd.Series(dtype=float))
    baseline = float(volumes.tail(200).median()) if not volumes.empty else 0.0
    zones: List[Dict[str, object]] = []

    for idx in range(window, len(session) - window):
        window_slice = session.iloc[idx - window : idx + window + 1]
        bar = session.iloc[idx]
        if kind == "supply":
            if bar["high"] >= window_slice["high"].max():
                high = float(bar["high"])
                low = float(max(bar["open"], bar["close"]))
            else:
                continue
        else:
            if bar["low"] <= window_slice["low"].min():
                low = float(bar["low"])
                high = float(min(bar["open"], bar["close"]))
            else:
                continue

        if math.isclose(high, low):
            continue
        zone = {
            "low": round(min(low, high), 4),
            "high": round(max(low, high), 4),
            "timeframe": _timeframe_label(interval),
            "strength": _strength_from_volume(float(bar.get("volume", 0.0)), baseline),
        }
        zones.append(zone)

    merged: List[Dict[str, object]] = []
    tolerance = max(0.01, float(session["close"].iloc[-1]) * 0.002)
    for zone in sorted(zones, key=lambda z: z["high"], reverse=(kind == "supply")):
        if not merged:
            merged.append(zone)
            continue
        last = merged[-1]
        overlap = not (zone["high"] < last["low"] - tolerance or zone["low"] > last["high"] + tolerance)
        if overlap:
            last["low"] = round(min(last["low"], zone["low"]), 4)
            last["high"] = round(max(last["high"], zone["high"]), 4)
        else:
            merged.append(zone)
        if len(merged) >= max_zones:
            break
    return merged


def _detect_liquidity_pools(session: pd.DataFrame, interval: str, tolerance_pct: float = 0.0015) -> List[Dict[str, object]]:
    if session.empty:
        return []
    pools: List[Dict[str, object]] = []
    price_ref = float(session["close"].iloc[-1])
    tolerance = max(0.02, price_ref * tolerance_pct)

    def _cluster(levels: Iterable[float], label: str):
        levels = np.array(list(levels), dtype=float)
        if levels.size == 0:
            return
        used = np.zeros(levels.shape, dtype=bool)
        for i, price in enumerate(levels):
            if used[i]:
                continue
            mask = np.abs(levels - price) <= tolerance
            count = int(mask.sum())
            if count >= 2:
                density = round(count / max(1, len(session)), 4)
                pools.append(
                    {
                        "level": round(float(np.mean(levels[mask])), 4),
                        "type": label,
                        "timeframe": _timeframe_label(interval),
                        "density": density,
                    }
                )
            used = used | mask

    _cluster(session["high"], "equal_highs")
    _cluster(session["low"], "equal_lows")
    pools.sort(key=lambda p: p["density"], reverse=True)
    return pools[:6]


def _detect_fvg(session: pd.DataFrame, interval: str) -> List[Dict[str, object]]:
    if len(session) < 3:
        return []
    fvgs: List[Dict[str, object]] = []
    highs = session["high"].to_numpy()
    lows = session["low"].to_numpy()
    for idx in range(2, len(session)):
        bull_gap = lows[idx] > highs[idx - 2]
        bear_gap = highs[idx] < lows[idx - 2]
        if bull_gap or bear_gap:
            low = round(float(highs[idx - 2]), 4)
            high = round(float(lows[idx]), 4)
            if bear_gap:
                low = round(float(highs[idx]), 4)
                high = round(float(lows[idx - 2]), 4)
            fvgs.append(
                {
                    "low": min(low, high),
                    "high": max(low, high),
                    "timeframe": _timeframe_label(interval),
                    "age": len(session) - idx,
                }
            )
    return fvgs[-6:]


def _relative_strength(history: pd.DataFrame, benchmark: pd.DataFrame | None, lookback: int = 15) -> Dict[str, object]:
    payload = {"benchmark": "SPY", "lookback_bars": lookback, "value": None}
    if benchmark is None or history.empty:
        return payload
    history = _ensure_datetime_index(history)
    benchmark = _ensure_datetime_index(benchmark)
    common_index = history.index.intersection(benchmark.index)
    if len(common_index) < lookback + 1:
        return payload
    asset_slice = history.loc[common_index]["close"].astype(float)
    bench_slice = benchmark.loc[common_index]["close"].astype(float)
    if asset_slice.isna().any() or bench_slice.isna().any():
        return payload
    lookback = min(lookback, len(asset_slice) - 1, len(bench_slice) - 1)
    asset_ret = asset_slice.pct_change().tail(lookback).dropna()
    bench_ret = bench_slice.pct_change().tail(lookback).dropna()
    if asset_ret.empty or bench_ret.empty:
        return payload
    asset_compound = float((asset_ret + 1.0).prod() - 1.0)
    bench_compound = float((bench_ret + 1.0).prod() - 1.0)
    if bench_compound == -1.0:
        return payload
    value = (1.0 + asset_compound) / (1.0 + bench_compound)
    payload["value"] = round(value, 4)
    return payload


def _anchored_vwap(history: pd.DataFrame, start_index: pd.Timestamp | None) -> Optional[float]:
    if start_index is None:
        return None
    if start_index.tzinfo is None:
        start_index = start_index.tz_localize("UTC")
    else:
        start_index = start_index.tz_convert("UTC")
    sub = history[history.index >= start_index]
    if sub.empty:
        return None
    closes = sub["close"].astype(float)
    volumes = sub.get("volume", pd.Series(dtype=float)).fillna(0.0).astype(float)
    if volumes.sum() == 0:
        return None
    cum_price_volume = (closes * volumes).cumsum()
    cum_volume = volumes.cumsum()
    vwap = cum_price_volume.iloc[-1] / cum_volume.iloc[-1]
    return round(float(vwap), 4)


def _volume_profile(session: pd.DataFrame, bins: int = 30) -> Dict[str, Optional[float]]:
    closes = session["close"].astype(float)
    volumes = session.get("volume", pd.Series(dtype=float)).fillna(0.0).astype(float)
    if closes.empty or volumes.sum() == 0:
        return {"vwap": None, "vah": None, "val": None, "poc": None}

    price_min = float(closes.min())
    price_max = float(closes.max())
    if math.isclose(price_min, price_max):
        return {"vwap": float(closes.iloc[-1]), "vah": None, "val": None, "poc": float(closes.iloc[-1])}

    bins = max(10, min(bins, len(closes)))
    hist, edges = np.histogram(closes, bins=bins, weights=volumes)
    if hist.sum() <= 0:
        return {"vwap": None, "vah": None, "val": None, "poc": None}

    poc_idx = int(hist.argmax())
    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0

    target_volume = hist.sum() * 0.7
    cum_volume = hist[poc_idx]
    low_idx = high_idx = poc_idx
    while cum_volume < target_volume and (low_idx > 0 or high_idx < len(hist) - 1):
        left_next = hist[low_idx - 1] if low_idx > 0 else -1
        right_next = hist[high_idx + 1] if high_idx < len(hist) - 1 else -1
        if right_next >= left_next:
            high_idx += 1
            cum_volume += max(right_next, 0)
        else:
            low_idx -= 1
            cum_volume += max(left_next, 0)

    vah = (edges[high_idx] + edges[high_idx + 1]) / 2.0
    val = (edges[low_idx] + edges[low_idx + 1]) / 2.0
    vwap = float((closes * volumes).sum() / volumes.sum())
    return {
        "vwap": round(vwap, 4),
        "vah": round(float(vah), 4),
        "val": round(float(val), 4),
        "poc": round(float(poc), 4),
    }


def _options_summary(chain: pd.DataFrame | None) -> Dict[str, Optional[float]]:
    summary = {
        "atm_iv": None,
        "iv_rank": None,
        "iv_pct": None,
        "skew_25d": None,
        "term_slope": None,
        "spread_bps": None,
    }
    if chain is None or chain.empty:
        return summary
    frame = chain.copy()
    if "mid" not in frame.columns or frame["mid"].isna().all():
        frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    if "spread_pct" not in frame.columns or frame["spread_pct"].isna().all():
        mid = frame["mid"].replace(0, np.nan)
        frame["spread_pct"] = (frame["ask"] - frame["bid"]) / mid

    # ATM IV (closest to 0.5 delta)
    delta_series = frame["delta"].astype(float)
    valid_delta = frame[delta_series.notna()]
    if not valid_delta.empty:
        idx = (valid_delta["delta"].abs() - 0.5).abs().idxmin()
        atm_row = valid_delta.loc[idx]
        iv_val = atm_row.get("implied_volatility")
        summary["atm_iv"] = round(float(iv_val), 4) if iv_val else None

    # 25-delta skew (call - put)
    calls = frame[(frame["option_type"] == "call") & frame["delta"].notna()].copy()
    puts = frame[(frame["option_type"] == "put") & frame["delta"].notna()].copy()
    call_25 = None
    put_25 = None
    if not calls.empty:
        call_idx = (calls["delta"] - 0.25).abs().idxmin()
        call_25 = calls.loc[call_idx]
    if not puts.empty:
        put_idx = (puts["delta"].abs() - 0.25).abs().idxmin()
        put_25 = puts.loc[put_idx]
    if call_25 is not None and put_25 is not None:
        call_iv = call_25.get("implied_volatility")
        put_iv = put_25.get("implied_volatility")
        if call_iv and put_iv:
            summary["skew_25d"] = round(float(call_iv) - float(put_iv), 4)

    # Term structure slope (front vs back)
    if "dte" in frame.columns and "implied_volatility" in frame.columns:
        near = frame[(frame["dte"] >= 0) & (frame["dte"] <= 7) & frame["implied_volatility"].notna()]
        far = frame[(frame["dte"] >= 21) & (frame["dte"] <= 60) & frame["implied_volatility"].notna()]
        if not near.empty and not far.empty:
            summary["term_slope"] = round(float(far["implied_volatility"].mean() - near["implied_volatility"].mean()), 4)

    spread_pct = frame["spread_pct"].dropna()
    if not spread_pct.empty:
        summary["spread_bps"] = round(float(spread_pct.mean() * 10000), 2)

    return summary


def _liquidity_metrics(summary: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    avg_spread = summary.get("spread_bps")
    return {
        "avg_spread_bps": avg_spread,
        "typical_slippage_bps": round(avg_spread / 2.0, 2) if avg_spread is not None else None,
        "lot_size_hint": None,
    }


def compute_context_overlays(
    history: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    benchmark_history: pd.DataFrame | None = None,
    options_chain: pd.DataFrame | None = None,
) -> Dict[str, object]:
    history = _ensure_datetime_index(history)
    session, prev_session = _latest_session(history)

    supply_zones = _detect_zones(session, interval, kind="supply")
    demand_zones = _detect_zones(session, interval, kind="demand")
    liquidity_pools = _detect_liquidity_pools(session, interval)
    fvgs = _detect_fvg(session, interval)
    rel_strength = _relative_strength(history, benchmark_history)

    session_et = session.index.tz_convert(TZ_ET) if not session.empty else None
    first_stamp = session_et[0] if session_et is not None and len(session_et) else None
    prev_close_stamp = prev_session.index[-1] if prev_session is not None and len(prev_session) else None
    session_low_ts = session["low"].idxmin() if not session.empty else None
    session_high_ts = session["high"].idxmax() if not session.empty else None

    avwap = {
        "from_open": _anchored_vwap(history, first_stamp.tz_convert("UTC")) if first_stamp is not None else None,
        "from_prev_close": _anchored_vwap(history, prev_close_stamp) if prev_close_stamp is not None else None,
        "from_session_low": _anchored_vwap(history, session_low_ts) if session_low_ts is not None else None,
        "from_session_high": _anchored_vwap(history, session_high_ts) if session_high_ts is not None else None,
    }

    volume_profile = _volume_profile(session)
    options_summary = _options_summary(options_chain)
    liquidity_metrics = _liquidity_metrics(options_summary)

    overlays: Dict[str, object] = {
        "supply_zones": supply_zones,
        "demand_zones": demand_zones,
        "liquidity_pools": liquidity_pools,
        "fvg": fvgs,
        "rel_strength_vs": rel_strength,
        "internals": {
            "adv_dec": None,
            "tick": None,
            "sector_perf": {},
            "index_bias": None,
        },
        "options_summary": options_summary,
        "liquidity_metrics": liquidity_metrics,
        "events": [],
        "avwap": avwap,
        "volume_profile": volume_profile,
    }

    return overlays
