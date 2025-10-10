"""Market scanning logic for detecting trade setups using live market data.

The scanner evaluates each configured strategy with fully realised
calculations (ATR, VWAP, anchored VWAPs, EMA stacks, etc.) and only produces
signals when the underlying market structure and statistics satisfy each
strategy's rule set.  No placeholder heuristics remain — every score, plan,
and directional hint comes directly from current intraday data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_library import Strategy, load_strategies
from .context_overlays import _volume_profile
from .calculations import atr, ema, vwap, adx

TZ_ET = "America/New_York"
RTH_START_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60


@dataclass(slots=True)
class Plan:
    direction: str
    entry: float
    stop: float
    targets: List[float]
    confidence: float
    risk_reward: float
    notes: str | None = None
    atr: float | None = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": round(float(self.entry), 4),
            "stop": round(float(self.stop), 4),
            "targets": [round(float(t), 4) for t in self.targets],
            "confidence": round(float(self.confidence), 3),
            "risk_reward": round(float(self.risk_reward), 3),
            "atr": round(float(self.atr), 4) if self.atr is not None else None,
            "notes": self.notes,
            "warnings": list(self.warnings),
        }


@dataclass
class Signal:
    """Represents a detected trade opportunity."""

    symbol: str
    strategy_id: str
    description: str
    score: float
    contract: Dict[str, Any] | None = None
    features: Dict[str, Any] = field(default_factory=dict)
    options_rules: Dict[str, Any] | None = None
    plan: Plan | None = None


def rr(entry: float, stop: float, tp: float, bias: str) -> float:
    risk = (entry - stop) if bias == "long" else (stop - entry)
    reward = (tp - entry) if bias == "long" else (entry - tp)
    if risk <= 0:
        return 0.0
    return max(0.0, reward / risk)


def _within_expected_move(entry: float, tp: float, em: Optional[float], prefer_cap: bool) -> bool:
    if not prefer_cap or em is None or not math.isfinite(em):
        return True
    return abs(tp - entry) <= em


def snap_targets_with_rr(
    *,
    entry: float,
    stop: float,
    bias: str,
    tp_raws: List[float],
    htf_levels: List[float],
    min_rr: float,
    em: Optional[float],
    prefer_em_cap: bool,
    snap_window_atr: float = 0.30,
    atr: Optional[float] = None,
) -> List[float]:
    window = (atr or 0.0) * snap_window_atr
    levels_sorted = sorted({float(level) for level in htf_levels if math.isfinite(level)})
    tps_final: List[float] = []

    for tp_raw in tp_raws:
        if not math.isfinite(tp_raw):
            continue
        candidate: Optional[float] = None
        if levels_sorted and atr:
            nearest = min(levels_sorted, key=lambda lvl: abs(lvl - tp_raw))
            if abs(nearest - tp_raw) <= window:
                candidate = nearest

        for tp_try in ([candidate] if candidate is not None else []) + [tp_raw]:
            if tp_try is None or not math.isfinite(tp_try):
                continue
            if not _within_expected_move(entry, tp_try, em, prefer_em_cap):
                continue
            if rr(entry, stop, tp_try, bias) >= min_rr:
                tps_final.append(tp_try)
                break

    tps_final = sorted(set(tps_final), reverse=(bias == "short"))
    return tps_final[:2]


def _normalize_trade_style(style: str | None) -> str:
    token = (style or "").strip().lower()
    if token == "leaps":
        token = "leap"
    if token not in {"scalp", "intraday", "swing", "leap"}:
        token = "intraday"
    return token


def _base_targets_for_style(
    *,
    style: str | None,
    bias: str,
    entry: float,
    stop: float,
    atr: float | None,
    expected_move: float | None,
    min_rr: float,
    prefer_em_cap: bool = True,
) -> List[float]:
    style_key = _normalize_trade_style(style)
    risk = abs(entry - stop)
    if risk <= 0:
        return []
    atr_val = float(atr or 0.0)
    expected_move_val = float(expected_move) if isinstance(expected_move, (int, float)) else None

    rules = {
        "scalp": {
            "tp1": {"atr_mult": 0.7, "em_mult": 0.35},
            "tp2": {"rr_mult": 2.0, "atr_mult": 1.2, "em_mult": 0.8},
        },
        "intraday": {
            "tp1": {"atr_mult": 0.9, "em_mult": 0.55},
            "tp2": {"rr_mult": 2.2, "atr_mult": 1.5, "em_mult": 0.9},
        },
        "swing": {
            "tp1": {"atr_mult": 1.0, "em_mult": 0.6},
            "tp2": {"rr_mult": 2.5, "atr_mult": 2.0, "em_mult": 1.0},
        },
        "leap": {
            "tp1": {"pct": 0.06, "rr_mult": 1.0, "em_mult": 0.6},
            "tp2": {"pct": 0.12, "rr_mult": 2.0, "em_mult": 1.0},
        },
    }

    style_rules = rules.get(style_key, rules["intraday"])
    offsets: List[float] = []

    def _compute_offset(spec: Dict[str, float]) -> float | None:
        base = min_rr * risk
        rr_mult = spec.get("rr_mult")
        if rr_mult is not None and rr_mult > 0:
            base = max(base, float(rr_mult) * risk)
        atr_mult = spec.get("atr_mult")
        if atr_mult is not None and atr_mult > 0 and atr_val > 0:
            base = max(base, float(atr_mult) * atr_val)
        pct_mult = spec.get("pct")
        if pct_mult is not None and pct_mult > 0:
            base = max(base, abs(entry) * float(pct_mult))
        if prefer_em_cap and expected_move_val is not None:
            em_mult = spec.get("em_mult")
            if em_mult is not None and em_mult > 0:
                base = min(base, expected_move_val * float(em_mult))
        if base <= 0 or not math.isfinite(base):
            return None
        return base

    for key in ("tp1", "tp2"):
        spec = style_rules.get(key)
        if not spec:
            continue
        offset = _compute_offset(spec)
        if offset is not None and offset > 0:
            offsets.append(offset)

    unique_offsets: List[float] = []
    for offset in offsets:
        if all(abs(offset - existing) > 1e-6 for existing in unique_offsets):
            unique_offsets.append(offset)

    targets: List[float] = []
    for offset in unique_offsets:
        if bias == "long":
            targets.append(entry + offset)
        else:
            targets.append(entry - offset)
    return targets


def _strategy_min_rr(strategy_id: str) -> float:
    default = 1.2
    mapping = {
        "orb_retest": 1.3,
        "power_hour_trend": 1.4,
        "vwap_avwap": 1.3,
        "gap_fill_open": 1.5,
        "midday_mean_revert": 1.25,
    }
    return mapping.get(strategy_id.lower(), default)


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    return frame


def _latest_sessions(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    if frame.empty:
        return frame, None
    frame = _ensure_datetime_index(frame).sort_index()
    et_index = frame.index.tz_convert(TZ_ET)
    session_dates = pd.Series(et_index.date, index=frame.index)
    if session_dates.empty:
        return frame, None
    latest_date = session_dates.iloc[-1]
    session_mask = session_dates.eq(latest_date).to_numpy()
    session_df = frame.iloc[session_mask]

    prev_df: pd.DataFrame | None = None
    unique_dates = session_dates.drop_duplicates().tolist()
    if len(unique_dates) >= 2:
        prev_date = unique_dates[-2]
        prev_mask = session_dates.eq(prev_date).to_numpy()
        prev_df = frame.iloc[prev_mask]
    return session_df, prev_df


def _minutes_from_midnight(index: pd.DatetimeIndex) -> np.ndarray:
    et_index = index.tz_convert(TZ_ET)
    return et_index.hour * 60 + et_index.minute


def _score_conditions(flags: Iterable[bool], bonus: float = 0.0, clamp: Tuple[float, float] = (0.0, 0.98)) -> float:
    flags = list(flags)
    if not flags:
        return round(max(clamp[0], min(clamp[1], 0.25 + bonus)), 3)
    positive = sum(1 for flag in flags if flag)
    ratio = positive / len(flags)
    confidence = 0.25 + ratio * 0.6 + bonus
    return round(max(clamp[0], min(clamp[1], confidence)), 3)


def _build_plan(
    direction: str,
    entry: float,
    stop: float,
    targets: List[float],
    *,
    atr_value: float | None,
    notes: str | None,
    conditions: Iterable[bool],
) -> Plan | None:
    if not math.isfinite(entry) or not math.isfinite(stop):
        return None
    # Clean and geometry-guard targets
    original_targets = [float(t) for t in targets if math.isfinite(t)]
    clean_targets = list(original_targets)
    geometry_reordered = False
    if direction == "long":
        ordered = sorted(clean_targets)
        if ordered != clean_targets:
            geometry_reordered = True
        clean_targets = ordered
    else:
        ordered = sorted(clean_targets, reverse=True)
        if ordered != clean_targets:
            geometry_reordered = True
        clean_targets = ordered
    if not clean_targets:
        return None
    # Geometry checks
    geometry_ok = True
    if direction == "long":
        if stop >= entry:
            geometry_ok = False
        risk = entry - stop
        reward = clean_targets[0] - entry
    else:
        if stop <= entry:
            geometry_ok = False
        risk = stop - entry
        reward = entry - clean_targets[0]
    if risk <= 0 or reward <= 0:
        return None
    risk_reward = reward / risk
    confidence = _score_conditions(conditions)
    # Append guardrail warning to notes if geometry was reordered or invalid
    warnings: List[str] = []
    if geometry_reordered or not geometry_ok:
        warnings.append("Geometry check: targets reordered or watch plan recommended")
        guard_note = "Geometry check: targets reordered or watch plan recommended"
    else:
        guard_note = None
    final_notes = notes
    if guard_note:
        final_notes = (notes + " | " + guard_note) if notes else guard_note

    return Plan(
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        targets=clean_targets,
        confidence=float(confidence),
        risk_reward=float(round(risk_reward, 3)),
        notes=final_notes,
        atr=float(atr_value) if atr_value is not None and math.isfinite(atr_value) else None,
        warnings=warnings,
    )


def _anchored_vwap(frame: pd.DataFrame, anchor_ts: pd.Timestamp) -> float | None:
    segment = frame.loc[frame.index >= anchor_ts]
    if segment.empty or "volume" not in segment.columns or segment["volume"].sum() <= 0:
        return None
    typical = segment["typical_price"]
    pv = (typical * segment["volume"]).cumsum()
    cum_volume = segment["volume"].cumsum()
    denom = float(cum_volume.iloc[-1])
    if denom <= 0:
        return None
    return float(pv.iloc[-1] / denom)


def _session_phase(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(TZ_ET)
    h, m = ts.hour, ts.minute
    wd = ts.weekday()
    if wd >= 5:
        return "off"
    if (h < 9) or (h == 9 and m < 30):
        return "premarket"
    if h == 9 and 30 <= m < 60:
        return "open_drive"
    if h == 10 or (h == 11 and m < 30):
        return "morning"
    if (h == 11 and m >= 30) or (12 <= h < 14):
        return "midday"
    if h == 14:
        return "afternoon"
    if h == 15:
        return "power_hour"
    if h >= 16:
        return "postmarket"
    return "other"


def _prepare_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_datetime_index(frame).sort_index().copy()
    if frame.empty:
        return frame
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in frame.columns:
            raise ValueError(f"Expected column '{column}' missing from OHLCV data.")
    frame["atr14"] = atr(frame["high"], frame["low"], frame["close"], 14)
    frame["ema9"] = ema(frame["close"], 9)
    frame["ema20"] = ema(frame["close"], 20)
    frame["ema50"] = ema(frame["close"], 50)
    frame["vwap"] = vwap(frame["close"], frame["volume"])
    frame["adx14"] = adx(frame["high"], frame["low"], frame["close"], 14)
    frame["typical_price"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    return frame


def _build_context(frame: pd.DataFrame) -> Dict[str, Any]:
    session_df, prev_session_df = _latest_sessions(frame)
    latest = frame.iloc[-1]
    atr_value = float(latest["atr14"]) if math.isfinite(latest["atr14"]) else math.nan
    volume_median = float(session_df["volume"].tail(40).median()) if not session_df.empty else math.nan
    minutes_vector = _minutes_from_midnight(session_df.index) if not session_df.empty else np.array([], dtype=int)
    # Expected move horizon (approx): median true range × bars over a short horizon
    expected_move_horizon = None
    try:
        df = frame.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        tr_med = float(tr.tail(20).median()) if not tr.empty else float("nan")
        # infer bar interval in minutes
        try:
            idx = df.index
            if len(idx) >= 2:
                delta_min = (idx[-1] - idx[-2]).total_seconds() / 60.0
            else:
                delta_min = 5.0
        except Exception:
            delta_min = 5.0
        horizon_minutes = 30 if delta_min <= 2.0 else 60
        horizon_bars = max(1, int(horizon_minutes / max(delta_min, 1.0)))
        if math.isfinite(tr_med):
            expected_move_horizon = tr_med * horizon_bars
    except Exception:
        expected_move_horizon = None
    # Key levels (session OR + previous session H/L/C)
    key_levels: Dict[str, float] = {}
    try:
        if not session_df.empty:
            # Opening range (first 15 minutes)
            mins = _minutes_from_midnight(session_df.index)
            mask_or = (mins >= RTH_START_MINUTE) & (mins < RTH_START_MINUTE + 15)
            if mask_or.any():
                or_slice = session_df.iloc[mask_or]
                key_levels["opening_range_high"] = float(or_slice["high"].max())
                key_levels["opening_range_low"] = float(or_slice["low"].min())
            key_levels["session_high"] = float(session_df["high"].max())
            key_levels["session_low"] = float(session_df["low"].min())
        if prev_session_df is not None and not prev_session_df.empty:
            key_levels["prev_high"] = float(prev_session_df["high"].max())
            key_levels["prev_low"] = float(prev_session_df["low"].min())
            key_levels["prev_close"] = float(prev_session_df["close"].iloc[-1])
    except Exception:
        key_levels = {}

    # Volume profile (session)
    vol_profile = {}
    try:
        if not session_df.empty:
            vol_profile = _volume_profile(session_df)
    except Exception:
        vol_profile = {}

    # Fib anchors (up/down) from recent swing window (last 50 bars)
    fib_up: Dict[str, float] = {}
    fib_down: Dict[str, float] = {}
    try:
        window = frame.tail(50)
        rng_low = float(window["low"].min())
        rng_high = float(window["high"].max())
        span = max(0.0, rng_high - rng_low)
        if span > 0:
            # Upward projections from high
            fib_up = {
                "FIB1.0": round(rng_high, 4),
                "FIB1.272": round(rng_high + 0.272 * span, 4),
                "FIB1.618": round(rng_high + 0.618 * span, 4),
            }
            # Downward projections from low
            fib_down = {
                "FIB1.0": round(rng_low, 4),
                "FIB1.272": round(rng_low - 0.272 * span, 4),
                "FIB1.618": round(rng_low - 0.618 * span, 4),
            }
    except Exception:
        fib_up, fib_down = {}, {}

    return {
        "frame": frame,
        "session": session_df,
        "prev_session": prev_session_df,
        "latest": latest,
        "atr": atr_value,
        "price": float(latest["close"]),
        "vwap": float(latest["vwap"]),
        "ema9": float(latest["ema9"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "adx": float(latest["adx14"]) if math.isfinite(latest["adx14"]) else math.nan,
        "volume_median": volume_median,
        "minutes_vector": minutes_vector,
        "timestamp": frame.index[-1],
        "session_phase": _session_phase(frame.index[-1]),
        "htf_levels": _collect_htf_levels(session_df, prev_session_df, latest),
        "expected_move_horizon": expected_move_horizon,
        "key": key_levels,
        "vol_profile": vol_profile,
        "fib_up": fib_up,
        "fib_down": fib_down,
    }


# Trade-style presets for TP1 selection
_TP1_RULES: Dict[str, Dict[str, Any]] = {
    "scalp":    {"minATR": 0.25, "maxHorizon": 0.5, "ratioTP2": (0.25, 0.45), "weights": {"vwap": 0.4,  "orb": 0.3,  "priorHL": 0.2, "volProfile": 0.2, "fib": 0.1, "microPivot": 0.3}},
    "intraday": {"minATR": 0.35, "maxHorizon": 0.8, "ratioTP2": (0.35, 0.60), "weights": {"vwap": 0.35, "orb": 0.35, "priorHL": 0.25, "volProfile": 0.35, "fib": 0.2, "microPivot": 0.15}},
    "swing":    {"minATR": 0.50, "maxHorizon": 1.0, "ratioTP2": (0.40, 0.65), "weights": {"vwap": 0.15, "orb": 0.15, "priorHL": 0.35, "volProfile": 0.45, "fib": 0.35, "microPivot": 0.05}},
    "leaps":    {"minATR": 0.60, "maxHorizon": 1.0, "ratioTP2": (0.40, 0.70), "weights": {"vwap": 0.05, "orb": 0.05, "priorHL": 0.35, "volProfile": 0.45, "fib": 0.45, "microPivot": 0.00}},
}


def _style_for_strategy_id(strategy_id: str) -> str:
    sid = (strategy_id or "").lower()
    if "power" in sid or "orb" in sid or "gap" in sid:
        return "intraday"
    if "midday" in sid:
        return "intraday"
    return "intraday"


def _build_tp_candidates_long(entry: float, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    c: List[Dict[str, Any]] = []
    key = ctx.get("key") or {}
    # Session structure
    for name, tag in (("prev_high", "PRIOR_HIGH"), ("opening_range_high", "ORB_HIGH")):
        val = key.get(name)
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": tag})
    # Volume Profile
    vp = ctx.get("vol_profile") or {}
    for lab, key in [("POC", "poc"), ("VAH", "vah"), ("VAL", "val")]:
        val = vp.get(key)
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": lab})
    # VWAP / EMAs
    v = ctx.get("vwap")
    if isinstance(v, (int, float)) and v > entry:
        c.append({"level": float(v), "tag": "VWAP"})
    for k in ("ema9", "ema20", "ema50"):
        val = ctx.get(k)
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": k.upper()})
    # Fib projections up
    fib_up = ctx.get("fib_up") or {}
    for tag, val in fib_up.items():
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": str(tag).upper()})
    # Dedupe by level within 1 cent
    seen: set[float] = set()
    uniq: List[Dict[str, Any]] = []
    for item in sorted(c, key=lambda x: x["level"]):
        keyf = round(item["level"], 2)
        if keyf in seen:
            continue
        seen.add(keyf)
        uniq.append(item)
    return uniq


def _build_tp_candidates_short(entry: float, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    c: List[Dict[str, Any]] = []
    key = ctx.get("key") or {}
    # Session structure below entry
    for name, tag in (("prev_low", "PRIOR_LOW"), ("opening_range_low", "ORB_LOW")):
        val = key.get(name)
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": tag})
    # Volume profile
    vp = ctx.get("vol_profile") or {}
    for lab, k in [("POC", "poc"), ("VAL", "val"), ("VAH", "vah")]:
        val = vp.get(k)
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": lab})
    # VWAP/EMAs
    v = ctx.get("vwap")
    if isinstance(v, (int, float)) and v < entry:
        c.append({"level": float(v), "tag": "VWAP"})
    for k in ("ema9", "ema20", "ema50"):
        val = ctx.get(k)
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": k.upper()})
    # Fib projections down
    fib_down = ctx.get("fib_down") or {}
    for tag, val in fib_down.items():
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": str(tag).upper()})
    # Dedupe
    seen: set[float] = set()
    uniq: List[Dict[str, Any]] = []
    for item in sorted(c, key=lambda x: x["level"], reverse=True):
        keyf = round(item["level"], 2)
        if keyf in seen:
            continue
        seen.add(keyf)
        uniq.append(item)
    return uniq


def _score_tp_candidate_long(level: Dict[str, Any], style: str, entry: float, stop: float, tp2: float, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    rules = _TP1_RULES.get(style, _TP1_RULES["intraday"])
    w = rules["weights"]
    s = 0.0
    tag = (level.get("tag") or "").upper()
    if tag == "VWAP":
        s += w.get("vwap", 0)
    if tag in {"ORB_HIGH"}:
        s += w.get("orb", 0)
    if tag in {"PRIOR_HIGH"}:
        s += w.get("priorHL", 0)
    if tag in {"EMA9", "EMA20", "EMA50"}:
        s += w.get("microPivot", 0)
    if tag in {"POC", "VAH", "VAL"}:
        s += w.get("volProfile", 0)
    if tag.startswith("FIB"):
        s += w.get("fib", 0)
    # MTF alignment proxy: EMA stack bullish → +0.1
    try:
        latest_bull = ctx.get("ema9") > ctx.get("ema20") > ctx.get("ema50")
        if latest_bull:
            s += 0.1
    except Exception:
        pass
    # Distance checks
    atr = ctx.get("atr") or 0.0
    min_atr = rules["minATR"] * float(atr or 0.0)
    dist = float(level.get("level") or 0.0) - entry
    if dist < max(0.0, min_atr):
        return None
    # EM cap
    em = ctx.get("expected_move_horizon")
    if isinstance(em, (int, float)) and dist > float(em) * float(rules.get("maxHorizon", 1.0)):
        return None
    # Ratio vs TP2
    span_tp2 = tp2 - entry
    if span_tp2 <= 0:
        return None
    ratio = dist / span_tp2
    rmin, rmax = rules["ratioTP2"]
    if not (rmin <= ratio <= rmax):
        return None
    # R:R
    risk = entry - stop
    if risk <= 0:
        return None
    rr_to_tp1 = dist / risk
    if rr_to_tp1 < float(min_rr):
        return None
    return round(float(s), 4)


def _score_tp_candidate_short(level: Dict[str, Any], style: str, entry: float, stop: float, tp2: float, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    rules = _TP1_RULES.get(style, _TP1_RULES["intraday"])
    w = rules["weights"]
    s = 0.0
    tag = (level.get("tag") or "").upper()
    if tag == "VWAP":
        s += w.get("vwap", 0)
    if tag in {"ORB_LOW"}:
        s += w.get("orb", 0)
    if tag in {"PRIOR_LOW"}:
        s += w.get("priorHL", 0)
    if tag in {"EMA9", "EMA20", "EMA50"}:
        s += w.get("microPivot", 0)
    if tag in {"POC", "VAH", "VAL"}:
        s += w.get("volProfile", 0)
    if tag.startswith("FIB"):
        s += w.get("fib", 0)
    # EMA stack bearish bonus
    try:
        latest_bear = ctx.get("ema9") < ctx.get("ema20") < ctx.get("ema50")
        if latest_bear:
            s += 0.1
    except Exception:
        pass
    # Distance checks
    atr = ctx.get("atr") or 0.0
    min_atr = rules["minATR"] * float(atr or 0.0)
    dist = entry - float(level.get("level") or 0.0)
    if dist < max(0.0, min_atr):
        return None
    em = ctx.get("expected_move_horizon")
    if isinstance(em, (int, float)) and dist > float(em) * float(rules.get("maxHorizon", 1.0)):
        return None
    span_tp2 = entry - tp2
    if span_tp2 <= 0:
        return None
    ratio = dist / span_tp2
    rmin, rmax = rules["ratioTP2"]
    if not (rmin <= ratio <= rmax):
        return None
    risk = stop - entry
    if risk <= 0:
        return None
    rr_to_tp1 = dist / risk
    if rr_to_tp1 < float(min_rr):
        return None
    return round(float(s), 4)


def _select_tp1_long(entry: float, stop: float, tp2: float, style: str, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    cands = _build_tp_candidates_long(entry, ctx)
    scored: List[Tuple[float, float]] = []  # (level, score)
    for L in cands:
        sc = _score_tp_candidate_long(L, style, entry, stop, tp2, ctx, min_rr)
        if sc is not None:
            scored.append((float(L["level"]), sc))
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return float(scored[0][0])


def _select_tp1_short(entry: float, stop: float, tp2: float, style: str, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    cands = _build_tp_candidates_short(entry, ctx)
    scored: List[Tuple[float, float]] = []
    for L in cands:
        sc = _score_tp_candidate_short(L, style, entry, stop, tp2, ctx, min_rr)
        if sc is not None:
            scored.append((float(L["level"]), sc))
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return float(scored[0][0])


def _collect_htf_levels(session: pd.DataFrame, prev_session: pd.DataFrame | None, latest: pd.Series) -> List[float]:
    levels: List[float] = []
    try:
        if session is not None and not session.empty:
            levels.extend(
                [
                    float(session["high"].max()),
                    float(session["low"].min()),
                    float(session["close"].iloc[-1]),
                ]
            )
            head_slice = session.head(3)
            if not head_slice.empty:
                levels.extend(
                    [
                        float(head_slice["high"].max()),
                        float(head_slice["low"].min()),
                    ]
                )
        if prev_session is not None and not prev_session.empty:
            levels.extend(
                [
                    float(prev_session["high"].max()),
                    float(prev_session["low"].min()),
                    float(prev_session["close"].iloc[-1]),
                ]
            )
        vwap_val = latest.get("vwap")
        if math.isfinite(vwap_val):
            levels.append(float(vwap_val))
        ema50_val = latest.get("ema50")
        if math.isfinite(ema50_val):
            levels.append(float(ema50_val))
    except Exception:
        pass
    clean = [lvl for lvl in levels if math.isfinite(lvl)]
    return sorted(set(clean))


def _detect_orb_retest(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    session = ctx["session"]
    if session.empty:
        return None
    minutes = ctx["minutes_vector"]
    if minutes.size == 0 or minutes.min() > RTH_START_MINUTE:
        return None

    window_minutes = 15
    range_mask = (minutes >= RTH_START_MINUTE) & (minutes < RTH_START_MINUTE + window_minutes)
    if not range_mask.any():
        return None
    opening_range = session.iloc[range_mask]
    post_range = session.iloc[~range_mask]
    if opening_range.empty or post_range.empty:
        return None

    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    or_high = float(opening_range["high"].max())
    or_low = float(opening_range["low"].min())
    latest = ctx["latest"]
    price = float(latest["close"])
    tolerance = max(atr_value * 0.25, price * 0.0015)
    ema_stack_long = latest["ema9"] > latest["ema20"] > latest["ema50"]
    ema_stack_short = latest["ema9"] < latest["ema20"] < latest["ema50"]
    adx_strong = ctx["adx"] >= 18 if math.isfinite(ctx["adx"]) else False
    volume_ok = math.isfinite(ctx["volume_median"]) and latest["volume"] >= ctx["volume_median"]

    notes: List[str] = []
    plan: Plan | None = None
    tp1_dbg: Dict[str, Any] | None = None

    recent_slice = post_range.tail(20)
    retest_low = float(recent_slice["low"].min()) if not recent_slice.empty else float("nan")
    retest_high = float(recent_slice["high"].max()) if not recent_slice.empty else float("nan")

    if price > or_high and math.isfinite(retest_low) and abs(retest_low - or_high) <= tolerance:
        entry = max(price, or_high)
        stop = retest_low - tolerance * 0.5
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        style = _style_for_strategy_id(strategy.id)
        base_targets = _base_targets_for_style(
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            base_targets = [entry + atr_value, entry + atr_value * 1.5]
        tp2 = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_long(entry, stop, tp2, style, ctx, min_rr)
        if tp1_struct is not None:
            targets = [tp1_struct, tp2]
            tp1_dbg = {"picked": tp1_struct, "tp2": tp2, "method": "structural", "style": style}
        else:
            targets = snap_targets_with_rr(
                entry=entry,
                stop=stop,
                bias="long",
                tp_raws=base_targets,
                htf_levels=ctx.get("htf_levels", []),
                min_rr=min_rr,
                em=expected_move,
                prefer_em_cap=True,
                atr=atr_value,
            )
        if not targets:
            return None
        plan = _build_plan(
            "long",
            entry,
            stop,
            targets,
            atr_value=atr_value,
            notes=f"Reclaimed OR high {or_high:.2f}; retest low {retest_low:.2f}",
            conditions=[ema_stack_long, adx_strong, volume_ok],
        )
        if plan:
            notes.append("Long OR retest validated")

    elif price < or_low and math.isfinite(retest_high) and abs(retest_high - or_low) <= tolerance:
        entry = min(price, or_low)
        stop = retest_high + tolerance * 0.5
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            base_targets = [entry - atr_value, entry - atr_value * 1.5]
        tp2 = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_short(entry, stop, tp2, style, ctx, min_rr)
        if tp1_struct is not None:
            targets = [tp1_struct, tp2]
            tp1_dbg = {"picked": tp1_struct, "tp2": tp2, "method": "structural", "style": style}
        else:
            targets = snap_targets_with_rr(
                entry=entry,
                stop=stop,
                bias="short",
                tp_raws=base_targets,
                htf_levels=ctx.get("htf_levels", []),
                min_rr=min_rr,
                em=expected_move,
                prefer_em_cap=True,
                atr=atr_value,
            )
        if not targets:
            return None
        plan = _build_plan(
            "short",
            entry,
            stop,
            targets,
            atr_value=atr_value,
            notes=f"Rejected OR low {or_low:.2f}; retest high {retest_high:.2f}",
            conditions=[ema_stack_short, adx_strong, volume_ok],
        )
        if plan:
            notes.append("Short OR retest validated")

    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "opening_range_high": or_high,
        "opening_range_low": or_low,
        "retest_extreme": retest_low if plan.direction == "long" else retest_high,
        "vwap": ctx["vwap"],
        "ema9": ctx["ema9"],
        "ema20": ctx["ema20"],
        "ema50": ctx["ema50"],
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    if tp1_dbg:
        features["tp1_struct_debug"] = tp1_dbg

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_power_hour_trend(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    if ctx["session_phase"] != "power_hour":
        return None
    session = ctx["session"]
    if session.empty:
        return None
    latest = ctx["latest"]
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    window = session.tail(30)
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    price = float(latest["close"])
    adx_strong = ctx["adx"] >= 20 if math.isfinite(ctx["adx"]) else False
    ema_stack_long = latest["ema9"] > latest["ema20"] > latest["ema50"]
    ema_stack_short = latest["ema9"] < latest["ema20"] < latest["ema50"]

    plan: Plan | None = None
    breakout_long = price >= range_high - 0.05 * atr_value
    breakout_short = price <= range_low + 0.05 * atr_value
    volume_ok = math.isfinite(ctx["volume_median"]) and latest["volume"] >= ctx["volume_median"]

    tp1_dbg_ph: Dict[str, Any] | None = None
    if price > ctx["vwap"] and ema_stack_long and breakout_long:
        entry = price
        stop = min(range_low, float(session["low"].tail(10).min())) - atr_value * 0.25
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            base_targets = [entry + atr_value, entry + atr_value * 1.6]
        tp2 = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_long(entry, stop, tp2, style, ctx, min_rr)
        if tp1_struct is not None:
            targets = [tp1_struct, tp2]
            tp1_dbg_ph = {"picked": tp1_struct, "tp2": tp2, "method": "structural", "style": style}
        else:
            targets = snap_targets_with_rr(
                entry=entry,
                stop=stop,
                bias="long",
                tp_raws=base_targets,
                htf_levels=ctx.get("htf_levels", []),
                min_rr=min_rr,
                em=expected_move,
                prefer_em_cap=True,
                atr=atr_value,
            )
        if not targets:
            return None
        plan = _build_plan(
            "long",
            entry,
            stop,
            targets,
            atr_value=atr_value,
            notes=f"VWAP support {ctx['vwap']:.2f}; afternoon range high {range_high:.2f}",
            conditions=[adx_strong, volume_ok, breakout_long],
        )
    elif price < ctx["vwap"] and ema_stack_short and breakout_short:
        entry = price
        stop = max(range_high, float(session["high"].tail(10).max())) + atr_value * 0.25
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            base_targets = [entry - atr_value, entry - atr_value * 1.6]
        tp2 = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_short(entry, stop, tp2, style, ctx, min_rr)
        if tp1_struct is not None:
            targets = [tp1_struct, tp2]
            tp1_dbg_ph = {"picked": tp1_struct, "tp2": tp2, "method": "structural", "style": style}
        else:
            targets = snap_targets_with_rr(
                entry=entry,
                stop=stop,
                bias="short",
                tp_raws=base_targets,
                htf_levels=ctx.get("htf_levels", []),
                min_rr=min_rr,
                em=expected_move,
                prefer_em_cap=True,
                atr=atr_value,
            )
        if not targets:
            return None
        plan = _build_plan(
            "short",
            entry,
            stop,
            targets,
            atr_value=atr_value,
            notes=f"VWAP resistance {ctx['vwap']:.2f}; afternoon range low {range_low:.2f}",
            conditions=[adx_strong, volume_ok, breakout_short],
        )

    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "range_high": range_high,
        "range_low": range_low,
        "vwap": ctx["vwap"],
        "ema9": ctx["ema9"],
        "ema20": ctx["ema20"],
        "ema50": ctx["ema50"],
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    if tp1_dbg_ph:
        features["tp1_struct_debug"] = tp1_dbg_ph

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_vwap_cluster(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    session = ctx["session"]
    prev_session = ctx["prev_session"]
    if session.empty or prev_session is None or prev_session.empty:
        return None
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    frame = ctx["frame"]
    prev_high_idx = prev_session["high"].idxmax()
    prev_low_idx = prev_session["low"].idxmin()
    open_idx = session.index[0]

    anchors = {
        "prev_high": _anchored_vwap(frame, prev_high_idx),
        "prev_low": _anchored_vwap(frame, prev_low_idx),
        "session_open": _anchored_vwap(frame, open_idx),
    }
    anchored_values = [val for val in anchors.values() if val is not None]
    if len(anchored_values) < 2:
        return None

    price = ctx["price"]
    cluster_mean = float(np.mean(anchored_values))
    cluster_spread = float(np.max(anchored_values) - np.min(anchored_values))
    tolerance = max(atr_value * 0.2, price * 0.001)
    cluster_tight = cluster_spread <= tolerance

    ema_stack_long = ctx["ema9"] > ctx["ema20"] > ctx["ema50"]
    ema_stack_short = ctx["ema9"] < ctx["ema20"] < ctx["ema50"]
    adx_ok = ctx["adx"] >= 16 if math.isfinite(ctx["adx"]) else False

    plan: Plan | None = None
    if price > ctx["vwap"] and ema_stack_long and cluster_tight and price > cluster_mean:
        entry = price
        stop = min(cluster_mean, np.min(anchored_values)) - tolerance
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            base_targets = [entry + atr_value * 1.1, entry + atr_value * 1.8]
        targets = snap_targets_with_rr(
            entry=entry,
            stop=stop,
            bias="long",
            tp_raws=base_targets,
            htf_levels=ctx.get("htf_levels", []),
            min_rr=min_rr,
            em=expected_move,
            prefer_em_cap=True,
            atr=atr_value,
        )
        if not targets:
            return None
        plan = _build_plan(
            "long",
            entry,
            stop,
            targets,
            atr_value=atr_value,
            notes=f"Above VWAP cluster (~{cluster_mean:.2f}); spread {cluster_spread:.2f}",
            conditions=[cluster_tight, adx_ok],
        )
    elif price < ctx["vwap"] and ema_stack_short and cluster_tight and price < cluster_mean:
        entry = price
        stop = max(cluster_mean, np.max(anchored_values)) + tolerance
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            base_targets = [entry - atr_value * 1.1, entry - atr_value * 1.8]
        targets = snap_targets_with_rr(
            entry=entry,
            stop=stop,
            bias="short",
            tp_raws=base_targets,
            htf_levels=ctx.get("htf_levels", []),
            min_rr=min_rr,
            em=expected_move,
            prefer_em_cap=True,
            atr=atr_value,
        )
        if not targets:
            return None
        plan = _build_plan(
            "short",
            entry,
            stop,
            targets,
            atr_value=atr_value,
            notes=f"Below VWAP cluster (~{cluster_mean:.2f}); spread {cluster_spread:.2f}",
            conditions=[cluster_tight, adx_ok],
        )

    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "session_vwap": ctx["vwap"],
        "anchored_vwap_prev_high": anchors["prev_high"],
        "anchored_vwap_prev_low": anchors["prev_low"],
        "anchored_vwap_session_open": anchors["session_open"],
        "cluster_span": cluster_spread,
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_gap_fill(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    session = ctx["session"]
    prev_session = ctx["prev_session"]
    if session.empty or prev_session is None or prev_session.empty:
        return None
    minutes = ctx["minutes_vector"]
    if minutes.size == 0 or minutes.min() > RTH_START_MINUTE:
        return None

    phase = ctx["session_phase"]
    if phase not in {"open_drive", "morning"}:
        return None

    latest = ctx["latest"]
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    prev_close = float(prev_session["close"].iloc[-1])
    first_open = float(session["open"].iloc[0])
    gap = first_open - prev_close
    gap_abs = abs(gap)
    min_gap = max(0.3 * atr_value, 0.003 * prev_close)
    if gap_abs < min_gap:
        return None

    price = float(latest["close"])
    filling = (gap > 0 and price < first_open) or (gap < 0 and price > first_open)
    if not filling:
        return None

    vwap_alignment = (gap > 0 and price < ctx["vwap"]) or (gap < 0 and price > ctx["vwap"])
    distance_to_close = abs(price - prev_close)
    progress = abs(price - first_open) / gap_abs if gap_abs else 0
    volume_ok = math.isfinite(ctx["volume_median"]) and latest["volume"] >= ctx["volume_median"]

    if gap > 0:
        direction = "short"
        entry = price
        stop = max(first_open + atr_value * 0.25, float(session["high"].head(3).max()))
        target_primary = prev_close
        target_secondary = prev_close - atr_value * 0.6
    else:
        direction = "long"
        entry = price
        stop = min(first_open - atr_value * 0.25, float(session["low"].head(3).min()))
        target_primary = prev_close
        target_secondary = prev_close + atr_value * 0.6

    plan = _build_plan(
        direction,
        entry,
        stop,
        [target_primary, target_secondary],
        atr_value=atr_value,
        notes=f"Gap {gap:+.2f} vs prev close {prev_close:.2f}; progress {progress:.2%}",
        conditions=[vwap_alignment, volume_ok, distance_to_close > atr_value * 0.2],
    )
    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "gap_points": gap,
        "prev_close": prev_close,
        "session_open": first_open,
        "vwap": ctx["vwap"],
        "gap_fill_progress": progress,
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_midday_mean_revert(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    if ctx["session_phase"] != "midday":
        return None
    session = ctx["session"]
    if session.empty:
        return None
    latest = ctx["latest"]
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    price = float(latest["close"])
    distance = price - ctx["vwap"]
    extension = abs(distance)
    threshold = atr_value * 0.6
    if extension < threshold:
        return None

    adx_weak = ctx["adx"] < 15 if math.isfinite(ctx["adx"]) else True
    contraction = session.tail(6)
    range_contraction = (contraction["high"].max() - contraction["low"].min()) < atr_value * 0.9
    volume_light = math.isfinite(ctx["volume_median"]) and latest["volume"] <= ctx["volume_median"]

    if distance < 0:  # price below VWAP -> look for mean reversion long
        direction = "long"
        entry = price
        stop = min(float(contraction["low"].min()), price - atr_value * 0.35)
        target_primary = ctx["vwap"]
        target_secondary = price + atr_value * 0.7
    else:
        direction = "short"
        entry = price
        stop = max(float(contraction["high"].max()), price + atr_value * 0.35)
        target_primary = ctx["vwap"]
        target_secondary = price - atr_value * 0.7

    plan = _build_plan(
        direction,
        entry,
        stop,
        [target_primary, target_secondary],
        atr_value=atr_value,
        notes=f"VWAP {ctx['vwap']:.2f}; extension {extension:.2f} ({extension/atr_value:.2f}× ATR)",
        conditions=[adx_weak, range_contraction, volume_light],
    )
    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "vwap": ctx["vwap"],
        "extension_points": extension,
        "extension_atr_multiple": extension / atr_value,
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


STRATEGY_DETECTORS: Dict[str, Callable[[str, Strategy, Dict[str, Any]], Optional[Signal]]] = {
    "orb_retest": _detect_orb_retest,
    "power_hour_trend": _detect_power_hour_trend,
    "vwap_avwap": _detect_vwap_cluster,
    "gap_fill_open": _detect_gap_fill,
    "midday_mean_revert": _detect_midday_mean_revert,
}


async def scan_market(tickers: List[str], market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
    """Scan the provided tickers for strategy setups using real indicator data."""

    strategies = load_strategies()
    signals: List[Signal] = []

    for symbol in tickers:
        raw_frame = market_data.get(symbol)
        if raw_frame is None or raw_frame.empty or len(raw_frame) < 30:
            continue

        try:
            frame = _prepare_symbol_frame(raw_frame)
        except ValueError:
            continue

        ctx = _build_context(frame)

        for strategy in strategies:
            detector = STRATEGY_DETECTORS.get(strategy.id)
            if detector is None:
                continue
            signal = detector(symbol, strategy, ctx)
            if signal is None:
                continue
            signals.append(signal)

    signals.sort(key=lambda sig: sig.score, reverse=True)
    return signals
