"""Multi-timeframe bias aggregation helpers."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from ..features.mtf import MTFBundle, TFState

# Default weights requested by product spec
MTF_WEIGHTS: Dict[str, float] = {"d": 0.50, "h60": 0.30, "m15": 0.15, "m5": 0.05}


def _score_ema(state: Optional[TFState]) -> float:
    if state is None:
        return 0.0
    if state.ema_up:
        return 1.0
    if state.ema_down:
        return -1.0
    return 0.0


def _score_vwap(state: Optional[TFState]) -> float:
    if state is None:
        return 0.0
    rel = getattr(state, "vwap_rel", "unknown") or "unknown"
    if rel == "above":
        return 0.6
    if rel == "below":
        return -0.6
    if rel == "near":
        return 0.15
    return 0.0


def _score_swing(price: Optional[float], swing_high: Optional[float], swing_low: Optional[float]) -> float:
    if price is None or swing_high is None or swing_low is None:
        return 0.0
    if price > swing_high:
        return 0.5
    if price < swing_low:
        return -0.5
    midpoint = (swing_high + swing_low) / 2.0
    return 0.25 if price >= midpoint else -0.25


def mtf_bias(ctx: Mapping[str, object]) -> Dict[str, object]:
    """Return aggregate bias + per timeframe components.

    Args:
        ctx: Mapping supporting keys `bundle`, `weights`, `price`, `swing_high`, `swing_low`.
    """

    bundle = ctx.get("bundle") if isinstance(ctx.get("bundle"), MTFBundle) else None
    weights = dict(MTF_WEIGHTS)
    overrides = ctx.get("weights")
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            if key in weights:
                try:
                    weights[key] = float(value)
                except (TypeError, ValueError):
                    continue

    try:
        price = float(ctx.get("price")) if ctx.get("price") is not None else None
    except (TypeError, ValueError):
        price = None
    swing_high = ctx.get("swing_high")
    swing_low = ctx.get("swing_low")
    try:
        swing_high_val = float(swing_high) if swing_high is not None else None
    except (TypeError, ValueError):
        swing_high_val = None
    try:
        swing_low_val = float(swing_low) if swing_low is not None else None
    except (TypeError, ValueError):
        swing_low_val = None

    score_acc = 0.0
    weight_sum = 0.0
    components: Dict[str, Dict[str, float]] = {}
    tf_map = {
        "d": ("D", weights.get("d", 0.50)),
        "h60": ("60m", weights.get("h60", 0.30)),
        "m15": ("15m", weights.get("m15", 0.15)),
        "m5": ("5m", weights.get("m5", 0.05)),
    }

    for token, (tf_name, weight) in tf_map.items():
        state = bundle.by_tf.get(tf_name) if bundle else None
        ema_score = _score_ema(state)
        vwap_score = _score_vwap(state)
        swing_score = _score_swing(price, swing_high_val, swing_low_val)
        composite = 0.6 * ema_score + 0.3 * vwap_score + 0.1 * swing_score
        composite = max(-1.0, min(1.0, composite))
        components[token] = {
            "tf": tf_name,
            "score": round(composite, 3),
            "ema": round(ema_score, 3),
            "vwap": round(vwap_score, 3),
            "swing": round(swing_score, 3),
            "weight": round(weight, 3),
        }
        score_acc += composite * weight
        weight_sum += weight

    aggregate = score_acc / weight_sum if weight_sum else 0.0
    if aggregate >= 0.2:
        direction = "long"
    elif aggregate <= -0.2:
        direction = "short"
    else:
        direction = "mixed"

    return {
        "dir": direction,
        "score": round(aggregate, 3),
        "components": components,
    }


__all__ = ["MTF_WEIGHTS", "mtf_bias"]
