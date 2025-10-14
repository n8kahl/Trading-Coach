"""Historical pattern statistics helpers."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

_PATTERN_CACHE: Dict[str, Dict[str, float | int | str]] = {}


def pattern_id(payload: Dict[str, object]) -> str:
    """Return a deterministic pattern identifier."""

    key = "|".join(f"{k}={payload.get(k)}" for k in sorted(payload))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{payload.get('symbol', 'UNKNOWN')}|{digest[:16]}"


def fetch_pattern_stats(pattern: str) -> Optional[Dict[str, float | int | str]]:
    stored = _PATTERN_CACHE.get(pattern)
    if stored:
        return dict(stored)
    # deterministic mock stats
    seed = int(hashlib.sha1(pattern.encode("utf-8")).hexdigest(), 16) % 1000
    stats = {
        "pattern_id": pattern,
        "sample_size": 120 + seed % 80,
        "win_rate": round(0.45 + (seed % 30) / 100.0, 2),
        "avg_r_multiple": round(1.2 + (seed % 50) / 100.0, 2),
        "avg_duration": f"{30 + seed % 60}m",
    }
    _PATTERN_CACHE[pattern] = stats
    return dict(stats)


def store_pattern_stats(pattern: str, stats: Dict[str, float | int | str]) -> None:
    _PATTERN_CACHE[pattern] = dict(stats)


__all__ = ["pattern_id", "fetch_pattern_stats", "store_pattern_stats"]
