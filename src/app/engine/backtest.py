"""Offline helpers for pattern statistics (stub implementation)."""

from __future__ import annotations

from typing import Dict


def simulate_pattern(pattern: str) -> Dict[str, float | int | str]:
    """Return placeholder statistics for the supplied pattern."""

    seed = hash(pattern) % 1000
    return {
        "pattern_id": pattern,
        "sample_size": 150 + seed % 120,
        "win_rate": round(0.5 + (seed % 25) / 100.0, 2),
        "avg_r_multiple": round(1.4 + (seed % 35) / 100.0, 2),
        "avg_duration": f"{35 + seed % 40}m",
    }


__all__ = ["simulate_pattern"]
