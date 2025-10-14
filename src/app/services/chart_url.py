"""Utilities for generating canonical Fancy Trader chart URLs."""

from __future__ import annotations

import urllib.parse as _u
from typing import Dict, List, Optional, Sequence


def _as_entry_dict(entry: Dict[str, object] | object | None) -> Dict[str, object]:
    """Normalise entry inputs to a flat dict with type/level keys."""
    if entry is None:
        return {}
    if isinstance(entry, dict):
        return entry
    entry_dict = {}
    entry_type = getattr(entry, "type", None)
    entry_level = getattr(entry, "level", None)
    if entry_type is not None:
        entry_dict["type"] = entry_type
    if entry_level is not None:
        entry_dict["level"] = entry_level
    return entry_dict


def _targets_slice(targets: Sequence[float] | None) -> List[Optional[float]]:
    payload: List[Optional[float]] = [None, None, None]
    if not targets:
        return payload
    for idx in range(min(3, len(targets))):
        try:
            payload[idx] = float(targets[idx])  # type: ignore[index]
        except (TypeError, ValueError):
            payload[idx] = None
    return payload


def build_chart_url(
    base: str,
    *,
    symbol: str,
    plan_id: str,
    as_of: str,
    entry: Dict[str, object] | object,
    stop: float,
    targets: Sequence[float] | None = None,
    plan_version: str | None = None,
) -> str:
    """Fancy Trader canonical chart URL with overlays & freeze-as-of."""

    base_token = base.split("?", 1)[0].rstrip("?")
    entry_dict = _as_entry_dict(entry)
    entry_level = entry_dict.get("level")
    entry_type = entry_dict.get("type")

    t1, t2, t3 = _targets_slice(targets)

    query = {
        "symbol": symbol,
        "plan_id": plan_id,
        "as_of": as_of,
        "entry_type": entry_type,
        "entry": entry_level,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "freeze": "1",  # instruct renderer to freeze tape at as_of
        "theme": "dark",  # optional UX flag
    }
    if plan_version is not None:
        query["plan_version"] = plan_version
    compact = {key: value for key, value in query.items() if value is not None}
    return f"{base_token}?{_u.urlencode(compact)}"


__all__ = ["build_chart_url"]
