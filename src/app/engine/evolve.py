"""Plan evolution helpers for real-time updates."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .risk import risk_model_payload


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _tp_hits(price: float, direction: str, targets: Iterable[float]) -> List[int]:
    hits: List[int] = []
    for idx, target in enumerate(targets, start=1):
        if direction == "long" and price >= target:
            hits.append(idx)
        elif direction == "short" and price <= target:
            hits.append(idx)
    return hits


def evolve_plan(setup: Dict[str, Any], tick: Dict[str, Any], as_of: str | None = None) -> Optional[Dict[str, Any]]:
    """Return an evolved plan payload or None if no changes are required."""

    price = _coerce_float(tick.get("p") or tick.get("price"))
    if price is None:
        return None

    direction = str(setup.get("direction") or "long").lower()
    targets = [float(t) for t in setup.get("targets") or [] if _coerce_float(t) is not None]
    hits = _tp_hits(price, direction, targets)
    if not hits:
        return None

    probabilities = setup.get("probabilities") or {}
    risk_payload = risk_model_payload(
        entry=float(setup["entry"]["level"]),
        stop=float(setup.get("stop")),
        targets=targets,
        probabilities=probabilities,
        direction=direction,
        atr_used=_coerce_float(setup.get("atr_used")),
        style=str(setup.get("style") or "intraday"),
    )
    evolved = dict(setup)
    evolved["risk_model"] = risk_payload
    evolved.setdefault("metadata", {})["last_evolved_at"] = (as_of or datetime.utcnow().isoformat())
    evolved["last_price"] = price
    evolved["event"] = {"type": "hit", "levels": hits}
    return evolved


__all__ = ["evolve_plan"]
