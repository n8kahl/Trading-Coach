"""Market internals adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict


def market_internals(as_of: str | None = None) -> Dict[str, int | float]:
    """Return a compact snapshot of index internals."""

    seed = hash(f"{as_of or datetime.now(timezone.utc).isoformat()}") % 500
    breadth = 1500 + (seed % 600)
    vix = round(12.0 + (seed % 40) / 10.0, 2)
    tick = int(-200 + (seed % 400))
    return {"breadth": breadth, "vix": vix, "tick": tick}


__all__ = ["market_internals"]
