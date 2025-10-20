"""Expected move helpers shared across planning flows."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

try:  # noqa: SIM105 - optional dependency
    from src.services.iv_model import daily_em_pct  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - defensive fallback

    def daily_em_pct(symbol: str, as_of: datetime) -> Optional[float]:  # type: ignore[no-redef]
        return None


try:  # noqa: SIM105 - optional dependency
    from src.services.market_data import latest_mid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - defensive fallback

    def latest_mid(symbol: str, as_of: datetime) -> Optional[float]:  # type: ignore[no-redef]
        return None


def session_expected_move(symbol: str, as_of: datetime, style: str) -> float:
    """
    Return session/day expected move in POINTS for the symbol at 'as_of'.

    Falls back to zero when upstream IV hooks are unavailable. The value is always rounded
    to four decimals for deterministic downstream comparisons.
    """

    pct = daily_em_pct(symbol, as_of)
    price = latest_mid(symbol, as_of)
    if pct is None or price is None:
        return 0.0
    try:
        em_points = float(price) * float(pct)
    except (TypeError, ValueError):
        return 0.0
    if not em_points or em_points < 0:
        return 0.0
    return round(em_points, 4)


__all__ = ["session_expected_move"]
