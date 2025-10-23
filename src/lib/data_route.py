from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class DataRoute:
    """Immutable description of how upstream market data should be sourced."""

    mode: Literal["live", "lkg"]
    as_of: datetime
    planning_context: Literal["live", "frozen"]


__all__ = ["DataRoute"]
