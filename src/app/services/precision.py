"""Backwards-compatible precision helpers."""

from __future__ import annotations

from .instrument_precision import DEFAULT_PRECISION_MAP, get_precision


def get_price_precision(symbol: str | None, *, precision_map: dict[str, int] | None = None) -> int:
    """Deprecated wrapper that forwards to :func:`get_precision`."""

    return get_precision(symbol, precision_map=precision_map)


__all__ = ["get_price_precision", "DEFAULT_PRECISION_MAP"]
