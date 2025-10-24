from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, List

from ...services.chart_levels import extract_supporting_levels


def _flatten_contexts(values: Sequence[Any]) -> List[Mapping[str, Any]]:
    """Return a flat list of mapping contexts extracted from the provided values."""

    flattened: List[Mapping[str, Any]] = []

    def _collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if value:
                flattened.append(value)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                _collect(item)

    for value in values:
        _collect(value)
    return flattened


def _extract_levels_for_chart(plan: Mapping[str, Any] | None, *extras: Any) -> str | None:
    """
    Produce a semicolon-delimited ``price|label`` token string for chart URLs.

    The helper defers to ``extract_supporting_levels`` for canonical ordering,
    deduplication, and tick precision clamping.  Extra mapping contexts (plan layers,
    structured plans, key level metadata) can be provided via ``extras`` and will be
    scanned for level data the same way the planner does when composing responses.
    """

    base: Mapping[str, Any]
    if isinstance(plan, Mapping):
        base = plan
    else:
        base = {}
    contexts = _flatten_contexts(extras)
    if not contexts:
        return extract_supporting_levels(base)
    return extract_supporting_levels(base, *contexts)


__all__ = ["_extract_levels_for_chart"]
