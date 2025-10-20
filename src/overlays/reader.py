"""Helpers for loading persisted chart overlays."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping


def _coerce_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _sanitize_layer_block(layers: Mapping[str, Any] | None, *, plan_id: str | None) -> Dict[str, Any]:
    if not isinstance(layers, Mapping):
        return {
            "plan_id": plan_id,
            "levels": [],
            "zones": [],
            "annotations": [],
            "meta": {},
        }
    payload: MutableMapping[str, Any] = deepcopy(dict(layers))
    payload.setdefault("plan_id", plan_id)
    payload["levels"] = _coerce_list(payload.get("levels"))
    payload["zones"] = _coerce_list(payload.get("zones"))
    payload["annotations"] = _coerce_list(payload.get("annotations"))
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        payload["meta"] = {}
    else:
        payload["meta"] = dict(meta)
    return dict(payload)


def extract_plan_layers(snapshot: Mapping[str, Any], *, plan_id: str) -> Dict[str, Any] | None:
    """Extract plan layers from a persisted snapshot."""

    if not isinstance(snapshot, Mapping):
        return None
    plan_block = snapshot.get("plan")
    layers = None
    if isinstance(plan_block, Mapping):
        layers = plan_block.get("plan_layers")
    if layers is None:
        layers = snapshot.get("plan_layers")
    if not isinstance(layers, Mapping):
        return None
    return _sanitize_layer_block(layers, plan_id=plan_id)


__all__ = ["extract_plan_layers"]

