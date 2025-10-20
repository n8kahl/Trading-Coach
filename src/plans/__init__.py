"""Plan geometry utilities."""

from .geometry import GeometryConfig, PlanGeometry, RunnerPolicy, StopResult, TargetMeta, build_plan_geometry
from .entry import select_structural_entry
from .expected_move import session_expected_move
from .clamp import clamp_targets_to_em, ensure_monotonic
from .levels import STRUCTURAL_ORDER, directional_nodes, profile_nodes, populate_recent_extrema
from .snap import snap_targets, stop_from_structure, build_key_levels_used
from .runner import compute_runner
from .actionability import actionability_score, compute_entry_candidates
from .invariants import assert_invariants, GeometryInvariantError
from .pipeline import StructuredGeometry, build_structured_geometry

__all__ = [
    "GeometryConfig",
    "PlanGeometry",
    "RunnerPolicy",
    "StopResult",
    "TargetMeta",
    "build_plan_geometry",
    "select_structural_entry",
    "session_expected_move",
    "clamp_targets_to_em",
    "ensure_monotonic",
    "STRUCTURAL_ORDER",
    "directional_nodes",
    "profile_nodes",
    "populate_recent_extrema",
    "snap_targets",
    "stop_from_structure",
    "build_key_levels_used",
    "compute_runner",
    "actionability_score",
    "compute_entry_candidates",
    "assert_invariants",
    "GeometryInvariantError",
    "StructuredGeometry",
    "build_structured_geometry",
]
