# Plan Refactor Roadmap

This document breaks the stop/TP/runner + MTF/strategy refactor into three deliverable phases. Phase 1 is complete in this commit; Phases 2 and 3 are outlined below for the next iteration.

## Phase 1 – Foundations ✅
- Added a dedicated `src/logic/` package containing pure helpers:
  - `levels.py`: structural level normalisation, round-number enrichment, snapping utilities.
  - `mtf.py`: aggregates the existing `MTFBundle` data into a weighted bias/score payload (default weights `0.50/0.30/0.15/0.05`).
  - `prob_calibration.py`: wraps the existing `CalibrationStore` with calibrated touch probabilities and monotonic enforcement.
  - `runner.py`: produces a hybrid chandelier/structure runner policy with give-back limiters.
  - `validators.py`: centralised plan guardrails (stop ATR bounds, RR floors, TP spacing/caps, probability monotonicity).
- Created `logic/__init__.py` to expose the helper modules for future imports.
- No wiring changes were made to `/gpt/plan` or `/gpt/scan`; API surface is untouched.

## Phase 2 – Planner Integration ✅
Implemented in this drop. Highlights:
- Centralised plan constants in `src/agent_server.py` and wired the new helpers (`build_stop`, `build_targets`) to produce structure-aware stops, targets, and metadata while preserving response schemas.
- Replaced runner construction with the hybrid policy from `logic.runner`, including chandelier/structure notes, give-back guardrails, and updated risk-block multiples.
- Routed target probabilities through `logic.prob_calibration` with monotone enforcement; existing calibration pass is skipped once calibrated data is present.
- Applied `logic.validators.validate_plan` before committing a candidate, falling back gracefully when guardrails fail.
- Surfaced multi-timeframe bias using `logic.mtf.mtf_bias`, threading its direction/score into plan notes, bias labelling, and actionability adjustments.
- Added guard-rails for event windows and MTF disagreement (ATR floors, TP RR floors, actionability downweighting).

Smoke tests still pending (manual run recommended) but OpenAPI contracts remain unchanged.
## Phase 3 – Strategy & Scan Enhancements ✅
Goal: Extend the integration to strategy selection, multi-candidate scoring, and scan parity.

Planned steps:
1. **Strategy gating ✅** – Rule-map implemented (Power Hour Continuation, VWAP Reclaim/Reject, Range Break & Retest, EMA Pullback Trend, Gap Fill Magnet). Plans now select the highest-scoring strategy using structural context + MTF bias and populate `strategy_id`, `strategy_profile.*`, and badges accordingly.
2. **Composite scoring ✅** – Probability × risk-reward × actionability × MTF multiplier blended into a composite score, with per-component diagnostics exposed in plan metadata.
3. **Event-window handling ✅** – ATR floors/RR minimums lifted during macro windows with accuracy-level annotations and runner notes describing the buffer.
4. **Runner give-back telemetry ✅** – Runner policies include telemetry (ADX slope, MTF score, event window), ensuring plan consumers can track momentum switches and give-back limits.
5. **Scan alignment ✅** – `/gpt/scan` now reuses the Phase 3 stop/TP/runner pipeline (probabilities, telemetry, composite scoring, guardrails) so candidates and plans emit matching metadata.
6. **Testing & docs ✅** – Added unit coverage for validator guardrails and runner telemetry, and refreshed the roadmap/README to note scan parity.
7. **Plan fallback parity ✅** – The `/gpt/plan` market-routing fallback path now executes the same refit geometry, so snapshot/LKG responses include target metadata, runner policy, probabilities, and strategy profile identical to scan-derived plans.

Deliverables: fully featured planner + scanner with strategy-aware outputs, robust guardrails, and parity across endpoints.
