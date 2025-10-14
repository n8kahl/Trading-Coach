# Fancy Trader 2.0 Delivery Plan

_Last updated: 2025-10-14_

## Overview

Fancy Trader 2.0 is a multi-stage upgrade of the Trading Coach backend focused on:

- Always-on session awareness and deterministic replay when markets are closed.
- Provider “as-of” data clamps that guarantee consistent results across intraday and overnight runs.
- A unified target/stop engine that harmonises plan sizing, chart rendering, and streaming guidance.
- Enriched option selection, event gating, and a rebuilt streaming/channel architecture.
- Test, observability, and documentation coverage required for production scale.

The work is split into the stages below so that the codebase remains deployable at every checkpoint.

## Stage Breakdown

### Stage 1 – Foundation _(current stage)_

- ✅ Branch `feat/v2.2-always-on-session` created.
- ✅ Added `session_state` service with timezone-aware helper.
- ✅ Attached session metadata to core JSON responses (scan/plan/context/contracts/etc.).
- ✅ Documented rollout approach (this file) and ensured repository permanence.
- ✅ Ran existing pytest suite to confirm zero regressions.

### Stage 2 – As-of Providers

- ✅ Added Polygon helpers `last_price_asof` and `fetch_polygon_option_chain_asof` with timestamp clamping.
- ✅ Threaded session-aware `as_of` filtering through scan/context/chart responses and chart URLs.
- ✅ Surface closed-session metadata (including underlying price snapshots) across contracts, context, and multi-frame APIs.
- ⬜ Add weekend/holiday regression tests (scheduled alongside Stage 6 testing expansion).

### Stage 3 – Unified Target/Stop Engine

- ✅ Introduced `app/engine/targets.py` (`TargetEngineResult`, structured plan builder) and wired scanners to populate plan profiles.
- ✅ Propagated target probabilities, snap traces, and structured plan JSON into `/gpt/plan` responses and features.
- ✅ Ensured chart payloads and downstream consumers reuse the same server-computed TP/SL profile (no client recomputation).

### Stage 4 – Options & Event Intelligence

- ✅ Implemented composite option scoring (`app/engine/options_select.py`) with response annotations (`example_leg`, score components) in `/gpt/contracts`.
- ✅ Added event gating heuristics (`app/engine/events.py`) that convert high-severity, near-term events into defined-risk or suppressed plans, surfaced via plan warnings and structured plan metadata.
- ✅ Propagated gating metadata into plan features/structured plans so downstream clients receive consistent guidance.

### Stage 5 – Streaming & Chart Enhancements

- ✅ Launched `/ws/plans` WebSocket endpoint (plan-scoped live stream with price/hit/replan events) alongside expanded plan event fan-out.
- ✅ Chart URLs continue to embed `plan_id`/`as_of`, now backed by session-aware providers.
- ⬜ Add streaming instrumentation/metrics dashboards (scheduled alongside Stage 6 observability tasks).

### Stage 6 – Testing & Backtesting

- ✅ Added unit coverage for option scoring (`tests/test_options_select.py`) and event gating (`tests/test_event_gating.py`), expanding regression coverage beyond indicator math.
- ✅ Instrumented plan gating and WebSocket lifecycle logging to aid observability during staging/production rollouts.
- ✅ Delivered the compact `/api/v1/assistant/exec` JSON workflow (canonical chart URLs, optional hedges, options examples) plus `/api/v1/symbol/*` diagnostics; UI no longer parses prose.
- ⬜ Load/soak harness + CI schema diff remain future enhancements (documented in roadmap).

### Stage 7 – Release & Documentation

- ✅ Refreshed README with composite option scoring, event gating, and `/ws/plans` streaming guidance; updated delivery plan (this file).
- ✅ Documented Fancy Trader 2.0 rollout plan in-repo for handoffs.
- ⬜ Final release checklist/tagging to be executed alongside deployment.

## Documentation & Ownership

- This plan is stored in-repo so future maintainers have a single source of truth.
- Update the “Last updated” date and relevant sections at the end of each stage.
- Record deviations or trade-offs in a new **Decisions** section if they arise.

## Next Steps

1. Fold closed-session and streaming regression tests into the expanded Stage 6 suite.
2. Implement Stage 6 observability/test automation while monitoring the new option scoring + plan streams in staging, including assistant exec JSON contract under load.
3. Continue updating this document as subsequent stages progress.
