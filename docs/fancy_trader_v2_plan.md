# Fancy Trader 2.0 Delivery Plan

_Last updated: 2025-10-13_

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

- Implement composite option scoring in `app/engine/options_select.py`.
- Add event gating heuristics to toggle between full-risk and defined-risk plays.
- Wire results into response formatting and documentation.

### Stage 5 – Streaming & Chart Enhancements

- Launch `/ws/plans` WebSocket endpoint with plan lifecycle events.
- Add canonical chart URLs that embed `plan_id` and `as_of`.
- Harden live streaming with shared caches and instrumentation.

### Stage 6 – Testing & Backtesting

- Build automated regression tests (unit, integration, load) covering closed-session replay, provider fallbacks, and streaming.
- Refactor the backtesting harness to align with the unified target engine.
- Add CI hooks for schema validation, linting, and coverage dashboards.

### Stage 7 – Release & Documentation

- Refresh READMEs, operational runbooks, and API schemas.
- Prepare migration notes for GPT actions and downstream clients.
- Create release checklist and tag production build.

## Documentation & Ownership

- This plan is stored in-repo so future maintainers have a single source of truth.
- Update the “Last updated” date and relevant sections at the end of each stage.
- Record deviations or trade-offs in a new **Decisions** section if they arise.

## Next Steps

1. Fold closed-session regression tests into the expanded Stage 6 suite.
2. Kick off Stage 4 (options/event intelligence) while monitoring new session + target pipelines in staging.
3. Continue updating this document as subsequent stages progress.
