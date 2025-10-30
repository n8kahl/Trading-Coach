# Trader Console Data Flow

This document captures the end-to-end flow that powers the Next.js trader console
(`trade-coach-ui/`). The goal is to make it easy to reason about live chart updates,
overlay rendering, and websocket lifecycles.

## High-level architecture

```
┌────────────┐    ┌──────────────┐    ┌────────────────┐    ┌───────────────────────┐
| /idea/{id} |──▶︎ | LivePlanClient |──▶︎| PlanPriceChart |──▶︎| Lightweight Charts DOM |
└────────────┘    └──────────────┘    └────────────────┘    └───────────────────────┘
       │                  │                         │
       │                  │                         │
       │                  ├── usesPlanLayers ───────┤
       │                  │                         │
       │                  ├── usePlanSocket ───────▶ plan stream manager (singleton WS)
       │                  │
       │                  └── usePriceSeries ──────▶ `/tv-api/bars`
       │                                              (60s cadence + WS-triggered refresh)
       │
       └── initial snapshot seeds plan + overlays, so the UI can render immediately.
```

## Data sources

| Source | Use | Notes |
| --- | --- | --- |
| `GET /idea/{plan_id}` | Hydrates `LivePlanClient` with the latest plan snapshot, badges, and timestamps. | Called on route load (Next.js server component). |
| `GET /api/v1/gpt/chart-layers?plan_id=...` | Provides persisted plan overlays (levels, zones, annotations). | Loaded once on mount; re-used when plan deltas arrive. |
| `GET /tv-api/bars?symbol=...&resolution=...` | Returns OHLCV bars consumed by `usePriceSeries`. | Fetches a ~45 day window capped at 1,800 candles. |
| `WS /ws/plans/{plan_id}` | Streams plan deltas, `bar`/`tick` hints, and heartbeats. | Managed by a singleton connection keyed by plan id. |

## `usePriceSeries`

- Sanitises the symbol/resolution, fetches OHLCV bars, and exposes `bars`, `status`, `error`, and `reload`.
- Responses are normalised and merged with existing bars (deduplicated by timestamp) to avoid flicker.
- Auto-refreshes every 60 seconds **only when streaming is enabled**. `LivePlanClient` can bump a
  `priceRefreshToken` to force an immediate reload whenever plan deltas reference `last_price` or
  a `bar`/`tick` event is observed.
- Consumers may call `reload()` directly; the hook memoises results for deterministic renders.

## Plan stream manager

- Implemented in `src/lib/streams/planStream.ts`.
- Ensures **one websocket per plan id** by memoising connections. Additional subscribers reuse the same
  socket (e.g., status pill + chart updates).
- Exponential backoff with jitter (0.75s base, capped at 15s).
- Sends pings every 20s and tracks the last heartbeat timestamp; `usePlanSocket` degrades status to
  `connecting` if no heartbeat is observed within 35s.
- Automatically tears down the socket a few seconds after the final subscriber unsubscribes.

## `PlanPriceChart`

- Renders a single Lightweight Charts candlestick series plus:
  - Volume histogram
  - EMA overlays (periods from plan `charts_params.ema`)
  - VWAP line (toggle via plan `charts_params.vwap`)
  - Plan overlays: entry/stop/trailing stop/targets/supporting levels/zones
- `followLive()` keeps the most recent ~120 candles in view (bounded at a 60–minute lookback).
- Replay mode walks back through cached candles with a configurable stride. Stopping replay automatically
  re-enables follow-live.
- Uses `useImperativeHandle` so parent controls (timeframe toggles, replay button, follow live) can trigger
  chart actions without forcing re-renders.

## Environment

- `PUBLIC_UI_BASE_URL` is respected when generating plan links (e.g., the “Open Plan” button) so that
  deep links remain correct in multi-host deployments.
- The console expects `NEXT_PUBLIC_API_BASE_URL` and `NEXT_PUBLIC_WS_BASE_URL` to be aligned with the
  backend host; the websocket manager will derive `wss://` if only the HTTP base is provided.

## Operational checklist

- Verify `/ws/plans/{id}` emits `heartbeat` every <20s; the status pill will fall back to `connecting`
  when heartbeats stop.
- Ensure `/tv-api/bars` returns data for the requested symbol+resolution; stale responses set the price
  status pill to `connecting` after ~2× the resolution interval.
- If charts render blank, confirm the plan still exists (`GET /idea/{id}`) and that overlays are available
  (`GET /api/v1/gpt/chart-layers?plan_id=...`).
- When deploying, append the console build to the main release notes with screenshots/GIFs to capture
  overlay fidelity and websocket stability.

Happy charting!
