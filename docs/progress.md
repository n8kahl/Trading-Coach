# Progress Log – Trading Coach Charts + GPT

**Last updated:** 2025‑10‑10  
Primary maintainer: _handoff to new developer_

This document captures what is live, where the sharp edges are, and the near-term roadmap. Treat it as the onboarding cheat sheet before diving into the code.

---

## 1. System snapshot

### 1.1 Shipping features

- **Scanning & plans**
  - `/gpt/scan` evaluates strategies with real indicators (anchored VWAPs, ATR, EMA stacks, breakout checks) and emits a complete plan payload (`entry`, `stop`, targets array, confidence, R:R, overlays).
  - Target snapping enforces strategy-specific minimum risk:reward and avoids micro take-profits by snapping only to higher-timeframe levels that preserve the R:R threshold (or dropping the target).
  - `/gpt/context/{symbol}` streams the latest interval bars and indicator series for ad‑hoc reasoning.
  - `/gpt/multi-context` bundles several intervals into one response and appends a volatility regime block (ATM IV, IV rank/percentile, HV20/60/120, IV↔HV ratio). Results are cached for 30 s.

- **Visualisation**
  - `/gpt/chart-url` normalises chart params and returns a `/tv` link. It supports plan rescaling (`scale_plan`), labelled EMAs, white VWAP, entry/stop/TP lines, and overlay bands (supply/demand, liquidity pools, FVGs, anchored VWAPs).
  - `/tv` serves TradingView Advanced Charts when bundled; otherwise, a Lightweight Charts fallback renders all overlays and plan labels. Autoscale + plan rescale keep old plans visible.
  - `/tv-api/*` fetches bars from Polygon with a Yahoo fallback and logs window choices for debugging.

- **Options & contracts**
  - `/gpt/contracts` ranks Tradier option chains using liquidity gates (spread, Δ window, DTE, OI) and computes scenario P/L with plan anchors. `risk_amount` (or legacy `max_price`) only informs projections—no contract is filtered for being “too expensive.”
  - Tradier chains and quotes are cached for 15 s; if Polygon snapshots are unavailable (HTTP 400), the service continues with Tradier data.

- **Volatility metrics**
  - `_compute_iv_metrics` builds ATM IV estimates and HV20/60/120 from daily bars, caching results for 120 s. Output feeds `volatility_regime` in multi-context responses.

- **Docs & tooling**
  - `docs/gpt_integration.md` documents request/response payloads for all GPT endpoints.
  - `README.md` now includes onboarding steps, architecture overview, and troubleshooting.

### 1.2 Recent enhancements (Oct 2025)

- Replaced the stubbed scanner with plan-aware detectors and confidence scoring.
- Added overlay parsing/rendering in `/tv` (supply/demand bands, liquidity pools, FVGs, AVWAP).
- Introduced `/gpt/multi-context` with caching and volatility regime metrics.
- Rebuilt `/gpt/contracts` with liquidity gates, scenario P/L, and tradeability scoring that ignores budget caps.
- Hardened IV metric calculations (safe `delta` coercion, better ATM IV detection).
- Documentation overhaul for onboarding, GPT wiring, and roadmap tracking.
- Refreshed the `/tv` viewer: multi-timeframe buttons, plan metadata banner, responsive drawer,
  and deduplicated level labels so toggling frames no longer stacks annotations.

---

## 2. Operational notes

- **Polygon snapshots (HTTP 400)**  
  Log spam such as `Polygon option snapshot failed ... 400 Bad Request` occurs when the Polygon account lacks the options snapshot entitlement. The failure is safe; Tradier data fills the gap. Consider adding a feature flag (`DISABLE_POLYGON_OPTIONS`) if logs become noisy.

- **Caching**  
  - Multi-context cache: 30 s per `(symbol, interval, lookback)`.
  - IV metrics cache: 120 s per symbol.
  - Tradier chain/quote cache: 15 s per request batch (chunked at 40 symbols).
  Restarting the container clears all caches.

- **Intervals**  
  Multi-context only accepts tokens normalised by `normalize_interval` (`1m`, `5m`, `15m`, `1h`, `4h`, `1D`, etc.). Invalid tokens trigger HTTP 400.

- **Chart rescale**  
  `/tv` defaults to `scale_plan=auto`. Disable with `scale_plan=off` if raw historical levels are required.

- **Auth**  
  Bearer auth is optional. When set, forward `X-User-Id` so future persistence can scope data correctly.

---

## 3. Roadmap & backlog

### 3.1 Immediate priorities

1. **Chart focus tooling**
   - Implement `focus=plan` vertical zoom and optional time centering so old plans automatically fill the viewport.
2. **Option data hygiene**
   - Add a configuration flag to suppress Polygon snapshot calls when the API key is missing or unauthorised.
   - Persist IV historical data (Redis or Postgres) to compute true IV rank instead of using HV proxies.
3. **Testing upgrades**
   - Add integration tests for `/gpt/scan`, `/gpt/contracts`, and `/gpt/multi-context` to guard against regressions.
4. **Observability**
   - Standardise structured logging (JSON) and add request IDs so GPT transcripts map cleanly to backend calls.

### 3.2 Near-term enhancements

- TradingView Advanced bundle: ship native overlays (`createStudy`, custom labels) when the bundle becomes available.
- Expand volatility regime metrics with skew (25Δ) and term slope once a reliable IV history source is wired.
- Persist user watchlists / journals and expose simple CRUD endpoints for GPT to manage them.
- Consider exposing a `/gpt/volatility` endpoint with a richer vol surface once data is ready.

### 3.3 Wishlist / longer term

- Backtesting API to replay plans on historical data.
- Portfolio-level risk dashboard (exposure, sector weightings, correlation).
- Webhook support to push alerts to Discord/Slack when plans trigger.

---

## 4. Testing & validation checklist

- Unit tests: `pytest` (currently indicator math only).
- Manual smoke tests:
  - `/gpt/scan` with `[AAPL, MSFT, TSLA]`.
  - `/gpt/multi-context` with `{"symbol":"SPY","intervals":["5m","1h","1D"]}`.
  - `/gpt/contracts` with `{"symbol":"AAPL","style":"intraday","risk_amount":150,"plan_anchor":{...}}`.
  - `/gpt/chart-url` using a plan from `/gpt/scan` and confirming overlays render on `/tv`.
  - `/tv-api/bars?symbol=SPY&resolution=5&range=5D` to confirm datafeed connectivity.

---

## 5. Reference map

- **Code**  
  - `src/agent_server.py` – FastAPI routes, caching, IV metrics, chart helpers.  
  - `src/scanner.py` + `src/strategy_library.py` – Strategy logic and metadata.  
  - `src/context_overlays.py` – Supply/demand, liquidity pools, FVGs, anchored VWAP calculations.  
  - `src/tradier.py` – Option chain/quote fetchers with caching.  
  - `static/tv/` – Lightweight Charts renderer and TradingView bootstrap code.

- **Docs**  
  - `README.md` – Onboarding, architecture, troubleshooting.  
  - `docs/gpt_integration.md` – Endpoint schemas and GPT usage patterns.  
  - `docs/progress.md` – (this file) history + roadmap.

- **Deployment**  
  - Railway (Nixpacks + Procfile). Runtime logs surfaced via Railway dashboard.

---

## 6. Owner hand-off

Focus first on:

1. Implementing `focus=plan` and optional time centering in `/tv` (tight UX win).
2. Suppressing Polygon snapshot noise and wiring IV history storage for accurate IV rank.
3. Adding integration tests around `/gpt/contracts` and `/gpt/multi-context`.

With those in place we can move toward richer overlays and persistence. Reach out if you need the prior GPT prompt or Action configs—the latest schema reference is in `docs/gpt_integration.md`.

Happy shipping!

- 2025-10-12 00:41:19Z – Schema documentation synced to OpenAPI 1.9.6 (docs/prompts/master_prompt_v2.1.md, docs/gpt_integration.md).

- 2025-10-11T20:37:56.623921-05:00 – Marked commit 23a45da as production ready baseline (docs/gpt_integration.md).

- 2025-10-16T14:45Z – Live plan/chart parity fix. Chart URLs now carry `live=1` and current timestamps whenever the plan context is live, `/gpt/context` links append `live=1`, and the TV surface polls in real time using the forwarded base URL (no more http→https blocking). Default universe for live fallbacks now rotates deterministically each RTH day.

- 2025-10-16T13:55Z – Live scan pipeline hardened. Diagnosed that FT-TopLiquidity expansion was intermittently returning empty results during RTH, causing the assistant to fall back to “frozen” lists despite the market being open. Added:
  - deterministic live-default universe (20 high-liquidity names) when expansion or market-data fetch fails.
  - better labeling (`Live liquidity leaders`) with no warning banner during RTH and up to 20 ranked symbols.
  - relaxed, logged handling for slightly stale bars instead of bailing to frozen context immediately.
  - doc note: if upstream watchlist service continues to fail, consider building an internal Top-Liquidity snapshot service for true live rankings.

- 2025-10-16T18:40Z – Stability + TV parity fixes. Addressed several production errors and made the chart viewer respect live session state.
  - Fixed NameError in fallback planner (`is_plan_live`) and normalized persistence to avoid NumPy/pandas JSON issues.
  - Corrected contracts helper call signature (`gpt_contracts`) to accept `(payload, user)` from server.
  - Embedded session metadata into chart links: `market_status`, `session_status`, `session_phase`, `session_banner`, plus `live=1` and `last_update` when applicable.
  - TV viewer now derives the symbol from `symbol`, `plan_meta`, or `plan_id`, writes it back into the URL, and connects the correct stream; no more fallback to AAPL when query is missing.
  - TV viewer treats session flags and data freshness properly: shows “Market Open” when session is live and only displays a degraded banner when `data_age_ms > 120s`.
- Bumped static bundle version (`tradingview-init.js?v=20251119`) to force fresh assets in production.

### 7. Release checklist

Use this when promoting to production:

- [ ] Merge to `main`; CI green
- [ ] Bump TV bundle query param in `static/tv/index.html` (forces clients to fetch latest JS)
- [ ] Deploy to Railway; confirm `/healthz` OK
- [ ] Smoke test:
  - [ ] `/gpt/scan` returns candidates for `[AAPL, MSFT, TSLA]`
  - [ ] `/gpt/plan` for a core symbol returns `trade_detail` with `symbol`, `plan_id`, `plan_version`, and session flags
  - [ ] Open the plan link; header shows the correct symbol and session status; levels + EMAs render
  - [ ] Switch timeframes; verify polling when `live=1`
- [ ] Check logs: no 500s, stale‑feed warnings within expected range
- [ ] Hard‑refresh `/tv` in browser to clear cached assets

### 8. Known issues / next work items

- Data freshness
  - Stale feed warnings during RTH indicate upstream data delays. Consider adding an adaptive retry/backfill and a “polygon_cached” mode label.
  - Add Prometheus/Grafana or simple heartbeat metrics for provider latency and last bar age.

- Options data
  - Sandbox 404s from Tradier spam logs; add feature flag to suppress snapshot/quotes when using sandbox keys and degrade quietly.
  - Persist IV history to compute true IV rank (see Roadmap 3.1).

- Chart URL service
  - Make `/gpt/chart-url` add session flags automatically (currently the `/gpt/plan` path injects them).
  - Add validation that `symbol` is present and normalize from `plan_meta.plan.symbol` if not.

- Viewer
  - Add a visible “Live” indicator when `live=1` is set and data age < 2 min.
  - Smooth-retry for SSE reconnects with jitter; display a transient banner when reconnecting.

- Tests & observability
  - Add integration tests for `/gpt/plan` fallback to cover plan embedding + chart param enrichment.
  - Add request IDs and structured logs to correlate GPT calls with backend traces.

#### Immediate fix plan – streaming UI, auto-replan, and Tradier noise (Oct 2025)

1. **Streaming status & chart updates**
   - Code to review: `static/tv/tradingview-init.js` (header clock, SSE handlers, `updateRealtimeBar`), `src/agent_server.py` (heartbeat logic in `_stream_generator`, `_publish_stream_event`).
   - Replace the multiline warning with a compact `Streaming Data` pill (green/yellow/red) that never resizes the header; keep verbose info in a tooltip. Drive colours from heartbeat age (≤15 s green, ≤60 s yellow, otherwise red) and surface a `Follow Live` toggle so manual panning disables auto-scroll.
   - Ensure heartbeat + tick events update the last bar even on 1 m charts while respecting the user’s scroll choice.
   - Test: mock SSE via `curl -N /stream/{symbol}`; check pill colours, last price, and chart stability with follow-live off/on.

2. **Confluence signals & level hygiene**
   - Code to review: `src/agent_server.py` (`build_plan_layers`, `_plan_meta_payload`), `static/tv/tradingview-init.js` (confluence mapper, level rendering around `renderPlanPanel`).
   - Always populate `plan_layers.meta` with `confidence_factors`, structured `confluence`, and key feature flags. Map them to up to five labelled dots (MTF alignment, EMA stack, VWAP context, liquidity pools, stats, runner) and show “No confluence signals” when empty.
   - Limit chart labels to priority levels (entry/stop/TP, ORH/L, session highs/lows, VWAP). Provide a “Show more levels” toggle for advanced view.
   - Test with a noisy ticker (IREN) to confirm default view is clean and confluence updates after scenario adoption.

3. **Scenario adoption & plan log UX**
   - Code to review: `static/tv/tradingview-init.js` (`applyPlanResponse`, new `adoptScenario`), `/ws/plans/{plan_id}` event handling.
   - On “Adopt This Scenario”, fetch `/idea/{plan_id}`, feed it through `applyPlanResponse`, update URL params, confluence, targets, and the plan log (limit visible entries to five with scroll for history; dedupe duplicate lines). Show a toast on success.
   - Stop auto-scroll during adoption, reset the overlay state, and leave streaming active on the new plan.
   - Test by adopting each scenario type; verify details/log update instantly without page reload.

4. **Auto-replan after stop-out**
   - Code to review: `src/live_plan_engine.py`, `_publish_stream_event`, plan delta handlers.
   - When a plan emits `status: exited`, kick off a background `/gpt/plan` for the same symbol/style, push a `plan_delta` with `next_plan_id`, and surface a CTA (“Reload plan”) unless the user enabled an Auto-rearm toggle.
   - Client should refresh confluence/targets once the new plan is adopted.
   - Test by injecting a fake `plan_delta` stop event over the plan websocket and confirming the CTA/new plan flow.

5. **Tradier sandbox noise & stale data handling**
   - Code to review: `src/tradier.py`, `src/app/services/options_select.py`, plus stale-data logs in `src/agent_server.py`.
   - Detect sandbox tokens and skip bulk quote lookups that 404; log once at INFO and show a “Sandbox quotes unavailable” banner in the UI. For production keys keep batching ≤50 symbols per call and add retry/backoff.
   - When Polygon bar age exceeds 15 min, automatically fall back to Yahoo equities before raising the stale-feed warning.
   - Test with sandbox credentials (`TRADIER_SANDBOX_TOKEN`) and production keys to ensure graceful degration.

##### Verification summary – Oct 2025
- Streaming pill + Follow Live: manual SSE replay confirmed green/yellow/red transitions, heartbeat tooltips, and that manual panning pauses auto-scroll until the toggle is re-enabled.
- Confluence & levels: plan layers now emit `level_groups`; default view shows trimmed primaries, and `Show more levels` reveals supplemental annotations without fetching new data.
- Scenario adoption: `/idea/{plan_id}` fetch runs during adoption, log history trimmed to five visible entries, success toast displayed, and stream continues on the new plan.
- Auto replan: stop-out delta delivers `next_plan_id`/`next_plan_version`; CTA renders with manual reload path and Auto Rearm applies the next plan automatically when toggled on.
- Tradier fallback: sandbox tokens skip quote batches with a single INFO log and surface a “Sandbox quotes unavailable” banner; production keys now request ≤25 symbols per batch and Yahoo fills intraday data when Polygon is >15 min stale.
- Tests: `pytest` (59 passed) covering calculations, plan rendering, live engine, and data-source fallbacks.
- Canonical roll-out: session middleware attaches `{status, as_of, next_open, tz}` to every request/response, plan payloads expose `confluence_tags` + `tp_reasons` and (behind `FF_OPTIONS_ALWAYS`) server-picked `options_contracts`, `/api/v1/gpt/chart-layers` now requires persisted overlays with as-of parity checks, and caches incorporate the session `as_of` so frozen/live scans stay deterministic.

##### Up next – canonical plan completeness & parity rollout
1. **Session SSOT middleware** (`src/agent_server.py`, new `session_middleware.py`): introduce `get_session()` that attaches `{status, as_of, next_open, tz}` to `request.state` and headers. Ensure all plan/scan endpoints consume this context and clamp frozen computations to `session.as_of`.
2. **Chart URL canonicalisation** (`src/chart_url.py`, `src/agent_server.py:gpt_chart_url`): implement allowlisted query builder with precision-aware formatting (`instrument_precision.py`). Replace existing inline URL assembly and add unit test `test_chart_url_allowlist_and_precision`.
3. **Plan layers persistence & endpoint** (`src/app/services/layers_service.py`, `src/agent_server.py`, `routers/chart_layers.py`): persist `plan_layers` during `/gpt/plan` generation, expose `GET /api/v1/gpt/chart-layers`, and validate `plan_layers.as_of == plan.as_of`. Add migration/backfill for historical plans.
4. **Plan payload completeness** (`src/live_plan_engine.py`, `src/agent_server.py`, `src/options_service.py`): populate deterministic `confluence` tags, `tp_reasons`, and `options_contracts` (or `options_note`) under `FF_OPTIONS_ALWAYS`. Integrate server-side contract picker and ensure hydration paths honour the flag.
5. **Frozen/live parity** (`src/scanner.py`, `src/plan_builder.py`): when `session.status == "closed"`, skip dynamic scan filters and surface frozen data using caches keyed by `as_of`. Add parity test comparing frozen vs. live-at-close outputs.
6. **Precision helper & canonical overlays** (`src/app/services/instrument_precision.py`, `src/app/services/chart_layers.py`): centralise tick precision and update URL/TP formatting. Adjust layers builder to use existing feature outputs and cache keys that include `as_of`.
7. **Testing & rollout** (tests under `tests/test_chart_url.py`, `tests/test_plan_integrity.py`, `tests/test_scan_policy.py`): cover URL allowlist, layers fetch parity, plan completeness, and frozen scan behaviour. Update README + prompts with new contracts and document flag rollout steps (`FF_CHART_CANONICAL_V1`, `FF_LAYERS_ENDPOINT`, `FF_OPTIONS_ALWAYS`).

Deliverables: updated frontend bundle with streaming indicator + confluence cleanup, backend heartbeat/auto-replan enhancements, Tradier sandbox guardrails, and docs (README + this log). Verify end-to-end with live symbols before promotion.

Deployment notes
- After pulling `main`, redeploy Railway to pick up the TV bundle bump and server fixes. Hard-refresh `/tv` pages to clear cached JS.
