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
