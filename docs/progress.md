# Progress Log – Trading Coach Charts + GPT

Last updated: 2025-10-08

## What’s live now

- GPT data surface
  - `/gpt/scan` and `/gpt/context` deliver market_snapshot, key_levels, features, and charts.params.
  - `charts.params` now supports: `symbol, interval, ema, vwap, view, theme, strategy, direction, atr, levels`.
  - Option chains from Polygon are summarized and attached when available; Tradier remains a fallback.

- Canonical chart link builder
  - `POST /gpt/chart-url` returns `{ "interactive": "<url>" }`.
  - Normalizes tokens: `5m→5`, `1h→60`, `1d→1D`; uppercases symbols while preserving exchange prefixes (`NASDAQ:TSLA`).
  - Accepts `levels` (comma floats) for dotted key lines on the chart.

- New `/tv` viewer
  - Serves TradingView Advanced Chart UI if `charting_library/` exists; otherwise falls back to Lightweight Charts.
  - Fallback features:
    - Candles + volume histogram
    - EMA overlays from `ema=` and VWAP toggle via `vwap=1`
    - Labeled Entry/Stop/TP lines using axis labels
    - Dotted yellow reference lines from `levels=`
    - Robust autoscale: ensures all plan lines and levels are included even when far from current prices
    - On‑screen debug when `debug=1` (shows data request + bar count)

- Datafeed for `/tv` – `/tv-api/*`
  - `GET /tv-api/bars` resolves resolution (`5`, `5m`, `1h`, `1D`) and fetches from Polygon, then Yahoo as fallback.
  - If intraday is missing, retries `15` then `D`.
  - `from`/`to` query params are optional; supports `range=5D/1W/1M` or explicit unix seconds/ms timestamps.
  - Server logs requests, chosen timeframe, and bar counts so Railway logs are useful for triage.

## Known behaviors / caveats

- Plans from a different price regime (e.g., pre/post split or stale context) will plot correctly but may sit far from current price. Autoscale keeps them visible, but they will appear away from the current candles.
- TradingView bundle isn’t deployed yet – `/tv/charting_library/charting_library.js` 404 is expected. The fallback renderer handles everything for now.
- Dotted key levels are auto‑included in `charts.params.levels` when links are produced by `/gpt/scan` or `/gpt/context`. Hand‑typed links must add `levels=` manually.

## Proposed improvements (next up)

1. Plan focus zoom (client only)
   - New query: `focus=plan`. Center/zoom vertically around min/max of {entry, stop, tps, levels} with a pad (e.g., 1.5× ATR or 0.8%).
2. Plan rescale (client only)
   - New query: `scale_plan=auto|<float>`. Auto = `last_close/entry` when sensible; otherwise respects an explicit factor. Useful when a plan was computed on a different basis.
3. TV labels + studies (when Advanced bundle arrives)
   - Add native study IDs and labels using `createStudy`/`createShape` and study titles.
4. Optional debug API echo
   - `debug=1` could surface the resolved timeframe and last close directly in the banner for faster checks.

## Testing notes

- Quick sanity for bars:
  - `/tv-api/bars?symbol=SPY&resolution=5&range=5D` → `{ "s":"ok", ... }`
- Example chart (fallback viewer):
  - `/tv/?symbol=QQQ&interval=5&ema=9,20,50&vwap=1&entry=486.40&stop=485.30&tp=488.20,489.40&levels=488.20,489.40&range=5D`
- Force cache bust + debug:
  - add `&_ts=1700000001&debug=1` to your `/tv` link.

## Implementation references

- Viewer: `static/tv/index.html`, `static/tv/tradingview-init.js`, `static/tv/tv-datafeed.js`
- Fallback renderer: `static/tv/tradingview-init.js`
- Datafeed: `src/agent_server.py` – `tv_api` router (`/tv-api/config`, `/tv-api/symbols`, `/tv-api/bars`)
- GPT endpoints: `src/agent_server.py` – `/gpt/scan`, `/gpt/context`, `/gpt/chart-url`
- Docs: `docs/gpt_integration.md`

## Open questions for follow‑up

- Do we want `focus=plan` and `scale_plan` enabled by default, or only when explicitly passed?
- Should we allow the GPT to request a centered time window around the plan (e.g., `center_time=latest` or a timestamp) in addition to vertical focus?
- When TradingView bundle is available, do we need parity for dotted key levels (TV shapes) or is fallback sufficient?

---

Owner handoff: continue from the “Proposed improvements” list. The highest impact for UX is `focus=plan` so the plan band fills the viewport even when the underlying has moved.

