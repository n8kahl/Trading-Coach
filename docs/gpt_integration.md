# GPT Integration Cheat Sheet

This backend now serves raw market context so that your GPT agent can make the
final trading decisions (entries, stops, targets, position sizing). The server
focuses on data prep; the agent performs the higher-level reasoning.

## Production Endpoint

- API host (production): `https://trading-coach-production.up.railway.app`
- Charts base (production): `https://trading-coach-production.up.railway.app/tv`
- Use `POST /gpt/chart-url` and render the returned `interactive` URL verbatim.
- Health: `GET /healthz`
- Quick checks:
  - `curl -sS https://trading-coach-production.up.railway.app/openapi.json | jq '.paths | keys'`
- `curl -sS -X POST https://trading-coach-production.up.railway.app/gpt/scan -H 'content-type: application/json' -d '{"tickers":["AAPL"],"style":"intraday"}'`

## Prompt & Schema (Live)

- Master prompt (**approved v2.1**): [`docs/prompts/master_prompt_v2.1.md`](prompts/master_prompt_v2.1.md)
- API schema (**OpenAPI 1.9.6**): mirrors `https://trading-coach-production.up.railway.app/openapi.json`
- Deployment status: **Production ready & approved** (2025-10-10 refresh)
- Persistent storage: set `DB_URL` (Postgres) so `/gpt/plan` snapshots and idea permalinks survive restarts.

---

## Offline Planning Mode

- `POST /gpt/plan?symbol=TSLA&style=swing&offline=true`
- Works outside RTH using last HTF snapshots (4h/1D) and a 5-day expected move horizon.
- Response extras when `offline=true`:
  - `planning_context: "offline"`
  - `offline_basis`: `{ htf_snapshot_time, volatility_regime, expected_move_days }`
  - `warnings` includes `Offline Planning Mode — Market Closed; using last valid HTF data.`
  - Chart URLs gain `offline_mode=true` query param (watermark rendered client-side).
- Prefer enabling only when the user explicitly asks for night / weekend plans.

## Production Snapshot (Frozen)

This section captures the exact surface area and behaviors considered “production ready” prior to the next significant change. Treat this as the last known‑good integration point.

- Timestamp (UTC): 2025-10-10
- Branch: `feature/ai-driven-planning`

What’s included
- `/gpt/plan`
  - Returns: `plan`, `charts`, `key_levels`, `market_snapshot`, `features`, `options`, `trade_detail`
  - Provenance: `calc_notes` (atr14, rr_inputs, em_cap_applied), `htf` (bias, snapped_targets), `debug.tp1` (when structural TP1 is used)
- `/gpt/multi-context`
  - Request: `include_series` (default false), `intervals` (aka `frames`)
  - Response: `contexts` (series trimmed when gated), `summary` (frames_used, confluence_score, trend_notes, volatility_regime+label, expected_move_horizon, nearby_levels), `decimals`, `data_quality`
- `/gpt/contracts`
  - Response adds `table` rows with: `label`, `dte`, `strike`, `price`, `bid`, `ask`, `delta`, `theta`, `iv`, `spread_pct`, `oi`, `liquidity_score`
- `/gpt/chart-url`
  - Server‑side validator: required fields, monotonic geometry, R:R gates, ATR distance w/ confluence override, whitelisted interval/view, percent‑encoding of notes/levels/strategy; BASE_URL respected verbatim; returns `/tv` link
- `/gpt/sentiment`
  - Latest‑video sentiment + `tickers_detail` (price, change_pct, 15m EMA stack, ATR, range), robust to transcript issues; always JSON (204/502/503 on errors)

Planner internals
- Structural TP1 selector (both long and short): candidate generation (ORB/prior H/L, VWAP/EMAs, VAH/VAL/POC, Fib projections), style‑aware scoring, EM/ATR/ratio/R:R constraints, with graceful fallback to HTF snapper.
- EM capping is applied during target snapping using expected_move_horizon.
- Geometry guardrail ensures ordered targets and appends a warning note when adjustments were required.

Client guidance
- Use `contracts.table` to render compact options; always show `bid`, `ask`, and `price`.
- Use `plan.calc_notes`, `plan.trade_detail`, and `plan.htf.snapped_targets` in explanations; the model should not recompute these locally.
- For charts, always call `POST /gpt/chart-url` and validate the returned URL before rendering.

This snapshot is the “last production point” before the upcoming significant change. Future updates may alter schemas and behaviors; use this section to maintain compatibility with current GPT prompts and tooling.


## `/gpt/scan`

`POST /gpt/scan` accepts the same payload as before (`tickers` plus an optional
`style`). The response has been reshaped to highlight market context instead of
pre-baked trade levels.

```jsonc
[
  {
    "symbol": "AAPL",
    "style": "scalp",
    "strategy_id": "power_hour_trend",
    "description": "During the last hour of RTH…",
    "score": 0.82,
    "contract_suggestion": {
      "symbol": "O:AAPL251010C00260000",
      "expiration": "2025-10-10",
      "option_type": "call",
      "strike": 260,
      "bid": 2.05,
      "ask": 2.15,
      "mid": 2.1,
      "spread_pct": 0.0476,
      "delta": 0.52,
      "open_interest": 1820,
      "volume": 340,
      "dte": 2,
      "implied_volatility": 0.41
    },
    "options": {
      "source": "polygon",
      "filters_applied": true,
      "filter_rules": {
        "dte_range": [0, 2],
        "delta_range": [0.45, 0.65],
        "max_spread_pct": 0.05,
        "min_open_interest": 500,
        "min_volume": 250
      },
      "chain_size": 192,
      "considered_size": 6,
      "underlying": {
        "symbol": "AAPL",
        "price": 258.3,
        "last_updated": "2025-10-08T19:29:58Z"
      },
      "best": {
        "symbol": "O:AAPL251010C00260000",
        "expiration": "2025-10-10",
        "option_type": "call",
        "strike": 260,
        "bid": 2.05,
        "ask": 2.15,
        "mid": 2.1,
        "spread_pct": 0.0476,
        "delta": 0.52,
        "open_interest": 1820,
        "volume": 340,
        "dte": 2,
        "implied_volatility": 0.41
      },
      "alternatives": [
        {
          "symbol": "O:AAPL251010C00257500",
          "expiration": "2025-10-10",
          "option_type": "call",
          "strike": 257.5,
          "bid": 3.4,
          "ask": 3.6,
          "mid": 3.5,
          "spread_pct": 0.0571,
          "delta": 0.63,
          "open_interest": 2410,
          "volume": 510,
          "dte": 2,
          "implied_volatility": 0.39
        },
        {
          "symbol": "O:AAPL251010C00262500",
          "expiration": "2025-10-10",
          "option_type": "call",
          "strike": 262.5,
          "bid": 1.41,
          "ask": 1.49,
          "mid": 1.45,
          "spread_pct": 0.0552,
          "delta": 0.43,
          "open_interest": 980,
          "volume": 220,
          "dte": 2,
          "implied_volatility": 0.42
        }
      ]
    },
    "direction_hint": "long",
    "key_levels": {
      "session_high": 261.4,
      "session_low": 255.9,
      "opening_range_high": 258.7,
      "opening_range_low": 257.3,
      "prev_close": 257.9,
      "prev_high": 259.8,
      "prev_low": 254.6,
      "gap_fill": 257.9
    },
    "market_snapshot": {
      "timestamp_utc": "2025-10-08T19:30:00+00:00",
      "price": {
        "open": 258.1,
        "high": 258.6,
        "low": 257.9,
        "close": 258.3,
        "volume": 1.28e6,
        "entry_reference": 258.3
      },
      "indicators": {
        "ema9": 258.2,
        "ema20": 257.7,
        "ema50": 256.9,
        "vwap": 257.8,
        "atr14": 1.92,
        "adx14": 21.4
      },
      "volatility": {
        "true_range_median": 0.42,
        "bollinger_width": 1.02,
        "keltner_width": 1.38,
        "in_squeeze": false,
        "expected_move_horizon": 2.52
      },
      "session": {
        "phase": "power_hour",
        "minutes_to_close": 28,
        "bar_interval_minutes": 1
      },
      "trend": {
        "ema_stack": "bullish",
        "direction_hint": "long"
      },
      "gap": {
        "points": 0.4,
        "percent": 0.16,
        "direction": "up"
      },
      "recent": {
        "closes": [257.9, 258.1, 258.3, 258.4, 258.3],
        "close_deltas": [0.2, 0.2, 0.1, -0.1]
      }
    },
    "features": {
      "atr": 1.92,
      "adx": 21.4,
      "ema9": 258.2,
      "ema20": 257.7,
      "ema50": 256.9,
      "vwap": 257.8,
      "direction_bias": "long",
      "plan_entry": 258.6,
      "plan_stop": 257.2,
      "plan_targets": [259.5, 260.4],
      "plan_confidence": 0.74,
      "plan_risk_reward": 1.63,
      "plan_notes": "VWAP hold + range break alignment"
    },
    "plan": {
      "direction": "long",
      "entry": 258.6,
      "stop": 257.2,
      "targets": [259.5, 260.4],
      "confidence": 0.74,
      "risk_reward": 1.63,
      "atr": 1.92,
      "notes": "VWAP hold + range break alignment"
    },
    "charts": {
      "params": {
        "symbol": "AAPL",
        "interval": "1",
        "ema": "9,20,50",
        "view": "30m",
        "title": "AAPL power_hour_trend",
        "strategy": "power_hour_trend",
        "direction": "long",
        "entry": "258.60",
        "stop": "257.20",
        "tp": "259.50,260.40",
        "atr": "1.9200",
        "vwap": "1",
        "theme": "dark",
        "levels": "259.40,255.90,258.70,257.30,259.80,254.60,257.90"
      }
    },
    "data": {
      "bars": "https://host/gpt/context/AAPL?interval=1m&lookback=300"
    },
    "context_overlays": {
      "supply_zones": [
        {"low": 259.4, "high": 260.1, "timeframe": "15m", "strength": "moderate"}
      ],
      "demand_zones": [
        {"low": 256.1, "high": 256.9, "timeframe": "15m", "strength": "strong"}
      ],
      "liquidity_pools": [
        {"level": 259.8, "type": "equal_highs", "timeframe": "1h", "density": 0.64}
      ],
      "fvg": [
        {"low": 257.6, "high": 258.0, "timeframe": "5m", "age": 7}
      ],
      "rel_strength_vs": {
        "benchmark": "SPY",
        "lookback_bars": 15,
        "value": 1.08
      },
      "internals": {
        "adv_dec": -320,
        "tick": 420,
        "sector_perf": {"XLK": 0.6, "XLF": -0.2},
        "index_bias": "bullish"
      },
      "options_summary": {
        "atm_iv": 0.41,
        "iv_rank": null,
        "iv_pct": null,
        "skew_25d": -0.08,
        "term_slope": 0.03,
        "spread_bps": 45
      },
      "liquidity_metrics": {
        "avg_spread_bps": 45,
        "typical_slippage_bps": 22.5,
        "lot_size_hint": null
      },
      "events": [
        {"type": "earnings", "time_utc": "2025-10-29T20:00:00Z", "severity": "high"}
      ],
      "avwap": {
        "from_open": 258.7,
        "from_prev_close": 257.9,
        "from_session_low": 258.1,
        "from_session_high": 259.6
      },
      "volume_profile": {
        "vwap": 257.8,
        "vah": 259.9,
        "val": 256.7,
        "poc": 258.4
      }
    }
  }
]
```

### Key points for the agent

- **Each signal ships with a fully calculated plan.** The `plan` payload (and
  mirrored `plan_*` feature fields) includes direction, entry, stop, two
  targets, confidence, risk:reward, ATR reference, and notes derived from the
  live data check.
- **Style tokens reflect the strategy library.** Expect one of `scalp`,
  `intraday`, `swing`, or `leaps` (index-focused strategies map to
  `intraday`). Use this value when filtering scans or requesting contracts.
- **`direction_hint` is advisory.** It reflects indicator alignment; override
  it if your analysis disagrees.
- **`options` bundle** surfaces Polygon's filtered option chain (best contract
  plus a few alternates) when `POLYGON_API_KEY` is configured. If Polygon is
  unavailable the field is omitted and `contract_suggestion` may fall back to
  Tradier metadata.
- **`charts.params`** exposes the exact query used to render charts (EMA stack,
  view, strategy, direction, ATR). Send those along with your computed entry,
  stop, and targets to `POST /gpt/chart-url` whenever you need a canonical link.
  The viewer accepts TradingView resolutions (`1`, `3`, `5`, `15`, `30`, `60`,
  `120`, `240`, `1D`); minute/hours strings like `1m` and `1h` are also
  normalized automatically. Key levels extracted from the response are included
  automatically in `levels` so the chart renders dotted reference lines for
  session/previous highs and lows. Provide optional labels via
  `levels=259.40|ORH;255.90|VAL` to have the viewer annotate each line. Include
  `scale_plan=auto` (default) to let
  the viewer rescale legacy entries/stops/targets/levels to the latest prices
  after splits or large gaps; set `scale_plan=off` to render the raw values or
  pass an explicit multiplier (e.g. `scale_plan=0.5`).
- **Plan rescaling.** The viewer defaults to `scale_plan=auto`, which rescales
  entry/stop/targets/levels to the latest close when the plan was built on a
  different price basis (splits, stale snapshots, etc.). Add `scale_plan=off`
  to disable, or supply an explicit multiplier (e.g., `scale_plan=0.5`) when
  you know the adjustment factor.
- **Overlay rendering.** When `charts.params` includes `supply`, `demand`,
  `liquidity`, `fvg`, or `avwap`, the `/tv` viewer plots labeled horizontal
  bands/lines to represent higher-timeframe zones, liquidity pools, fair value
  gaps, and anchored VWAP references. When `ema` is provided (e.g. `"9,20,50"`)
  the viewer overlays those EMAs on the price action; add `vwap=true` to plot
  the session VWAP curve.
- **`context_overlays`** packages higher-timeframe zones, liquidity pools, FVGs,
  relative strength, internals, options/volatility summaries, liquidity
  frictions, event hooks, anchored VWAPs, and volume profile magnets. These
  default to empty/null until analytics are wired in.
- **Volatility ranks/percentiles** need historical IV data. Until a volatility
  history store is connected those specific fields stay null even when other
  option metrics are populated.
- **`expected_move_horizon`** approximates the move that is typically achievable
  over ~30 minutes (1–2 minute bars) or ~60 minutes (5 minute bars).
- **`data.bars`** provides a ready-made URL for deeper context.

## `/gpt/chart-url`

Use this helper whenever you need a shareable chart link. Provide the baseline
`charts.params` from the scan/context response and append your computed trade
plan fields.

**Required body fields:** `symbol`, `interval`, `direction`, `entry`, `stop`,
and `tp` (comma-separated list for multiple targets). Missing any of these will
return a `422` validation error.

Depending on the data available, `charts.params` may also include overlay keys
such as `supply`, `demand`, `liquidity`, `fvg`, and `avwap`; the chart viewer
will render these automatically.

> **Deployment note:** Set the `BASE_URL` environment variable on Railway to the
> fully qualified TradingView page (e.g. `https://your-app.up.railway.app/tv`).
> The `/gpt/chart-url` endpoint appends query parameters to that URL.

If the TradingView Advanced Charting Library isn't available on the server yet,
the `/tv` page automatically falls back to the open-source Lightweight Charts
renderer (candlesticks, EMA overlays, VWAP approximation, and plan lines).

```bash
curl -s -X POST https://host/gpt/chart-url \
  -H "content-type: application/json" \
  -d '{
        "symbol":"AAPL",
        "interval":"1",
        "direction":"long",
        "ema":"9,20,50",
        "view":"30m",
        "entry":"258.40",
        "stop":"257.80",
        "tp":"259.60,260.10",
        "notes":"Breakout plan"
      }'
```

Response:

```json
{"interactive":"https://host/tv?symbol=AAPL&interval=1m&ema=9%2C20%2C50&view=30m&entry=258.40&stop=257.80&tp=259.60%2C260.10&notes=Breakout+plan"}
```

If any required parameter is missing or geometry/risk constraints fail you will
receive a `422` response with a descriptive error message.

### Client-side URL validation (optional but recommended)

If you render the returned link in a client, you can enforce a canonical base
to avoid accidental redirects or stale hosts. Example (TypeScript/JavaScript):

```ts
// After calling POST /gpt/chart-url
const url: string = resp.interactive;

try {
  const u = new URL(url);
  if (
    u.protocol !== 'https:' ||
    u.hostname !== 'trading-coach-production.up.railway.app' ||
    u.pathname !== '/tv'
  ) {
    throw new Error('Non-canonical chart URL');
  }
  // ✅ safe to render
  showLink('📈 Chart: View Interactive Setup', url);
} catch (e) {
  showText('📈 Chart: unavailable (URL failed validation; ask me to retry)');
}
```

If you later switch to the lightweight renderer, update the pathname check to
`'/charts/html'` and consider keeping the allowed host/domain in a config
variable instead of hardcoding it.

## `/gpt/contracts`

Ranks Tradier option contracts according to your style/budget rules and
returns plug-and-play picks with liquidity and greeks baked in.

### Request body

```json
{
  "symbol": "NVDA",
  "side": "call",
  "style": "intraday",
  "min_dte": 1,
  "max_dte": 5,
  "min_delta": 0.45,
  "max_delta": 0.55,
  "max_spread_pct": 10,
  "min_oi": 500,
  "risk_amount": 120,
  "expiry": null
}
```

`max_price` is still accepted for backwards compatibility; when provided it is
treated as the fallback `risk_amount` for projections.

Style defaults applied server-side when you omit filters:

| Style    | DTE window | Delta window | Max spread % | Min OI |
|----------|------------|--------------|--------------|--------|
| scalp    | 0–2        | 0.55–0.65    | 8            | 500    |
| intraday | 1–5        | 0.45–0.55    | 10           | 500    |
| swing    | 7–45       | 0.30–0.55    | 12           | 500    |
| leaps    | ≥180       | 0.25–0.45    | 12           | 500    |

No budget-based filtering is applied. The optional `risk_amount` (default
**$100**) is only used to size the P/L projections returned for each contract.
If no contracts pass, the service widens the delta window by ±0.05, then the
DTE window by ±2 days before returning an empty result.

### Response shape

```jsonc
{
  "symbol": "NVDA",
  "side": "call",
  "style": "intraday",
  "risk_amount": 120.0,
  "filters": {
    "min_dte": 1,
    "max_dte": 5,
    "min_delta": 0.45,
    "max_delta": 0.55,
    "max_spread_pct": 10.0,
    "min_oi": 500
  },
  "relaxed_filters": false,
  "best": [
    {
      "label": "NVDA 2024-10-18 460C",
      "symbol": "NVDA241018C00460000",
      "expiry": "2024-10-18",
      "dte": 9,
      "strike": 460.0,
      "type": "CALL",
      "price": 5.6,
      "bid": 5.5,
      "ask": 5.7,
      "spread_pct": 3.57,
      "volume": 1823,
      "oi": 9410,
      "delta": 0.51,
      "gamma": 0.04,
      "theta": -0.08,
      "vega": 0.32,
      "iv": 0.38,
      "iv_rank": null,
      "tradeability": 87.4,
      "pnl": {
        "per_contract_cost": 560.0,
        "at_stop": -55.0,
        "at_tp1": 120.0,
        "at_tp2": 260.0,
        "rr_to_tp1": 2.18
      },
      "pl_projection": {
        "risk_per_contract": 560.0,
        "contracts_possible": 1,
        "max_profit_est": 260.0,
        "max_loss_est": 55.0
      }
    }
  ],
  "alternatives": [
    { "symbol": "…", "tradeability": 81.2 }
  ]
}
```

`tradeability` (0–100) weights spread (40 %), delta fit (30 %), open interest
(20 %), and implied-vol regime (10 %). The endpoint returns up to three
contracts in `best` and the next seven in `alternatives`.

Field summary:

- **label** – human-friendly identifier (`SYMBOL YYYY-MM-DD STRIKE<Call/Put>`).
- **price** – mid/mark rounded to cents.
- **spread_pct** – bid/ask spread as a percentage.
- **volume / oi** – latest Tradier values.
- **delta/gamma/theta/vega/iv** – Tradier quote greeks.
- **iv_rank** – `null` for now (left for GPT to down-rank when IV history is
  available).
- **tradeability** – blended liquidity/fit score used for sorting.
- **pnl / pl_projection** – per-contract scenario results and risk sizing based
  on the supplied `risk_amount` (defaults to $100).

## `/gpt/context/{symbol}`

## `/gpt/multi-context`

Fetches market context across multiple timeframes plus implied-volatility
metrics in a single round-trip.

### Request body

```json
{
  "symbol": "AAPL",
  "intervals": ["1m", "5m", "1h", "1D"],
  "lookback": 300
}
```

Each interval is normalised using the same rules as `/gpt/context`. `lookback`
defaults to 300 bars when omitted.

### Response

```jsonc
{
  "symbol": "AAPL",
  "snapshots": [
    {
      "interval": "1",
      "requested": "1m",
      "lookback": 300,
      "cached": false,
      "bars": [ {"time": "…", "open": 175.3, …} ],
      "key_levels": { "session_high": 176.4, … },
      "snapshot": { /* same shape as single-context snapshot */ },
      "indicators": {
        "ema9": [ {"time": "…", "value": 175.28}, … ],
        "ema20": […],
        "vwap": […],
        "atr14": […],
        "adx14": […]
      }
    },
    { "interval": "60", "requested": "1h", … }
  ],
  "contexts": [ /* deprecated alias, mirrors snapshots for older clients */ ],
  "volatility_regime": {
    "timestamp": "2025-10-08T19:40:05.123456Z",
    "iv_atm": 0.37,
    "iv_rank": 62.4,
    "iv_percentile": 68.5,
    "hv_20": 0.29,
    "hv_60": 0.32,
    "hv_120": 0.34,
    "hv_20_percentile": 54.1,
    "iv_to_hv_ratio": 1.28,
    "skew_25d": -0.08
  },
  "sentiment": {
    "symbol_sentiment": 0.14,
    "news_count_24h": 12,
    "news_bias_24h": 0.08,
    "headline_risk": "normal"
  },
  "events": {
    "next_fomc_minutes": 4320,
    "next_cpi_minutes": 120,
    "next_nfp_minutes": null,
    "within_event_window": true,
    "label": "watch"
  },
  "earnings": {
    "next_earnings_at": "2025-10-28T20:00:00+00:00",
    "dte_to_earnings": 19.0,
    "pre_or_post": "post",
    "earnings_flag": "near"
  }
}
```

Intervals pulled from cache include `"cached": true`. IV metrics fall back to
`null` when the chain/price data is unavailable.
`sentiment`, `events`, and `earnings` are populated when the enrichment sidecar
(`enrich_service.py`) is running with a valid `FINNHUB_API_KEY`. They are `null`
when the sidecar is disabled or the upstream APIs fail temporarily.

Legacy note: the response still includes a `contexts` array (mirroring
`snapshots`) so existing GPT tooling keeps working. New integrations should
read from `snapshots`.

## `/gpt/futures-snapshot`

```
GET /gpt/futures-snapshot
```

Returns Finnhub-backed ETF proxies for the major index futures plus VIX. The
endpoint now lives on the primary FastAPI service (no sidecar round-trip).

```jsonc
{
  "es_proxy": { "symbol": "SPY", "last": 565.2, "percent": -0.003, "time_utc": "…", "stale": false },
  "nq_proxy": { "symbol": "QQQ", … },
  "ym_proxy": { "symbol": "DIA", … },
  "rty_proxy": { "symbol": "IWM", … },
  "vix": { "symbol": "CBOE:VIX", … },
  "market_phase": "afterhours",
  "stale_seconds": 0
}
```

Responses are cached for three minutes. A missing `FINNHUB_API_KEY` raises a
`503` with `{ "code": "UNAVAILABLE", "message": "FINNHUB_API_KEY missing" }`.

## Suggested GPT system prompt

Embed the following guidance in your GPT action configuration:

```
You are a trading assistant. Always:
1. Start with POST /gpt/scan (provide 3–6 tickers and optional style hints).
2. Use the returned market_snapshot + key_levels to decide if a setup is valid.
   - Evaluate momentum (EMA stack, ADX), volatility (ATR, squeeze state),
     session phase, and gap context.
   - Derive stop-loss and take-profit levels using those metrics. Target at
     least 0.8 R:R unless the user requests otherwise.
   - Review option liquidity, spreads, and greeks via the `options.best`
     bundle (Polygon) or `contract_suggestion` fallback if Polygon is absent.
   - Inspect `context_overlays` for supply/demand zones, liquidity pools, FVGs,
     internals, and event hooks before locking the plan.
   - Estimate whether the target is achievable within the expected_move_horizon.
3. When you need deeper context, GET /gpt/context/{symbol} using the provided
   data.bars URL.
4. Present the play with entry, stop, PT1/PT2, holding expectations, and a
   rationale. Mention the metrics you used.
5. When you need a chart link, call POST /gpt/chart-url with charts.params plus
   your chosen entry/stop/targets and any key `levels`, then share the returned URL.
6. Re-check setup validity whenever the timestamp or session phase changes.

Never invent fills or executions. Always remind the user to manage risk and
confirm levels against live price action.
```

Feel free to tailor the tone/wording, but keep the sequencing: **scan → analyse
metrics → compute plan → optional chart → deliver recommendation**.
