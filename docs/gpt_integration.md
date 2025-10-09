# GPT Integration Cheat Sheet

This backend now serves raw market context so that your GPT agent can make the
final trading decisions (entries, stops, targets, position sizing). The server
focuses on data prep; the agent performs the higher-level reasoning.

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
      "direction_bias": "long"
    },
    "charts": {
      "interactive": "https://host/charts/html?symbol=AAPL&interval=1m&ema=9,20,50&strategy=power_hour_trend"
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

- **No server-generated stops/targets.** Use `market_snapshot`, `key_levels`,
  and `/gpt/context` to compute your own playbook.
- **`direction_hint` is advisory.** It reflects indicator alignment; override
  it if your analysis disagrees.
- **`options` bundle** surfaces Polygon's filtered option chain (best contract
  plus a few alternates) when `POLYGON_API_KEY` is configured. If Polygon is
  unavailable the field is omitted and `contract_suggestion` may fall back to
  Tradier metadata.
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

## `/gpt/context/{symbol}`

Example response:

```jsonc
{
  "symbol": "AAPL",
  "interval": "1m",
  "lookback": 300,
  "bars": [
    {"time": "2025-10-08T14:02:00+00:00", "open": 257.4, "high": 257.6, "low": 257.2, "close": 257.5, "volume": 820000},
    …
  ],
  "indicators": {
    "ema9": [{"time": "2025-10-08T14:02:00+00:00", "value": 257.48}, …],
    "ema20": […],
    "ema50": […],
    "vwap": […],
    "atr14": […],
    "adx14": […]
  },
  "key_levels": { "session_high": 261.4, … },
  "snapshot": { /* same structure as scan payload */ },
  "supply_zones": [],
  "demand_zones": [],
  "liquidity_pools": [],
  "fvg": [],
  "rel_strength_vs": { "benchmark": "SPY", "lookback_bars": 15, "value": null },
  "internals": { "adv_dec": null, "tick": null, "sector_perf": {}, "index_bias": null },
  "options_summary": {
    "atm_iv": null,
    "iv_rank": null,
    "iv_pct": null,
    "skew_25d": null,
    "term_slope": null,
    "spread_bps": null
  },
  "liquidity_metrics": {
    "avg_spread_bps": null,
    "typical_slippage_bps": null,
    "lot_size_hint": null
  },
  "events": [],
  "avwap": {
    "from_open": null,
    "from_prev_close": null,
    "from_session_low": null,
    "from_session_high": null
  },
  "volume_profile": {
    "vwap": null,
    "vah": null,
    "val": null,
    "poc": null
  },
  "options": { /* populated when Polygon chains are available, same structure as scan */ },
  "context_overlays": { /* same as the fields above for convenience */ }
}
```

Use this when you need raw bars for custom indicator calculations, measuring
momentum, or validating expected time-to-target.

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
5. If you want to render a chart, call /charts/html and pass your chosen entry
   and targets as query parameters.
6. Re-check setup validity whenever the timestamp or session phase changes.

Never invent fills or executions. Always remind the user to manage risk and
confirm levels against live price action.
```

Feel free to tailor the tone/wording, but keep the sequencing: **scan → analyse
metrics → compute plan → optional chart → deliver recommendation**.
