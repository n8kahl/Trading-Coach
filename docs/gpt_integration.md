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
    "contract_suggestion": { /* optional Tradier contract meta */ },
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
    }
  }
]
```

### Key points for the agent

- **No server-generated stops/targets.** Use `market_snapshot`, `key_levels`,
  and `/gpt/context` to compute your own playbook.
- **`direction_hint` is advisory.** It reflects indicator alignment; override
  it if your analysis disagrees.
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
  "snapshot": { /* same structure as scan payload */ }
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

