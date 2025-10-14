# FT-Backtest (Historical Pattern Edge)

**Purpose:** Attach deterministic statistics for recurring setups. These statistics inform the assistant’s commentary and risk framing.

## Pattern ID

Create a stable hash using:

- `symbol`
- `style`
- `direction`
- `entry` type (`break`, `retest`, `reclaim`, `reject`, `limit`)
- HTF bias bucket (aligned / opposed / neutral)
- Regime bucket (`low`, `normal`, `elevated`, `extreme`)

Serialise the key as `k=v|k=v` (sorted keys), then SHA-1 → `SYMBOL|hash[:16]`.

## Data Hygiene

- Filter out incomplete trades (missing stop/target fills).
- Wins/losses measured in **R** (target distance divided by stop distance).
- Duration measured in minutes from entry to exit.
- Minimum sample size: 50. If below threshold, omit the block.

## Stored Fields

```json
"historical_stats": {
  "pattern_id": "AAPL|9a4d1e4f7c83a6b1",
  "sample_size": 186,
  "win_rate": 0.64,
  "avg_r_multiple": 1.87,
  "avg_duration": "36m"
}
```

Values are cached in-memory (and optionally persisted) for deterministic responses. Offline refresh jobs populate the cache using the same calculation path.

## GPT Guidance

- Reference `sample_size` to contextualise statistical weight.
- Couple `win_rate` with `avg_r_multiple` when discussing expectancy.
- Do **not** extrapolate beyond the supplied fields.
