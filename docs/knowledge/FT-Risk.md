# FT-Risk (Expected Value • Kelly • MFE)

**Purpose:** Quantify risk/reward quality for each setup using deterministic calculations. Returned as the `risk_model` block in API responses.

## Inputs

- Entry price, stop, target ladder (server-side).
- Probabilities per target (`probabilities.tp1`, `probabilities.tp2`, …).
- ATR snapshot (`atr_used`) and expected move cap (`em_used` when available).
- Trade style (`scalp`, `intraday`, `swing`, `leap`) for MFE heuristics.

## Formulas

### Expected Value (expressed in R)

```
RR_stop = abs(entry - stop)
R_i = abs(target_i - entry) / RR_stop
EV = Σ_i (p_hit_i - p_hit_{i-1}) * R_i  -  (1 - p_hit_max)
```

Where `p_hit_0` is 0. Cap targets using expected move logic before evaluation.

### Kelly Fraction (scaled)

```
kelly_raw = (p_win * R - (1 - p_win)) / R
kelly = clamp(kelly_raw * 0.5, 0, 1)
```

Use `p_win = probabilities.tp1` and `R = average(R_i)` above 1e-6.

### MFE Projection

Heuristic labels by style:

- `scalp` → ≈1.2× ATR
- `intraday` → ≈1.6× ATR
- `swing` → ≈2.4× ATR
- `leap` → ≈3.0× ATR

Append “before reversal” when ATR is available; otherwise include “heuristic”.

## Example Payload

```json
"risk_model": {
  "expected_value_r": 0.42,
  "kelly_fraction": 0.23,
  "mfe_projection": "≈1.9× ATR before reversal"
}
```

## GPT Guidance

- Use `expected_value_r` to explain expectancy.
- Treat `kelly_fraction` as **position sizing guidance**, not a command.
- Reference the MFE label when describing runner or scaling expectations.
