# Strategy Library and Indicator Calculations

This document summarises every predefined trading strategy in the starter kit and the core technical indicators used by the market scanner.  Each strategy lists its category, entry logic, contract selection rules and risk‑management guidelines.  The calculations section explains how key indicators are computed with citations to authoritative sources.

> **Disclaimer**
>
> The strategies outlined here are for educational purposes only and do **not** constitute financial advice.  Historical win‑rate targets are aspirational and must be validated against your own backtests before trading live.

## Indicator calculations

### Average True Range (ATR)

The **Average True Range** measures volatility by averaging the *true range* over a lookback period.  The true range for each bar is the maximum of:

* the current high minus the current low;
* the absolute value of the current high minus the previous close; and
* the absolute value of the current low minus the previous close【617695039259081†L287-L333】.

To calculate ATR, compute the true range for each period and then take a moving average (often a 14‑period average) of those values【617695039259081†L287-L333】.  Shorter lookbacks make the ATR more sensitive; longer periods smooth out volatility swings.

### Volume‑Weighted Average Price (VWAP)

**VWAP** is the ratio of the total value traded to the total volume traded over a session.  First compute the *typical price* for each bar: `(high + low + close) / 3`.  Multiply this by volume to get PV (price × volume).  The VWAP at a given time is the cumulative sum of PV divided by the cumulative volume【893378759195020†L323-L343】.  Traders often anchor VWAP to significant events (earnings, previous day high/low) to create **anchored VWAPs (AVWAP)**.

### Exponential Moving Average (EMA)

An EMA is a moving average that weights recent prices more heavily.  It is calculated by applying a smoothing multiplier to the current price and the previous EMA.  In code we use `pandas.Series.ewm(span=n, adjust=False).mean()` to compute an EMA of period `n`.

### Average Directional Index (ADX)

The **ADX** quantifies trend strength by combining positive and negative directional indicators.  The positive directional indicator (+DI) equals 100 times the exponential moving average of positive directional movement (+DM) divided by the ATR; the negative indicator (–DI) is computed analogously.  The ADX itself is 100 times the exponential moving average of the absolute difference between +DI and –DI divided by their sum【498039660128542†L325-L346】.  Higher ADX values indicate a stronger trend.

### Bollinger Bands

Bollinger Bands consist of a middle line (a simple or exponential moving average) and two bands placed a certain number of standard deviations above and below the middle line.  A common setting uses a 20‑period SMA and bands two standard deviations away.  The **upper band** is the SMA plus two standard deviations; the **lower band** is the SMA minus two standard deviations【894760941611558†L390-L399】.  Bands widen when volatility increases and contract when it decreases【894760941611558†L398-L399】.

### Keltner Channels

Keltner Channels are volatility bands around an exponential moving average.  A typical formula uses a 20‑period EMA for the midline and offsets it by a multiple of the Average True Range.  The **upper channel** equals `EMA + 2 × ATR` and the **lower channel** equals `EMA – 2 × ATR`【109929969071151†L330-L341】.  Traders sometimes adjust the ATR multiplier (e.g. 1.5 or 2.5) to tighten or widen the channel【109929969071151†L340-L341】.

## Strategy summary

The table below summarises the strategies defined in `src/strategy_library.py`.  Each entry lists the category (scalp, swing, **leaps**, index or generic), the core idea, triggering conditions, option selection rules and high‑level risk management.  See the code for exact parameter ranges.

| ID | Category | Idea | Trigger highlights | Option rules | Stops & take‑profit | Target win rate |
|---|---|---|---|---|---|---|
| **orb_retest** | Scalp | Opening range breakout with retest.  After the first 5–15 min, price breaks the opening range, retests it and reclaims with volume. | Close above/below opening range; volume ≥ 1.5× 20‑bar median; EMAs aligned; ADX > 18; VWAP support/resistance. | 0–2 DTE, δ 0.50–0.65, spread ≤ 5%, OI ≥ 500, volume ≥ 250. | Stop beyond retest candle; take profit at 0.75–1× ATR; trail remainder with chandelier stop. | ~58% |
| **vwap_avwap** | Scalp | Session VWAP with anchored VWAP cluster continuation. | Price closes above session VWAP and at least two AVWAPs; distance to AVWAP cluster narrowing; rising volume. | 0–3 DTE, δ 0.45–0.60, spread ≤ 5%, OI ≥ 500, volume ≥ 250. | Stop beyond AVWAP cluster; first target 1–1.5× ATR; scale out and trail. | ~56% |
| **liquidity_sweep** | Scalp | Fade liquidity sweeps (stop runs). | Large wick piercing a swing level and closing back inside; volume spike Z‑score > 2; RSI crosses back above/below 35/65. | 0–2 DTE, δ 0.35–0.50, spread ≤ 6%, OI ≥ 300, volume ≥ 200. | Stop beyond wick; target VWAP then range midpoint. | ~54% |
| **inside_breakout** | Scalp | Inside bar (NR7) breakout. | 5‑minute inside bar or narrowest of last 7; rising ADX; breakout with volume. | 0–3 DTE, δ 0.45–0.55, spread ≤ 5%, OI ≥ 500, volume ≥ 250. | Stop on opposite side of inside bar; target 1.2–1.8× ATR. | ~55% |
| **ema_pullback** | Swing | Trend pullback to the 20‑EMA with 50/200 EMA alignment. | EMAs stacked bullish/bearish; pullback to 20‑EMA on declining volume; stochastic RSI turns with trend. | 20–45 DTE debit spread, δ 0.30–0.40, spread ≤ 5%, OI ≥ 1000, volume ≥ 500. | Stop below pullback swing; target 1.5–2.5× daily ATR; vertical spread reduces theta. | ~52% |
| **earnings_drift** | Swing | Post‑earnings drift with IV crush. | Earnings gap in direction of surprise; price holds above/below AVWAP anchored to earnings; news sentiment confirms. | 15–30 DTE debit or calendar, δ 0.30–0.50, spread ≤ 5%, OI ≥ 1000, volume ≥ 500. | Stop at low/high of earnings candle; target 1.6× ATR; calendars capture IV reversion. | ~55% |
| **gap_fill_breakaway** | Swing | Distinguish exhaustion vs breakaway gaps. | Gap into higher‑time‑frame supply/demand or away from it; volume/AVWAP behaviour signals fill vs trend. | 7–21 DTE, δ 0.35–0.50, spread ≤ 5%, OI ≥ 800, volume ≥ 400. | Stop depends on gap type; target 1.4–2× ATR. | ~55% |
| **high_iv_rank** | Swing | High implied‑volatility rank credit spreads. | IV rank > 50; no major catalysts within two weeks; price not at extreme trend. | 30–45 DTE credit spread, δ 0.25–0.30, width ≈ 2 strikes, OI ≥ 1500, volume ≥ 800. | Risk capped at width minus credit; take profits at 50–60% of credit. | ~65% |
| **pmcc** | Leap | Poor Man’s Covered Call (LEAPS + short calls). | Long‑term bullish bias; short‑term overbought to sell call; IV term structure steep. | 180–365 DTE deep ITM call (δ 0.70–0.85) + short 7–14 DTE calls; OI ≥ 1000, volume ≥ 500. | Use LEAPS as core; roll or buy back short calls at 50% max gain or delta > 0.80; take profits when trend exhausts. | ~60% |
| **gex_flip** | Index | Gamma exposure flip on SPX/NDX. | Price crosses zero‑gamma level computed from net gamma exposure; rising futures volume; news sentiment supports move. | 0–5 DTE, δ 0.40–0.60, OI ≥ 2000, volume ≥ 1000. | Tight stop (0.5× ATR) intraday; trend days above zero‑gamma target 1.5× ATR; mean‑reversion days fade at VWAP. | ~56% |
| **vix_gate** | Index | VWAP with VIX regime gating. | For VIX below threshold: look for breakouts above VWAP; for VIX above threshold: fade moves back toward VWAP. | 0–2 DTE, δ 0.45–0.60, OI ≥ 1500, volume ≥ 800. | Breakouts: stop below VWAP; fades: stop above session high/low; target 1.3–1.8× ATR. | ~57% |
| **inside_day_rth** | Index | Inside‑day then RTH expansion. | Daily range is inside the previous day’s range; regular session break aligns with higher‑time‑frame bias; volume confirms. | 0–3 DTE, δ 0.45–0.60, OI ≥ 1500, volume ≥ 800. | Stop on opposite side of daily inside bar; target 1.5× ATR or key level. | ~55% |
| **break_and_retest** | Generic | Generic break & retest pattern for any timeframe. | Break of key level with conviction; retest without closing beyond; volume confirms. | 0–10 DTE, δ 0.40–0.60, OI ≥ 300, volume ≥ 200. | Stop beyond retest low/high; target measured move from range plus ATR/pivots. | – |
| **momentum_play** | Generic | Trade strong momentum in the direction of widening range and increasing volume. | Consecutive higher highs or lower lows; rising volume; short‑term EMAs align with direction. | 0–5 DTE, δ 0.50–0.65, OI ≥ 500, volume ≥ 300. | Dynamic stop using ATR or supertrend; scale out as momentum slows or at key levels. | – |

### Notes

* **DTE** stands for “days to expiration.”  Short expirations (0–2 DTE) are used for scalps and intraday plays, whereas swings and LEAPS use longer maturities.  Short‑dated contracts are more responsive but decay faster.
* **Δ (delta)** measures an option’s sensitivity to the underlying price.  Higher deltas provide more directional exposure; lower deltas reduce directional risk but increase the impact of volatility and time decay.
* **OI/volume filters** ensure liquidity so that traders can enter and exit positions without excessive slippage.  Spread percentages limit how wide the bid–ask spread can be relative to the option price.
* The **win‑rate target** column is a historical benchmark used to rank signals.  Actual performance varies by symbol, market regime and execution.

## Using these strategies

These strategy definitions are consumed by the scanner and follower services.  The scanner computes indicator values (ATR, EMA, VWAP, ADX, etc.) from live and historical data, applies the triggers and option rules, and produces a ranked feed of signals.  The follower monitors open trades, updates stops and targets based on ATR/volatility, and streams coaching instructions back to the user.

For details on implementing the scanner and follower, refer to `src/scanner.py` and `src/follower.py`.  To modify a strategy or add your own, edit `src/strategy_library.py`—the rest of the system will automatically pick up the changes.
