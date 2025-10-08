"""Chart rendering endpoints for interactive HTML and static PNG outputs."""

from __future__ import annotations

import html
import io
import json
import math
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlencode

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/charts", tags=["charts"])

INTERVAL_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "d": "1D"}
INTERVAL_ALIASES = {
    "1": "1m",
    "1m": "1m",
    "1min": "1m",
    "5": "5m",
    "5m": "5m",
    "5min": "5m",
    "15": "15m",
    "15m": "15m",
    "15min": "15m",
    "60": "1h",
    "1h": "1h",
    "60m": "1h",
    "1hr": "1h",
    "h": "1h",
    "d": "d",
    "1d": "d",
    "day": "d",
    "daily": "d",
}
MAX_LOOKBACK = 360
MAX_TPS = 5
MAX_EMAS = 5


def normalize_interval(interval: str) -> str:
    """Return canonical interval tokens like '5m' or raise ValueError."""
    key = (interval or "").strip().lower()
    if not key:
        raise ValueError("Interval is required.")
    normalized = INTERVAL_ALIASES.get(key, key)
    if normalized not in INTERVAL_FREQ:
        raise ValueError(f"Unsupported interval '{interval}'")
    return normalized


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def get_candles(symbol: str, interval: str, lookback: int = 300) -> pd.DataFrame:
    """Return stub candle data (replace with Polygon/DB fetch later)."""

    normalized = normalize_interval(interval)

    lookback = int(max(20, min(lookback, MAX_LOOKBACK)))
    freq = INTERVAL_FREQ[normalized]
    now = pd.Timestamp.utcnow().ceil("min")
    idx = pd.date_range(end=now, periods=lookback, freq=freq)
    base = 430 + np.random.uniform(-3, 3)
    random_walk = np.cumsum(np.random.normal(0, 0.25, size=lookback))
    prices = base + random_walk
    high = prices + np.random.uniform(0.05, 0.35, size=lookback)
    low = prices - np.random.uniform(0.05, 0.35, size=lookback)
    open_ = prices + np.random.uniform(-0.15, 0.15, size=lookback)
    close = prices + np.random.uniform(-0.15, 0.15, size=lookback)
    volume = np.random.randint(100_000, 600_000, size=lookback)
    df = pd.DataFrame(
        {"time": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    return df


def parse_floats(csv: Optional[str]) -> List[float]:
    if not csv:
        return []
    values: List[float] = []
    for chunk in csv.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values


def parse_ints(csv: Optional[str]) -> List[int]:
    if not csv:
        return []
    spans: List[int] = []
    for chunk in csv.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            val = int(chunk)
        except ValueError:
            continue
        if 1 <= val <= 400:
            spans.append(val)
    return sorted(set(spans))


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _price_band(df: pd.DataFrame) -> tuple[float, float]:
    lo = float(df["low"].min())
    hi = float(df["high"].max())
    span = max(hi - lo, max(abs(lo), abs(hi)) * 0.02 or 1.0)
    return lo - span * 0.5, hi + span * 0.5


# --------------------------------------------------------------------------- #
# Interactive HTML renderer
# --------------------------------------------------------------------------- #


@router.get("/html", response_class=HTMLResponse)
def chart_html(
    symbol: str,
    interval: str = Query("1m"),
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    tp: Optional[str] = None,
    ema: Optional[str] = "9,21",
    title: Optional[str] = None,
    lookback: int = Query(300, ge=50, le=MAX_LOOKBACK),
    direction: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    atr: Optional[float] = Query(None),
    risk_reward: Optional[float] = Query(None),
) -> HTMLResponse:
    try:
        interval_normalized = normalize_interval(interval)
        df = get_candles(symbol, interval_normalized, lookback=lookback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if df.empty:
        raise HTTPException(status_code=500, detail="No candle data available.")

    tps = parse_floats(tp)[:MAX_TPS]
    emas = parse_ints(ema)[:MAX_EMAS] or [9, 21]

    price_lo, price_hi = _price_band(df)

    def _clamp_or_none(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        return round(clamp(float(val), price_lo, price_hi), 4)

    entry_val = _clamp_or_none(entry)
    stop_val = _clamp_or_none(stop)
    tp_vals = [_clamp_or_none(v) for v in tps if v is not None]

    candles = []
    volumes = []
    for _, row in df.iterrows():
        ts = int(pd.Timestamp(row["time"]).timestamp())
        candles.append(
            {
                "time": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
        volumes.append(
            {
                "time": ts,
                "value": float(row["volume"]),
                "color": "#22c55e" if row["close"] >= row["open"] else "#ef4444",
            }
        )

    ema_series: Dict[int, List[Dict[str, float]]] = {}
    for span in emas:
        ema_values = compute_ema(df["close"], span)
        ema_series[span] = [
            {"time": int(pd.Timestamp(ts).timestamp()), "value": float(val)}
            for ts, val in zip(df["time"], ema_values)
            if not math.isnan(val)
        ]

    levels = []
    if entry_val is not None:
        levels.append({"value": entry_val, "label": "Entry", "color": "#e5e54b"})
    if stop_val is not None:
        levels.append({"value": stop_val, "label": "Stop", "color": "#e74c3c"})
    for idx, val in enumerate(tp_vals):
        if val is not None:
            levels.append({"value": val, "label": f"TP{idx+1}", "color": "#2ecc71"})

    plan_info = {
        "entry": entry_val,
        "stop": stop_val,
        "tps": tp_vals,
        "direction": (direction or "").lower() or None,
        "strategy": strategy,
        "atr": float(atr) if atr is not None else None,
        "risk_reward": float(risk_reward) if risk_reward is not None else None,
    }

    payload = {
        "symbol": symbol.upper(),
        "candles": candles,
        "volume": volumes,
        "ema_series": ema_series,
        "levels": levels,
        "title": title or f"{symbol.upper()} {interval_normalized}",
        "interval": interval_normalized,
        "plan": plan_info,
    }

    safe_title = html.escape(payload["title"])
    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{safe_title}</title>
    <style>
      :root {{
        color-scheme: dark;
      }}
      html, body {{
        height: 100%;
      }}
      body {{
        margin: 0;
        background: #0b0f14;
        color: #e6edf3;
        font-family: 'Inter', 'Segoe UI', sans-serif;
      }}
      #chart-container {{
        position: relative;
        height: 100vh;
        width: 100vw;
      }}
      .legend {{
        position: absolute;
        top: 16px;
        left: 16px;
        background: rgba(11, 15, 20, 0.88);
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
      }}
      .legend h1 {{
        margin: 0 0 8px;
        font-size: 20px;
      }}
      .legend p {{
        margin: 4px 0;
        font-size: 13px;
        opacity: 0.85;
      }}
      .legend .plan-grid {{
        margin-top: 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 12px;
      }}
      .legend .plan-grid span {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
      }}
      .legend .plan-grid .label {{
        opacity: 0.7;
        font-weight: 600;
      }}
      .badge-group {{
        margin-top: 10px;
        display: flex;
        flex-wrap: wrap;
      }}
      .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 6px;
        margin-top: 4px;
        background: rgba(255, 255, 255, 0.08);
      }}
    </style>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
  </head>
  <body>
    <div id="chart-container"></div>
    <div class="legend" id="legend"></div>
    <script>
      window.addEventListener('error', (event) => {{
        const existing = document.querySelector('.chart-error');
        if (existing) return;
        const div = document.createElement('div');
        div.className = 'chart-error';
        Object.assign(div.style, {{
          position: 'fixed',
          bottom: '16px',
          right: '16px',
          background: 'rgba(239, 68, 68, 0.92)',
          color: '#fff',
          padding: '10px 14px',
          borderRadius: '8px',
          fontFamily: 'monospace',
          fontSize: '12px',
          maxWidth: '320px',
          zIndex: '1000',
        }});
        div.textContent = 'Chart error: ' + (event?.message || 'Unknown');
        document.body.appendChild(div);
      }}, {{ once: true }});

      const payload = {json.dumps(payload)};
      if (!window.LightweightCharts) {{
        document.body.innerHTML = '<p style="padding:16px;color:#f87171">Failed to load chart library.</p>';
      }} else {{
        const container = document.getElementById('chart-container');
        console.info('Chart payload', {{
          candles: payload.candles.length,
          volume: payload.volume.length,
          emaSeries: Object.keys(payload.ema_series || {{}}).length,
          levels: (payload.levels || []).length,
        }});
        const initialWidth = container.clientWidth || window.innerWidth || 1200;
        const initialHeight = container.clientHeight || window.innerHeight || 700;
        if (!payload.candles.length) {{
          container.innerHTML = '<p style="padding:16px;color:#f97316;font-family:monospace;">No candle data available.</p>';
        }} else {{
          const chart = LightweightCharts.createChart(container, {{
            width: initialWidth,
            height: initialHeight,
            layout: {{
              background: {{ type: 'Solid', color: '#0b0f14' }},
              textColor: '#e6edf3',
            }},
            rightPriceScale: {{ borderColor: 'rgba(46, 51, 64, 0.6)' }},
            timeScale: {{ borderColor: 'rgba(46, 51, 64, 0.6)' }},
            grid: {{
              vertLines: {{ color: 'rgba(35, 43, 60, 0.7)' }},
              horzLines: {{ color: 'rgba(35, 43, 60, 0.7)' }},
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
          }});

          const resize = () => {{
            const width = container.clientWidth || window.innerWidth || initialWidth;
            const height = container.clientHeight || window.innerHeight || initialHeight;
            chart.resize(width, height, false);
          }};
          window.addEventListener('resize', resize);
          resize();

          const candleSeries = chart.addCandlestickSeries({{
            upColor: '#22c55e',
            downColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
            borderVisible: false,
          }});
          candleSeries.setData(payload.candles);

          chart.priceScale('right').applyOptions({{
            scaleMargins: {{ top: 0.1, bottom: 0.3 }},
          }});

          const volumeSeries = chart.addHistogramSeries({{
            priceScaleId: 'volume',
            priceFormat: {{ type: 'volume' }},
            priceLineVisible: false,
            lastValueVisible: false,
          }});
          chart.priceScale('volume').applyOptions({{
            scaleMargins: {{ top: 0.8, bottom: 0 }},
            visible: false,
          }});
          volumeSeries.setData(payload.volume);

          const emaColors = ['#f39c12', '#00bcd4', '#9b59b6', '#cddc39', '#ff6f61'];
          Object.entries(payload.ema_series || {{}}).forEach(([span, data], idx) => {{
            if (!data.length) return;
            const color = emaColors[idx % emaColors.length];
            const line = chart.addLineSeries({{
              color,
              lineWidth: 2,
              priceLineVisible: false,
            }});
            line.setData(data);
            line.applyOptions({{
              lastValueVisible: true,
              priceLineVisible: true,
              priceLineColor: color,
              priceLineWidth: 1,
            }});
            const lastPoint = data[data.length - 1];
            if (lastPoint) {{
              line.createPriceLine({{
                price: lastPoint.value,
                color,
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: `EMA${{span}}`,
              }});
            }}
          }});

          (payload.levels || []).forEach(level => {{
            candleSeries.createPriceLine({{
              price: level.value,
              color: level.color,
              lineWidth: 2,
              lineStyle: LightweightCharts.LineStyle.Solid,
              axisLabelVisible: true,
              title: level.label,
            }});
          }});

          chart.timeScale().fitContent();

          const legend = document.getElementById('legend');
          const plan = payload.plan || {{}};
          const formatPrice = value => (typeof value === 'number' && isFinite(value) ? value.toFixed(2) : '—');
          const planRows = [];
          if (typeof plan.entry === 'number') planRows.push(['Entry', formatPrice(plan.entry)]);
          if (typeof plan.stop === 'number') planRows.push(['Stop', formatPrice(plan.stop)]);
          (plan.tps || []).forEach((val, idx) => {{
            if (typeof val === 'number' && isFinite(val)) {{
              planRows.push([`TP${{idx + 1}}`, formatPrice(val)]);
            }}
          }});
          if (typeof plan.atr === 'number' && isFinite(plan.atr)) {{
            planRows.push(['ATR(14)', Number(plan.atr).toFixed(2)]);
          }}
          if (typeof plan.risk_reward === 'number' && isFinite(plan.risk_reward)) {{
            planRows.push(['R:R', Number(plan.risk_reward).toFixed(2)]);
          }}
          const strategyLabel = [plan.direction, plan.strategy]
            .filter(Boolean)
            .map(token => String(token).toUpperCase())
            .join(' · ');
          const levelBadges = (payload.levels || []).map(level => `
            <span class="badge" style="background:${{level.color}}22;color:${{level.color}}">${{level.label}}</span>
          `).join('');
          const indicatorBadges = Object.keys(payload.ema_series || {{}}).map(span => `
            <span class="badge">EMA${{span}}</span>
          `).join('');
          legend.innerHTML = `
            <h1>${{payload.symbol}} · ${{payload.interval.toUpperCase()}}</h1>
            ${{strategyLabel ? `<p>${{strategyLabel}}</p>` : ''}}
            ${{planRows.length ? `<div class="plan-grid">${{planRows.map(([label, value]) => `<span><strong class="label">${{label}}</strong><span>${{value}}</span></span>`).join('')}}</div>` : ''}}
            <div class="badge-group">
              ${{levelBadges}}${{indicatorBadges}}
            </div>
          `;
        }}
      }}
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html_doc)


# --------------------------------------------------------------------------- #
# Static PNG renderer
# --------------------------------------------------------------------------- #


@router.get("/png")
def chart_png(
    symbol: str,
    interval: str = Query("1m"),
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    tp: Optional[str] = None,
    ema: Optional[str] = "9,21",
    title: Optional[str] = None,
    lookback: int = Query(300, ge=50, le=MAX_LOOKBACK),
) -> Response:
    try:
        interval_normalized = normalize_interval(interval)
        df = get_candles(symbol, interval_normalized, lookback=lookback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if df.empty:
        raise HTTPException(status_code=500, detail="No candle data available.")

    tps = parse_floats(tp)[:MAX_TPS]
    emas = parse_ints(ema)[:MAX_EMAS] or [9, 21]

    for span in emas:
        df[f"EMA{span}"] = compute_ema(df["close"], span)

    df = df.copy()
    df["time_num"] = mdates.date2num(pd.to_datetime(df["time"]))

    width = 12
    height = 6
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(width, height), dpi=160, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    bar_width = _candle_width(interval_normalized)

    for _, row in df.iterrows():
        color = "#22c55e" if row["close"] >= row["open"] else "#ef4444"
        ax_price.plot([row["time_num"], row["time_num"]], [row["low"], row["high"]], color=color, linewidth=1.0)
        rect = Rectangle(
            (row["time_num"] - bar_width / 2),
            min(row["open"], row["close"]),
            bar_width,
            abs(row["close"] - row["open"]) or bar_width / 10,
            facecolor=color,
            edgecolor=color,
            linewidth=0.6,
        )
        ax_price.add_patch(rect)

    for span in emas:
        ax_price.plot(df["time_num"], df[f"EMA{span}"], linewidth=1.2, label=f"EMA{span}")

    if entry is not None:
        ax_price.axhline(entry, color="#e5e54b", linestyle="--", linewidth=1.2, label="Entry")
    if stop is not None:
        ax_price.axhline(stop, color="#e74c3c", linestyle="--", linewidth=1.2, label="Stop")
    for idx, val in enumerate(tps):
        ax_price.axhline(val, color="#2ecc71", linestyle="--", linewidth=1.0, label="TP" if idx == 0 else None)

    ax_price.set_title(title or f"{symbol.upper()} {interval_normalized.upper()}", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=8, ncol=3)
    ax_price.grid(alpha=0.2)

    ax_vol.bar(df["time_num"], df["volume"], width=bar_width, color="#3e9cff", alpha=0.7)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(alpha=0.15)

    ax_vol.xaxis_date()
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)

    headers = {"Cache-Control": "public, max-age=60"}
    return Response(content=buffer.getvalue(), media_type="image/png", headers=headers)


def _candle_width(interval: str) -> float:
    """Return matplotlib-friendly candle width based on interval."""
    if interval == "1m":
        return 1 / (24 * 60) * 0.9
    if interval == "5m":
        return 5 / (24 * 60) * 0.9
    if interval == "15m":
        return 15 / (24 * 60) * 0.9
    if interval == "1h":
        return 1 / 24 * 0.9
    if interval == "d":
        return 1.0 * 0.8
    return 1 / (24 * 60) * 0.9


# --------------------------------------------------------------------------- #
# URL builder
# --------------------------------------------------------------------------- #


def build_chart_url(
    base: str,
    kind: str,
    symbol: str,
    *,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    tps: Optional[Sequence[float]] = None,
    emas: Optional[Sequence[int]] = None,
    interval: str = "1m",
    title: Optional[str] = None,
    renderer_params: Optional[Dict[str, str]] = None,
    lookback: Optional[int] = None,
) -> str:
    """Construct a URL for the chart endpoints with proper encoding."""

    if kind not in {"html", "png"}:
        raise ValueError("kind must be 'html' or 'png'")

    try:
        interval_normalized = normalize_interval(interval)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    params: Dict[str, str] = {"symbol": symbol, "interval": interval_normalized}
    if entry is not None:
        params["entry"] = f"{entry:.4f}"
    if stop is not None:
        params["stop"] = f"{stop:.4f}"
    if tps:
        params["tp"] = ",".join(f"{tp:.4f}" for tp in tps)
    if emas:
        params["ema"] = ",".join(str(int(e)) for e in emas)
    if title:
        params["title"] = title
    if lookback:
        params["lookback"] = str(int(lookback))
    if renderer_params:
        params.update(renderer_params)

    query = urlencode(params, doseq=False, safe=",")
    base = base.rstrip("/")
    return f"{base}/charts/{kind}?{query}"
