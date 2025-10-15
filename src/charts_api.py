"""Chart rendering endpoints for interactive HTML outputs."""

from __future__ import annotations

import io
import html
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlencode

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import httpx

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from .config import get_settings

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
MAX_LOOKBACK = 2400
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


def _polygon_range_params(interval: str, lookback: int, session_padding: int = 2) -> tuple[int, str, pd.Timestamp, pd.Timestamp]:
    """Calculate Polygon aggs request parameters covering the desired lookback.

    Returns multiplier, timespan, start timestamp, end timestamp.
    """
    interval = interval.lower()
    if interval == "d":
        multiplier, timespan = 1, "day"
        days_back = max(lookback + session_padding, 10)
    elif interval.endswith("h"):
        hours = max(int(interval.rstrip("h")), 1)
        multiplier, timespan = hours, "hour"
        total_hours = lookback * hours
        days_back = max(math.ceil(total_hours / 12) + session_padding, 5)
    elif interval.endswith("m"):
        minutes = max(int(interval.rstrip("m")), 1)
        multiplier, timespan = minutes, "minute"
        total_minutes = lookback * minutes
        # cover at least `lookback` bars plus two extra days to handle gaps
        days_back = max(math.ceil(total_minutes / (60 * 6)) + session_padding, 3)
    else:
        raise ValueError(f"Unsupported interval '{interval}' for Polygon aggregation.")

    now = pd.Timestamp.utcnow()
    end = now + pd.Timedelta(minutes=multiplier)
    start = now - pd.Timedelta(days=days_back)
    return multiplier, timespan, start.normalize(), end.normalize() + pd.Timedelta(days=1)


def _fetch_polygon_candles(symbol: str, interval: str, lookback: int) -> pd.DataFrame | None:
    """Fetch candles from Polygon.io; return None if credentials missing or no data."""
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return None

    multiplier, timespan, start, end = _polygon_range_params(interval, lookback)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{start.date()}/{end.date()}"
    params = {
        "adjusted": "true",
        "sort": "desc",
        "limit": max(lookback, 500),
        "apiKey": api_key,
    }
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
    except httpx.HTTPError:
        return None

    payload = resp.json()
    results = payload.get("results") or []
    if not results:
        return None

    frame = pd.DataFrame(results)
    if frame.empty:
        return None
    frame = frame.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.tail(lookback)
    return frame


def _fetch_yahoo_candles(symbol: str, interval: str) -> pd.DataFrame | None:
    """Fetch candles from Yahoo Finance as a secondary live data source."""
    interval_map = {
        "1m": ("5d", "1m"),
        "5m": ("5d", "5m"),
        "15m": ("1mo", "15m"),
        "1h": ("3mo", "60m"),
        "d": ("1y", "1d"),
    }
    range_span, yahoo_interval = interval_map.get(interval, ("5d", "5m"))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
    params = {"interval": yahoo_interval, "range": range_span, "includePrePost": "false"}
    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
    except httpx.HTTPError:
        return None

    payload = resp.json()
    try:
        chart = payload["chart"]
        if chart.get("error"):
            return None
        result = chart["result"][0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    timestamps = result.get("timestamp")
    if not timestamps:
        return None
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    ).dropna()
    if frame.empty:
        return None
    frame = frame.set_index("timestamp").sort_index()
    return frame


def _is_stale(frame: pd.DataFrame, interval: str) -> bool:
    """Return True if the latest candle timestamp is older than expected for the interval."""
    if frame.empty:
        return True
    last_ts = frame.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")
    now = pd.Timestamp.utcnow()
    age = now - last_ts
    if interval in {"1m", "5m", "15m", "1h"}:
        return age > pd.Timedelta(hours=36)
    return age > pd.Timedelta(days=10)


def get_candles(symbol: str, interval: str, lookback: int = 300) -> pd.DataFrame:
    """Return live OHLCV candles sourced from Polygon.io or Yahoo Finance."""
    normalized = normalize_interval(interval)
    lookback = int(max(20, min(lookback, MAX_LOOKBACK)))

    frame = _fetch_polygon_candles(symbol, normalized, lookback)
    if frame is None or _is_stale(frame, normalized):
        frame = _fetch_yahoo_candles(symbol, normalized)
    if frame is None or frame.empty:
        raise HTTPException(status_code=502, detail=f"No market data available for {symbol.upper()} ({normalized}).")

    frame = frame.sort_index().tail(lookback)
    frame = frame.reset_index().rename(columns={"timestamp": "time"})
    expected_cols = {"time", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(frame.columns)
    if missing:
        raise HTTPException(status_code=502, detail=f"Incomplete market data for {symbol.upper()}: missing {missing}")
    return frame


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
    tp2: Optional[str] = None,
    ema: Optional[str] = "9,21",
    title: Optional[str] = None,
    lookback: int = Query(600, ge=50, le=MAX_LOOKBACK),
    direction: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    atr: Optional[float] = Query(None),
    risk_reward: Optional[float] = Query(None),
    view: Optional[str] = Query("fit"),
    notes: Optional[str] = Query(None),
) -> HTMLResponse:
    try:
        interval_normalized = normalize_interval(interval)
        df = get_candles(symbol, interval_normalized, lookback=lookback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if df.empty:
        raise HTTPException(status_code=500, detail="No candle data available.")

    tp_primary = parse_floats(tp)[:MAX_TPS]
    tp_secondary = parse_floats(tp2)[:MAX_TPS]
    tps = tp_primary or []
    if tp_secondary:
        tps.extend(tp_secondary)
    tps = tps[:MAX_TPS]
    emas = parse_ints(ema)[:MAX_EMAS] or [9, 21]

    def _to_level(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        try:
            parsed = float(val)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return round(parsed, 4)

    entry_val = _to_level(entry)
    stop_val = _to_level(stop)
    tp_vals = [_to_level(v) for v in tps if v is not None]

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
        "notes": notes or None,
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
        "view": (view or "fit").lower(),
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
        position: relative;
        overflow: hidden;
      }}
      #chart-container {{
        position: absolute;
        inset: 0;
      }}
      .legend {{
        position: absolute;
        top: 16px;
        left: 16px;
        background: rgba(11, 15, 20, 0.88);
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        z-index: 20;
        max-width: min(320px, 30vw);
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
      #chart-toolbar {{
        position: absolute;
        top: 16px;
        right: 16px;
        display: flex;
        gap: 8px;
        z-index: 20;
      }}
      #chart-toolbar button {{
        background: rgba(15, 23, 42, 0.85);
        color: #e6edf3;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        cursor: pointer;
        transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
      }}
      #chart-toolbar button:hover {{
        background: rgba(37, 99, 235, 0.6);
      }}
      #chart-toolbar button.active {{
        background: rgba(59, 130, 246, 0.9);
        border-color: rgba(191, 219, 254, 0.9);
        color: #fff;
      }}
    </style>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
  </head>
  <body>
    <div id="chart-container"></div>
    <div class="legend" id="legend"></div>
    <div id="chart-toolbar"></div>
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
          const isIntraday = ['1m', '5m', '15m', '1h'].includes(payload.interval);
          const chart = LightweightCharts.createChart(container, {{
            width: initialWidth,
            height: initialHeight,
            layout: {{
              background: {{ type: 'Solid', color: '#0b0f14' }},
              textColor: '#e6edf3',
            }},
            rightPriceScale: {{ borderColor: 'rgba(46, 51, 64, 0.6)' }},
            timeScale: {{
              borderColor: 'rgba(46, 51, 64, 0.6)',
              timeVisible: true,
              secondsVisible: isIntraday && payload.interval !== '1h',
              fixRightEdge: false,
              rightOffset: 2,
            }},
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

          const priceValues = [
            ...payload.candles.map(candle => candle.low),
            ...payload.candles.map(candle => candle.high),
            ...(payload.levels || []).map(level => level.value),
          ].filter(value => typeof value === 'number' && isFinite(value));

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

          const addVWAP = candleData => {{
            if (!candleData.length) return;
            let volSum = 0;
            let priceVolSum = 0;
            const vwapPoints = [];
            candleData.forEach(candle => {{
              const volPoint = payload.volume.find(v => v.time === candle.time);
              const vol = volPoint ? Number(volPoint.value) : 0;
              const typicalPrice = (candle.high + candle.low + candle.close) / 3;
              volSum += vol;
              priceVolSum += typicalPrice * vol;
              if (volSum > 0) {{
                vwapPoints.push({{ time: candle.time, value: priceVolSum / volSum }});
              }}
            }});
            if (!vwapPoints.length) return;
            const vwapSeries = chart.addLineSeries({{
              color: '#ffffff',
              lineWidth: 2,
              lineStyle: LightweightCharts.LineStyle.Solid,
              priceLineVisible: true,
              lastValueVisible: true,
            }});
            vwapSeries.setData(vwapPoints);
            const lastVWAP = vwapPoints[vwapPoints.length - 1];
            if (lastVWAP) {{
              vwapSeries.createPriceLine({{
                price: lastVWAP.value,
                color: '#ffffff',
                lineWidth: 1,
                axisLabelVisible: true,
                title: 'VWAP',
              }});
            }}
          }};

          addVWAP(payload.candles);

          const firstTime = payload.candles[0]?.time;
          const lastTime = payload.candles[payload.candles.length - 1]?.time ?? firstTime;
          const totalBars = payload.candles.length;

          if (priceValues.length && firstTime != null && lastTime != null) {{
            const minPrice = Math.min(...priceValues);
            const maxPrice = Math.max(...priceValues);
            const padding = (maxPrice - minPrice) * 0.05 || 1.0;
            chart.priceScale('right').applyOptions({{
              autoScale: true,
              scaleMargins: {{ top: 0.1, bottom: 0.3 }},
            }});
            candleSeries.applyOptions({{
              autoscaleInfoProvider: () => {{
                return {{
                  priceRange: {{
                    minValue: minPrice - padding,
                    maxValue: maxPrice + padding,
                  }},
                }};
              }},
            }});
          }}

          const toolbar = document.getElementById('chart-toolbar');
          const VIEW_PRESETS = [
            {{ key: '30m', label: '30m', seconds: 30 * 60 }},
            {{ key: '1h', label: '1H', seconds: 60 * 60 }},
            {{ key: '4h', label: '4H', seconds: 4 * 60 * 60 }},
            {{ key: '1d', label: '1D', seconds: 24 * 60 * 60 }},
            {{ key: '5d', label: '5D', seconds: 5 * 24 * 60 * 60 }},
            {{ key: 'fit', label: 'All', type: 'fit' }},
          ];
          const VIEW_MAP = Object.fromEntries(VIEW_PRESETS.map(preset => [preset.key, preset]));
          const viewButtons = [];

          const setActiveButton = key => {{
            viewButtons.forEach(button => {{
              button.classList.toggle('active', button.dataset.key === key);
            }});
          }};

          const resolveView = token => {{
            const normalized = (token || '').trim().toLowerCase();
            if (!normalized) return {{ ...VIEW_MAP.fit, key: 'fit', baseKey: 'fit' }};
            if (VIEW_MAP[normalized]) return {{ ...VIEW_MAP[normalized], key: normalized, baseKey: normalized }};
            if (['fit', 'all', 'auto', 'full'].includes(normalized)) {{
              return {{ ...VIEW_MAP.fit, key: normalized, baseKey: 'fit' }};
            }}
            let match = normalized.match(/^bars:(\\d+)$/);
            if (match) {{
              const bars = Math.max(1, parseInt(match[1], 10));
              return {{ key: `bars:${{bars}}`, bars, baseKey: null }};
            }}
            match = normalized.match(/^(\\d+)([smhd])$/);
            if (match) {{
              const value = parseInt(match[1], 10);
              const unit = match[2];
              const factor = unit === 's' ? 1 : unit === 'm' ? 60 : unit === 'h' ? 3600 : 86400;
              return {{ key: normalized, seconds: value * factor, baseKey: null }};
            }}
            match = normalized.match(/^seconds:(\\d+)$/);
            if (match) {{
              return {{ key: normalized, seconds: parseInt(match[1], 10), baseKey: null }};
            }}
            match = normalized.match(/^minutes:(\\d+)$/);
            if (match) {{
              return {{ key: normalized, seconds: parseInt(match[1], 10) * 60, baseKey: null }};
            }}
            match = normalized.match(/^hours:(\\d+)$/);
            if (match) {{
              return {{ key: normalized, seconds: parseInt(match[1], 10) * 3600, baseKey: null }};
            }}
            match = normalized.match(/^days:(\\d+)$/);
            if (match) {{
              return {{ key: normalized, seconds: parseInt(match[1], 10) * 86400, baseKey: null }};
            }}
            return {{ ...VIEW_MAP.fit, key: 'fit', baseKey: 'fit' }};
          }};

          const applyView = token => {{
            const preset = resolveView(token);
            if (!preset) return;
            if (preset.baseKey) {{
              setActiveButton(preset.baseKey);
            }} else {{
              setActiveButton(null);
            }}

            if (preset.type === 'fit' || !lastTime || !isFinite(lastTime)) {{
              chart.timeScale().fitContent();
              return;
            }}

            if (preset.seconds && preset.seconds > 0) {{
              const from = Math.max(firstTime ?? lastTime - preset.seconds, lastTime - preset.seconds);
              const to = lastTime;
              chart.timeScale().setVisibleRange({{ from, to }});
              return;
            }}

            if (preset.bars && totalBars > 0) {{
              const lastIndex = totalBars - 1;
              const fromIndex = Math.max(0, lastIndex - preset.bars + 1);
              chart.timeScale().setVisibleLogicalRange({{
                from: fromIndex - 0.5,
                to: lastIndex + 0.1,
              }});
              return;
            }}

            chart.timeScale().fitContent();
          }};

          if (toolbar) {{
            VIEW_PRESETS.forEach(preset => {{
              const button = document.createElement('button');
              button.type = 'button';
              button.dataset.key = preset.key;
              button.textContent = preset.label;
              button.addEventListener('click', () => applyView(preset.key));
              toolbar.appendChild(button);
              viewButtons.push(button);
            }});
          }}

          const initialView = payload.view || 'fit';
          applyView(initialView);

          const observerTarget = document.body || container;
          if (window.ResizeObserver && observerTarget) {{
            const resizeObserver = new ResizeObserver(() => resize());
            resizeObserver.observe(observerTarget);
          }}

          if (priceValues.length) {{
            const phantom = chart.addLineSeries({{
              color: 'rgba(0,0,0,0)',
              lineWidth: 0,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
            }});
            const firstTime = payload.candles[0]?.time;
            const lastTime = payload.candles[payload.candles.length - 1]?.time || firstTime;
            const minPrice = Math.min(...priceValues);
            const maxPrice = Math.max(...priceValues);
            const padding = (maxPrice - minPrice) * 0.05 || 1.0;
            phantom.setData([
              {{ time: firstTime, value: minPrice - padding }},
              {{ time: lastTime, value: maxPrice + padding }},
            ]);
          }}

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

          if (firstTime != null && lastTime != null) {{
            (payload.levels || []).forEach(level => {{
              if (typeof level.value !== 'number' || !isFinite(level.value)) return;
              const levelSeries = chart.addLineSeries({{
                color: level.color,
                lineWidth: 2,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              }});
              levelSeries.setData([
                {{ time: firstTime, value: level.value }},
                {{ time: lastTime, value: level.value }},
              ]);
              levelSeries.createPriceLine({{
                price: level.value,
                color: level.color,
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true,
                title: level.label,
              }});
            }});
          }}

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
          const indicatorBadges = [
            ...Object.keys(payload.ema_series || {{}}).map(span => `<span class="badge">EMA${{span}}</span>`),
            '<span class="badge">VWAP</span>',
          ].join('');
          const notesBlock = plan.notes ? `<div class="badge-group" style="opacity:0.85">${{plan.notes}}</div>` : '';
          legend.innerHTML = `
            <h1>${{payload.symbol}} · ${{payload.interval.toUpperCase()}}</h1>
            ${{strategyLabel ? `<p>${{strategyLabel}}</p>` : ''}}
            ${{planRows.length ? `<div class="plan-grid">${{planRows.map(([label, value]) => `<span><strong class="label">${{label}}</strong><span>${{value}}</span></span>`).join('')}}</div>` : ''}}
            <div class="badge-group">
              ${{levelBadges}}${{indicatorBadges}}
            </div>
            ${{notesBlock}}
          `;
        }}
      }}
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html_doc)


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
    view: Optional[str] = None,
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
    if view:
        params["view"] = view
    if renderer_params:
        params.update(renderer_params)

    query = urlencode(params, doseq=False, safe=",")
    base = base.rstrip("/")
    return f"{base}/charts/{kind}?{query}"


# --------------------------------------------------------------------------- #
# PNG renderer
# --------------------------------------------------------------------------- #


@router.get("/png")
def chart_png(
    symbol: str,
    interval: str = Query("1m"),
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    tp: Optional[str] = None,
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

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)

    tps = parse_floats(tp)[:MAX_TPS]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(df.index, df["close"], color="#22c55e", linewidth=1.6, label="Close")
    ax.fill_between(df.index, df["low"], df["high"], color="#22c55e", alpha=0.12)

    def _plot_line(level: Optional[float], color: str, label: str):
        if level is None or not math.isfinite(level):
            return
        ax.axhline(level, color=color, linestyle="--", linewidth=1.1, alpha=0.9, label=label)

    _plot_line(entry, "#60a5fa", "Entry")
    _plot_line(stop, "#f87171", "Stop")
    for idx, target in enumerate(tps, start=1):
        _plot_line(target, "#34d399", f"TP{idx}")

    title_text = title or f"{symbol.upper()} {interval_normalized.upper()}"
    ax.set_title(title_text, fontsize=13, pad=14)
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
