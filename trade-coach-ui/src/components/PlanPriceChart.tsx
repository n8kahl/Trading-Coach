"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type {
  CandlestickData,
  HistogramData,
  IChartApi,
  IPriceLine,
  ISeriesApi,
  LineData,
  PriceScaleMargins,
  Time,
} from "lightweight-charts";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanLayerLevel, PlanLayerZone, PlanLayers } from "@/lib/types";
import type { PriceSeriesCandle } from "@/lib/hooks/usePriceSeries";

type ChartLib = typeof import("lightweight-charts");

export type OverlayTarget = {
  price: number;
  label?: string | null;
};

export type ChartOverlayState = {
  entry?: number | null;
  stop?: number | null;
  trailingStop?: number | null;
  targets?: OverlayTarget[];
  emaPeriods?: number[];
  showVWAP?: boolean;
  layers?: PlanLayers | null;
  supportingLevels?: SupportingLevel[];
};

type PlanPriceChartProps = {
  planId: string;
  symbol: string;
  resolution: string;
  theme: "dark" | "light";
  data: PriceSeriesCandle[];
  overlays: ChartOverlayState;
  onLastBarTimeChange?: (time: number | null) => void;
  devMode?: boolean;
};

export type PlanPriceChartHandle = {
  followLive: () => void;
  startReplay: () => void;
  stopReplay: () => void;
  refreshOverlays: () => void;
  getLastBarTime: () => number | null;
};

type CandlesSeries = ISeriesApi<"Candlestick">;
type VolumeSeries = ISeriesApi<"Histogram">;
type LineSeries = ISeriesApi<"Line">;

const GREEN = "#22c55e";
const RED = "#ef4444";
const SURFACE_DARK = "#050709";
const SURFACE_LIGHT = "#ffffff";
const TEXT_DARK = "#e5e7eb";
const TEXT_LIGHT = "#1e293b";
const GRID_DARK = "rgba(148, 163, 184, 0.12)";
const GRID_LIGHT = "rgba(148, 163, 184, 0.25)";

const EMA_COLORS = ["#93c5fd", "#f97316", "#a855f7", "#14b8a6"];

function resolutionToSeconds(resolution: string): number {
  const token = resolution.trim().toUpperCase();
  if (!token) return 60;
  if (token.endsWith("D")) {
    const days = Number.parseInt(token.replace("D", ""), 10);
    return Number.isFinite(days) && days > 0 ? days * 24 * 60 * 60 : 24 * 60 * 60;
  }
  if (token.endsWith("W")) {
    const weeks = Number.parseInt(token.replace("W", ""), 10);
    return Number.isFinite(weeks) && weeks > 0 ? weeks * 7 * 24 * 60 * 60 : 7 * 24 * 60 * 60;
  }
  if (token.endsWith("H")) {
    const hours = Number.parseInt(token.replace("H", ""), 10);
    return Number.isFinite(hours) && hours > 0 ? hours * 60 * 60 : 60 * 60;
  }
  const minutes = Number.parseInt(token, 10);
  return Number.isFinite(minutes) && minutes > 0 ? minutes * 60 : 60;
}

function computeEMA(bars: PriceSeriesCandle[], length: number): LineData[] {
  if (!Number.isFinite(length) || length <= 1 || bars.length === 0) return [];
  let ema = bars[0]?.close ?? 0;
  const multiplier = 2 / (length + 1);
  return bars.map((bar, index) => {
    if (index === 0) {
      ema = bar.close;
    } else {
      ema = (bar.close - ema) * multiplier + ema;
    }
    return { time: bar.time as Time, value: ema };
  });
}

function computeVWAP(bars: PriceSeriesCandle[]): LineData[] {
  let cumulativePriceVolume = 0;
  let cumulativeVolume = 0;
  return bars.map((bar) => {
    const typical = (bar.high + bar.low + bar.close) / 3;
    const volume = Number.isFinite(bar.volume) && bar.volume != null ? bar.volume : 1;
    cumulativePriceVolume += typical * volume;
    cumulativeVolume += volume;
    const vwap = cumulativeVolume > 0 ? cumulativePriceVolume / cumulativeVolume : typical;
    return { time: bar.time as Time, value: vwap };
  });
}

function pickLevelColor(kind?: string | null): string {
  const token = (kind || "").toLowerCase();
  if (token.includes("entry")) return "#facc15";
  if (token.includes("stop")) return "#fb7185";
  if (token.includes("trail")) return "#fb923c";
  if (token.includes("target") || token.includes("tp")) return "#22c55e";
  if (token.includes("vwap")) return "#eab308";
  if (token.includes("ema")) return "#93c5fd";
  if (token.includes("vah")) return "#84cc16";
  if (token.includes("val")) return "#38bdf8";
  if (token.includes("poc")) return "#f97316";
  if (token.includes("session_high") || token.includes("high")) return "#facc15";
  if (token.includes("session_low") || token.includes("low")) return "#fb7185";
  return "#94a3b8";
}

function toPriceLineId(base: string, value: number | string, index?: number): string {
  const suffix = index != null ? `:${index}` : "";
  return `${base}:${value}${suffix}`;
}

const PlanPriceChart = forwardRef<PlanPriceChartHandle, PlanPriceChartProps>(
  ({ planId, symbol, resolution, theme, data, overlays, onLastBarTimeChange, devMode = false }, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const chartLibRef = useRef<ChartLib | null>(null);
    const candleSeriesRef = useRef<CandlesSeries | null>(null);
    const volumeSeriesRef = useRef<VolumeSeries | null>(null);
    const emaSeriesRef = useRef<Map<number, LineSeries>>(new Map());
    const vwapSeriesRef = useRef<LineSeries | null>(null);
    const overlayLinesRef = useRef<Map<string, IPriceLine>>(new Map());
    const overlaysRef = useRef<ChartOverlayState>(overlays);
    const resizeObserverRef = useRef<ResizeObserver | null>(null);
    const replayTimerRef = useRef<number | null>(null);
    const replayActiveRef = useRef(false);
    const autoFollowRef = useRef(true);
    const lastBarTimeRef = useRef<number | null>(null);
    const [chartReady, setChartReady] = useState(false);
    const hasInitialLoadRef = useRef(false);
    const resolutionSecondsRef = useRef(resolutionToSeconds(resolution));

    useEffect(() => {
      resolutionSecondsRef.current = resolutionToSeconds(resolution);
    }, [resolution]);

    useEffect(() => {
      if (!containerRef.current || chartRef.current) return;
      let disposed = false;

      (async () => {
        const lib = await import("lightweight-charts");
        if (disposed || !containerRef.current) return;

        chartLibRef.current = lib;
        const chart = lib.createChart(containerRef.current, {
          layout: {
            background: { type: lib.ColorType.Solid, color: theme === "light" ? SURFACE_LIGHT : SURFACE_DARK },
            textColor: theme === "light" ? TEXT_LIGHT : TEXT_DARK,
          },
          grid: {
            vertLines: { color: theme === "light" ? GRID_LIGHT : GRID_DARK },
            horzLines: { color: theme === "light" ? GRID_LIGHT : GRID_DARK },
          },
          crosshair: {
            mode: lib.CrosshairMode.Magnet,
          },
          rightPriceScale: {
            borderColor: theme === "light" ? "rgba(148,163,184,0.25)" : "rgba(148,163,184,0.2)",
            autoScale: true,
            scaleMargins: { top: 0.15, bottom: 0.2 } as PriceScaleMargins,
          },
          timeScale: {
            borderColor: theme === "light" ? "rgba(148,163,184,0.25)" : "rgba(148,163,184,0.2)",
            timeVisible: true,
            secondsVisible: true,
          },
          autoSize: true,
        });

        const candleSeries = chart.addCandlestickSeries({
          upColor: GREEN,
          wickUpColor: GREEN,
          borderUpColor: GREEN,
          downColor: RED,
          wickDownColor: RED,
          borderDownColor: RED,
        });

        const volumeSeries = chart.addHistogramSeries({
          priceScaleId: "",
          color: "rgba(148, 163, 184, 0.4)",
          base: 0,
        });

        volumeSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.8, bottom: 0 } as PriceScaleMargins,
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;
        volumeSeriesRef.current = volumeSeries;
        resizeObserverRef.current = new ResizeObserver(() => {
          chart.applyOptions({
            width: containerRef.current?.clientWidth,
            height: containerRef.current?.clientHeight,
          });
        });
        resizeObserverRef.current.observe(containerRef.current);
        setChartReady(true);
      })().catch((error) => {
        if (devMode) {
          console.error("[PlanPriceChart] chart init failed", error);
        }
      });

      return () => {
        disposed = true;
        setChartReady(false);
        replayActiveRef.current = false;
        if (replayTimerRef.current) {
          window.clearInterval(replayTimerRef.current);
          replayTimerRef.current = null;
        }
        resizeObserverRef.current?.disconnect();
        resizeObserverRef.current = null;
        candleSeriesRef.current = null;
        volumeSeriesRef.current = null;
        // eslint-disable-next-line react-hooks/exhaustive-deps
        const emaSeries = emaSeriesRef.current;
        emaSeries.forEach((series) => series.priceScale().chart()?.removeSeries(series));
        emaSeries.clear();
        const vwapSeries = vwapSeriesRef.current;
        if (vwapSeries) {
          vwapSeries.priceScale().chart()?.removeSeries(vwapSeries);
          vwapSeriesRef.current = null;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
        const overlayLines = overlayLinesRef.current;
        overlayLines.clear();
        chartRef.current?.remove();
        chartRef.current = null;
        chartLibRef.current = null;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
      const lib = chartLibRef.current;
      const chart = chartRef.current;
      if (!chart || !lib) return;
      chart.applyOptions({
        layout: {
          background: { type: lib.ColorType.Solid, color: theme === "light" ? SURFACE_LIGHT : SURFACE_DARK },
          textColor: theme === "light" ? TEXT_LIGHT : TEXT_DARK,
        },
        grid: {
          vertLines: { color: theme === "light" ? GRID_LIGHT : GRID_DARK },
          horzLines: { color: theme === "light" ? GRID_LIGHT : GRID_DARK },
        },
        rightPriceScale: {
          borderColor: theme === "light" ? "rgba(148,163,184,0.25)" : "rgba(148,163,184,0.2)",
        },
        timeScale: {
          borderColor: theme === "light" ? "rgba(148,163,184,0.25)" : "rgba(148,163,184,0.2)",
        },
      });
    }, [theme]);

    useEffect(() => {
      if (!chartReady) return;
      const candleSeries = candleSeriesRef.current;
      const volumeSeries = volumeSeriesRef.current;
      if (!candleSeries || !volumeSeries) return;

      const candles: CandlestickData[] = data.map((bar) => ({
        time: bar.time as Time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));

      candleSeries.setData(candles);

      const volumes: HistogramData[] = data.map((bar) => ({
        time: bar.time as Time,
        value: Number.isFinite(bar.volume) && bar.volume != null ? bar.volume : 0,
        color: bar.close >= bar.open ? `${GREEN}55` : `${RED}55`,
      }));
      volumeSeries.setData(volumes);

      if (candles.length) {
        const last = candles[candles.length - 1];
        const lastMs = Number(last.time) * 1000;
        lastBarTimeRef.current = Number.isFinite(lastMs) ? lastMs : null;
        onLastBarTimeChange?.(lastBarTimeRef.current);
      } else {
        lastBarTimeRef.current = null;
        onLastBarTimeChange?.(null);
      }

      if (!hasInitialLoadRef.current) {
        hasInitialLoadRef.current = true;
        chartRef.current?.timeScale().fitContent();
      } else if (autoFollowRef.current) {
        chartRef.current?.timeScale().scrollToRealTime();
      }
    }, [chartReady, data, onLastBarTimeChange]);

    const syncDerivedSeries = useCallback(() => {
      if (!chartReady || data.length === 0) return;
      const lib = chartLibRef.current;
      const chart = chartRef.current;
      const candleSeries = candleSeriesRef.current;
      if (!lib || !chart || !candleSeries) return;

      const overlayState = overlaysRef.current;
      const periods = (overlayState.emaPeriods ?? []).filter((value) => Number.isFinite(value) && value > 1);
      const desired = new Set(periods);

      emaSeriesRef.current.forEach((series, period) => {
        if (!desired.has(period)) {
          chart.removeSeries(series);
          emaSeriesRef.current.delete(period);
        }
      });

      periods.slice(0, EMA_COLORS.length).forEach((period, index) => {
        let series = emaSeriesRef.current.get(period);
        if (!series) {
          series = chart.addLineSeries({
            color: EMA_COLORS[index % EMA_COLORS.length],
            lineWidth: 2,
          });
          emaSeriesRef.current.set(period, series);
        }
        series.setData(computeEMA(data, period));
      });

      if (overlayState.showVWAP) {
        if (!vwapSeriesRef.current) {
          vwapSeriesRef.current = chart.addLineSeries({
            color: "#eab308",
            lineWidth: 2,
            lineStyle: lib.LineStyle.Solid,
          });
        }
        vwapSeriesRef.current.setData(computeVWAP(data));
      } else if (vwapSeriesRef.current) {
        chart.removeSeries(vwapSeriesRef.current);
        vwapSeriesRef.current = null;
      }

      overlayLinesRef.current.forEach((line) => {
        try {
          candleSeries.removePriceLine(line);
        } catch {
          /* ignore */
        }
      });
      overlayLinesRef.current.clear();

      const registerLine = (id: string, options: Parameters<CandlesSeries["createPriceLine"]>[0]) => {
        const line = candleSeries.createPriceLine(options);
        overlayLinesRef.current.set(id, line);
      };

      const addLevelLine = (id: string, price: number, opts: Parameters<CandlesSeries["createPriceLine"]>[0]) => {
        if (!Number.isFinite(price)) return;
        registerLine(id, { price, axisLabelVisible: true, ...opts });
      };

      const entryPrice = overlayState.entry;
      if (Number.isFinite(entryPrice ?? null)) {
        addLevelLine("plan:entry", Number(entryPrice), {
          color: "#facc15",
          lineWidth: 2,
          lineStyle: lib.LineStyle.Dotted,
          title: "Entry",
        });
      }

      const stopPrice = overlayState.stop;
      if (Number.isFinite(stopPrice ?? null)) {
        addLevelLine("plan:stop", Number(stopPrice), {
          color: "#f87171",
          lineWidth: 2,
          lineStyle: lib.LineStyle.Solid,
          title: "Stop",
        });
      }

      const trailing = overlayState.trailingStop;
      if (Number.isFinite(trailing ?? null)) {
        addLevelLine("plan:trail", Number(trailing), {
          color: "#fb923c",
          lineWidth: 2,
          lineStyle: lib.LineStyle.Dashed,
          title: "Trailing",
        });
      }

      overlayState.targets?.forEach((target, index) => {
        if (!Number.isFinite(target?.price)) return;
        const label = target?.label || `TP${index + 1}`;
        addLevelLine(toPriceLineId("plan:tp", target.price, index), Number(target.price), {
          color: "#22c55e",
          lineWidth: 1,
          lineStyle: lib.LineStyle.Dashed,
          title: label,
        });
      });

      const allLevels: PlanLayerLevel[] = overlayState.layers?.levels ?? [];
      allLevels.forEach((level, idx) => {
        if (!Number.isFinite(level.price ?? null)) return;
        const label = level.label || level.kind || `Level ${idx + 1}`;
        addLevelLine(toPriceLineId("layer:level", level.price, idx), Number(level.price), {
          color: pickLevelColor(level.kind),
          lineWidth: 1,
          lineStyle: lib.LineStyle.Dotted,
          title: label,
        });
      });

      const zones: PlanLayerZone[] = overlayState.layers?.zones ?? [];
      zones.forEach((zone, idx) => {
        const high = Number.isFinite(zone.high ?? null) ? Number(zone.high) : null;
        const low = Number.isFinite(zone.low ?? null) ? Number(zone.low) : null;
        const label = zone.label || zone.kind || `Zone ${idx + 1}`;
        if (high != null) {
          addLevelLine(toPriceLineId("layer:zone-high", high, idx), high, {
            color: `${pickLevelColor(zone.kind)}aa`,
            lineWidth: 1,
            lineStyle: lib.LineStyle.Solid,
            title: `${label} High`,
          });
        }
        if (low != null) {
          addLevelLine(toPriceLineId("layer:zone-low", low, idx), low, {
            color: `${pickLevelColor(zone.kind)}aa`,
            lineWidth: 1,
            lineStyle: lib.LineStyle.Solid,
            title: `${label} Low`,
          });
        }
      });
    }, [chartReady, data]);

    useEffect(() => {
      overlaysRef.current = overlays;
      syncDerivedSeries();
    }, [overlays, syncDerivedSeries]);

    useEffect(() => {
      syncDerivedSeries();
    }, [syncDerivedSeries]);

    const stopReplay = useCallback(() => {
      replayActiveRef.current = false;
      if (replayTimerRef.current) {
        window.clearInterval(replayTimerRef.current);
        replayTimerRef.current = null;
      }
    }, []);

    const followLive = useCallback(() => {
      const chart = chartRef.current;
      if (!chart || !data.length) return;
      stopReplay();
      autoFollowRef.current = true;
      const last = data[data.length - 1];
      const lastTime = Number(last.time);
      if (!Number.isFinite(lastTime)) return;
      const lookbackSeconds = Math.max(resolutionSecondsRef.current * 120, 300);
      const from = Math.max(lastTime - lookbackSeconds, lastTime - 60 * 60);
      try {
        chart.timeScale().setVisibleRange({ from, to: lastTime });
      } catch (error) {
        if (devMode) {
          console.warn("[PlanPriceChart] followLive range failed", error);
        }
      }
      chart.timeScale().scrollToRealTime();
    }, [data, devMode, stopReplay]);

    const startReplay = useCallback(() => {
      const chart = chartRef.current;
      if (!chart || data.length < 60) return;
      stopReplay();
      autoFollowRef.current = false;
      replayActiveRef.current = true;
      let index = Math.max(data.length - 200, 0);
      const step = Math.max(1, Math.floor(data.length / 120));

      const tick = () => {
        if (!replayActiveRef.current || !chart) {
          stopReplay();
          return;
        }
        index += step;
        if (index >= data.length) {
          followLive();
          return;
        }
        const end = data[index];
        const startIdx = Math.max(index - 120, 0);
        const start = data[startIdx];
        chart.timeScale().setVisibleRange({
          from: Number(start.time),
          to: Number(end.time),
        });
      };

      tick();
      replayTimerRef.current = window.setInterval(tick, 400);
    }, [data, followLive, stopReplay]);

    useImperativeHandle(
      ref,
      () => ({
        followLive: () => followLive(),
        startReplay: () => startReplay(),
        stopReplay: () => stopReplay(),
        refreshOverlays: () => {
          syncDerivedSeries();
        },
        getLastBarTime: () => lastBarTimeRef.current,
      }),
      [syncDerivedSeries, followLive, startReplay, stopReplay],
    );

    return <div ref={containerRef} className="h-[360px] w-full rounded-2xl border border-neutral-800/70" data-symbol={symbol} data-plan-id={planId} />;
  },
);

PlanPriceChart.displayName = "PlanPriceChart";

export default PlanPriceChart;
