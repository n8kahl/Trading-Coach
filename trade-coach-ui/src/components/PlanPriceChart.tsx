"use client";

import clsx from "clsx";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type {
  AutoscaleInfo,
  AutoscaleInfoProvider,
  CandlestickData,
  HistogramData,
  IChartApi,
  IPriceLine,
  ISeriesApi,
  LineData,
  LogicalRange,
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
type LogicalRangeInput = { from: number; to: number };

const GREEN = "#22c55e";
const RED = "#ef4444";
const SURFACE_DARK = "#050709";
const SURFACE_LIGHT = "#ffffff";
const TEXT_DARK = "#e5e7eb";
const TEXT_LIGHT = "#1e293b";
const GRID_DARK = "rgba(148, 163, 184, 0.12)";
const GRID_LIGHT = "rgba(148, 163, 184, 0.25)";

const EMA_COLORS = ["#93c5fd", "#f97316", "#a855f7", "#14b8a6"];

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
  ({ planId, symbol, resolution: _resolution, theme, data, overlays, onLastBarTimeChange, devMode = false }, ref) => {
    const debug =
      devMode || (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("dev") !== null);
    const [debugMsgs, setDebugMsgs] = useState<string[]>([]);
    const addDbg = useCallback(
      (m: string) => {
        if (!debug) return;
        setDebugMsgs((prev) => (prev.length > 30 ? [...prev.slice(-20), m] : [...prev, m]));
        console.log(m);
      },
      [debug],
    );
    const containerRef = useRef<HTMLDivElement | null>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const chartLibRef = useRef<ChartLib | null>(null);
    const candleSeriesRef = useRef<CandlesSeries | null>(null);
    const volumeSeriesRef = useRef<VolumeSeries | null>(null);
    const emaSeriesRef = useRef<Map<number, LineSeries>>(new Map());
    const vwapSeriesRef = useRef<LineSeries | null>(null);
    const overlayLinesRef = useRef<Map<string, IPriceLine>>(new Map());
    const overlaysRef = useRef<ChartOverlayState>(overlays);
    const overlayBoundsRef = useRef<{ min: number | null; max: number | null }>({ min: null, max: null });
    const resizeObserverRef = useRef<ResizeObserver | null>(null);
    const replayTimerRef = useRef<number | null>(null);
    const replayActiveRef = useRef(false);
    const autoFollowRef = useRef(true);
    const lastBarTimeRef = useRef<number | null>(null);
    const suppressInteractionRef = useRef(false);
    const [chartReady, setChartReady] = useState(false);
    const [isAutoFollow, setIsAutoFollow] = useState(true);
    const hasInitialLoadRef = useRef(false);

    const setAutoFollow = useCallback((next: boolean) => {
      autoFollowRef.current = next;
      setIsAutoFollow(next);
    }, []);

    const overlayLegendItems = useMemo(() => {
      const items: Array<{ key: string; label: string; tone: "vwap" | "ema" }> = [];
      const periods = (overlays.emaPeriods ?? [])
        .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 1)
        .map((value) => Math.round(value))
        .sort((a, b) => a - b);
      const uniquePeriods = Array.from(new Set(periods));
      if (overlays.showVWAP !== false) {
        items.push({ key: "vwap", label: "VWAP", tone: "vwap" });
      }
      if (uniquePeriods.length) {
        items.push({ key: "ema", label: `EMA ${uniquePeriods.join("/")}`, tone: "ema" });
      }
      return items;
    }, [overlays]);

    const computeOverlayBounds = useCallback((overlayState: ChartOverlayState | null | undefined) => {
      if (!overlayState) return { min: null, max: null };
      const values: number[] = [];
      const push = (value: number | null | undefined) => {
        if (typeof value === "number" && Number.isFinite(value)) {
          values.push(value);
        }
      };
      push(overlayState.entry);
      push(overlayState.stop);
      push(overlayState.trailingStop);
      overlayState.targets?.forEach((target) => {
        if (target) push(target.price);
      });
      return values.length
        ? { min: Math.min(...values), max: Math.max(...values) }
        : { min: null, max: null };
    }, []);

    const updateOverlayBounds = useCallback(() => {
      overlayBoundsRef.current = computeOverlayBounds(overlaysRef.current);
    }, [computeOverlayBounds]);

    const applyLogicalRange = useCallback(
      (range: LogicalRangeInput, context: string) => {
        const chart = chartRef.current;
        if (!chart) return false;
        const timeScale = chart.timeScale();
        suppressInteractionRef.current = true;
        try {
          timeScale.setVisibleLogicalRange(range as LogicalRange);
          return true;
        } catch (error) {
          addDbg(`[PlanPriceChart] ${context} logical range failed: ${String(error)}`);
          return false;
        } finally {
          const release = () => {
            suppressInteractionRef.current = false;
          };
          if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(release);
          } else {
            release();
          }
        }
      },
      [addDbg],
    );

    const applyTimeRange = useCallback(
      (range: LogicalRangeInput, context: string) => {
        const chart = chartRef.current;
        if (!chart) return false;
        const timeScale = chart.timeScale();
        suppressInteractionRef.current = true;
        try {
          timeScale.setVisibleRange(range);
          return true;
        } catch (error) {
          addDbg(`[PlanPriceChart] ${context} range failed: ${String(error)}`);
          return false;
        } finally {
          const release = () => {
            suppressInteractionRef.current = false;
          };
          if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(release);
          } else {
            release();
          }
        }
      },
      [addDbg],
    );

    const markManualInteraction = useCallback(() => {
      if (suppressInteractionRef.current) return;
      if (!autoFollowRef.current) return;
      setAutoFollow(false);
    }, [setAutoFollow]);


    const safeData = useMemo(() => {
      if (!Array.isArray(data)) return [];
      return data.filter((bar): bar is PriceSeriesCandle => {
        if (!bar) return false;
        const timeValue = typeof bar.time === "number" ? bar.time : Number(bar.time);
        return (
          Number.isFinite(timeValue) &&
          Number.isFinite(bar.open) &&
          Number.isFinite(bar.high) &&
          Number.isFinite(bar.low) &&
          Number.isFinite(bar.close)
        );
      });
    }, [data]);

    useEffect(() => {
      if (!debug || !Array.isArray(data)) return;
      if (safeData.length !== data.length) {
        addDbg(`[PlanPriceChart] filtered invalid bars: in=${data.length} out=${safeData.length}`);
      }
    }, [debug, data, safeData, addDbg]);

    useEffect(() => {
      if (!containerRef.current || chartRef.current) return;
      let disposed = false;

      (async () => {
        addDbg(`[PlanPriceChart] mount: container=${!!containerRef.current}`);
        const imported = await import("lightweight-charts");
        if (disposed || !containerRef.current) return;

        // Handle different module shapes (ESM vs UMD default)
        const lib: any = (imported && (imported as any).createChart)
          ? imported
          : (imported as any)?.default || imported;
        const createChartFn: any = lib?.createChart;
        if (typeof createChartFn !== "function") {
          addDbg(`[PlanPriceChart] error: createChart not a function on module; keys=${Object.keys(imported || {}).join(',')}`);
          return;
        }

        chartLibRef.current = lib;

        // Wait for container to have non-zero size to avoid internal asserts
        const waitForSize = async () => {
          const start = Date.now();
          while (!disposed && containerRef.current) {
            const el = containerRef.current;
            const w = el.clientWidth;
            const h = el.clientHeight;
            if (w > 10 && h > 10) return { w, h };
            if (Date.now() - start > 800) break;
            await new Promise((r) => requestAnimationFrame(r));
          }
          const el = containerRef.current!;
          return { w: el?.clientWidth ?? 0, h: el?.clientHeight ?? 0 };
        };

        const size = await waitForSize();
        addDbg(`[PlanPriceChart] container size w=${size.w} h=${size.h}`);

        const ColorTypeSolid = (lib as any)?.ColorType?.Solid ?? "solid";
        const CrosshairModeMagnet = (lib as any)?.CrosshairMode?.Magnet ?? "magnet";

        // Create chart preferring autosize (legacy/stable path), then fallback to width/height
        let chart: any = null;
        const el = containerRef.current as HTMLDivElement;
        // StrictMode guard to prevent double init
        if ((el as any).__lw_attached) {
          addDbg(`[PlanPriceChart] init skipped: container already has a chart`);
          return;
        }
        (el as any).__lw_attached = true;

        const w = el.clientWidth || 800;
        const h = el.clientHeight || 360;
        try {
          chart = createChartFn(el, { autoSize: true });
          addDbg(`[PlanPriceChart] createChart autosize ok`);
        } catch (e1) {
          addDbg(`[PlanPriceChart] createChart autosize failed: ${String(e1)}`);
          chart = createChartFn(el, { width: w, height: h });
          addDbg(`[PlanPriceChart] createChart width/height ok`);
        }
        if (!chart) {
          addDbg(`[PlanPriceChart] error: unable to create chart; libKeys=${Object.keys(lib || {}).join(',')}`);
          return;
        }

        const layoutColorType = typeof ColorTypeSolid === "string" ? ColorTypeSolid : "solid";
        const crosshairMode = typeof CrosshairModeMagnet === "string" ? CrosshairModeMagnet : 1;

        (chart as any).applyOptions?.({
          layout: {
            background: { type: layoutColorType, color: theme === "light" ? SURFACE_LIGHT : SURFACE_DARK },
            textColor: theme === "light" ? TEXT_LIGHT : TEXT_DARK,
          },
          grid: {
            vertLines: { color: theme === "light" ? GRID_LIGHT : GRID_DARK },
            horzLines: { color: theme === "light" ? GRID_LIGHT : GRID_DARK },
          },
          crosshair: {
            mode: crosshairMode,
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
        });
        const chartApi: any = chart as any;
        if (!chartApi || typeof chartApi.addCandlestickSeries !== "function" || typeof chartApi.addHistogramSeries !== "function") {
          addDbg(`[PlanPriceChart] error: legacy series APIs not available; chartKeys=${Object.keys(chartApi || {}).join(',')}`);
          return;
        }

        const autoscaleProvider: AutoscaleInfoProvider = (baseImplementation) => {
          const baseInfo = baseImplementation();
          const bounds = overlayBoundsRef.current;
          if (bounds.min == null || bounds.max == null) {
            return baseInfo;
          }
          const span = Math.max(bounds.max - bounds.min, 0);
          const fallback = Math.abs(bounds.max ?? bounds.min ?? 0) * 0.001;
          const padding = span > 0 ? span * 0.08 : Math.max(fallback, 0.5);
          const minValue = bounds.min - padding;
          const maxValue = bounds.max + padding;
          if (baseInfo?.priceRange) {
            return {
              ...baseInfo,
              priceRange: {
                minValue: Math.min(baseInfo.priceRange.minValue, minValue),
                maxValue: Math.max(baseInfo.priceRange.maxValue, maxValue),
              },
            } satisfies AutoscaleInfo;
          }
          return {
            priceRange: {
              minValue,
              maxValue,
            },
          } satisfies AutoscaleInfo;
        };

        const candleSeries = chartApi.addCandlestickSeries({
          upColor: GREEN,
          wickUpColor: GREEN,
          borderUpColor: GREEN,
          downColor: RED,
          wickDownColor: RED,
          borderDownColor: RED,
          autoscaleInfoProvider: autoscaleProvider,
        });

        const volumeSeries = chartApi.addHistogramSeries({
          priceScaleId: "",
          color: "rgba(148, 163, 184, 0.4)",
        });

        addDbg(`[PlanPriceChart] series created via addCandlestickSeries/addHistogramSeries`);

        volumeSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.8, bottom: 0 } as PriceScaleMargins,
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;
        volumeSeriesRef.current = volumeSeries;
        // autoSize=true; no need to keep a ResizeObserver writing dimensions
        resizeObserverRef.current = null;
        setChartReady(true);
        addDbg(`[PlanPriceChart] chart ready: ${symbol} plan=${planId}`);
      })().catch((error) => {
        addDbg(`[PlanPriceChart] chart init failed: ${String(error)}`);
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
        const chart = chartRef.current;
        emaSeries.forEach((series) => {
          try {
            chart?.removeSeries(series);
          } catch {
            /* ignore */
          }
        });
        emaSeries.clear();
        const vwapSeries = vwapSeriesRef.current;
        if (vwapSeries) {
          try {
            chart?.removeSeries(vwapSeries);
          } catch {
            /* ignore */
          }
          vwapSeriesRef.current = null;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
        const overlayLines = overlayLinesRef.current;
        overlayLines.clear();
        chartRef.current?.remove();
        chartRef.current = null;
        chartLibRef.current = null;
        if (containerRef.current) delete (containerRef.current as any).__lw_attached;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
      const lib = chartLibRef.current as any;
      const chart = chartRef.current as any;
      if (!chart || !lib) return;
      const layoutColorType = typeof lib?.ColorType?.Solid === "string" ? lib.ColorType.Solid : "solid";
      chart.applyOptions({
        layout: {
          background: { type: layoutColorType, color: theme === "light" ? SURFACE_LIGHT : SURFACE_DARK },
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
      const chart = chartRef.current;
      if (!chart) return;
      const timeScale = chart.timeScale();
      const handleLogicalChange = () => {
        markManualInteraction();
      };
      const handleTimeChange = () => {
        markManualInteraction();
      };
      timeScale.subscribeVisibleLogicalRangeChange(handleLogicalChange);
      timeScale.subscribeVisibleTimeRangeChange(handleTimeChange);
      return () => {
        timeScale.unsubscribeVisibleLogicalRangeChange(handleLogicalChange);
        timeScale.unsubscribeVisibleTimeRangeChange(handleTimeChange);
      };
    }, [chartReady, markManualInteraction]);

    useEffect(() => {
      if (!chartReady) return;
      const candleSeries = candleSeriesRef.current;
      const volumeSeries = volumeSeriesRef.current;
      if (!candleSeries || !volumeSeries) return;

      const timeCheck = safeData.every((bar) => bar && Number.isFinite(Number((bar as any).time)));
      const candleCheck = safeData.every(
        (bar) =>
          bar &&
          Number.isFinite(bar.open) &&
          Number.isFinite(bar.high) &&
          Number.isFinite(bar.low) &&
          Number.isFinite(bar.close),
      );
      if (!timeCheck || !candleCheck) {
        addDbg(
          `[PlanPriceChart] skipped setData: invalid bars detected timeCheck=${timeCheck} candleCheck=${candleCheck} length=${safeData.length}`,
        );
        safeData.forEach((bar, index) => {
          if (
            !bar ||
            !Number.isFinite(Number((bar as any).time)) ||
            !Number.isFinite(bar.open) ||
            !Number.isFinite(bar.high) ||
            !Number.isFinite(bar.low) ||
            !Number.isFinite(bar.close)
          ) {
            addDbg(
              `[PlanPriceChart] bad bar at ${index}: ${JSON.stringify({
                time: bar?.time,
                open: bar?.open,
                high: bar?.high,
                low: bar?.low,
                close: bar?.close,
                volume: bar?.volume,
              })}`,
            );
          }
        });
        const snapshot = safeData.slice(0, 5).map((bar) => ({
          time: bar?.time,
          open: bar?.open,
          high: bar?.high,
          low: bar?.low,
          close: bar?.close,
        }));
        addDbg(`[PlanPriceChart] first bars snapshot: ${JSON.stringify(snapshot)}`);
        return;
      }

      const candles: CandlestickData[] = safeData.map((bar) => ({
        time: bar.time as Time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));

      candleSeries.setData(candles);
      {
        const n = candles.length;
        const first = n ? Number(candles[0].time) : null;
        const last = n ? Number(candles[n - 1].time) : null;
        addDbg(`[PlanPriceChart] setData count=${n} first=${first} last=${last}`);
      }

      const volumes: HistogramData[] = safeData.map((bar) => ({
        time: bar.time as Time,
        value: Number.isFinite(bar.volume) && bar.volume != null ? bar.volume : 0,
        color: bar.close >= bar.open ? `${GREEN}55` : `${RED}55`,
      }));
      volumeSeries.setData(volumes);

      if (candles.length) {
        const lastIndex = candles.length - 1;
        const last = candles[lastIndex];
        const lastMs = Number(last.time) * 1000;
        lastBarTimeRef.current = Number.isFinite(lastMs) ? lastMs : null;
        onLastBarTimeChange?.(lastBarTimeRef.current);
        addDbg(`[PlanPriceChart] last bar ms=${lastBarTimeRef.current}`);
        const logicalRange = {
          from: Math.max(lastIndex - 120, 0),
          to: lastIndex + 2,
        };
        const chart = chartRef.current;
        if (!chart) {
          addDbg("[PlanPriceChart] chart missing during range apply");
          return;
        }
        if (!hasInitialLoadRef.current) {
          hasInitialLoadRef.current = true;
          if (applyLogicalRange(logicalRange, "initial")) {
            chart.timeScale().scrollToRealTime();
          }
        } else if (autoFollowRef.current) {
          if (applyLogicalRange(logicalRange, "update")) {
            chart.timeScale().scrollToRealTime();
          }
        }
      } else {
        lastBarTimeRef.current = null;
        onLastBarTimeChange?.(null);
        addDbg(`[PlanPriceChart] no candles`);
      }
    }, [chartReady, safeData, onLastBarTimeChange, applyLogicalRange]);

    const syncDerivedSeries = useCallback(() => {
      if (!chartReady || safeData.length === 0) return;
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

      const emaLineMarkers: Array<{ id: string; price: number; color: string; title: string }> = [];

      periods.slice(0, EMA_COLORS.length).forEach((period, index) => {
        const color = EMA_COLORS[index % EMA_COLORS.length];
        let series = emaSeriesRef.current.get(period);
        if (!series) {
          series = chart.addLineSeries({
            color,
            lineWidth: 2,
          });
          emaSeriesRef.current.set(period, series);
        } else {
          series.applyOptions({ color, lineWidth: 2 });
        }
        const emaData = computeEMA(safeData, period);
        series.setData(emaData);
        const latest = emaData[emaData.length - 1];
        if (latest) {
          const rounded = Math.round(period);
          emaLineMarkers.push({
            id: `indicator:ema:${rounded}`,
            price: latest.value,
            color,
            title: `EMA ${rounded}`,
          });
        }
      });

      let vwapMarker: { price: number } | null = null;
      if (overlayState.showVWAP) {
        if (!vwapSeriesRef.current) {
          vwapSeriesRef.current = chart.addLineSeries({
            color: "#ffffff",
            lineWidth: 2,
            lineStyle: lib.LineStyle.Solid,
          });
        } else {
          vwapSeriesRef.current.applyOptions({ color: "#ffffff", lineWidth: 2 });
        }
        const vwapData = computeVWAP(safeData);
        vwapSeriesRef.current.setData(vwapData);
        const vwapLatest = vwapData[vwapData.length - 1];
        if (vwapLatest) {
          vwapMarker = { price: vwapLatest.value };
        }
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

      if (vwapMarker) {
        addLevelLine("indicator:vwap", vwapMarker.price, {
          color: "#ffffff",
          lineWidth: 2,
          lineStyle: lib.LineStyle.Solid,
          title: "VWAP",
        });
      }

      emaLineMarkers.forEach((marker) => {
        addLevelLine(marker.id, marker.price, {
          color: marker.color,
          lineWidth: 1,
          lineStyle: lib.LineStyle.Solid,
          title: marker.title,
        });
      });

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
    }, [chartReady, safeData, overlays]);

    useEffect(() => {
      overlaysRef.current = overlays;
      updateOverlayBounds();
      syncDerivedSeries();
    }, [overlays, syncDerivedSeries, safeData, updateOverlayBounds]);

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

    useEffect(() => {
      if (safeData.length === 0) {
        stopReplay();
        setAutoFollow(true);
      }
    }, [safeData.length, stopReplay, setAutoFollow]);

    const followLive = useCallback(() => {
      const chart = chartRef.current;
      if (!chart || safeData.length === 0) return;
      stopReplay();
      const lastIndex = safeData.length - 1;
      const last = safeData[lastIndex];
      if (!Number.isFinite(Number(last.time))) return;
      const logicalRange = {
        from: Math.max(lastIndex - 120, 0),
        to: lastIndex + 2,
      };
      setAutoFollow(true);
      if (applyLogicalRange(logicalRange, "followLive")) {
        chart.timeScale().scrollToRealTime();
      }
    }, [safeData, stopReplay, setAutoFollow, applyLogicalRange]);

    const startReplay = useCallback(() => {
      const chart = chartRef.current;
      if (!chart || safeData.length < 60) return;
      stopReplay();
      setAutoFollow(false);
      replayActiveRef.current = true;
      let index = Math.max(safeData.length - 200, 0);
      const step = Math.max(1, Math.floor(safeData.length / 120));

      const tick = () => {
        if (!replayActiveRef.current || !chart) {
          stopReplay();
          return;
        }
        if (safeData.length < 2) {
          stopReplay();
          setAutoFollow(true);
          return;
        }
        index += step;
        if (index >= safeData.length) {
          followLive();
          return;
        }
        const end = safeData[index];
        const startIdx = Math.max(index - 120, 0);
        const start = safeData[startIdx];
        applyTimeRange(
          {
            from: Number(start.time),
            to: Number(end.time),
          },
          "replay",
        );
      };

      tick();
      replayTimerRef.current = window.setInterval(tick, 400);
    }, [safeData, followLive, stopReplay, setAutoFollow, applyTimeRange]);

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

    return (
      <div
        ref={containerRef}
        className="relative h-full min-h-[360px] w-full rounded-2xl border border-neutral-800/70"
        data-symbol={symbol}
        data-plan-id={planId}
        data-resolution={_resolution}
      >
        {overlayLegendItems.length ? (
          <div
            className={clsx(
              "pointer-events-none absolute left-4 top-4 z-20 flex flex-wrap items-center gap-2 rounded-lg px-3 py-2 text-[0.68rem] font-semibold",
              theme === "light"
                ? "bg-white/85 text-slate-700 shadow-sm"
                : "bg-neutral-950/70 text-neutral-200 backdrop-blur",
            )}
          >
            {overlayLegendItems.map((item) => (
              <span
                key={item.key}
                className={clsx(
                  item.tone === "ema"
                    ? theme === "light"
                      ? "text-sky-700"
                      : "text-sky-200"
                    : theme === "light"
                      ? "text-slate-800"
                      : "text-neutral-100",
                )}
              >
                {item.label}
              </span>
            ))}
          </div>
        ) : null}
        {!isAutoFollow ? (
          <div className="pointer-events-none absolute bottom-4 right-4 z-30">
            <button
              type="button"
              onClick={followLive}
              className="pointer-events-auto rounded-full border border-emerald-500/60 bg-emerald-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-100 shadow-sm transition hover:border-emerald-400 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
            >
              Follow Live
            </button>
          </div>
        ) : null}
        {debug && debugMsgs.length ? (
          <div className="pointer-events-none absolute right-3 top-3 z-30 max-w-[50%] rounded-md bg-neutral-900/70 p-2 text-[10px] leading-snug text-neutral-200">
            {debugMsgs.map((m, i) => (
              <div key={i}>{m}</div>
            ))}
          </div>
        ) : null}
      </div>
    );
  },
);

PlanPriceChart.displayName = "PlanPriceChart";

export default PlanPriceChart;
