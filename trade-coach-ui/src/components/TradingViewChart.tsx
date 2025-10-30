"use client";

import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import type { PlanLayers } from "@/lib/types";
import { TradingViewDatafeed, TVBar } from "@/lib/tradingview/datafeed";

type TradingViewShape = {
  remove(): void;
};

type TradingViewActiveChart = {
  setTimezone(zone: string): void;
  createStudy(name: string, overlay: boolean, inputs: boolean, options?: Record<string, unknown>): void;
  createShape(points: Array<{ time: number; price: number }>, options: Record<string, unknown>): TradingViewShape;
  setVisibleRange(range: { from: number; to: number }, options?: Record<string, unknown>): void;
  scrollToRealTime(): void;
  setResolution(resolution: string, callback?: () => void): void;
};

type TradingViewWidget = {
  onChartReady(callback: () => void): void;
  remove(): void;
  activeChart(): TradingViewActiveChart;
};

type TradingViewWidgetConstructor = new (options: Record<string, unknown>) => TradingViewWidget;

declare global {
  interface Window {
    TradingView?: {
      widget: TradingViewWidgetConstructor;
    };
  }
}

type TradingViewChartProps = {
  symbol: string;
  planId: string;
  resolution: string;
  theme?: "dark" | "light";
  planLayers?: PlanLayers | null;
  onBarsLoaded?: (bars: TVBar[]) => void;
  onRealtimeBar?: (bar: TVBar) => void;
  devMode?: boolean;
};

export type TradingViewChartHandle = {
  setResolution: (resolution: string) => void;
  startReplay: () => void;
  stopReplay: () => void;
  followLive: () => void;
  refreshOverlays: () => void;
  getLastBarTime: () => number | null;
};

const SCRIPT_URL = "https://s3.tradingview.com/tv.js";
let tradingViewScriptPromise: Promise<void> | null = null;

function loadTradingViewScript(): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.resolve();
  }
  if (window.TradingView) {
    return Promise.resolve();
  }
  if (!tradingViewScriptPromise) {
    tradingViewScriptPromise = new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = SCRIPT_URL;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = (err) => reject(err);
      document.body.appendChild(script);
    });
  }
  return tradingViewScriptPromise;
}

function pickLevelColor(kind?: string | null): string {
  const token = (kind || "").toLowerCase();
  if (token.includes("poc")) return "#f97316";
  if (token.includes("vah") || token.includes("valu")) return "#84cc16";
  if (token.includes("val")) return "#22d3ee";
  if (token.includes("high") || token.includes("session_high")) return "#facc15";
  if (token.includes("low") || token.includes("session_low")) return "#fb7185";
  if (token.includes("demand")) return "#38bdf8";
  if (token.includes("supply")) return "#f97316";
  return "#cbd5f5";
}

const TradingViewChart = forwardRef<TradingViewChartHandle, TradingViewChartProps>(
  ({ symbol, planId, resolution, theme = "dark", planLayers, onBarsLoaded, onRealtimeBar, devMode = false }, ref) => {
    const containerId = useMemo(
      () => `tv-chart-${planId.replace(/[^a-zA-Z0-9]/g, "")}-${Math.random().toString(36).slice(2)}`,
      [planId],
    );
  const widgetRef = useRef<TradingViewWidget | null>(null);
  const chartRef = useRef<TradingViewActiveChart | null>(null);
  const overlaysRef = useRef<TradingViewShape[]>([]);
  const datafeedRef = useRef<TradingViewDatafeed | null>(null);
  const lastBarTimeRef = useRef<number | null>(null);
  const studiesAppliedRef = useRef(false);
  const replayTimerRef = useRef<number | null>(null);
  const replayActiveRef = useRef(false);
  const barsCacheRef = useRef<TVBar[]>([]);
  const [chartReady, setChartReady] = useState(false);
  const [barsReady, setBarsReady] = useState(false);

    const applyPlanLayers = React.useCallback(
      (layers: PlanLayers | null | undefined) => {
        const chart = chartRef.current;
        if (!chart) return;
        overlaysRef.current.forEach((shape) => {
          try {
            shape.remove();
          } catch {
            /* ignore */
          }
        });
        overlaysRef.current = [];
        if (!layers || !layers.levels?.length) return;

        const anchorTimeSec =
          lastBarTimeRef.current != null ? Math.floor(lastBarTimeRef.current / 1000) : Math.floor(Date.now() / 1000);
        layers.levels.forEach((level) => {
          if (typeof level.price !== "number" || !Number.isFinite(level.price)) return;
          const label = level.label || level.kind || "";
          const color = pickLevelColor(level.kind);
          try {
            const shape = chart.createShape(
              [{ time: anchorTimeSec, price: level.price }],
              {
                shape: "horizontal_line",
                disableSelection: true,
                lock: true,
                text: label,
                overrides: {
                  linecolor: color,
                  linewidth: 2,
                  horzLabelsBackgroundColor: color,
                },
              },
            );
            overlaysRef.current.push(shape);
          } catch (error) {
            if (devMode) {
              console.warn("[tv] unable to create overlay", { level, error });
            }
          }
        });

        if (Array.isArray(layers.zones)) {
          layers.zones.forEach((zone) => {
            const high = typeof zone?.high === "number" ? zone.high : null;
            const low = typeof zone?.low === "number" ? zone.low : null;
            if (high == null || low == null) return;
            const label = zone?.label || zone?.kind || "Zone";
            const color = pickLevelColor(zone?.kind);
            try {
              const shape = chart.createShape(
                [
                  { time: anchorTimeSec - 60, price: high },
                  { time: anchorTimeSec + 60, price: low },
                ],
                {
                  shape: "rectangle",
                  disableSelection: true,
                  lock: true,
                  text: label,
                  overrides: {
                    color,
                    backgroundColor: `${color}1A`,
                    linewidth: 1,
                  },
                },
              );
              overlaysRef.current.push(shape);
            } catch (error) {
              if (devMode) {
                console.warn("[tv] unable to create zone overlay", { zone, error });
              }
            }
          });
        }
      },
      [devMode],
    );

    useEffect(() => {
      let disposed = false;
      let datafeed: TradingViewDatafeed | null = null;

      (async () => {
        try {
          await loadTradingViewScript();
          if (disposed) return;
          datafeed = new TradingViewDatafeed({
            onBatchLoaded: (_, __, bars) => {
              if (disposed) return;
              barsCacheRef.current = bars;
              if (bars.length) {
                lastBarTimeRef.current = bars[bars.length - 1].time;
              }
              setBarsReady(bars.length > 0);
              onBarsLoaded?.(bars);
            },
            onRealtimeBar: (_, __, bar) => {
              if (disposed) return;
              lastBarTimeRef.current = bar.time;
              const existing = barsCacheRef.current;
              if (!existing.length || existing[existing.length - 1].time !== bar.time) {
                barsCacheRef.current = [...existing.slice(-600), bar];
              }
              setBarsReady(true);
              onRealtimeBar?.(bar);
            },
          });
          datafeedRef.current = datafeed;

          const tv = window.TradingView;
          if (!tv || typeof tv.widget !== "function") {
            console.error("[tv] TradingView widget library unavailable");
            return;
          }
          const widget = new tv.widget({
            symbol: symbol.toUpperCase(),
            interval: resolution,
            container_id: containerId,
            autosize: true,
            timezone: "America/New_York",
            theme: theme,
            style: "1",
            allow_symbol_change: false,
            hide_legend: false,
            hide_side_toolbar: false,
            hide_top_toolbar: true,
            save_image: false,
            studies_overrides: {},
            datafeed,
            locale: "en",
            enable_publishing: false,
            disabled_features: [
              "header_symbol_search",
              "header_compare",
              "header_saveload",
              "timeframes_toolbar",
              "context_menus",
              "display_market_status",
              "border_around_the_chart",
            ],
            enabled_features: ["use_localstorage_for_settings", "seconds_resolution", "create_volume_indicator"],
            overrides: {
              "paneProperties.background": theme === "dark" ? "#050709" : "#ffffff",
              "paneProperties.vertGridProperties.color": theme === "dark" ? "rgba(148,163,184,0.12)" : "rgba(71,85,105,0.18)",
              "paneProperties.horzGridProperties.color": theme === "dark" ? "rgba(148,163,184,0.08)" : "rgba(71,85,105,0.12)",
            },
          });

          widgetRef.current = widget;
          widget.onChartReady(() => {
            if (disposed) return;
            chartRef.current = widget.activeChart();
            setChartReady(true);
          });
        } catch (error) {
          console.error("[tv] widget init failed", error);
        }
      })();

      return () => {
        disposed = true;
        setChartReady(false);
        setBarsReady(false);
        if (replayTimerRef.current) {
          window.clearInterval(replayTimerRef.current);
          replayTimerRef.current = null;
        }
        overlaysRef.current.forEach((shape) => {
          try {
            shape.remove();
          } catch {
            /* ignore */
          }
        });
        overlaysRef.current = [];
        replayActiveRef.current = false;
        studiesAppliedRef.current = false;
        widgetRef.current?.remove();
        widgetRef.current = null;
        chartRef.current = null;
        datafeedRef.current = null;
      };
    }, [applyPlanLayers, containerId, devMode, onBarsLoaded, onRealtimeBar, planLayers, resolution, symbol, theme]);

    useEffect(() => {
      setBarsReady(false);
      studiesAppliedRef.current = false;
    }, [symbol, resolution]);

    useEffect(() => {
      if (!chartReady || !barsReady) return;
      const chart = chartRef.current;
      if (!chart) return;
      if (!studiesAppliedRef.current) {
        try {
          chart.createStudy("VWAP", false, false);
        } catch (error) {
          if (devMode) {
            console.warn("[tv] unable to add VWAP study", error);
          }
        }
        studiesAppliedRef.current = true;
      }
      applyPlanLayers(planLayers ?? null);
    }, [chartReady, barsReady, applyPlanLayers, planLayers, devMode]);

    const stopReplayInternal = React.useCallback(() => {
      replayActiveRef.current = false;
      if (replayTimerRef.current) {
        window.clearInterval(replayTimerRef.current);
        replayTimerRef.current = null;
      }
    }, []);

    const followLiveInternal = React.useCallback(() => {
      const chart = chartRef.current;
      if (!chart) return;
      stopReplayInternal();
      const lastTime = lastBarTimeRef.current;
      if (!lastTime) return;
      const lookbackSeconds = resolutionToSeconds(resolution) * 60;
      const from = Math.max(lastTime - lookbackSeconds * 1000, lastTime - 60 * 60 * 1000);
      try {
        chart.setVisibleRange({ from: from / 1000, to: lastTime / 1000 }, { applyDefaultRightMargin: true });
        chart.scrollToRealTime();
      } catch (error) {
        if (devMode) {
          console.warn("[tv] followLive failed", error);
        }
      }
    }, [devMode, resolution, stopReplayInternal]);

    const startReplayInternal = React.useCallback(() => {
      const chart = chartRef.current;
      const bars = barsCacheRef.current;
      if (!chart || bars.length < 10) return;
      stopReplayInternal();
      replayActiveRef.current = true;
      let index = Math.max(bars.length - 200, 0);
      const step = Math.max(1, Math.floor(bars.length / 120));

      const tick = () => {
        if (!replayActiveRef.current || !chart) {
          stopReplayInternal();
          return;
        }
        index += step;
        if (index >= bars.length) {
          followLiveInternal();
          return;
        }
        const end = bars[index].time;
        const startIdx = Math.max(index - 120, 0);
        const start = bars[startIdx].time;
        try {
          chart.setVisibleRange({ from: start / 1000, to: end / 1000 }, { applyDefaultRightMargin: false });
        } catch (error) {
          if (devMode) {
            console.warn("[tv] replay range failed", error);
          }
        }
      };

      tick();
      replayTimerRef.current = window.setInterval(tick, 500);
    }, [devMode, followLiveInternal, stopReplayInternal]);

    useImperativeHandle(
      ref,
      () => ({
        setResolution: (nextResolution: string) => {
          if (!widgetRef.current) return;
          try {
            widgetRef.current.activeChart().setResolution(nextResolution, () => undefined);
          } catch (error) {
            if (devMode) {
              console.warn("[tv] setResolution failed", error);
            }
          }
        },
        startReplay: () => startReplayInternal(),
        stopReplay: () => stopReplayInternal(),
        followLive: () => followLiveInternal(),
        refreshOverlays: () => {
          if (!chartReady || !barsReady) return;
          applyPlanLayers(planLayers ?? null);
        },
        getLastBarTime: () => lastBarTimeRef.current,
      }),
      [applyPlanLayers, barsReady, chartReady, devMode, followLiveInternal, planLayers, startReplayInternal, stopReplayInternal],
    );

    return <div id={containerId} className="h-[360px] w-full rounded-2xl border border-neutral-800/70" />;
  },
);

TradingViewChart.displayName = "TradingViewChart";

function resolutionToSeconds(resolution: string): number {
  const token = (resolution || "").toString().trim().toUpperCase();
  if (!token) return 60;
  if (token.endsWith("D")) {
    const days = Number.parseInt(token.replace("D", ""), 10) || 1;
    return days * 24 * 60 * 60;
  }
  if (token.endsWith("W")) {
    const weeks = Number.parseInt(token.replace("W", ""), 10) || 1;
    return weeks * 7 * 24 * 60 * 60;
  }
  const minutes = Number.parseInt(token, 10);
  if (!Number.isFinite(minutes) || minutes <= 0) {
    return 60;
  }
  return minutes * 60;
}

export default TradingViewChart;
