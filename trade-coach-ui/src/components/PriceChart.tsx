'use client';

import { MutableRefObject, useEffect, useMemo, useRef, useState } from "react";
import type { IChartApi, ISeriesApi, LineData } from "lightweight-charts";
import type { SupportingLevel } from "@/lib/chart";

type ChartLib = typeof import("lightweight-charts");

type PriceChartProps = {
  data: LineData[];
  lastPrice?: number | null;
  entry?: number | null;
  stop?: number | null;
  trailingStop?: number | null;
  targets?: number[];
  supportingLevels?: SupportingLevel[];
  showSupportingLevels?: boolean;
  onHighlightLevel?: (level: SupportingLevel | null) => void;
  compare?: {
    entry?: number | null;
    stop?: number | null;
    targets?: number[];
    label?: string;
  } | null;
};

export default function PriceChart({
  data,
  lastPrice,
  entry,
  stop,
  trailingStop,
  targets,
  supportingLevels,
  showSupportingLevels = true,
  onHighlightLevel,
  compare,
}: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const chartLibRef = useRef<ChartLib | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const entryLineRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const stopLineRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const trailLineRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const targetLinesRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]>[]>([]);
  const supportingLinesRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]>[]>([]);
  const supportingLevelsRef = useRef<SupportingLevel[]>([]);
  const ghostEntryRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const ghostStopRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const ghostTargetRefs = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]>[]>([]);
  const [libReady, setLibReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current || chartRef.current) return;

    let disposed = false;
    let resizeObserver: ResizeObserver | null = null;

    (async () => {
      const lib = await import("lightweight-charts");
      if (disposed || !containerRef.current || chartRef.current) return;
      chartLibRef.current = lib;

      const chart = (chartRef.current = lib.createChart(containerRef.current, {
        layout: {
          background: { type: lib.ColorType.Solid, color: "transparent" },
          textColor: "#E5E7EB",
        },
        grid: {
          vertLines: { color: "rgba(148, 163, 184, 0.15)" },
          horzLines: { color: "rgba(148, 163, 184, 0.08)" },
        },
        rightPriceScale: {
          visible: true,
          borderColor: "rgba(148, 163, 184, 0.2)",
          scaleMargins: { top: 0.2, bottom: 0.2 },
        },
        timeScale: {
          visible: true,
          borderColor: "rgba(148, 163, 184, 0.2)",
          timeVisible: true,
          secondsVisible: true,
        },
        crosshair: {
          mode: lib.CrosshairMode.Magnet,
          vertLine: { color: "rgba(74, 222, 128, 0.2)", style: lib.LineStyle.Solid },
          horzLine: { color: "rgba(74, 222, 128, 0.2)", style: lib.LineStyle.Solid },
        },
        autoSize: true,
      }));

      const areaSeries = chart.addAreaSeries({
        lineColor: "rgba(74, 222, 128, 0.8)",
        topColor: "rgba(74, 222, 128, 0.25)",
        bottomColor: "rgba(74, 222, 128, 0.02)",
      });

      seriesRef.current = areaSeries;

      resizeObserver = new ResizeObserver(() => {
        chart.applyOptions({ width: containerRef.current?.clientWidth, height: containerRef.current?.clientHeight });
      });
      resizeObserver.observe(containerRef.current);
      setLibReady(true);
    })();

    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      resizeObserver = null;
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      seriesRef.current = null;
      chartLibRef.current = null;
      supportingLinesRef.current = [];
      supportingLevelsRef.current = [];
      ghostTargetRefs.current = [];
      ghostEntryRef.current = null;
      ghostStopRef.current = null;
      setLibReady(false);
    };
  }, []);

  useEffect(() => {
    if (!libReady || !seriesRef.current || data.length === 0) return;
    seriesRef.current.setData(data);
  }, [data, libReady]);

  useEffect(() => {
    if (!libReady || !seriesRef.current || lastPrice === undefined || lastPrice === null) return;
    seriesRef.current.applyOptions({
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
    });
  }, [lastPrice, libReady]);

  useEffect(() => {
    const series = seriesRef.current;
    const lib = chartLibRef.current;
    if (!libReady || !series || !lib) return;

    const ensureLine = (
      ref: MutableRefObject<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>,
      price: number | null | undefined,
      options: Parameters<ISeriesApi<"Area">["createPriceLine"]>[0],
    ) => {
      if (price === undefined || price === null || !Number.isFinite(price)) {
        if (ref.current) {
          series.removePriceLine(ref.current);
          ref.current = null;
        }
        return;
      }
      if (!ref.current) {
        ref.current = series.createPriceLine({ ...options, price });
      } else {
        ref.current.applyOptions({ ...options, price });
      }
    };

    ensureLine(entryLineRef, entry, {
      color: "rgba(59, 130, 246, 0.7)",
      lineWidth: 2,
      lineStyle: lib.LineStyle.Dotted,
      title: "Entry",
    });

    ensureLine(stopLineRef, stop, {
      color: "rgba(248, 113, 113, 0.8)",
      lineWidth: 2,
      lineStyle: lib.LineStyle.Solid,
      title: "Stop",
    });

    ensureLine(trailLineRef, trailingStop, {
      color: "rgba(249, 115, 22, 0.9)",
      lineWidth: 2,
      lineStyle: lib.LineStyle.Dashed,
      title: "Trail",
    });

    targetLinesRef.current.forEach((line) => series.removePriceLine(line));
    targetLinesRef.current = [];

    if (targets?.length) {
      targets.forEach((target, index) => {
        if (!Number.isFinite(target)) return;
        const line = series.createPriceLine({
          price: Number(target),
          color: "rgba(16, 185, 129, 0.7)",
          lineWidth: 1,
          lineStyle: lib.LineStyle.Dashed,
          title: `TP${index + 1}`,
        });
        targetLinesRef.current.push(line);
      });
    }

    return () => {
      targetLinesRef.current.forEach((line) => series.removePriceLine(line));
      targetLinesRef.current = [];
    };
  }, [entry, stop, trailingStop, targets, libReady]);

  useEffect(() => {
    const series = seriesRef.current;
    const lib = chartLibRef.current;
    if (!libReady || !series || !lib) return;

    supportingLinesRef.current.forEach((line) => {
      series.removePriceLine(line);
    });
    supportingLinesRef.current = [];
    supportingLevelsRef.current = Array.isArray(supportingLevels) ? supportingLevels : [];

    if (!showSupportingLevels || !supportingLevelsRef.current.length) {
      if (onHighlightLevel) onHighlightLevel(null);
      return;
    }

    supportingLevelsRef.current.forEach((level) => {
      const line = series.createPriceLine({
        price: level.price,
        color: "rgba(148, 163, 184, 0.5)",
        lineStyle: lib.LineStyle.Dotted,
        lineWidth: 1,
        axisLabelVisible: true,
        title: level.label,
      });
      supportingLinesRef.current.push(line);
    });

    return () => {
      supportingLinesRef.current.forEach((line) => series.removePriceLine(line));
      supportingLinesRef.current = [];
    };
  }, [supportingLevels, showSupportingLevels, onHighlightLevel, libReady]);

  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!libReady || !chart || !series || !onHighlightLevel || !supportingLevelsRef.current.length) return;

    const handler = (param: Parameters<IChartApi["subscribeCrosshairMove"]>[0]) => {
      if (!param) {
        onHighlightLevel(null);
        return;
      }
      const price = param.seriesPrices?.get(series);
      if (typeof price !== "number") {
        onHighlightLevel(null);
        return;
      }
      const candidate = supportingLevelsRef.current.reduce<SupportingLevel | null>((closest, level) => {
        const delta = Math.abs(level.price - price);
        if (delta > computeTolerance(level.price)) {
          return closest;
        }
        if (!closest) return level;
        const currentDelta = Math.abs(closest.price - price);
        return delta < currentDelta ? level : closest;
      }, null);
      onHighlightLevel(candidate ?? null);
    };

    chart.subscribeCrosshairMove(handler);
    return () => {
      chart.unsubscribeCrosshairMove(handler);
    };
  }, [onHighlightLevel, libReady]);

  useEffect(() => {
    const series = seriesRef.current;
    const lib = chartLibRef.current;
    if (!libReady || !series || !lib) return;

    const clearGhost = () => {
      if (ghostEntryRef.current) {
        series.removePriceLine(ghostEntryRef.current);
        ghostEntryRef.current = null;
      }
      if (ghostStopRef.current) {
        series.removePriceLine(ghostStopRef.current);
        ghostStopRef.current = null;
      }
      ghostTargetRefs.current.forEach((line) => series.removePriceLine(line));
      ghostTargetRefs.current = [];
    };

    clearGhost();

    if (!compare) return;

    const ghostLabel = compare.label ? `${compare.label} ` : "Scenario ";

    if (Number.isFinite(compare.entry as number)) {
      ghostEntryRef.current = series.createPriceLine({
        price: Number(compare.entry),
        color: "rgba(56, 189, 248, 0.5)",
        lineWidth: 1,
        lineStyle: lib.LineStyle.Dotted,
        title: `${ghostLabel}Entry`,
      });
    }
    if (Number.isFinite(compare.stop as number)) {
      ghostStopRef.current = series.createPriceLine({
        price: Number(compare.stop),
        color: "rgba(248, 113, 113, 0.6)",
        lineWidth: 1,
        lineStyle: lib.LineStyle.Dashed,
        title: `${ghostLabel}Stop`,
      });
    }
    (compare.targets || []).forEach((target, idx) => {
      if (!Number.isFinite(target)) return;
      const line = series.createPriceLine({
        price: Number(target),
        color: "rgba(96, 165, 250, 0.5)",
        lineWidth: 1,
        lineStyle: lib.LineStyle.Dashed,
        title: `${ghostLabel}TP${idx + 1}`,
      });
      ghostTargetRefs.current.push(line);
    });

    return () => {
      clearGhost();
    };
  }, [libReady, compare ? JSON.stringify(compare) : ""]);

  const chartClass = useMemo(() => "h-[360px] w-full", []);

  return <div ref={containerRef} className={chartClass} />;
}

function computeTolerance(price: number): number {
  if (!Number.isFinite(price) || price <= 0) return 0.1;
  return Math.max(price * 0.0015, 0.05);
}
