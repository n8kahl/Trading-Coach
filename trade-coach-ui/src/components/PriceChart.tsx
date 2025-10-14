'use client';

import { MutableRefObject, useEffect, useRef } from "react";
import {
  ColorType,
  CrosshairMode,
  LineStyle,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
} from "lightweight-charts";

type PriceChartProps = {
  data: LineData[];
  lastPrice?: number | null;
  entry?: number | null;
  stop?: number | null;
  trailingStop?: number | null;
  targets?: number[];
};

export default function PriceChart({ data, lastPrice, entry, stop, trailingStop, targets }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const entryLineRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const stopLineRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const trailLineRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]> | null>(null);
  const targetLinesRef = useRef<ReturnType<ISeriesApi<"Area">["createPriceLine"]>[]>([]);

  useEffect(() => {
    if (!containerRef.current || chartRef.current) return;

    const chart = (chartRef.current = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
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
        mode: CrosshairMode.Magnet,
        vertLine: { color: "rgba(74, 222, 128, 0.2)", style: LineStyle.Solid },
        horzLine: { color: "rgba(74, 222, 128, 0.2)", style: LineStyle.Solid },
      },
      autoSize: true,
    }));

    const areaSeries = chart.addAreaSeries({
      lineColor: "rgba(74, 222, 128, 0.8)",
      topColor: "rgba(74, 222, 128, 0.25)",
      bottomColor: "rgba(74, 222, 128, 0.02)",
    });

    seriesRef.current = areaSeries;

    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({ width: containerRef.current?.clientWidth, height: containerRef.current?.clientHeight });
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      seriesRef.current = null;
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return;
    seriesRef.current.setData(data);
  }, [data]);

  useEffect(() => {
    if (!seriesRef.current || lastPrice === undefined || lastPrice === null) return;
    const series = seriesRef.current;
    series.applyOptions({
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
    });
  }, [lastPrice]);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

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
      lineStyle: LineStyle.Dotted,
      title: "Entry",
    });

    ensureLine(stopLineRef, stop, {
      color: "rgba(248, 113, 113, 0.8)",
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      title: "Stop",
    });

    ensureLine(trailLineRef, trailingStop, {
      color: "rgba(249, 115, 22, 0.9)",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      title: "Trail",
    });

    targetLinesRef.current.forEach((line) => {
      series.removePriceLine(line);
    });
    targetLinesRef.current = [];

    if (targets && targets.length) {
      targets.forEach((target, index) => {
        if (!Number.isFinite(target)) return;
        const line = series.createPriceLine({
          price: target,
          color: "rgba(16, 185, 129, 0.7)",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          title: `TP${index + 1}`,
        });
        targetLinesRef.current.push(line);
      });
    }

    return () => {
      targetLinesRef.current.forEach((line) => series.removePriceLine(line));
      targetLinesRef.current = [];
    };
  }, [entry, stop, trailingStop, targets]);

  return <div ref={containerRef} className="h-[360px] w-full" />;
}
