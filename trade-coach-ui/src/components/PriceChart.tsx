'use client';

import { useEffect, useRef } from "react";
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
};

export default function PriceChart({ data, lastPrice }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

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

  return <div ref={containerRef} className="h-[360px] w-full" />;
}
