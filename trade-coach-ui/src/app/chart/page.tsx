"use client";

import { Suspense, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import PlanPriceChart, { type ChartOverlayState } from "@/components/PlanPriceChart";
import { usePriceSeries } from "@/lib/hooks/usePriceSeries";
import { parseSupportingLevels, parseTargets } from "@/lib/chart";
import type { PlanLayers } from "@/lib/types";
import { PUBLIC_UI_BASE_URL } from "@/lib/env";

function normalizeResolution(raw: string | null): string {
  if (!raw) return "5";
  const token = raw.toString().trim();
  if (!token) return "5";
  const upper = token.toUpperCase();
  if (upper === "D") return "1D";
  if (upper === "W") return "1W";
  return upper;
}

export default function ChartPageWrapper() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-[#050709] text-neutral-200">
          Loading chart…
        </div>
      }
    >
      <ChartPage />
    </Suspense>
  );
}

function ChartPage() {
  const params = useSearchParams();

  const { symbol, planId, resolution, theme, overlays, planLink } = useMemo(() => {
    const parseNumeric = (value: string | null): number | null => {
      if (!value) return null;
      const parsed = Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : null;
    };

    const symbolParam = params.get("symbol") || params.get("ticker") || "SPY";
    const normalizedSymbol = symbolParam.toUpperCase();
    const planParam = params.get("plan_id") || params.get("planId");
    const resolutionParam = normalizeResolution(params.get("interval") || params.get("resolution"));
    const requestedTheme = (params.get("theme") || "").toLowerCase();
    const normalizedTheme = requestedTheme === "light" ? "light" : "dark";
    const entry = parseNumeric(params.get("entry"));
    const stop = parseNumeric(params.get("stop"));
    const trailingStop =
      parseNumeric(params.get("trail")) ??
      parseNumeric(params.get("trailing")) ??
      parseNumeric(params.get("trail_stop")) ??
      parseNumeric(params.get("trailingStop"));
    const targetList = parseTargets(params.get("tp")) ?? [];
    const emaParam = params.get("ema") || "";
    const emaPeriods = emaParam
      ? emaParam
          .split(",")
          .map((token) => Number.parseInt(token.trim(), 10))
          .filter((value) => Number.isFinite(value) && value > 1)
      : [];
    const showVWAP = (() => {
      const raw = params.get("vwap");
      if (raw == null) return true;
      const token = raw.trim().toLowerCase();
      if (token === "0" || token === "false" || token === "off") return false;
      if (token === "1" || token === "true" || token === "on") return true;
      return true;
    })();
    const supportingLevels = parseSupportingLevels(params.get("levels"));
    const layers: PlanLayers | null = supportingLevels.length
      ? {
          plan_id: planParam || normalizedSymbol,
          symbol: normalizedSymbol,
          interval: resolutionParam,
          as_of: null,
          planning_context: null,
          precision: null,
          levels: supportingLevels.map((level) => ({
            price: level.price,
            label: level.label,
            kind: null,
          })),
          zones: [],
          annotations: [],
          meta: {},
        }
      : null;

    const planLinkAbsolute = planParam ? `${PUBLIC_UI_BASE_URL}/plan/${encodeURIComponent(planParam)}` : null;

    return {
      symbol: normalizedSymbol,
      planId: planParam || normalizedSymbol,
      resolution: resolutionParam,
      theme: normalizedTheme as "dark" | "light",
      overlays: {
        entry,
        stop,
        trailingStop,
        targets: targetList.map((price, index) => ({ price, label: `TP${index + 1}` })),
        emaPeriods,
        showVWAP,
        layers,
      } as ChartOverlayState,
      planLink: planLinkAbsolute,
    };
  }, [params]);

  const {
    bars,
    status,
    error,
  } = usePriceSeries(symbol, resolution, [symbol, resolution]);

  const chartStatusMessage = useMemo(() => {
    if (status === "loading") return "Loading price data…";
    if (status === "error") return error?.message ?? "Price data unavailable";
    if (status === "ready" && bars.length === 0) return "No market data available";
    return null;
  }, [status, error, bars.length]);

  return (
    <div
      className={`min-h-screen ${theme === "light" ? "bg-slate-50 text-slate-900" : "bg-[#050709] text-neutral-100"}`}
    >
      <header
        className="flex items-center justify-between border-b border-slate-800/60 px-6 py-4"
        style={theme === "light" ? { borderColor: "rgba(15, 23, 42, 0.1)" } : undefined}
      >
        <div>
          <h1 className="text-xl font-semibold tracking-[0.35em] uppercase">{symbol}</h1>
          <p className="mt-1 text-xs uppercase tracking-[0.25em] text-neutral-400">
            Interval {resolution}
          </p>
        </div>
        {planLink ? (
          <a
            href={planLink}
            className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-emerald-200 transition hover:border-emerald-400 hover:text-emerald-100"
          >
            Open Plan Console
          </a>
        ) : null}
      </header>
      <main className="px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-5xl rounded-3xl border border-neutral-800/70 bg-neutral-950/50 p-4 shadow-[0_0_25px_rgba(15,118,110,0.15)]">
          <div className="relative min-h-[360px]">
            {chartStatusMessage ? (
              <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-neutral-950/70 text-sm text-neutral-300">
                {chartStatusMessage}
              </div>
            ) : null}
            <PlanPriceChart planId={planId} symbol={symbol} resolution={resolution} theme={theme} data={bars} overlays={overlays} />
          </div>
        </div>
      </main>
    </div>
  );
}

export const dynamic = "force-dynamic";
