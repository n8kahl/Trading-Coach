"use client";

import { useMemo } from "react";
import { useSearchParams } from "next/navigation";
import TradingViewChart from "@/components/TradingViewChart";

function normalizeResolution(raw: string | null): string {
  if (!raw) return "5";
  const token = raw.toString().trim();
  if (!token) return "5";
  const upper = token.toUpperCase();
  if (upper === "D") return "1D";
  if (upper === "W") return "1W";
  return upper;
}

export default function ChartPage() {
  const params = useSearchParams();

  const { symbol, planId, resolution, theme } = useMemo(() => {
    const symbolParam = params.get("symbol") || params.get("ticker") || "SPY";
    const normalizedSymbol = symbolParam.toUpperCase();
    const planParam = params.get("plan_id") || params.get("planId");
    const resolutionParam = normalizeResolution(params.get("interval") || params.get("resolution"));
    const requestedTheme = (params.get("theme") || "").toLowerCase();
    const normalizedTheme = requestedTheme === "light" ? "light" : "dark";
    return {
      symbol: normalizedSymbol,
      planId: planParam || normalizedSymbol,
      resolution: resolutionParam,
      theme: normalizedTheme as "dark" | "light",
    };
  }, [params]);

  const planLink = params.get("plan_id") ? `/plan/${encodeURIComponent(params.get("plan_id") as string)}` : null;

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
          <TradingViewChart symbol={symbol} planId={planId} resolution={resolution} planLayers={null} theme={theme} />
        </div>
      </main>
    </div>
  );
}
