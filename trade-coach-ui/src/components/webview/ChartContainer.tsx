"use client";

import clsx from "clsx";
import Link from "next/link";
import PriceChart from "../PriceChart";
import type { LineData } from "lightweight-charts";
import type { SupportingLevel, ParsedUiState } from "@/lib/chart";
import { ensureCanonicalChartUrl } from "@/lib/chartUrl";

type ChartContainerProps = {
  symbol: string;
  interval?: string;
  chartTheme?: string;
  uiTheme?: "dark" | "light";
  uiState: ParsedUiState;
  priceSeries: LineData[];
  lastPrice?: number | null;
  entry?: number | null;
  stop?: number | null;
  trailingStop?: number | null;
  targets?: number[];
  supportingLevels: SupportingLevel[];
  showSupportingLevels: boolean;
  onToggleSupportingLevels: () => void;
  onHighlightLevel: (level: SupportingLevel | null) => void;
  highlightedLevel: SupportingLevel | null;
  interactiveUrl?: string | null;
  children?: React.ReactNode;
};

export default function ChartContainer({
  symbol,
  interval,
  chartTheme,
  uiTheme = "dark",
  uiState,
  priceSeries,
  lastPrice,
  entry,
  stop,
  trailingStop,
  targets,
  supportingLevels,
  showSupportingLevels,
  onToggleSupportingLevels,
  onHighlightLevel,
  highlightedLevel,
  interactiveUrl,
  children,
}: ChartContainerProps) {
  const isLight = uiTheme === "light";
  let safeInteractiveUrl: string | null = null;
  let chartLinkError = false;
  if (interactiveUrl) {
    const canonical = ensureCanonicalChartUrl(interactiveUrl);
    if (canonical) {
      try {
        const parsed = new URL(canonical);
        if (/[?&]plan_id=/.test(parsed.search)) {
          safeInteractiveUrl = canonical;
        } else {
          chartLinkError = true;
        }
      } catch {
        chartLinkError = true;
      }
    } else {
      chartLinkError = true;
    }
  }
  const toggleClasses = showSupportingLevels
    ? isLight
      ? "border-emerald-500/50 bg-emerald-400/20 text-emerald-700 hover:bg-emerald-400/30"
      : "border-emerald-500/50 bg-emerald-500/15 text-emerald-100 hover:bg-emerald-500/25"
    : isLight
      ? "border-slate-300 bg-white text-slate-700 hover:border-emerald-500 hover:text-emerald-600"
      : "border-neutral-700/70 bg-neutral-900/70 text-neutral-300 hover:border-neutral-600 hover:bg-neutral-800/80";

  const surfaceChart = isLight
    ? "border-slate-200 bg-white/85"
    : "border-neutral-800/80 bg-neutral-950/30";

  const markerSurface = isLight
    ? "border border-emerald-500/40 bg-emerald-400/20 text-emerald-700"
    : "border border-emerald-500/40 bg-emerald-500/15 text-emerald-100";

  return (
    <div className="flex h-full flex-col gap-4">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className={clsx("text-xs uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-500")}>
            Live chart
          </div>
          <div className="mt-1 flex items-baseline gap-3">
            <h2 className={clsx("text-2xl font-semibold", isLight ? "text-slate-900" : "text-white")}>{symbol}</h2>
            {interval ? (
              <span
                className={clsx(
                  "rounded-full px-3 py-1 text-xs uppercase tracking-[0.3em]",
                  isLight ? "border border-slate-300 bg-white text-slate-600" : "border border-neutral-700/70 bg-neutral-900/80 text-neutral-300",
                )}
              >
                {interval}
              </span>
            ) : null}
          </div>
          <p
            className={clsx(
              "mt-2 text-xs uppercase tracking-[0.3em]",
              isLight ? "text-slate-500" : "text-neutral-500",
            )}
          >
            Session:{" "}
            <span className={clsx(isLight ? "text-slate-700" : "text-neutral-200")}>{uiState.session.toUpperCase()}</span> · Style:{" "}
            <span className={clsx(isLight ? "text-slate-700" : "text-neutral-200")}>{uiState.style.toUpperCase()}</span> · Confidence{" "}
            <span className={clsx(isLight ? "text-emerald-600" : "text-emerald-200")}>{Math.round(uiState.confidence * 100)}%</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            type="button"
            onClick={onToggleSupportingLevels}
            className={clsx(
              "rounded-full border px-4 py-2 uppercase tracking-[0.25em] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/80",
              toggleClasses,
            )}
            aria-pressed={showSupportingLevels}
          >
            {showSupportingLevels ? "Hide Supporting Levels" : "Show Supporting Levels"}
          </button>
          {safeInteractiveUrl ? (
            <Link
              href={safeInteractiveUrl}
              target="_blank"
              className={clsx(
                "rounded-full border px-4 py-2 uppercase tracking-[0.25em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/70",
                isLight
                  ? "border-cyan-500/50 bg-cyan-500/15 text-cyan-700 hover:bg-cyan-500/25"
                  : "border-cyan-500/50 bg-cyan-500/15 text-cyan-100 hover:bg-cyan-500/25",
              )}
            >
              Open TV View
            </Link>
          ) : chartLinkError ? (
            <div className="rounded-full border border-rose-500/40 bg-rose-500/15 px-4 py-2 text-xs uppercase tracking-[0.25em] text-rose-200">
              Chart link unavailable (non-canonical). Regenerate plan.
            </div>
          ) : null}
        </div>
      </header>

      <div
        className={clsx(
          "relative flex-1 overflow-hidden rounded-3xl border transition-colors duration-300",
          surfaceChart,
        )}
      >
        <PriceChart
          data={priceSeries}
          lastPrice={lastPrice}
          entry={entry}
          stop={stop}
          trailingStop={trailingStop}
          targets={targets}
          supportingLevels={supportingLevels}
          showSupportingLevels={showSupportingLevels}
          onHighlightLevel={onHighlightLevel}
        />
        {highlightedLevel ? (
          <div
            className={clsx(
              "pointer-events-none absolute left-4 top-4 rounded-full px-3 py-1 text-xs uppercase tracking-[0.25em]",
              markerSurface,
            )}
          >
            {highlightedLevel.label} · {highlightedLevel.price.toFixed(2)}
          </div>
        ) : null}
      </div>

      {children}

      {chartTheme ? (
        <p
          className={clsx(
            "text-right text-[0.65rem] uppercase tracking-[0.3em]",
            isLight ? "text-slate-500" : "text-neutral-500",
          )}
        >
          Theme · <span className={clsx(isLight ? "text-slate-700" : "text-neutral-300")}>{chartTheme}</span>
        </p>
      ) : null}
    </div>
  );
}
