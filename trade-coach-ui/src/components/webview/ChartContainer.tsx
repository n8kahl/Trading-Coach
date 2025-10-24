"use client";

import Link from "next/link";
import PriceChart from "../PriceChart";
import type { LineData } from "lightweight-charts";
import type { SupportingLevel, ParsedUiState } from "@/lib/chart";

type ChartContainerProps = {
  symbol: string;
  interval?: string;
  theme?: string;
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
  theme,
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
  return (
    <div className="flex h-full flex-col gap-4">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-neutral-500">Live chart</div>
          <div className="mt-1 flex items-baseline gap-3">
            <h2 className="text-2xl font-semibold text-white">{symbol}</h2>
            {interval ? <span className="rounded-full border border-neutral-700/70 bg-neutral-900/80 px-3 py-1 text-xs uppercase tracking-[0.3em] text-neutral-300">{interval}</span> : null}
          </div>
          <p className="mt-2 text-xs uppercase tracking-[0.3em] text-neutral-500">
            Session: <span className="text-neutral-200">{uiState.session.toUpperCase()}</span> · Style:{" "}
            <span className="text-neutral-200">{uiState.style.toUpperCase()}</span> · Confidence{" "}
            <span className="text-emerald-200">{Math.round(uiState.confidence * 100)}%</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            type="button"
            onClick={onToggleSupportingLevels}
            className={`rounded-full border px-4 py-2 uppercase tracking-[0.25em] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/80 ${
              showSupportingLevels
                ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-100 hover:bg-emerald-500/25"
                : "border-neutral-700/70 bg-neutral-900/70 text-neutral-300 hover:border-neutral-600 hover:bg-neutral-800/80"
            }`}
          >
            {showSupportingLevels ? "Hide Supporting Levels" : "Show Supporting Levels"}
          </button>
          {interactiveUrl ? (
            <Link
              href={interactiveUrl}
              target="_blank"
              className="rounded-full border border-cyan-500/50 bg-cyan-500/15 px-4 py-2 uppercase tracking-[0.25em] text-cyan-100 transition hover:bg-cyan-500/25"
            >
              Open TV View
            </Link>
          ) : null}
        </div>
      </header>

      <div className="relative flex-1 overflow-hidden rounded-3xl border border-neutral-800/80 bg-neutral-950/30">
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
          <div className="pointer-events-none absolute left-4 top-4 rounded-full border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 text-xs uppercase tracking-[0.25em] text-emerald-100">
            {highlightedLevel.label} · {highlightedLevel.price.toFixed(2)}
          </div>
        ) : null}
      </div>

      {children}

      {theme ? (
        <p className="text-right text-[0.65rem] uppercase tracking-[0.3em] text-neutral-500">
          Theme · <span className="text-neutral-300">{theme}</span>
        </p>
      ) : null}
    </div>
  );
}
