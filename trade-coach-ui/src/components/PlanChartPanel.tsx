"use client";

import clsx from "clsx";
import type { LineData } from "lightweight-charts";
import PriceChart from "@/components/PriceChart";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";

type TimeframeOption = {
  value: string;
  label: string;
};

type PlanChartPanelProps = {
  plan: PlanSnapshot["plan"];
  layers: PlanLayers | null;
  primaryLevels: SupportingLevel[];
  supportingLevels: SupportingLevel[];
  supportingVisible: boolean;
  followLive: boolean;
  streamingEnabled: boolean;
  onToggleFollowLive(): void;
  onToggleStreaming(): void;
  onToggleSupporting(): void;
  highlightedLevel: SupportingLevel | null;
  onHighlightLevel(level: SupportingLevel | null): void;
  priceSeries: LineData[];
  priceSeriesStatus: "idle" | "loading" | "ready" | "error";
  timeframe: string;
  timeframeOptions: TimeframeOption[];
  onSelectTimeframe(value: string): void;
  onReloadSeries(): void;
  lastPrice?: number | null;
};

function resolveEntry(plan: PlanSnapshot["plan"]): number | null {
  const structured = plan.structured_plan as any;
  if (structured?.entry?.level != null) return Number(structured.entry.level);
  if (plan.entry != null) return Number(plan.entry);
  return null;
}

function resolveStop(plan: PlanSnapshot["plan"]): number | null {
  const structured = plan.structured_plan as any;
  if (structured?.stop != null) return Number(structured.stop);
  if (plan.stop != null) return Number(plan.stop);
  return null;
}

function resolveTargets(plan: PlanSnapshot["plan"]): number[] {
  if (Array.isArray(plan.targets) && plan.targets.length) {
    return plan.targets.filter((value): value is number => typeof value === "number");
  }
  const structured = plan.structured_plan as any;
  if (structured?.targets?.length) {
    return structured.targets.filter((value: unknown): value is number => typeof value === "number");
  }
  return [];
}

function extractExecutionRule(plan: PlanSnapshot["plan"], key: string): string | null {
  const block = (plan as Record<string, unknown>).execution_rules;
  if (block && typeof block === "object" && typeof (block as Record<string, unknown>)[key] === "string") {
    return ((block as Record<string, unknown>)[key] as string) || null;
  }
  return null;
}

export default function PlanChartPanel({
  plan,
  layers,
  primaryLevels,
  supportingLevels,
  supportingVisible,
  followLive,
  streamingEnabled,
  onToggleFollowLive,
  onToggleStreaming,
  onToggleSupporting,
  highlightedLevel,
  onHighlightLevel,
  priceSeries,
  priceSeriesStatus,
  timeframe,
  timeframeOptions,
  onSelectTimeframe,
  onReloadSeries,
  lastPrice,
}: PlanChartPanelProps) {
  const symbol = plan.symbol?.toUpperCase() ?? "—";
  const style = plan.style ?? plan.structured_plan?.style ?? null;
  const session = plan.session_state?.status ?? null;
  const entry = resolveEntry(plan);
  const stop = resolveStop(plan);
  const targets = resolveTargets(plan);
  const trailingStop = (plan as Record<string, unknown>).trailing_stop ?? null;
  const triggerRule = extractExecutionRule(plan, "trigger");
  const invalidationRule = extractExecutionRule(plan, "invalidation");
  const reloadRule = extractExecutionRule(plan, "reload");
  const scaleRule = extractExecutionRule(plan, "scale");
  const planAsOf = plan.session_state?.as_of ?? layers?.as_of ?? null;
  const planVersion = plan.version ?? (plan as Record<string, unknown>).version ?? null;
  const planId = plan.plan_id;

  const renderSeries = () => {
    if (priceSeriesStatus === "loading" && priceSeries.length === 0) {
      return (
        <div className="flex h-full items-center justify-center text-sm text-neutral-400">
          Loading price history…
        </div>
      );
    }
    if (priceSeriesStatus === "error" && priceSeries.length === 0) {
      return (
        <div className="flex h-full items-center justify-center text-sm text-rose-300">
          Unable to load price history. Retry or check data feed.
        </div>
      );
    }
    return (
      <PriceChart
        data={priceSeries}
        lastPrice={lastPrice ?? undefined}
        entry={entry ?? undefined}
        stop={stop ?? undefined}
        trailingStop={typeof trailingStop === "number" ? trailingStop : undefined}
        targets={targets}
        supportingLevels={supportingLevels}
        showSupportingLevels={supportingVisible}
        onHighlightLevel={onHighlightLevel}
      />
    );
  };

  return (
    <div className="flex h-full flex-col gap-6">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-[0.4em] text-emerald-300">{symbol}</h1>
            {style ? (
              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs uppercase tracking-[0.3em] text-emerald-200">
                {style}
              </span>
            ) : null}
            {session ? (
              <span className="rounded-full border border-sky-500/40 bg-sky-500/10 px-3 py-1 text-xs uppercase tracking-[0.3em] text-sky-200">
                {session}
              </span>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-400">
            <span>
              Plan {planId}
              {planVersion ? ` · v${planVersion}` : null}
            </span>
            {planAsOf ? <span>As of {new Date(planAsOf).toLocaleString()}</span> : null}
            {layers?.planning_context ? <span>{layers.planning_context.toUpperCase()}</span> : null}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em] text-neutral-400">
            <ToggleChip label="Follow Live" active={followLive} onClick={onToggleFollowLive} />
            <ToggleChip label="Streaming" active={streamingEnabled} onClick={onToggleStreaming} />
            <ToggleChip
              label={supportingVisible ? "Supporting On" : "Supporting Off"}
              active={supportingVisible}
              onClick={onToggleSupporting}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {timeframeOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => onSelectTimeframe(option.value)}
              className={clsx(
                "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                option.value === timeframe
                  ? "border border-emerald-400/60 bg-emerald-400/10 text-emerald-200"
                  : "border border-neutral-800/60 bg-neutral-900/70 text-neutral-300 hover:border-emerald-400/40 hover:text-emerald-200",
              )}
            >
              {option.label}
            </button>
          ))}
          <button
            type="button"
            onClick={onReloadSeries}
            className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-neutral-300 transition hover:border-emerald-400/50 hover:text-emerald-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
          >
            Refresh
          </button>
        </div>
      </header>

      <section className="space-y-3">
        <div
          className={clsx(
            "rounded-2xl border border-neutral-800/70 bg-neutral-950/40 p-3 text-xs uppercase tracking-[0.25em] text-neutral-400",
            supportingVisible ? "shadow-[0_0_25px_rgba(16,185,129,0.15)]" : "",
          )}
        >
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-neutral-500">Primary levels:</span>
            {primaryLevels.length === 0 ? (
              <span className="text-neutral-500">None persisted</span>
            ) : (
              primaryLevels.map((level) => (
                <span
                  key={`${level.label}-${level.price}`}
                  className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-emerald-200"
                >
                  {level.label} {level.price.toFixed(2)}
                </span>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="relative min-h-[360px] overflow-hidden rounded-3xl border border-neutral-800/70 bg-neutral-950/40 p-2">
        {renderSeries()}
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <PlanControlCard title="Trigger" body={triggerRule} tone="emerald" />
        <PlanControlCard title="Invalidate" body={invalidationRule} tone="rose" />
        <PlanControlCard title="Scale" body={scaleRule} tone="sky" />
        <PlanControlCard title="Reload" body={reloadRule} tone="amber" />
      </section>

      {highlightedLevel ? (
        <div className="rounded-2xl border border-emerald-500/40 bg-emerald-500/10 p-4 text-sm text-emerald-100">
          <div className="flex items-center justify-between">
            <div className="font-semibold uppercase tracking-[0.4em] text-emerald-300">{highlightedLevel.label}</div>
            <div className="font-mono text-lg text-emerald-200">{highlightedLevel.price.toFixed(2)}</div>
          </div>
          <p className="mt-2 text-xs text-emerald-200/80">Hovering level — toggle supporting levels to lock view.</p>
        </div>
      ) : null}
    </div>
  );
}

function ToggleChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
        active
          ? "border-emerald-400/60 bg-emerald-400/10 text-emerald-200"
          : "border-neutral-800/60 bg-neutral-900/60 text-neutral-300 hover:border-emerald-400/40 hover:text-emerald-200",
      )}
    >
      {label}
    </button>
  );
}

function PlanControlCard({ title, body, tone }: { title: string; body: string | null; tone: "emerald" | "rose" | "sky" | "amber" }) {
  if (!body) {
    return (
      <div className="rounded-2xl border border-neutral-800/60 bg-neutral-950/30 p-4 text-sm text-neutral-400">
        <div className="text-xs font-semibold uppercase tracking-[0.3em] text-neutral-500">{title}</div>
        <p className="mt-2 text-neutral-500">No guidance published.</p>
      </div>
    );
  }
  const toneMap: Record<typeof tone, string> = {
    emerald: "border-emerald-500/40 bg-emerald-500/10 text-emerald-100",
    rose: "border-rose-500/40 bg-rose-500/10 text-rose-100",
    sky: "border-sky-500/40 bg-sky-500/10 text-sky-100",
    amber: "border-amber-500/40 bg-amber-500/10 text-amber-100",
  };
  return (
    <div className={clsx("rounded-2xl border p-4 text-sm leading-relaxed", toneMap[tone])}>
      <div className="text-xs font-semibold uppercase tracking-[0.3em] opacity-80">{title}</div>
      <p className="mt-2">{body}</p>
    </div>
  );
}
