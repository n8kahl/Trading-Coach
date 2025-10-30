"use client";

import clsx from "clsx";
import { useMemo, useRef, useState } from "react";
import TradingViewChart, { TradingViewChartHandle } from "@/components/TradingViewChart";
import type { TVBar } from "@/lib/tradingview/datafeed";
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
  supportingVisible: boolean;
  followLive: boolean;
  streamingEnabled: boolean;
  onSetFollowLive(value: boolean): void;
  onToggleStreaming(): void;
  onToggleSupporting(): void;
  timeframe: string;
  timeframeOptions: TimeframeOption[];
  onSelectTimeframe(value: string): void;
  onLastBarTimeChange(time: number | null): void;
  onReplayStateChange?: (state: "idle" | "playing") => void;
  devMode?: boolean;
  theme: "dark" | "light";
};

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
  supportingVisible,
  followLive,
  streamingEnabled,
  onSetFollowLive,
  onToggleStreaming,
  onToggleSupporting,
  timeframe,
  timeframeOptions,
  onSelectTimeframe,
  onLastBarTimeChange,
  onReplayStateChange,
  devMode = false,
  theme,
}: PlanChartPanelProps) {
  const chartHandle = useRef<TradingViewChartHandle | null>(null);
  const [replayActive, setReplayActive] = useState(false);

  const symbol = plan.symbol?.toUpperCase() ?? "—";
  const style = plan.style ?? plan.structured_plan?.style ?? null;
  const session = plan.session_state?.status ?? null;
  const triggerRule = extractExecutionRule(plan, "trigger");
  const invalidationRule = extractExecutionRule(plan, "invalidation");
  const reloadRule = extractExecutionRule(plan, "reload");
  const scaleRule = extractExecutionRule(plan, "scale");
  const planAsOf = plan.session_state?.as_of ?? layers?.as_of ?? null;
  const planVersion = plan.version ?? (plan as Record<string, unknown>).version ?? null;
  const planId = plan.plan_id;

  const filteredLayers = useMemo(() => {
    if (!layers) return null;
    if (supportingVisible) return layers;
    const groups = (layers.meta?.level_groups ?? {}) as Record<string, unknown>;
    type LevelGroupEntry = { price?: number | null };
    const primaryEntries = Array.isArray(groups.primary)
      ? (groups.primary as LevelGroupEntry[])
      : [];
    const priceSet = new Set<number>();
    primaryEntries.forEach((entry) => {
      if (entry && typeof entry.price === "number") {
        priceSet.add(entry.price);
      }
    });
    const filteredLevels = Array.isArray(layers.levels)
      ? layers.levels.filter((level) => typeof level?.price === "number" && priceSet.has(level.price))
      : [];
    return { ...layers, levels: filteredLevels };
  }, [layers, supportingVisible]);

  const handleResolution = (value: string) => {
    onSelectTimeframe(value);
    chartHandle.current?.setResolution(value);
    if (replayActive) {
      chartHandle.current?.stopReplay();
      setReplayActive(false);
      onReplayStateChange?.("idle");
    }
    if (followLive) {
      chartHandle.current?.followLive();
    }
  };

  const handleReplayToggle = () => {
    if (replayActive) {
      chartHandle.current?.stopReplay();
      setReplayActive(false);
      onReplayStateChange?.("idle");
      chartHandle.current?.followLive();
      onSetFollowLive(true);
    } else {
      chartHandle.current?.startReplay();
      setReplayActive(true);
      onReplayStateChange?.("playing");
      onSetFollowLive(false);
    }
  };

  const handleFollowLiveClick = () => {
    const next = !followLive;
    if (next) {
      chartHandle.current?.followLive();
    } else {
      chartHandle.current?.stopReplay();
      setReplayActive(false);
      onReplayStateChange?.("idle");
    }
    onSetFollowLive(next);
  };

  const handleStreamingClick = () => {
    onToggleStreaming();
  };

  const handleSupportingClick = () => {
    onToggleSupporting();
    chartHandle.current?.refreshOverlays();
  };

  const handleBarsLoaded = (bars: TVBar[]) => {
    if (!bars.length) {
      onLastBarTimeChange(null);
      return;
    }
    const lastTime = bars[bars.length - 1]?.time ?? null;
    onLastBarTimeChange(lastTime);
    if (followLive && lastTime != null) {
      chartHandle.current?.followLive();
    }
  };

  const handleRealtimeBar = (bar: TVBar) => {
    const barTime = bar?.time ?? null;
    onLastBarTimeChange(barTime);
    if (followLive && barTime != null) {
      chartHandle.current?.followLive();
    }
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
            <ToggleChip label="Follow Live" active={followLive} onClick={handleFollowLiveClick} />
            <ToggleChip label="Streaming" active={streamingEnabled} onClick={handleStreamingClick} />
            <ToggleChip
              label={supportingVisible ? "Supporting On" : "Supporting Off"}
              active={supportingVisible}
              onClick={handleSupportingClick}
            />
            <ToggleChip
              label={replayActive ? "Stop Replay" : "Replay"}
              active={replayActive}
              onClick={handleReplayToggle}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {timeframeOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleResolution(option.value)}
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
        <TradingViewChart
          ref={chartHandle}
          symbol={symbol}
          planId={planId}
          resolution={timeframe}
          planLayers={filteredLayers}
          theme={theme}
          devMode={devMode}
          onBarsLoaded={handleBarsLoaded}
          onRealtimeBar={handleRealtimeBar}
        />
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <PlanControlCard title="Trigger" body={triggerRule} tone="emerald" />
        <PlanControlCard title="Invalidate" body={invalidationRule} tone="rose" />
        <PlanControlCard title="Scale" body={scaleRule} tone="sky" />
        <PlanControlCard title="Reload" body={reloadRule} tone="amber" />
      </section>

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
