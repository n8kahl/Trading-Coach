"use client";

import clsx from "clsx";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlanPriceChart, { type ChartOverlayState, type PlanPriceChartHandle } from "@/components/PlanPriceChart";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { usePriceSeries } from "@/lib/hooks/usePriceSeries";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanLayers, PlanSnapshot, TargetMetaEntry } from "@/lib/types";
import { subscribePlanEvents } from "@/lib/plan/events";

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
  timeframe: string;
  timeframeOptions: TimeframeOption[];
  onSelectTimeframe(value: string): void;
  onLastBarTimeChange(time: number | null): void;
  onReplayStateChange?: (state: "idle" | "playing") => void;
  devMode?: boolean;
  theme: "dark" | "light";
  priceRefreshToken?: number;
  highlightLevelId?: string | null;
  hiddenLevelIds?: string[];
};

const PLAN_AS_OF_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  year: "numeric",
  month: "short",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: true,
});

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
  timeframe,
  timeframeOptions,
  onSelectTimeframe,
  onLastBarTimeChange,
  onReplayStateChange,
  devMode = false,
  theme,
  priceRefreshToken = 0,
  highlightLevelId = null,
  hiddenLevelIds = [],
}: PlanChartPanelProps) {
  const chartHandle = useRef<PlanPriceChartHandle | null>(null);
  const [replayActive, setReplayActive] = useState(false);
  const [recentPlanEvent, setRecentPlanEvent] = useState<{ type: string; at: number } | null>(null);
  const [chartExpanded, setChartExpanded] = useState(false);

  const symbol = plan.symbol?.toUpperCase() ?? "—";
  const style = plan.style ?? plan.structured_plan?.style ?? null;
  const session = plan.session_state?.status ?? null;
  const triggerRule = extractExecutionRule(plan, "trigger");
  const invalidationRule = extractExecutionRule(plan, "invalidation");
  const reloadRule = extractExecutionRule(plan, "reload");
  const scaleRule = extractExecutionRule(plan, "scale");
  const planAsOf = plan.session_state?.as_of ?? layers?.as_of ?? null;
  const planAsOfLabel = useMemo(() => {
    if (!planAsOf) return null;
    const date = new Date(planAsOf);
    if (Number.isNaN(date.getTime())) return null;
    return PLAN_AS_OF_FORMATTER.format(date);
  }, [planAsOf]);
  const planVersion = plan.version ?? (plan as Record<string, unknown>).version ?? null;
  const planId = plan.plan_id;
  const priceSymbol = plan.symbol ? plan.symbol.toUpperCase() : null;

  useEffect(() => {
    if (!planId) return;
    const unsubscribe = subscribePlanEvents(planId, (event) => {
      if (event.type === "tp_hit" || event.type === "stop_hit") {
        setRecentPlanEvent({ type: event.type, at: Date.now() });
      }
    });
    return unsubscribe;
  }, [planId]);

  useEffect(() => {
    if (!recentPlanEvent) return;
    if (typeof window === "undefined") return () => undefined;
    const timer = window.setTimeout(() => setRecentPlanEvent(null), 1500);
    return () => {
      window.clearTimeout(timer);
    };
  }, [recentPlanEvent]);

  const previousFollowLiveRef = useRef(followLive);
  useEffect(() => {
    if (!previousFollowLiveRef.current && followLive) {
      chartHandle.current?.followLive();
    }
    previousFollowLiveRef.current = followLive;
  }, [followLive]);

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
    return {
      ...layers,
      levels: filteredLevels,
      zones: [],
    };
  }, [layers, supportingVisible]);

  const chartParams = useMemo(() => {
    const fromCharts = plan.charts?.params;
    const candidate =
      (fromCharts && typeof fromCharts === "object" ? fromCharts : null) ??
      (plan.charts_params && typeof plan.charts_params === "object" ? plan.charts_params : null);
    if (!candidate) return {};
    return { ...(candidate as Record<string, unknown>) };
  }, [plan.charts?.params, plan.charts_params]);

  const structuredPlan = plan.structured_plan ?? null;

  const parseNumeric = (value: unknown): number | null => {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const parsed = Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  };

  const targetMeta = useMemo(
    () => (Array.isArray(plan.target_meta) ? (plan.target_meta as TargetMetaEntry[]) : []),
    [plan.target_meta],
  );

  const overlayTargets = useMemo(() => {
    const planTargets =
      Array.isArray(plan.targets) && plan.targets.length
        ? plan.targets
        : Array.isArray(structuredPlan?.targets)
          ? structuredPlan.targets
          : [];
    return planTargets
      .map((value, index) => {
        const numeric = parseNumeric(value);
        if (numeric == null) return null;
        const meta = targetMeta[index];
        return { price: numeric, label: meta?.label ?? `TP${index + 1}` };
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
  }, [plan.targets, structuredPlan?.targets, targetMeta]);

  const targetDetails = useMemo<Array<{ label?: string | null; price: number; rationale: string | null }>>(
    () =>
      overlayTargets.map((target, index) => ({
        ...target,
        rationale: extractTargetRationale(targetMeta[index]),
      })),
    [overlayTargets, targetMeta],
  );

  const confluenceTokens = useMemo(() => {
    const tokens: string[] = [];
    const layerConfluence = layers?.meta?.confluence;
    if (Array.isArray(layerConfluence)) {
      layerConfluence.forEach((item) => {
        if (typeof item === "string") {
          tokens.push(item);
        } else if (item && typeof item === "object" && typeof (item as { label?: string }).label === "string") {
          tokens.push(String((item as { label?: string }).label));
        }
      });
    } else if (typeof layerConfluence === "string") {
      tokens.push(...layerConfluence.split(","));
    }
    if (tokens.length === 0) {
      const planConfluence = Array.isArray((plan as Record<string, unknown>).confluence)
        ? ((plan as Record<string, unknown>).confluence as unknown[])
        : [];
      planConfluence.forEach((item) => {
        if (typeof item === "string") tokens.push(item);
      });
    }
    return tokens
      .map((token) => token.trim())
      .filter(Boolean)
      .slice(0, 5)
      .map((token) => token.toUpperCase());
  }, [layers?.meta?.confluence, plan]);


  const emaPeriods = useMemo(() => {
    const raw = chartParams["ema"] ?? chartParams["emas"] ?? chartParams["emaPeriods"];
    if (!raw) return [];
    if (Array.isArray(raw)) {
      return raw
        .map((value) => parseNumeric(value))
        .filter((value): value is number => value != null && value > 1)
        .map((value) => Math.round(value));
    }
    if (typeof raw === "string") {
      return raw
        .split(",")
        .map((token) => parseNumeric(token.trim()))
        .filter((value): value is number => value != null && value > 1)
        .map((value) => Math.round(value));
    }
    const numeric = parseNumeric(raw);
    return numeric != null && numeric > 1 ? [Math.round(numeric)] : [];
  }, [chartParams]);

  const showVWAP = useMemo(() => {
    const raw = chartParams["vwap"] ?? chartParams["showVWAP"];
    if (raw == null) return true;
    if (typeof raw === "boolean") return raw;
    if (typeof raw === "number") return raw !== 0;
    if (typeof raw === "string") {
      const token = raw.trim().toLowerCase();
      if (token === "0" || token === "false" || token === "off") return false;
      if (token === "1" || token === "true" || token === "on") return true;
    }
    return true;
  }, [chartParams]);

  const entryPrice = useMemo(() => {
    const fromPlan = parseNumeric(plan.entry);
    if (fromPlan != null) return fromPlan;
    const structuredEntry = parseNumeric(structuredPlan?.entry?.level);
    if (structuredEntry != null) return structuredEntry;
    return parseNumeric(chartParams["entry"]);
  }, [plan.entry, structuredPlan?.entry?.level, chartParams]);

  const stopPrice = useMemo(() => {
    const fromPlan = parseNumeric(plan.stop);
    if (fromPlan != null) return fromPlan;
    const structuredStop = parseNumeric(structuredPlan?.stop);
    if (structuredStop != null) return structuredStop;
    return parseNumeric(chartParams["stop"]);
  }, [plan.stop, structuredPlan?.stop, chartParams]);

  const trailingStop = useMemo(() => {
    const direct = parseNumeric((plan as Record<string, unknown>).trailing_stop);
    if (direct != null) return direct;
    const details = (plan as Record<string, unknown>).details;
    if (details && typeof details === "object") {
      const detailStop = parseNumeric((details as Record<string, unknown>).stop);
      if (detailStop != null) return detailStop;
    }
    const altTrail = parseNumeric((plan as Record<string, unknown>).trail_stop);
    if (altTrail != null) return altTrail;
    return (
      parseNumeric(chartParams["trailingStop"]) ??
      parseNumeric(chartParams["trailing_stop"]) ??
      parseNumeric(chartParams["trail_stop"]) ??
      parseNumeric(chartParams["trail"])
    );
  }, [plan, chartParams]);

  const targetRationales = useMemo(
    () =>
      targetDetails
        .map((detail) => {
          if (typeof detail.rationale === "string") {
            const trimmed = detail.rationale.trim();
            if (trimmed.length > 0) {
              return { ...detail, rationale: trimmed };
            }
          }
          return null;
        })
        .filter(
          (detail): detail is { label?: string | null; price: number; rationale: string } =>
            detail !== null,
        ),
    [targetDetails],
  );

  const chartOverlays = useMemo<ChartOverlayState>(
    () => ({
      entry: entryPrice,
      stop: stopPrice,
      trailingStop,
      targets: overlayTargets,
      emaPeriods,
      showVWAP,
      layers: (filteredLayers ?? layers) ?? null,
    }),
    [entryPrice, stopPrice, trailingStop, overlayTargets, emaPeriods, showVWAP, filteredLayers, layers],
  );

  const {
    bars: priceBars,
    status: priceStatus,
    error: priceError,
    reload: reloadPriceSeries,
  } = usePriceSeries(priceSymbol, timeframe, [planId, priceRefreshToken]);

  useEffect(() => {
    if (!devMode) return;
    console.log("[PlanChartPanel] bars", { symbol: priceSymbol, timeframe, count: priceBars.length });
  }, [devMode, priceBars.length, priceSymbol, timeframe]);

  useEffect(() => {
    if (priceStatus === "ready" && followLive) {
      chartHandle.current?.followLive();
    }
  }, [priceStatus, followLive]);

  useEffect(() => {
    if (followLive) {
      chartHandle.current?.followLive();
    }
  }, [followLive]);

  const handleResolution = (value: string) => {
    onSelectTimeframe(value);
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

  const handleLatestBarTime = useCallback(
    (time: number | null) => {
      onLastBarTimeChange(time);
      if (followLive && time != null) {
        chartHandle.current?.followLive();
      }
      if (devMode) {
        console.log("[PlanChartPanel] lastBarTime", { time });
      }
    },
    [followLive, onLastBarTimeChange, devMode],
  );

  const chartStatusMessage = useMemo(() => {
    if (priceStatus === "loading") return "Loading price data…";
    if (priceStatus === "error") return priceError?.message ?? "Price data unavailable";
    if (priceStatus === "ready" && priceBars.length === 0) return "No market data available";
    if (priceStatus === "idle" && !priceSymbol) return "Awaiting symbol…";
    return null;
  }, [priceStatus, priceError, priceBars.length, priceSymbol]);

  useEffect(() => {
    if (!priceSymbol || !streamingEnabled) return;
    if (typeof window === "undefined") return;
    const interval = window.setInterval(() => {
      reloadPriceSeries();
    }, 60_000);
    return () => {
      window.clearInterval(interval);
    };
  }, [priceSymbol, timeframe, reloadPriceSeries, streamingEnabled]);

  const highlightInvalidate = recentPlanEvent?.type === "stop_hit";
  const highlightScale = recentPlanEvent?.type === "tp_hit";

  const chartSectionClass = clsx(
    "relative overflow-hidden rounded-3xl border border-neutral-800/70 bg-neutral-950/40 p-2 transition-all duration-300",
    chartExpanded ? "chart-expanded min-h-[75vh] md:min-h-[80vh]" : "chart-default min-h-[60vh] md:min-h-[520px]",
  );

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
            {planAsOfLabel ? <span>As of {planAsOfLabel}</span> : null}
            {layers?.planning_context ? <span>{layers.planning_context.toUpperCase()}</span> : null}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {timeframeOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleResolution(option.value)}
              className={clsx(
                "inline-flex h-10 items-center justify-center rounded-full px-4 text-xs font-semibold uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
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
            onClick={handleReplayToggle}
            className={clsx(
              "inline-flex h-10 items-center justify-center rounded-full px-4 text-xs font-semibold uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
              replayActive
                ? "border border-emerald-400/60 bg-emerald-400/15 text-emerald-200"
                : "border border-neutral-800/60 bg-neutral-900/70 text-neutral-300 hover:border-emerald-400/40 hover:text-emerald-200",
            )}
            aria-pressed={replayActive}
          >
            {replayActive ? "Stop Replay" : "Replay"}
          </button>
        </div>
      </header>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-neutral-950/20 px-3 py-2 text-[11px] uppercase tracking-[0.24em] text-neutral-400">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-semibold text-neutral-300">Confluence</span>
            {confluenceTokens.length ? (
              <div className="flex flex-wrap items-center gap-1">
                {confluenceTokens.map((token) => (
                  <span
                    key={token}
                    className="inline-flex items-center gap-1 rounded-full border border-neutral-800/60 bg-neutral-900/60 px-2 py-0.5 text-[10px] text-neutral-200"
                  >
                    <span className="block h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden />
                    <span>{token}</span>
                  </span>
                ))}
              </div>
            ) : (
              <span className="text-[10px] text-neutral-500">None noted</span>
            )}
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-neutral-300">
            <span className="font-semibold uppercase tracking-[0.28em] text-neutral-400">Confidence</span>
            <ConfidenceBadge value={plan.confidence} className="h-8 w-8 text-[10px]" />
          </div>
        </div>
        <div
          className={clsx(
            "rounded-xl bg-neutral-950/20 p-3",
            supportingVisible ? "shadow-[0_0_25px_rgba(16,185,129,0.15)]" : "",
          )}
        >
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="space-y-2.5">
              <h3 className="text-[0.68rem] uppercase tracking-[0.3em] text-neutral-500">Target notes</h3>
              {targetRationales.length ? (
                <ul className="space-y-1 text-[11px] leading-relaxed text-neutral-300">
                  {targetRationales.map((detail, index) => {
                    const label = detail.label ?? `TP${index + 1}`;
                    return (
                      <li key={`${label}-${detail.price}-note`} className="space-y-0.5">
                        <div className="flex flex-wrap items-baseline gap-1.5">
                          <span className="font-semibold uppercase tracking-[0.18em] text-neutral-100">{label}</span>
                          <span className="tabular-nums text-neutral-500">{detail.price.toFixed(2)}</span>
                        </div>
                        <p className="text-[11px] leading-snug text-neutral-400">{detail.rationale}</p>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="text-[11px] text-neutral-500">No target rationales published.</p>
              )}
            </div>
            <div className="space-y-2.5">
              <h3 className="text-[0.68rem] uppercase tracking-[0.3em] text-neutral-500">Primary levels</h3>
              {primaryLevels.length ? (
                <ul className="divide-y divide-neutral-800/60 text-[11px]">
                  {primaryLevels.map((level, index) => {
                    const label = level.label ?? level.kind ?? `Level ${index + 1}`;
                    return (
                      <li key={`${label}-${level.price}`} className="flex items-center justify-between px-2 py-1.5 text-neutral-200">
                        <span className="uppercase tracking-[0.15em] text-neutral-500">{label}</span>
                        <span className="font-semibold text-neutral-100">{level.price.toFixed(2)}</span>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="text-[11px] text-neutral-500">No primary levels published.</p>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className={chartSectionClass}>
        <button
          type="button"
          onClick={() => setChartExpanded((prev) => !prev)}
          className="absolute right-4 top-4 z-30 rounded-full border border-neutral-700/60 bg-neutral-900/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-neutral-200 transition hover:border-emerald-400 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
          aria-pressed={chartExpanded}
        >
          {chartExpanded ? "Collapse" : "Expand"}
        </button>
        {chartStatusMessage ? (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-neutral-950/70 text-sm text-neutral-300">
            {chartStatusMessage}
          </div>
        ) : null}
        {devMode ? (
          <div className="pointer-events-none absolute left-3 top-3 z-20 rounded bg-neutral-900/70 px-2 py-1 text-[10px] text-neutral-300">
            dev: {symbol} · {timeframe} · status={priceStatus} · bars={priceBars.length}
          </div>
        ) : null}
        <PlanPriceChart
          ref={chartHandle}
          planId={planId}
          symbol={symbol}
          resolution={timeframe}
          theme={theme}
          data={priceBars}
          overlays={chartOverlays}
          onLastBarTimeChange={handleLatestBarTime}
          devMode={devMode}
          followLive={followLive}
          onFollowLiveChange={onSetFollowLive}
          levelsExpanded={supportingVisible}
          expanded={chartExpanded}
          highlightedLevelId={highlightLevelId}
          hiddenLevelIds={hiddenLevelIds}
        />
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <PlanControlCard title="Trigger" body={triggerRule} tone="emerald" />
        <PlanControlCard title="Invalidate" body={invalidationRule} tone="rose" highlight={highlightInvalidate} />
        <PlanControlCard title="Scale" body={scaleRule} tone="sky" highlight={highlightScale} />
        <PlanControlCard title="Reload" body={reloadRule} tone="amber" />
      </section>

    </div>
  );
}

function PlanControlCard({
  title,
  body,
  tone,
  highlight = false,
}: {
  title: string;
  body: string | null;
  tone: "emerald" | "rose" | "sky" | "amber";
  highlight?: boolean;
}) {
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
  const ringMap: Record<typeof tone, string> = {
    emerald: "ring-emerald-400/70",
    rose: "ring-rose-400/70",
    sky: "ring-sky-400/70",
    amber: "ring-amber-400/70",
  };
  return (
    <div
      className={clsx(
        "rounded-2xl border p-4 text-sm leading-relaxed transition",
        toneMap[tone],
        highlight && ["ring-2 ring-offset-2 ring-offset-neutral-950", ringMap[tone]],
      )}
    >
      <div className="text-xs font-semibold uppercase tracking-[0.3em] opacity-80">{title}</div>
      <p className="mt-2">{body}</p>
    </div>
  );
}

function extractTargetRationale(meta: TargetMetaEntry | undefined): string | null {
  if (!meta) return null;
  const record = meta as Record<string, unknown>;
  const candidateKeys = ["rationale", "reason", "context", "note", "summary", "basis", "basis_label", "snap_tag", "tag"];
  for (const key of candidateKeys) {
    const value = record[key];
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed) return trimmed;
    }
    if (Array.isArray(value)) {
      const joined = value
        .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
        .filter(Boolean)
        .join(" · ");
      if (joined) return joined;
    }
  }
  return null;
}
