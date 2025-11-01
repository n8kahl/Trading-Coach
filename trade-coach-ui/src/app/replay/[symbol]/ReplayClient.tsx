"use client";

import clsx from "clsx";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlanPriceChart, { type ChartOverlayState } from "@/components/PlanPriceChart";
import PlanPanel from "@/components/webview/PlanPanel";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";
import { extractPrimaryLevels, extractSupportingLevels } from "@/lib/utils/layers";
import { extractPlanLevels, resolveTrailingStop } from "@/lib/plan/levels";
import { resolvePlanEntry, resolvePlanStop, resolvePlanTargets } from "@/lib/plan/coach";
import { usePriceSeries, type PriceBar } from "@/lib/hooks/usePriceSeries";
import { API_BASE_URL, withAuthHeaders } from "@/lib/env";
import { useChartUrl } from "@/lib/hooks/useChartUrl";

const TIMEFRAME_OPTIONS = [
  { value: "1", label: "1m" },
  { value: "3", label: "3m" },
  { value: "5", label: "5m" },
  { value: "15", label: "15m" },
  { value: "60", label: "1h" },
  { value: "240", label: "4h" },
  { value: "1D", label: "1D" },
];

const PLAYBACK_SPEEDS = [0.5, 1, 2, 4]; // bars per second

type ReplayClientProps = {
  symbol: string;
  initialSnapshot: PlanSnapshot | null;
};

type SessionInfo = PlanSnapshot["plan"]["session_state"] | null;

type PriceSeriesCandle = PriceBar;

function normalizeTimeframeFromPlan(plan: PlanSnapshot["plan"]): string {
  const raw = (plan.charts_params as Record<string, unknown> | undefined)?.interval ?? plan.chart_timeframe ?? "5";
  const value = typeof raw === "string" ? raw.trim() : "";
  if (!value) return "5";
  const lower = value.toLowerCase();
  if (lower.endsWith("m")) {
    const minutes = Number.parseInt(lower.replace("m", ""), 10);
    return Number.isFinite(minutes) && minutes > 0 ? String(minutes) : "5";
  }
  if (lower.endsWith("h")) {
    const hours = Number.parseInt(lower.replace("h", ""), 10);
    return Number.isFinite(hours) && hours > 0 ? String(hours * 60) : "60";
  }
  if (lower === "d" || lower === "1d") return "1D";
  if (lower === "w" || lower === "1w") return "1W";
  const numeric = Number.parseInt(lower, 10);
  return Number.isFinite(numeric) && numeric > 0 ? String(numeric) : "5";
}

function formatSessionStatus(session: SessionInfo): string {
  if (!session?.status) return "Status unknown";
  return session.status;
}

function formatAsOf(session: SessionInfo): string {
  if (!session?.as_of) return "—";
  const date = new Date(session.as_of);
  if (!Number.isFinite(date.getTime())) return "—";
  const tz = session.tz || "America/New_York";
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
    timeZone: tz,
    timeZoneName: "short",
  }).format(date);
}

function formatTimestamp(value: PriceSeriesCandle | null, tz?: string | null): string {
  if (!value) return "—";
  const time = typeof value.time === "number" ? value.time * 1000 : Number(value.time) * 1000;
  if (!Number.isFinite(time)) return "—";
  const date = new Date(time);
  if (!Number.isFinite(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
    timeZone: tz || "America/New_York",
  }).format(date);
}

function percentLabel(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

export default function ReplayClient({ symbol, initialSnapshot }: ReplayClientProps) {
  const plan = initialSnapshot?.plan ?? null;

  if (!plan) {
    return (
      <div className="px-6 py-12 text-center text-sm text-[color:var(--tc-neutral-400)]">
        Simulated plan unavailable for {symbol.toUpperCase()}.
      </div>
    );
  }

  const initialLayers = (plan.plan_layers as PlanLayers | undefined) ?? null;
  const [planLayers, setPlanLayers] = useState<PlanLayers | null>(initialLayers);
  const [supportVisible, setSupportVisible] = useState(true);
  const [highlightedLevel, setHighlightedLevel] = useState<SupportingLevel | null>(null);
  const [timeframe, setTimeframe] = useState(() => normalizeTimeframeFromPlan(plan));
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const session = plan.session_state ?? null;
  const chartUrl = useChartUrl(plan);

  const trailingStopValue = useMemo(() => resolveTrailingStop(plan), [plan]);
  const levelSummary = useMemo(() => extractPlanLevels(plan, { trailingStop: trailingStopValue }), [plan, trailingStopValue]);

  const primaryLevels = useMemo(() => extractPrimaryLevels(planLayers), [planLayers]);
  const supportingLevels = useMemo(() => extractSupportingLevels(planLayers), [planLayers]);

  const filteredLayers = useMemo(() => {
    if (!planLayers) return null;
    if (supportVisible) return planLayers;
    const groups = (planLayers.meta?.level_groups ?? {}) as Record<string, unknown>;
    type LevelGroupEntry = { price?: number | null };
    const primaryEntries = Array.isArray(groups.primary) ? (groups.primary as LevelGroupEntry[]) : [];
    const primarySet = new Set<number>();
    primaryEntries.forEach((entry) => {
      if (entry && typeof entry.price === "number") {
        primarySet.add(entry.price);
      }
    });
    const filteredLevelsList = Array.isArray(planLayers.levels)
      ? planLayers.levels.filter((level) => typeof level?.price === "number" && primarySet.has(level.price))
      : [];
    return {
      ...planLayers,
      levels: filteredLevelsList,
      zones: [],
    } as PlanLayers;
  }, [planLayers, supportVisible]);

  useEffect(() => {
    if (!plan.plan_id) return;
    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        const qs = new URLSearchParams({ plan_id: plan.plan_id });
        const res = await fetch(`${API_BASE_URL}/api/v1/gpt/chart-layers?${qs.toString()}`, {
          cache: "no-store",
          signal: controller.signal,
          headers: withAuthHeaders({ Accept: "application/json" }),
        });
        if (!res.ok) return;
        const payload = (await res.json()) as PlanLayers;
        if (!cancelled) {
          setPlanLayers(payload ?? null);
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        if (process.env.NODE_ENV !== "production") {
          console.warn("[ReplayClient] layers fetch failed", error);
        }
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [plan.plan_id]);

  const overlayTargets = useMemo(() => resolvePlanTargets(plan), [plan]);
  const entryPrice = useMemo(() => resolvePlanEntry(plan), [plan]);
  const stopPrice = useMemo(() => resolvePlanStop(plan, trailingStopValue), [plan, trailingStopValue]);

  const chartOverlays = useMemo<ChartOverlayState>(() => ({
    entry: entryPrice,
    stop: stopPrice,
    trailingStop: trailingStopValue,
    targets: overlayTargets,
    layers: (supportVisible ? planLayers : filteredLayers) ?? null,
  }), [entryPrice, stopPrice, trailingStopValue, overlayTargets, filteredLayers, planLayers, supportVisible]);

  const symbolToken = plan.symbol ?? symbol;
  const {
    bars: priceBars,
    status: priceStatus,
    error: priceError,
    reload: reloadSeries,
  } = usePriceSeries(symbolToken, timeframe, [plan.plan_id]);

  useEffect(() => {
    setPlaybackIndex(0);
    setIsPlaying(false);
  }, [timeframe, priceBars.length]);

  useEffect(() => {
    if (!isPlaying) return;
    if (priceBars.length === 0) return;
    const intervalMs = Math.max(75, Math.round(1000 / playbackSpeed));
    const timer = window.setInterval(() => {
      setPlaybackIndex((prev) => {
        if (prev >= priceBars.length - 1) {
          window.clearInterval(timer);
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => {
      window.clearInterval(timer);
    };
  }, [isPlaying, playbackSpeed, priceBars.length]);

  const maxIndex = Math.max(0, priceBars.length - 1);
  useEffect(() => {
    if (playbackIndex > maxIndex) {
      setPlaybackIndex(maxIndex);
    }
  }, [maxIndex, playbackIndex]);

  const currentBar = priceBars.length ? priceBars[Math.min(playbackIndex, maxIndex)] : null;
  const displayBars = useMemo(() => {
    if (!priceBars.length) return [] as PriceSeriesCandle[];
    return priceBars.slice(0, Math.min(playbackIndex + 1, priceBars.length));
  }, [priceBars, playbackIndex]);

  const timelineProgress = maxIndex === 0 ? 0 : (playbackIndex / maxIndex) * 100;

  const handleToggleSupport = useCallback(() => {
    setSupportVisible((prev) => !prev);
  }, []);

  const handleSelectLevel = useCallback((level: SupportingLevel | null) => {
    setHighlightedLevel(level);
  }, []);

  const handleSetPlaybackIndex = useCallback((value: number) => {
    const next = Number.isFinite(value) ? Math.max(0, Math.min(value, maxIndex)) : 0;
    setPlaybackIndex(next);
  }, [maxIndex]);

  const handleStep = useCallback((delta: number) => {
    setPlaybackIndex((prev) => {
      const next = Math.max(0, Math.min(prev + delta, maxIndex));
      return next;
    });
  }, [maxIndex]);

  const banner = plan.session_state?.banner ?? null;
  const objectiveMeta = planLayers?.meta?.next_objective as Record<string, unknown> | undefined;
  const objectiveProgress = typeof objectiveMeta?.progress === "number" ? objectiveMeta.progress : null;

  const statusMessage = useMemo(() => {
    if (priceError) return "Price data unavailable";
    if (priceStatus === "loading") return "Loading price history…";
    if (priceStatus === "error") return "Failed to load price data";
    if (!priceBars.length) return "Waiting for historical data";
    return null;
  }, [priceStatus, priceError, priceBars.length]);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-10">
      <header className="flex flex-col gap-4 rounded-3xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/90 p-5 backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="text-xs uppercase tracking-[0.3em] text-[color:var(--tc-neutral-400)]">Simulated Dojo</div>
            <h1 className="text-2xl font-semibold text-[color:var(--tc-neutral-50)]">{symbolToken.toUpperCase()}</h1>
            {banner ? (
              <p className="text-sm text-[color:var(--tc-neutral-300)]">{banner}</p>
            ) : null}
          </div>
          <div className="flex items-center gap-2 text-sm text-[color:var(--tc-neutral-300)]">
            <span>{formatSessionStatus(session)}</span>
            <span className="inline-flex items-center gap-1 rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-xs uppercase tracking-[0.2em] text-[color:var(--tc-neutral-200)]">
              As of {formatAsOf(session)}
            </span>
            {chartUrl ? (
              <a
                href={chartUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center rounded-full border border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-emerald-200)] transition hover:border-[color:var(--tc-emerald-400)] hover:text-[color:var(--tc-emerald-100)]"
              >
                Canonical Chart ↗
              </a>
            ) : null}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-[color:var(--tc-neutral-400)]">
          <span>Objective progress: {percentLabel(objectiveProgress)}</span>
          <span>Playback bar {priceBars.length ? playbackIndex + 1 : 0}/{priceBars.length}</span>
          <span>Timestamp: {formatTimestamp(currentBar, session?.tz)}</span>
        </div>
      </header>

      <section className="relative rounded-3xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-muted)]/90 p-4 backdrop-blur">
        {statusMessage ? (
          <div className="absolute inset-0 z-10 flex items-center justify-center rounded-3xl bg-[color:var(--tc-surface-primary)]/85 text-sm text-[color:var(--tc-neutral-200)]">
            {statusMessage}
          </div>
        ) : null}
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <label className="text-xs uppercase tracking-[0.22em] text-[color:var(--tc-neutral-400)]" htmlFor="dojo-timeframe">
              Timeframe
            </label>
            <select
              id="dojo-timeframe"
              value={timeframe}
              onChange={(event) => setTimeframe(event.target.value)}
              className="rounded-full border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)] px-3 py-1 text-xs text-[color:var(--tc-neutral-200)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
            >
              {TIMEFRAME_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2 text-xs text-[color:var(--tc-neutral-400)]">
            <button
              type="button"
              onClick={() => reloadSeries()}
              className="rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-neutral-200)] transition hover:border-[color:var(--tc-emerald-400)] hover:text-[color:var(--tc-emerald-200)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
            >
              Refresh Bars
            </button>
            <button
              type="button"
              onClick={handleToggleSupport}
              className={clsx(
                "rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]",
                supportVisible
                  ? "border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] text-[color:var(--tc-emerald-200)]"
                  : "border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)] text-[color:var(--tc-neutral-300)]",
              )}
            >
              Levels {supportVisible ? "On" : "Off"}
            </button>
          </div>
        </div>
        <PlanPriceChart
          planId={plan.plan_id}
          symbol={symbolToken}
          resolution={timeframe}
          theme="dark"
          data={displayBars}
          overlays={chartOverlays}
          onLastBarTimeChange={() => {}}
          levelsExpanded={supportVisible}
        />
        <div className="mt-4 space-y-3">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setIsPlaying((prev) => !prev)}
              className="rounded-full border border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] px-4 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-emerald-200)] transition hover:border-[color:var(--tc-emerald-400)] hover:text-[color:var(--tc-emerald-100)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              onClick={() => handleStep(-1)}
              className="rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-neutral-200)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
            >
              Step −1
            </button>
            <button
              type="button"
              onClick={() => handleStep(1)}
              className="rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-neutral-200)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
            >
              Step +1
            </button>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-[0.2em] text-[color:var(--tc-neutral-400)]">Speed</span>
              <select
                value={playbackSpeed}
                onChange={(event) => setPlaybackSpeed(Number(event.target.value) || 1)}
                className="rounded-full border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)] px-2 py-1 text-xs text-[color:var(--tc-neutral-200)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
              >
                {PLAYBACK_SPEEDS.map((speed) => (
                  <option key={speed} value={speed}>
                    {speed}×
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={maxIndex}
              value={playbackIndex}
              onChange={(event) => handleSetPlaybackIndex(Number(event.target.value))}
              className="w-full accent-[color:var(--tc-emerald-400)]"
            />
            <span className="w-16 text-right text-xs text-[color:var(--tc-neutral-400)]">{Math.round(timelineProgress)}%</span>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr),minmax(0,1fr)]">
        <div className="space-y-4 rounded-3xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/90 p-5 backdrop-blur">
          <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-[color:var(--tc-neutral-400)]">Plan Overview</h2>
          <PlanPanel
            plan={plan}
            supportingLevels={supportingLevels}
            highlightedLevel={highlightedLevel}
            onSelectLevel={handleSelectLevel}
            theme="dark"
          />
        </div>
        <div className="space-y-4 rounded-3xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/90 p-5 backdrop-blur">
          <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-[color:var(--tc-neutral-400)]">Primary Levels</h2>
          {primaryLevels.length ? (
            <ul className="space-y-2 text-sm text-[color:var(--tc-neutral-100)]">
              {primaryLevels.map((level) => (
                <li key={`${level.label}-${level.price}`} className="flex items-center justify-between rounded-xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-muted)]/80 px-3 py-2">
                  <span className="text-xs uppercase tracking-[0.24em] text-[color:var(--tc-neutral-400)]">{level.label ?? level.kind ?? "Level"}</span>
                  <span className="font-semibold text-[color:var(--tc-neutral-50)]">{level.price.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-[color:var(--tc-neutral-400)]">No primary levels provided.</p>
          )}
        </div>
      </section>
    </div>
  );
}
