"use client";

import clsx from "clsx";
import * as React from "react";
import { useRouter } from "next/navigation";
import WebviewShell from "@/components/webview/WebviewShell";
import PlanPanel from "@/components/webview/PlanPanel";
import PlanChartPanel from "@/components/PlanChartPanel";
import CoachNote from "@/components/CoachNote";
import HeaderMarkers from "@/components/HeaderMarkers";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { usePlanSocket } from "@/lib/hooks/usePlanSocket";
import { usePlanLayers } from "@/lib/hooks/usePlanLayers";
import { extractPrimaryLevels, extractSupportingLevels } from "@/lib/utils/layers";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanDeltaEvent, PlanLayers, PlanSnapshot } from "@/lib/types";
import { API_BASE_URL, WS_BASE_URL, BUILD_SHA, withAuthHeaders } from "@/lib/env";
import { deriveCoachMessage, resolvePlanTargets, type CoachGoal, type CoachNote as CoachNoteModel } from "@/lib/plan/coach";
import { extractPlanLevels, resolveTrailingStop } from "@/lib/plan/levels";
import { emitPlanEvent } from "@/lib/plan/events";

const TIMEFRAME_OPTIONS = [
  { value: '1', label: '1m' },
  { value: '3', label: '3m' },
  { value: '5', label: '5m' },
  { value: '15', label: '15m' },
  { value: '60', label: '1h' },
  { value: '240', label: '4h' },
  { value: '1D', label: '1D' },
];

const HEADER_AS_OF_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  month: "short",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: true,
});

type LivePlanClientProps = {
  initialSnapshot: PlanSnapshot;
  planId: string;
};

type StatusTone = "green" | "yellow" | "red";

function sanitizeSymbolToken(value: string): string {
  return value.replace(/[^A-Za-z0-9]/g, "").toUpperCase().slice(0, 6);
}

export default function LivePlanClient({ initialSnapshot, planId }: LivePlanClientProps) {
  const router = useRouter();
  const [snapshot, setSnapshot] = React.useState(initialSnapshot);
  const [plan, setPlan] = React.useState(initialSnapshot.plan);
  const [followLive, setFollowLive] = React.useState(true);
  const [streamingEnabled, setStreamingEnabled] = React.useState(true);
  const [supportVisible, setSupportVisible] = React.useState(() => extractSupportVisible(initialSnapshot.plan));
  const [highlightedLevel, setHighlightedLevel] = React.useState<SupportingLevel | null>(null);
  const [layerSeed, setLayerSeed] = React.useState<PlanLayers | null>(() => extractInitialLayers(initialSnapshot));
  const [timeframe, setTimeframe] = React.useState(() => normalizeTimeframeFromPlan(initialSnapshot.plan));
  const [nowTick, setNowTick] = React.useState(Date.now());
  const [lastBarTime, setLastBarTime] = React.useState<number | null>(null);
  const [devMode, setDevMode] = React.useState(() => process.env.NEXT_PUBLIC_DEVTOOLS === "1");
  const [priceRefreshToken, setPriceRefreshToken] = React.useState(0);
  const [lastPlanHeartbeat, setLastPlanHeartbeat] = React.useState<number | null>(() => Date.now());
  const [symbolDraft, setSymbolDraft] = React.useState(() => (plan.symbol ? plan.symbol.toUpperCase() : ""));
  const [symbolSubmitting, setSymbolSubmitting] = React.useState(false);
  const [coachNote, setCoachNote] = React.useState<CoachNoteModel>(() => ({
    text: "Awaiting tick…",
    goal: "neutral",
    progressPct: 0,
    updatedAt: Date.now(),
  }));
  const [coachLoading, setCoachLoading] = React.useState(false);
  const [activeLevelId, setActiveLevelId] = React.useState<string | null>(null);
  const [hiddenLevelIds, setHiddenLevelIds] = React.useState<Set<string>>(() => new Set());
  const [collapsed, setCollapsed] = React.useState(false);
  const lastPriceRefreshRef = React.useRef(0);
  const lastDeltaAtRef = React.useRef(Date.now());
  const symbolRequestRef = React.useRef(0);
  const replanPendingRef = React.useRef(false);
  const lastCoachGoalRef = React.useRef<CoachGoal | null>(null);
  const lastCoachUpdateRef = React.useRef<number>(Date.now());
  const touchStartYRef = React.useRef<number | null>(null);

  const theme = "dark" as const;

  const requestPriceRefresh = React.useCallback(() => {
    const now = Date.now();
    if (now - lastPriceRefreshRef.current < 4000) return;
    lastPriceRefreshRef.current = now;
    setPriceRefreshToken((token) => token + 1);
  }, []);

  const activePlanId = plan.plan_id || planId;
  const markPlanHeartbeat = React.useCallback(() => {
    const now = Date.now();
    lastDeltaAtRef.current = now;
    setLastPlanHeartbeat(now);
  }, []);

  React.useEffect(() => {
    const interval = window.setInterval(() => setNowTick(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  React.useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get('dev') === '1') {
        setDevMode(true);
      }
    } catch {
      // ignore
    }
  }, []);

  const planSocketUrl = React.useMemo(() => `${WS_BASE_URL}/ws/plans/${encodeURIComponent(activePlanId)}`, [activePlanId]);

  const streamingEnabledRef = React.useRef(streamingEnabled);
  React.useEffect(() => {
    streamingEnabledRef.current = streamingEnabled;
  }, [streamingEnabled]);

  const { layers } = usePlanLayers(activePlanId, layerSeed);

  const refreshPlanSnapshot = React.useCallback(
    async (targetPlanId: string) => {
      if (!targetPlanId) return;
      try {
        const response = await fetch(`${API_BASE_URL}/idea/${encodeURIComponent(targetPlanId)}`, {
          headers: withAuthHeaders({ Accept: "application/json" }),
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`refreshPlanSnapshot failed (${response.status})`);
        }
        const payload = (await response.json()) as PlanSnapshot;
        if (!payload?.plan) return;
        setSnapshot(payload);
        setPlan(payload.plan);
        setLayerSeed(extractInitialLayers(payload));
        setHighlightedLevel(null);
        setTimeframe((prev) => prev || normalizeTimeframeFromPlan(payload.plan));
        setLastPlanHeartbeat(Date.now());
        lastDeltaAtRef.current = Date.now();
        setSymbolDraft(payload.plan.symbol ? payload.plan.symbol.toUpperCase() : "");
        requestPriceRefresh();
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("[LivePlanClient] refreshPlanSnapshot", error);
        }
      }
    },
    [requestPriceRefresh],
  );

  const queueReplan = React.useCallback(
    (targetPlanId: string | null | undefined) => {
      if (!targetPlanId || replanPendingRef.current) return;
      replanPendingRef.current = true;
      refreshPlanSnapshot(targetPlanId).finally(() => {
        replanPendingRef.current = false;
      });
    },
    [refreshPlanSnapshot],
  );

  const socketStatus = usePlanSocket(
    planSocketUrl,
    activePlanId,
    React.useCallback(
      (payload: unknown) => {
        if (!streamingEnabledRef.current) return;
        if (!payload || typeof payload !== 'object') return;
        const message = payload as { plan_id?: string; event?: Record<string, unknown> };
        const event = message.event;
        if (!event || typeof event !== "object") return;
        const type = typeof event.t === "string" ? event.t : null;
        if (!type) return;
        markPlanHeartbeat();
        if (type === 'plan_delta') {
          const delta = event as unknown as PlanDeltaEvent;
          setPlan((prev) => {
            const merged = mergePlanWithDelta(prev, delta);
            if (merged !== prev) {
              const statusChange = typeof delta.changes?.status === "string" ? delta.changes.status.toLowerCase() : null;
              if (statusChange === "invalid") {
                const targetPlanId = merged.plan_id || prev.plan_id || activePlanId;
                queueReplan(targetPlanId ?? activePlanId);
              }
            }
            return merged;
          });
          if (delta.changes && (delta.changes.last_price !== undefined || delta.changes.trailing_stop !== undefined)) {
            requestPriceRefresh();
          }
          return;
        }
        if (type === 'plan_full') {
          const payloadSnapshot = (event as Record<string, unknown>).payload as PlanSnapshot | undefined;
          if (!payloadSnapshot || !payloadSnapshot.plan) return;
          setSnapshot(payloadSnapshot);
          setPlan(payloadSnapshot.plan);
          setLayerSeed(extractInitialLayers(payloadSnapshot));
          setTimeframe((prev) => {
            const inferred = normalizeTimeframeFromPlan(payloadSnapshot.plan);
            return prev || inferred;
          });
          setSymbolDraft(payloadSnapshot.plan.symbol ? payloadSnapshot.plan.symbol.toUpperCase() : "");
          requestPriceRefresh();
          return;
        }
        if (type === "plan_state") {
          return;
        }
        if (type === "tick" || type === "bar") {
          requestPriceRefresh();
          return;
        }
      },
      [activePlanId, queueReplan, requestPriceRefresh, markPlanHeartbeat],
    ),
  );

  const primaryLevels = React.useMemo(() => extractPrimaryLevels(layers), [layers]);
  const supportingLevels = React.useMemo(() => extractSupportingLevels(layers), [layers]);

  const resolutionMs = React.useMemo(() => timeframeToMs(timeframe), [timeframe]);
  const dataAgeSeconds = React.useMemo(() => {
    if (!lastBarTime) return null;
    return Math.max(0, (nowTick - lastBarTime) / 1000);
  }, [lastBarTime, nowTick]);

  const planHeartbeatAgeSeconds = React.useMemo(() => {
    if (!lastPlanHeartbeat) return null;
    return Math.max(0, (nowTick - lastPlanHeartbeat) / 1000);
  }, [lastPlanHeartbeat, nowTick]);

  const resolutionSeconds = React.useMemo(() => {
    if (resolutionMs == null) return null;
    return Math.max(1, Math.round(resolutionMs / 1000));
  }, [resolutionMs]);

  const planStatusTone = React.useMemo<StatusTone>(() => {
    const statusValue = ((plan as Record<string, unknown>).status ?? plan.session_state?.status ?? "") as string;
    const normalized = typeof statusValue === "string" ? statusValue.toLowerCase() : "";
    if (!normalized) return "yellow";
    if (normalized.includes("invalid") || normalized.includes("closed") || normalized.includes("cancel")) return "red";
    if (normalized.includes("pending") || normalized.includes("watch") || normalized.includes("prep")) return "yellow";
    return "green";
  }, [plan]);

  const planStatusTitle = React.useMemo(() => {
    const statusValue = ((plan as Record<string, unknown>).status ?? plan.session_state?.status ?? "") as string;
    return statusValue ? `Plan status ${statusValue}` : "Plan status unavailable";
  }, [plan]);

  const streamStatusTone = React.useMemo<StatusTone>(() => {
    if (socketStatus === "disconnected") return "red";
    if (planHeartbeatAgeSeconds == null) return "yellow";
    if (planHeartbeatAgeSeconds <= 15) return "green";
    if (planHeartbeatAgeSeconds <= 60) return "yellow";
    return "red";
  }, [socketStatus, planHeartbeatAgeSeconds]);

  const streamStatusTitle = React.useMemo(() => {
    if (socketStatus === "disconnected") return "Stream disconnected";
    if (planHeartbeatAgeSeconds == null) return "Stream heartbeat pending";
    return `Last stream ${planHeartbeatAgeSeconds.toFixed(1)}s ago`;
  }, [planHeartbeatAgeSeconds, socketStatus]);

  const dataStatusTone = React.useMemo<StatusTone>(() => {
    if (!streamingEnabled) return "red";
    if (dataAgeSeconds == null || resolutionSeconds == null) return "yellow";
    if (dataAgeSeconds <= resolutionSeconds * 2) return "green";
    if (dataAgeSeconds <= resolutionSeconds * 6) return "yellow";
    return "red";
  }, [streamingEnabled, dataAgeSeconds, resolutionSeconds]);

  const dataStatusTitle = React.useMemo(() => {
    if (dataAgeSeconds == null) return "Price data pending";
    return `Last price update ${dataAgeSeconds.toFixed(1)}s ago`;
  }, [dataAgeSeconds]);

  const sessionBanner = plan.session_state?.banner ?? null;
  const riskBanner =
    Array.isArray(plan.warnings) && plan.warnings.length ? String(plan.warnings[0]) : plan.session_state?.message ?? null;

  const resolvedTargets = React.useMemo(() => resolvePlanTargets(plan), [plan]);
  const planTargets = React.useMemo(() => resolvedTargets.map((target) => target.price), [resolvedTargets]);

  React.useEffect(() => {
    setSymbolDraft(plan.symbol ? sanitizeSymbolToken(String(plan.symbol)) : "");
  }, [plan.symbol]);

  const trailingStopValue = React.useMemo(() => resolveTrailingStop(plan), [plan]);

  const levelSummary = React.useMemo(
    () => extractPlanLevels(plan, { trailingStop: trailingStopValue }),
    [plan, trailingStopValue],
  );

  const headerLevels = React.useMemo(
    () =>
      levelSummary.levels.map((level) => ({
        ...level,
        hidden: hiddenLevelIds.has(level.id),
      })),
    [levelSummary.levels, hiddenLevelIds],
  );

  const hiddenLevelIdList = React.useMemo(() => Array.from(hiddenLevelIds), [hiddenLevelIds]);

  React.useEffect(() => {
    setCollapsed(false);
  }, [plan.plan_id]);

  const lastPrice = React.useMemo(() => {
    const details = (plan.details ?? {}) as Record<string, unknown>;
    return (
      getNumber((plan as Record<string, unknown>).last_price) ??
      getNumber(details.last) ??
      getNumber((plan as Record<string, unknown>).mark) ??
      null
    );
  }, [plan]);

  const computeCoachMessage = React.useCallback(() => {
    return deriveCoachMessage({
      plan,
      price: lastPrice,
      trailingStop: levelSummary.trailingStop,
      now: Date.now(),
    });
  }, [plan, lastPrice, levelSummary.trailingStop]);

  const applyCoachMessage = React.useCallback(
    (reason: string) => {
      const next = computeCoachMessage();
      const finishLoading = () => {
        if (typeof window === "undefined") {
          setCoachLoading(false);
          return;
        }
        window.setTimeout(() => setCoachLoading(false), 220);
      };
      setCoachNote((prev) => {
        const prevProgress = Math.round(prev.progressPct);
        const nextProgress = Math.round(next.progressPct);
        if (prev.text === next.text && prev.goal === next.goal && prevProgress === nextProgress) {
          if (reason !== "initial") {
            lastCoachUpdateRef.current = Date.now();
            finishLoading();
          }
          return { ...next, updatedAt: prev.updatedAt };
        }
        lastCoachUpdateRef.current = next.updatedAt;
        finishLoading();
        return next;
      });
    },
    [computeCoachMessage],
  );

  React.useEffect(() => {
    applyCoachMessage("initial");
  }, [applyCoachMessage]);

  React.useEffect(() => {
    if (typeof window === "undefined") return undefined;
    const interval = window.setInterval(() => {
      setCoachLoading(true);
      applyCoachMessage("cadence");
    }, 5000);
    return () => {
      window.clearInterval(interval);
    };
  }, [applyCoachMessage]);

  React.useEffect(() => {
    const goal = coachNote.goal;
    if (!activePlanId) return;
    if (goal === lastCoachGoalRef.current) return;
    lastCoachGoalRef.current = goal;
    if (goal === "tp_hit" || goal === "stop_hit") {
      emitPlanEvent({
        type: goal,
        planId: activePlanId,
        payload: {
          price: lastPrice ?? null,
          updatedAt: coachNote.updatedAt,
        },
      });
    }
  }, [coachNote.goal, coachNote.updatedAt, activePlanId, lastPrice]);

  const confluenceTokens = React.useMemo(() => {
    const tokens: string[] = [];
    const layerConfluence = layers?.meta?.confluence;
    if (Array.isArray(layerConfluence)) {
      layerConfluence.forEach((item) => {
        if (typeof item === "string") {
          tokens.push(item);
          return;
        }
        if (item && typeof item === "object") {
          const label = (item as { label?: string }).label;
          if (typeof label === "string") tokens.push(label);
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

  const priceFormatter = React.useMemo(
    () =>
      new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
        minimumFractionDigits: 2,
      }),
    [],
  );

  const lastPriceLabel = lastPrice != null ? priceFormatter.format(lastPrice) : "—";
  const timeframeLabel = React.useMemo(() => {
    const match = TIMEFRAME_OPTIONS.find((item) => item.value === timeframe);
    return match?.label ?? timeframe;
  }, [timeframe]);
  const planSymbol = plan.symbol?.toUpperCase() ?? "—";
  const planVersion = plan.version ?? (plan as Record<string, unknown>).version ?? null;
  const planAsOfLabel = React.useMemo(() => {
    const asOf = plan.session_state?.as_of ?? null;
    if (!asOf) return null;
    const date = new Date(asOf);
    if (Number.isNaN(date.getTime())) return null;
    return HEADER_AS_OF_FORMATTER.format(date);
  }, [plan.session_state?.as_of]);
  const planIdLabel = plan.plan_id || activePlanId;

  const handleSymbolInputChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setSymbolDraft(sanitizeSymbolToken(event.target.value));
    },
    [],
  );

  const handleSymbolSubmit = React.useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const token = sanitizeSymbolToken(symbolDraft);
      if (!token || symbolSubmitting) return;
      const now = Date.now();
      if (now - symbolRequestRef.current < 300) return;
      symbolRequestRef.current = now;
      setSymbolSubmitting(true);
      try {
        const response = await fetch(`${API_BASE_URL}/gpt/plan`, {
          method: "POST",
          headers: withAuthHeaders({ "Content-Type": "application/json", Accept: "application/json" }),
          body: JSON.stringify({ symbol: token }),
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`symbol plan failed (${response.status})`);
        }
        const payload = await response.json();
        const nextPlanId: string | undefined = payload?.plan?.plan_id ?? payload?.plan_id;
        if (nextPlanId) {
          router.push(`/plan/${encodeURIComponent(nextPlanId)}`);
        }
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("[LivePlanClient] symbol submit", error);
        }
      } finally {
        setSymbolSubmitting(false);
      }
    },
    [router, symbolDraft, symbolSubmitting],
  );

  const handleSetFollowLive = React.useCallback((value: boolean) => {
    setFollowLive(value);
  }, []);

  const handleToggleStreaming = React.useCallback(() => {
    setStreamingEnabled((prev) => {
      if (!prev) {
        requestPriceRefresh();
      }
      return !prev;
    });
  }, [requestPriceRefresh]);

  const handleToggleSupporting = React.useCallback(() => {
    setSupportVisible((prev) => !prev);
  }, []);

  const handleSelectTimeframe = React.useCallback((value: string) => {
    setTimeframe(value);
    requestPriceRefresh();
  }, [requestPriceRefresh]);

  const handleSelectLevel = React.useCallback((level: SupportingLevel | null) => {
    setHighlightedLevel(level);
  }, []);

  const handleHighlightLevel = React.useCallback((levelId: string) => {
    setActiveLevelId((prev) => (prev === levelId ? null : levelId));
  }, []);

  const handleToggleLevelVisibility = React.useCallback((levelId: string) => {
    setHiddenLevelIds((prev) => {
      const next = new Set(prev);
      if (next.has(levelId)) {
        next.delete(levelId);
      } else {
        next.add(levelId);
      }
      return next;
    });
  }, []);

  const handleLastBarTime = React.useCallback((time: number | null) => {
    setLastBarTime(time);
    if (time != null) {
      lastDeltaAtRef.current = Date.now();
    }
  }, []);

  React.useEffect(() => {
    setTimeframe((current) => current || normalizeTimeframeFromPlan(plan));
  }, [plan]);

  React.useEffect(() => {
    markPlanHeartbeat();
  }, [markPlanHeartbeat]);

  React.useEffect(() => {
    if (socketStatus === "connected") {
      markPlanHeartbeat();
    }
  }, [socketStatus, markPlanHeartbeat]);

  const indicatorItems = React.useMemo(
    () => [
      { key: "plan", label: "Plan", tone: planStatusTone, title: planStatusTitle },
      { key: "data", label: "Data", tone: dataStatusTone, title: dataStatusTitle },
      { key: "stream", label: "Stream", tone: streamStatusTone, title: streamStatusTitle },
    ],
    [planStatusTone, planStatusTitle, dataStatusTone, dataStatusTitle, streamStatusTone, streamStatusTitle],
  );

  const handleStatusTouchStart = React.useCallback((event: React.TouchEvent<HTMLElement>) => {
    if (event.touches.length !== 1) return;
    touchStartYRef.current = event.touches[0]?.clientY ?? null;
  }, []);

  const handleStatusTouchEnd = React.useCallback((event: React.TouchEvent<HTMLElement>) => {
    if (touchStartYRef.current == null) return;
    const endY = event.changedTouches[0]?.clientY ?? touchStartYRef.current;
    const delta = endY - touchStartYRef.current;
    touchStartYRef.current = null;
    if (delta < -60) {
      setCollapsed(true);
    }
    if (delta > 60) {
      setCollapsed(false);
    }
  }, []);

  const statusStrip = (
    <section
      className="flex flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8"
      role="region"
      aria-label="Fancy Trader status"
      onTouchStart={handleStatusTouchStart}
      onTouchEnd={handleStatusTouchEnd}
    >
      {!collapsed ? (
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.25em] text-neutral-400">
              <span className="text-lg font-semibold uppercase tracking-[0.35em] text-emerald-300">Fancy Trader</span>
              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-100">
                {planSymbol}
              </span>
              <span className="rounded-full border border-neutral-700/60 px-2 py-0.5 text-[0.68rem] text-neutral-300">
                {timeframeLabel}
              </span>
              {sessionBanner ? (
                <span className="rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-[0.68rem] text-sky-200">
                  {sessionBanner.toUpperCase()}
                </span>
              ) : null}
              {riskBanner ? (
                <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[0.68rem] text-amber-200">
                  {riskBanner}
                </span>
              ) : null}
              <span className="text-xs text-neutral-300">
                Last&nbsp;
                <span className="font-semibold text-white">{lastPriceLabel}</span>
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-[0.62rem] uppercase tracking-[0.3em] text-neutral-400">
              {indicatorItems.map((indicator) => (
                <button
                  key={indicator.key}
                  type="button"
                  className="flex flex-col items-center gap-1 transition hover:text-neutral-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
                  aria-label={indicator.title}
                  title={indicator.title}
                >
                  <span className="sr-only">{indicator.label}</span>
                  <StatusDot color={indicator.tone} label={`${indicator.label} status ${indicator.tone}`} />
                  <span className="hidden sm:block">{indicator.label}</span>
                </button>
              ))}
              <button
                type="button"
                onClick={() => {
                  const next = !followLive;
                  handleSetFollowLive(next);
                }}
                className={clsx(
                  "rounded-full border px-3 py-1 text-[0.62rem] font-semibold uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                  followLive
                    ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-50"
                    : "border-neutral-700/60 bg-neutral-900/60 text-neutral-300 hover:border-emerald-400/60 hover:text-emerald-50",
                )}
                aria-pressed={followLive}
              >
                Follow {followLive ? "On" : "Off"}
              </button>
              <button
                type="button"
                onClick={() => setCollapsed((prev) => !prev)}
                className="rounded-full border border-neutral-700/60 bg-neutral-900/60 px-3 py-1 text-[0.62rem] font-semibold uppercase tracking-[0.2em] text-neutral-200 transition hover:border-emerald-400/60 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
                aria-pressed={collapsed}
              >
                {collapsed ? "Expand Layout" : "Collapse"}
              </button>
            </div>
          </div>
          <form className="flex flex-wrap items-center gap-2" onSubmit={handleSymbolSubmit}>
            <input
              type="text"
              inputMode="text"
              autoComplete="off"
              maxLength={6}
              placeholder="SYM"
              value={symbolDraft}
              onChange={handleSymbolInputChange}
              disabled={symbolSubmitting}
              className="h-10 w-28 rounded-lg border border-neutral-700 bg-neutral-950/70 px-3 text-sm font-semibold uppercase tracking-[0.25em] text-neutral-100 shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
            />
            <button
              type="submit"
              disabled={symbolSubmitting || !symbolDraft}
              className={clsx(
                "h-10 min-w-[72px] rounded-lg border px-4 text-xs font-semibold uppercase tracking-[0.25em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                symbolSubmitting || !symbolDraft
                  ? "cursor-not-allowed border-neutral-700 bg-neutral-900/60 text-neutral-500"
                  : "border-emerald-500/60 bg-emerald-500/15 text-emerald-50 hover:border-emerald-400",
              )}
              aria-label="Generate plan for symbol"
            >
              {symbolSubmitting ? "..." : "Plan"}
            </button>
          </form>
        </div>
      ) : null}
      <CoachNote
        note={coachNote}
        subdued={!streamingEnabled || dataStatusTone !== "green"}
        loading={coachLoading}
        actions={
          collapsed ? (
            <button
              type="button"
              onClick={() => setCollapsed(false)}
              className="rounded-full border border-neutral-700/60 bg-neutral-900/60 px-3 py-1 text-[0.62rem] font-semibold uppercase tracking-[0.2em] text-neutral-200 transition hover:border-emerald-400/60 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
            >
              Expand Layout
            </button>
          ) : null
        }
      />
      {!collapsed ? (
        <div className="space-y-2">
          <HeaderMarkers
            levels={headerLevels}
            highlightedId={activeLevelId}
            onHighlight={handleHighlightLevel}
            onToggleVisibility={handleToggleLevelVisibility}
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[0.68rem] uppercase tracking-[0.25em] text-neutral-500">Confluence</span>
              {confluenceTokens.length ? (
                <div className="flex flex-wrap items-center gap-2 text-[0.68rem] text-neutral-300">
                  {confluenceTokens.map((token, index) => (
                    <span
                      key={`${token}-${index}`}
                      className="inline-flex items-center gap-1 rounded-md border border-neutral-800/60 bg-neutral-900/50 px-2 py-1"
                    >
                      <span className="block h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden />
                      <span>{token}</span>
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-[0.68rem] text-neutral-600">No confluence noted</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[0.68rem] uppercase tracking-[0.25em] text-neutral-500">Confidence</span>
              <ConfidenceBadge value={plan.confidence} />
            </div>
          </div>
        </div>
      ) : null}
      {!collapsed ? (
        <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.2em] text-neutral-400">
          <span>
            Plan&nbsp;
            <span className="font-semibold text-neutral-100">{planIdLabel}</span>
            {planVersion ? <span>&nbsp;· v{planVersion}</span> : null}
          </span>
          {planAsOfLabel ? <span>As of {planAsOfLabel} UTC</span> : null}
          <span>
            Data&nbsp;
            <span className="font-semibold text-neutral-100">{dataStatusTitle}</span>
          </span>
        </div>
      ) : null}
    </section>
  );

  const planPanel = (
    <div className="flex h-full flex-col gap-4">
      <PlanPanel
        plan={plan}
        structured={plan.structured_plan ?? null}
        badges={Array.isArray(plan.badges) ? plan.badges : undefined}
        confidence={typeof plan.confidence === 'number' ? plan.confidence : null}
        supportingLevels={supportingLevels}
        highlightedLevel={highlightedLevel}
        onSelectLevel={handleSelectLevel}
        targetsAwaiting={planTargets.length === 0}
        theme={theme}
      />
    </div>
  );

  const chartPanel = (
    <PlanChartPanel
      plan={plan}
      layers={layers}
      primaryLevels={primaryLevels}
      supportingVisible={supportVisible}
      followLive={followLive}
      streamingEnabled={streamingEnabled}
      onSetFollowLive={handleSetFollowLive}
      onToggleStreaming={handleToggleStreaming}
      onToggleSupporting={handleToggleSupporting}
      timeframe={timeframe}
      timeframeOptions={TIMEFRAME_OPTIONS}
      onSelectTimeframe={handleSelectTimeframe}
      onLastBarTimeChange={handleLastBarTime}
      onReplayStateChange={(state) => {
        if (state === 'playing') {
          handleSetFollowLive(false);
        }
      }}
      theme={theme}
      devMode={!!devMode}
      priceRefreshToken={priceRefreshToken}
      highlightLevelId={activeLevelId}
      hiddenLevelIds={hiddenLevelIdList}
    />
  );

  const debugPanel = devMode ? (
    <div className="fixed bottom-4 left-4 z-50 rounded-lg border border-neutral-800 bg-neutral-950/85 px-4 py-3 text-xs text-neutral-200 shadow-lg">
      <div className="font-semibold uppercase tracking-[0.2em] text-neutral-400">Dev Stats</div>
      <div>Bundle: {BUILD_SHA ? BUILD_SHA.slice(0, 7) : 'n/a'}</div>
      <div>WS: {socketStatus}</div>
      <div>Data age: {dataAgeSeconds != null ? `${dataAgeSeconds.toFixed(1)}s` : 'n/a'}</div>
      <div>Last bar: {lastBarTime ? new Date(lastBarTime).toLocaleTimeString() : 'n/a'}</div>
      <div>Follow Live: {followLive ? 'yes' : 'no'}</div>
    </div>
  ) : null;

  return (
    <>
      <WebviewShell
        theme={theme}
        statusStrip={statusStrip}
        chartPanel={chartPanel}
        planPanel={planPanel}
        collapsed={collapsed}
      />
      {debugPanel}
    </>
  );
}

function normalizeTimeframeFromPlan(plan: PlanSnapshot['plan']): string {
  const raw = (plan.charts_params as Record<string, unknown> | undefined)?.interval ?? plan.chart_timeframe ?? '5';
  return normalizeTimeframe(raw);
}

function normalizeTimeframe(token: unknown): string {
  const value = typeof token === 'string' ? token.trim() : '';
  if (!value) return '5';
  const lower = value.toLowerCase();
  if (lower.endsWith('m')) {
    return String(Number.parseInt(lower.replace('m', ''), 10) || 5);
  }
  if (lower.endsWith('h')) {
    const minutes = (Number.parseInt(lower.replace('h', ''), 10) || 1) * 60;
    return String(minutes);
  }
  if (lower === 'd' || lower === '1d') return '1D';
  if (lower === 'w' || lower === '1w') return '1W';
  return value.toUpperCase();
}

function timeframeToMs(resolution: string): number | null {
  const token = resolution.toUpperCase();
  if (token === '1D' || token === 'D') return 24 * 60 * 60 * 1000;
  if (token === '1W' || token === 'W') return 7 * 24 * 60 * 60 * 1000;
  const minutes = Number.parseInt(token, 10);
  if (Number.isFinite(minutes) && minutes > 0) {
    return minutes * 60 * 1000;
  }
  return null;
}

function extractInitialLayers(snapshot: PlanSnapshot): PlanLayers | null {
  const planLayers = snapshot.plan?.plan_layers;
  if (planLayers) return planLayers as PlanLayers;
  const rootLayers = (snapshot as unknown as { plan_layers?: PlanLayers }).plan_layers;
  return rootLayers ?? null;
}

function extractSupportVisible(plan: PlanSnapshot['plan']): boolean {
  const params = plan.charts?.params ?? plan.charts_params;
  if (params && typeof params === 'object' && 'supportingLevels' in params) {
    const raw = (params as Record<string, unknown>).supportingLevels;
    if (typeof raw === 'string') {
      return raw !== '0';
    }
  }
  return true;
}

function mergePlanWithDelta(prev: PlanSnapshot['plan'], event: PlanDeltaEvent): PlanSnapshot['plan'] {
  const { status, next_step, note, rr_to_t1, trailing_stop, last_price } = event.changes;
  let mutated = false;
  const next: PlanSnapshot['plan'] = { ...prev };

  if (status !== undefined && status !== prev.status) {
    next.status = status;
    mutated = true;
  }
  if (next_step !== undefined && (prev as Record<string, unknown>).next_step !== next_step) {
    (next as Record<string, unknown>).next_step = next_step ?? undefined;
    mutated = true;
  }
  if (note !== undefined && prev.notes !== (note ?? null)) {
    next.notes = note ?? null;
    mutated = true;
  }
  if (rr_to_t1 !== undefined && prev.rr_to_t1 !== (rr_to_t1 ?? null)) {
    next.rr_to_t1 = rr_to_t1 ?? null;
    mutated = true;
  }
  if (trailing_stop !== undefined && (prev as Record<string, unknown>).trailing_stop !== (trailing_stop ?? null)) {
    (next as Record<string, unknown>).trailing_stop = trailing_stop ?? null;
    if (trailing_stop != null) {
      const detailsPrev = (prev.details ?? {}) as Record<string, unknown>;
      next.details = { ...detailsPrev, stop: trailing_stop };
    }
    mutated = true;
  }
  if (last_price !== undefined && (prev as Record<string, unknown>).last_price !== (last_price ?? null)) {
    (next as Record<string, unknown>).last_price = last_price ?? null;
    if (last_price != null) {
      const detailsPrev = (prev.details ?? {}) as Record<string, unknown>;
      next.details = { ...detailsPrev, last: last_price };
    }
    mutated = true;
  }

  return mutated ? next : prev;
}

const STATUS_TONE_COLOR: Record<StatusTone, string> = {
  green: "#34d399",
  yellow: "#f59e0b",
  red: "#ef4444",
};

export function StatusDot({ color, label, className }: { color: StatusTone; label: string; className?: string }) {
  return (
    <span className={clsx("inline-flex h-2 w-2 items-center justify-center", className)}>
      <span
        aria-label={label}
        role="status"
        className="block h-2 w-2 rounded-full"
        style={{ backgroundColor: STATUS_TONE_COLOR[color] }}
      />
    </span>
  );
}

function getNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}
