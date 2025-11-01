"use client";

import clsx from "clsx";
import * as React from "react";
import { useRouter } from "next/navigation";
import WebviewShell from "@/components/webview/WebviewShell";
import PlanPanel from "@/components/webview/PlanPanel";
import PlanChartPanel from "@/components/PlanChartPanel";
import HeaderMarkers from "@/components/HeaderMarkers";
import { WatchlistRail, WatchlistDrawer, useWatchlist } from "@/features/watchlist";
import { PlanInsightsSheet } from "@/features/plan";
import { CoachPanel } from "@/features/coach";
import SessionChip from "@/components/SessionChip";
import ObjectiveProgress from "@/components/ObjectiveProgress";
import { extractPrimaryLevels, extractSupportingLevels } from "@/lib/utils/layers";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";
import { API_BASE_URL, BUILD_SHA, withAuthHeaders } from "@/lib/env";
import { extractPlanLevels, resolveTrailingStop } from "@/lib/plan/levels";
import { useStore } from "@/store/useStore";
import { wsMux, type ConnectionState } from "@/lib/wsMux";
import { useChartUrl } from "@/lib/hooks/useChartUrl";
import headerStyles from "./LivePlanHeader.module.css";

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
  const planState = useStore((state) => state.plan);
  const planLayersState = useStore((state) => state.planLayers);
  const applyEvent = useStore((state) => state.applyEvent);
  const hydratePlan = useStore((state) => state.hydratePlan);
  const setPlanLayers = useStore((state) => state.setPlanLayers);
  const setSession = useStore((state) => state.setSession);
  const setConnection = useStore((state) => state.setConnection);
  const connectionState = useStore((state) => state.connection);
  const wsStats = useStore((state) => state.wsStats);
  const initialLayers = React.useMemo(() => extractInitialLayers(initialSnapshot), [initialSnapshot]);
  const plan = planState ?? initialSnapshot.plan;
  const layers = planLayersState ?? initialLayers;
  const planConnectionStatus = connectionState.plan;
  const barsConnectionStatus = connectionState.bars;
  const coachConnectionStatus = connectionState.coach;
  const chartUrl = useChartUrl(plan);
  const coachPulse = useStore((state) => state.coach);
  const {
    items: watchlistItems,
    status: watchlistStatus,
    error: watchlistError,
    lastUpdated: watchlistUpdated,
    refresh: refreshWatchlist,
  } = useWatchlist();
  const [watchlistOpen, setWatchlistOpen] = React.useState(false);
  const [insightsOpen, setInsightsOpen] = React.useState(false);
  const [followLive, setFollowLive] = React.useState(true);
  const [streamingEnabled, setStreamingEnabled] = React.useState(true);
  const [supportVisible, setSupportVisible] = React.useState(() => extractSupportVisible(initialSnapshot.plan));
  const [highlightedLevel, setHighlightedLevel] = React.useState<SupportingLevel | null>(null);
  const [timeframe, setTimeframe] = React.useState(() => normalizeTimeframeFromPlan(initialSnapshot.plan));
  const [nowTick, setNowTick] = React.useState(Date.now());
  const [lastBarTime, setLastBarTime] = React.useState<number | null>(null);
  const [devMode, setDevMode] = React.useState(() => process.env.NEXT_PUBLIC_DEVTOOLS === "1");
  const [priceRefreshToken, setPriceRefreshToken] = React.useState(0);
  const [lastPlanHeartbeat, setLastPlanHeartbeat] = React.useState<number | null>(() => Date.now());
  const [symbolDraft, setSymbolDraft] = React.useState(() => (plan.symbol ? plan.symbol.toUpperCase() : ""));
  const [symbolSubmitting, setSymbolSubmitting] = React.useState(false);
  const [addingSetup, setAddingSetup] = React.useState(false);
  const [activeLevelId, setActiveLevelId] = React.useState<string | null>(null);
  const [hiddenLevelIds, setHiddenLevelIds] = React.useState<Set<string>>(() => new Set());
  const [collapsed, setCollapsed] = React.useState(false);
  const lastPriceRefreshRef = React.useRef(0);
  const lastDeltaAtRef = React.useRef(Date.now());
  const symbolRequestRef = React.useRef(0);
  const replanPendingRef = React.useRef(false);
  const touchStartYRef = React.useRef<number | null>(null);

  const theme = "dark" as const;

  const bootstrappedRef = React.useRef(false);
  const perfRef = React.useRef<{ mark: (name: string) => void; end: (name: string) => void } | null>(null);
  const diagModuleRef = React.useRef<{ updatePerf: () => void; updateDataAge: (date: Date | null) => void } | null>(null);
  const prevPlanStateRef = React.useRef<ConnectionState | null>(null);

  React.useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const mod = await import("../../../../../webview/diag.js");
        if (disposed) return;
        diagModuleRef.current = {
          updatePerf: () => mod.updatePerf(),
          updateDataAge: (date: Date | null) => mod.updateDataAge(date as unknown as Date),
        };
        perfRef.current = mod.Perf;
        mod.Perf.mark("webview:init");
        await mod.mountObservabilityPanel("#diag-panel");
        mod.Perf.end("webview:init");
        mod.updatePerf();
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[LivePlanClient] diagnostics mount failed", error);
        }
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  React.useEffect(() => {
    if (bootstrappedRef.current) return;
    hydratePlan(initialSnapshot.plan);
    setPlanLayers(initialLayers ?? null);
    const sessionBlock = initialSnapshot.plan?.session_state ?? null;
    if (sessionBlock) {
      setSession(sessionBlock);
    }
    bootstrappedRef.current = true;
  }, [hydratePlan, initialSnapshot.plan, initialLayers, setPlanLayers, setSession]);

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

  const streamingEnabledRef = React.useRef(streamingEnabled);
  React.useEffect(() => {
    streamingEnabledRef.current = streamingEnabled;
  }, [streamingEnabled]);

  React.useEffect(() => {
    if (!streamingEnabled) {
      setConnection("plan", "idle");
      return;
    }
    const release = wsMux.connectPlan(activePlanId);
    const releaseState = wsMux.onState("plan", activePlanId, (state) => {
      setConnection("plan", state);
      if (state === "connected") {
        markPlanHeartbeat();
      }
    });
    return () => {
      release();
      releaseState();
    };
  }, [activePlanId, markPlanHeartbeat, setConnection, streamingEnabled]);

  const planSymbol = React.useMemo(() => (plan.symbol ? plan.symbol.toUpperCase() : null), [plan.symbol]);
  const barsBySymbol = useStore((state) => state.barsBySymbol);
  const streamingBars = React.useMemo(() => (planSymbol ? barsBySymbol[planSymbol] : undefined), [barsBySymbol, planSymbol]);

  React.useEffect(() => {
    if (!streamingEnabled || !planSymbol) {
      setConnection("bars", "idle");
      return;
    }
    const release = wsMux.connectBars(planSymbol);
    const releaseState = wsMux.onState("bars", planSymbol, (state) => setConnection("bars", state));
    return () => {
      release();
      releaseState();
    };
  }, [planSymbol, setConnection, streamingEnabled]);

  React.useEffect(() => {
    if (!streamingEnabled) {
      setConnection("coach", "idle");
      return;
    }
    const release = wsMux.connectCoach(activePlanId);
    const releaseState = wsMux.onState("coach", activePlanId, (state) => setConnection("coach", state));
    return () => {
      release();
      releaseState();
    };
  }, [activePlanId, setConnection, streamingEnabled]);

  React.useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const load = async () => {
      try {
        const qs = new URLSearchParams({ plan_id: activePlanId });
        const response = await fetch(`${API_BASE_URL}/api/v1/gpt/chart-layers?${qs.toString()}`, {
          cache: "no-store",
          signal: controller.signal,
          headers: withAuthHeaders({ Accept: "application/json" }),
        });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as PlanLayers;
        if (!cancelled && payload) {
          setPlanLayers(payload);
        }
      } catch (error) {
        if (cancelled || (error instanceof DOMException && error.name === "AbortError")) {
          return;
        }
        if (process.env.NODE_ENV !== "production") {
          console.warn("[LivePlanClient] chart layers fetch failed", error);
        }
      }
    };
    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [activePlanId, setPlanLayers]);

  const refreshPlanSnapshot = React.useCallback(
    async (targetPlanId: string) => {
      if (!targetPlanId) return;
      try {
        perfRef.current?.mark("plan:fetch");
        const response = await fetch(`${API_BASE_URL}/idea/${encodeURIComponent(targetPlanId)}`, {
          headers: withAuthHeaders({ Accept: "application/json" }),
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`refreshPlanSnapshot failed (${response.status})`);
        }
        const payload = (await response.json()) as PlanSnapshot;
        if (!payload?.plan) return;
        hydratePlan(payload.plan);
        const overlays = extractInitialLayers(payload);
        setPlanLayers(overlays ?? null);
        if (payload.plan.session_state) {
          setSession(payload.plan.session_state);
        }
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
      } finally {
        perfRef.current?.end("plan:fetch");
      }
    },
    [hydratePlan, requestPriceRefresh, setPlanLayers, setSession],
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

  React.useEffect(() => {
    const unsubscribe = wsMux.onEvent((event) => {
      if (!streamingEnabledRef.current) return;
      if (event.t === "plan_delta") {
        markPlanHeartbeat();
        applyEvent(event);
        const statusToken = typeof event.fields.status === "string" ? event.fields.status.toLowerCase() : null;
        if (statusToken === "invalid") {
          const targetPlanId = plan.plan_id || activePlanId;
          queueReplan(targetPlanId);
        }
        return;
      }
      applyEvent(event);
    });
    return () => {
      unsubscribe();
    };
  }, [applyEvent, markPlanHeartbeat, plan.plan_id, activePlanId, queueReplan]);

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
    if (planConnectionStatus === "error") return "red";
    if (planConnectionStatus === "idle") return "yellow";
    if (planConnectionStatus === "connecting") return "yellow";
    if (planHeartbeatAgeSeconds == null) return "yellow";
    if (planHeartbeatAgeSeconds <= 15) return "green";
    if (planHeartbeatAgeSeconds <= 60) return "yellow";
    return "red";
  }, [planConnectionStatus, planHeartbeatAgeSeconds]);

  const streamStatusTitle = React.useMemo(() => {
    if (planConnectionStatus === "error") return "Stream disconnected";
    if (planConnectionStatus === "idle") return "Stream idle";
    if (planConnectionStatus === "connecting") return "Stream connecting";
    if (planHeartbeatAgeSeconds == null) return "Stream heartbeat pending";
    return `Last stream ${planHeartbeatAgeSeconds.toFixed(1)}s ago`;
  }, [planConnectionStatus, planHeartbeatAgeSeconds]);

  const dataStatusTone = React.useMemo<StatusTone>(() => {
    if (!streamingEnabled) return "red";
    if (barsConnectionStatus === "error" || barsConnectionStatus === "idle") return "red";
    if (dataAgeSeconds == null || resolutionSeconds == null) return "yellow";
    if (dataAgeSeconds <= resolutionSeconds * 2) return "green";
    if (dataAgeSeconds <= resolutionSeconds * 6) return "yellow";
    return "red";
  }, [streamingEnabled, dataAgeSeconds, resolutionSeconds, barsConnectionStatus]);

  const dataStatusTitle = React.useMemo(() => {
    if (dataAgeSeconds == null) return "Price data pending";
    return `Last price update ${dataAgeSeconds.toFixed(1)}s ago`;
  }, [dataAgeSeconds]);

  React.useEffect(() => {
    if (!diagModuleRef.current) return;
    diagModuleRef.current.updatePerf();
  }, [planConnectionStatus, barsConnectionStatus, streamingEnabled, streamStatusTone, dataStatusTone]);

  React.useEffect(() => {
    if (!diagModuleRef.current) return;
    if (lastBarTime) {
      diagModuleRef.current.updateDataAge(new Date(lastBarTime));
    } else {
      diagModuleRef.current.updateDataAge(null);
    }
  }, [lastBarTime]);

  React.useEffect(() => {
    if (!diagModuleRef.current || lastBarTime) return;
    const asOf = plan.session_state?.as_of;
    if (!asOf) {
      diagModuleRef.current.updateDataAge(null);
      return;
    }
    const date = new Date(asOf);
    if (Number.isNaN(date.getTime())) {
      diagModuleRef.current.updateDataAge(null);
    } else {
      diagModuleRef.current.updateDataAge(date);
    }
  }, [plan.session_state?.as_of, lastBarTime]);

  const sessionBanner = plan.session_state?.banner ?? null;
  const riskBanner =
    Array.isArray(plan.warnings) && plan.warnings.length ? String(plan.warnings[0]) : plan.session_state?.message ?? null;

  const trailingStopValue = React.useMemo(() => resolveTrailingStop(plan), [plan]);
  React.useEffect(() => {
    setSymbolDraft(plan.symbol ? sanitizeSymbolToken(String(plan.symbol)) : "");
  }, [plan.symbol]);

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
    setWatchlistOpen(false);
    setInsightsOpen(false);
  }, [plan.plan_id]);

  React.useEffect(() => {
    if (collapsed) {
      setAddingSetup(false);
    }
  }, [collapsed]);

  const lastPrice = React.useMemo(() => {
    const details = (plan.details ?? {}) as Record<string, unknown>;
    return (
      getNumber((plan as Record<string, unknown>).last_price) ??
      getNumber(details.last) ??
      getNumber((plan as Record<string, unknown>).mark) ??
      null
    );
  }, [plan]);

  const coachActionText = React.useMemo(() => {
    const diff = coachPulse.diff;
    if (diff?.next_action && diff.next_action.trim()) {
      return diff.next_action.trim();
    }
    if (diff?.waiting_for) {
      return diff.waiting_for.replace(/[_-]/g, " ");
    }
    return "Awaiting guidance…";
  }, [coachPulse.diff]);

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
  const planSymbolLabel = plan.symbol?.toUpperCase() ?? "—";
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
          setAddingSetup(false);
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
  const handleCancelSetup = React.useCallback(() => {
    setAddingSetup(false);
  }, []);

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
    if (planConnectionStatus === "connected") {
      markPlanHeartbeat();
    }
  }, [planConnectionStatus, markPlanHeartbeat]);

  React.useEffect(() => {
    const previous = prevPlanStateRef.current;
    if (planConnectionStatus === "connecting" && previous !== "connecting") {
      perfRef.current?.mark("ws:connect");
    }
    if (planConnectionStatus === "connected" && previous !== "connected") {
      perfRef.current?.end("ws:connect");
    }
    prevPlanStateRef.current = planConnectionStatus;
  }, [planConnectionStatus]);

  const indicatorItems = React.useMemo(
    () => [
      { key: "plan", label: "Plan", tone: planStatusTone, title: planStatusTitle },
      { key: "data", label: "Data", tone: dataStatusTone, title: dataStatusTitle },
      { key: "stream", label: "Stream", tone: streamStatusTone, title: streamStatusTitle },
    ],
    [planStatusTone, planStatusTitle, dataStatusTone, dataStatusTitle, streamStatusTone, streamStatusTitle],
  );
  const headerStatusItems = React.useMemo(
    () => indicatorItems.filter((item) => item.key !== "plan"),
    [indicatorItems],
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

  const toneBadgeClass: Record<StatusTone, string> = {
    green: "border-emerald-500/60 bg-emerald-500/10 text-emerald-100",
    yellow: "border-amber-500/60 bg-amber-500/10 text-amber-100",
    red: "border-rose-500/60 bg-rose-500/10 text-rose-100",
  };

  const renderSetupControl = (variant: "expanded" | "collapsed") => {
    const buttonClasses = clsx(
      "inline-flex items-center justify-center rounded-full border border-neutral-700/60 bg-neutral-900/60 text-neutral-200 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
      variant === "collapsed" ? "px-3 py-1 text-[0.58rem] font-semibold uppercase tracking-[0.22em]" : "px-3.5 py-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.24em]",
    );
    if (!addingSetup) {
      return (
        <button
          type="button"
          className={buttonClasses}
          onClick={() => {
            if (variant === "collapsed") {
              setCollapsed(false);
            }
            setAddingSetup(true);
          }}
        >
          Add Setup
        </button>
      );
    }
    if (variant === "collapsed") {
      return null;
    }
    return (
      <form className="flex flex-wrap items-center gap-2" onSubmit={handleSymbolSubmit}>
        <label className="sr-only" htmlFor="setup-symbol-input">
          Symbol
        </label>
        <input
          id="setup-symbol-input"
          type="text"
          inputMode="text"
          autoComplete="off"
          maxLength={6}
          placeholder="SYM"
          value={symbolDraft}
          onChange={handleSymbolInputChange}
          disabled={symbolSubmitting}
          className="h-9 w-28 rounded-md border border-neutral-700 bg-neutral-950/70 px-3 text-[0.7rem] font-semibold uppercase tracking-[0.25em] text-neutral-100 shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
        />
        <div className="flex items-center gap-1">
          <button
            type="submit"
            disabled={symbolSubmitting || !symbolDraft}
            className={clsx(
              "rounded-full border px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
              symbolSubmitting || !symbolDraft
                ? "cursor-not-allowed border-neutral-700/60 bg-neutral-900/60 text-neutral-500"
                : "border-emerald-500/60 bg-emerald-500/15 text-emerald-50 hover:border-emerald-400",
            )}
          >
            {symbolSubmitting ? "..." : "Find Setup"}
          </button>
          <button
            type="button"
            onClick={handleCancelSetup}
            className="rounded-full border border-neutral-700/60 px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] text-neutral-300 transition hover:border-neutral-500 hover:text-neutral-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
          >
            Cancel
          </button>
        </div>
      </form>
    );
  };

  const statusStrip = (
    <section
      className="flex flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8"
      role="region"
      aria-label="Fancy Trader status"
      onTouchStart={handleStatusTouchStart}
      onTouchEnd={handleStatusTouchEnd}
    >
      {collapsed ? (
        <div className={headerStyles.collapsedBanner}>
          <button
            type="button"
            onClick={() => setCollapsed(false)}
            className={headerStyles.collapsedButton}
            aria-expanded="false"
          >
            <span className={headerStyles.collapsedDot} aria-hidden="true" />
            <span className={headerStyles.collapsedTextWrapper}>
            <span className={headerStyles.collapsedText}>{coachActionText}</span>
            </span>
            <span className="sr-only">Expand coach guidance</span>
            <svg
              aria-hidden
              className={headerStyles.collapsedChevron}
              width="10"
              height="10"
              viewBox="0 0 10 10"
              fill="none"
            >
              <path
                d="M2 3.5L5 6.5L8 3.5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <div className={headerStyles.collapsedActions}>{renderSetupControl("collapsed")}</div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.25em] text-neutral-400">
            <span className="text-lg font-semibold uppercase tracking-[0.35em] text-emerald-300">Fancy Trader</span>
            <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-100">
              {planSymbolLabel}
            </span>
            <span className="rounded-full border border-neutral-700/60 px-2 py-0.5 text-[0.65rem] text-neutral-300">
              {timeframeLabel}
            </span>
            {sessionBanner ? (
              <span className="rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-[0.65rem] text-sky-200">
                {sessionBanner.toUpperCase()}
              </span>
            ) : null}
            {riskBanner ? (
              <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[0.65rem] text-amber-200">
                {riskBanner}
              </span>
            ) : null}
            <span className="text-xs text-neutral-300">
              Last&nbsp;
              <span className="font-semibold text-white">{lastPriceLabel}</span>
            </span>
          </div>
          <CoachPanel plan={plan} layers={layers} />
          <div className="flex flex-col gap-3 md:flex-row md:items-start">
            <SessionChip />
            <div className="min-w-[220px] flex-1">
              <ObjectiveProgress />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[0.6rem] uppercase tracking-[0.24em] text-neutral-400">
            <div className="flex flex-wrap items-center gap-2">
              {headerStatusItems.map((indicator) => (
                <span
                  key={indicator.key}
                  className={clsx(
                    "inline-flex items-center gap-1 rounded-full border px-3 py-1 transition",
                    toneBadgeClass[indicator.tone],
                  )}
                  title={indicator.title}
                >
                  <StatusDot color={indicator.tone} label={indicator.title} />
                  <span>{indicator.label}</span>
                </span>
              ))}
            </div>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setWatchlistOpen(true)}
                className="inline-flex items-center justify-center rounded-full border border-neutral-700/60 bg-neutral-900/60 px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] text-neutral-200 transition hover:border-emerald-400/60 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 md:hidden"
                aria-haspopup="dialog"
                aria-expanded={watchlistOpen}
              >
                Watchlist
              </button>
              <button
                type="button"
                onClick={() => setInsightsOpen(true)}
                className="inline-flex items-center justify-center rounded-full border border-neutral-700/60 bg-neutral-900/60 px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] text-neutral-200 transition hover:border-emerald-400/60 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 md:hidden"
                aria-haspopup="dialog"
                aria-expanded={insightsOpen}
              >
                Insights
              </button>
              <button
                type="button"
                onClick={() => handleSetFollowLive(!followLive)}
                className={clsx(
                  "inline-flex items-center justify-center rounded-full border px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
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
                onClick={handleToggleSupporting}
                className={clsx(
                  "inline-flex items-center justify-center rounded-full border px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                  supportVisible
                    ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-100"
                    : "border-neutral-700/60 bg-neutral-900/60 text-neutral-300 hover:border-emerald-400/60 hover:text-emerald-50",
                )}
                aria-pressed={supportVisible}
              >
                Levels {supportVisible ? "On" : "Off"}
              </button>
              <button
                type="button"
                onClick={handleToggleStreaming}
                className={clsx(
                  "inline-flex items-center justify-center rounded-full border px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                  streamingEnabled
                    ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-100"
                    : "border-neutral-700/60 bg-neutral-900/60 text-neutral-300 hover:border-emerald-400/60 hover:text-emerald-50",
                )}
                aria-pressed={streamingEnabled}
              >
                Stream {streamingEnabled ? "On" : "Off"}
              </button>
              {chartUrl ? (
                <a
                  href={chartUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={clsx(
                    "inline-flex items-center justify-center rounded-full border px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                    "border-emerald-500/50 bg-emerald-500/10 text-emerald-100 hover:border-emerald-400/70 hover:text-emerald-50",
                  )}
                >
                  Open Chart
                </a>
              ) : null}
              {renderSetupControl("expanded")}
              <button
                type="button"
                onClick={() => setCollapsed(true)}
                className="inline-flex items-center justify-center rounded-full border border-neutral-700/60 bg-neutral-900/60 px-3 py-1 text-[0.6rem] font-semibold uppercase tracking-[0.24em] text-neutral-200 transition hover:border-emerald-400/60 hover:text-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
                aria-expanded={!collapsed}
              >
                Collapse
              </button>
            </div>
          </div>
          <div className="space-y-2">
            <HeaderMarkers
              levels={headerLevels}
              highlightedId={activeLevelId}
              onHighlight={handleHighlightLevel}
              onToggleVisibility={handleToggleLevelVisibility}
            />
            <div className="flex flex-wrap items-center gap-3 text-[0.62rem] uppercase tracking-[0.2em] text-neutral-400">
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
            <div id="diag-panel" className="mt-2 w-full" />
          </div>
        </div>
      )}
    </section>
  );

  const planPanel = (
    <div className="flex h-full flex-col gap-4">
      <PlanPanel
        plan={plan}
        supportingLevels={supportingLevels}
        highlightedLevel={highlightedLevel}
        onSelectLevel={handleSelectLevel}
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
      liveBars={streamingBars}
    />
  );

  const debugPanel = devMode ? (
    <div className="fixed bottom-4 left-4 z-50 rounded-lg border border-neutral-800 bg-neutral-950/85 px-4 py-3 text-xs text-neutral-200 shadow-lg">
      <div className="font-semibold uppercase tracking-[0.2em] text-neutral-400">Dev Stats</div>
      <div>Bundle: {BUILD_SHA ? BUILD_SHA.slice(0, 7) : 'n/a'}</div>
      <div>WS: {planConnectionStatus}</div>
      <div>WS uptime: {(wsStats.plan.uptimeMs / 1000).toFixed(1)}s · reconnects {wsStats.plan.reconnects}</div>
      <div>Data age: {dataAgeSeconds != null ? `${dataAgeSeconds.toFixed(1)}s` : 'n/a'}</div>
      <div>Last bar: {lastBarTime ? new Date(lastBarTime).toLocaleTimeString() : 'n/a'}</div>
      <div>Follow Live: {followLive ? 'yes' : 'no'}</div>
    </div>
  ) : null;

  const watchlistPanel = (
    <WatchlistRail
      items={watchlistItems}
      status={watchlistStatus}
      error={watchlistError}
      lastUpdated={watchlistUpdated}
      onRefresh={refreshWatchlist}
    />
  );

  const handleWatchlistSelect = React.useCallback(
    (planUrl: string) => {
      setWatchlistOpen(false);
      router.push(planUrl);
    },
    [router],
  );

  const mobileSheets = (
    <>
      <WatchlistDrawer
        open={watchlistOpen}
        onClose={() => setWatchlistOpen(false)}
        items={watchlistItems}
        status={watchlistStatus}
        error={watchlistError}
        onSelect={handleWatchlistSelect}
      />
      <PlanInsightsSheet
        plan={plan}
        layers={layers}
        trailingStop={levelSummary.trailingStop ?? null}
        open={insightsOpen}
        onOpenChange={setInsightsOpen}
      />
    </>
  );

  return (
    <>
      <WebviewShell
        theme={theme}
        statusStrip={statusStrip}
        chartPanel={chartPanel}
        planPanel={planPanel}
        watchlistPanel={watchlistPanel}
        collapsed={collapsed}
        mobileSheet={mobileSheets}
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
