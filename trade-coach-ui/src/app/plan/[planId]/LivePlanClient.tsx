"use client";

import clsx from "clsx";
import * as React from "react";
import { useRouter } from "next/navigation";
import PlanShell from "@/components/PlanShell";
import PlanChartPanel from "@/components/PlanChartPanel";
import SessionChip, { type SessionSSOT, safeTime } from "@/components/SessionChip";
import ObjectiveProgress, { type NextObjectiveMeta } from "@/components/ObjectiveProgress";
import WatchlistRail, { type WatchItem as WatchlistRailItem } from "@/components/WatchlistRail";
import CoachNote from "@/components/CoachNote";
import ConfluenceStrength from "@/components/ConfluenceStrength";
import { extractPrimaryLevels } from "@/lib/utils/layers";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";
import { API_BASE_URL, BUILD_SHA, withAuthHeaders } from "@/lib/env";
import { useStore } from "@/store/useStore";
import { wsMux } from "@/lib/wsMux";
import { useWatchlist } from "@/features/watchlist";

const TIMEFRAME_OPTIONS = [
  { value: "1", label: "1m" },
  { value: "3", label: "3m" },
  { value: "5", label: "5m" },
  { value: "15", label: "15m" },
  { value: "60", label: "1h" },
  { value: "240", label: "4h" },
  { value: "1D", label: "1D" },
];

type CoachSnapshot = {
  text: string;
  progressPct?: number | null;
  metrics?: Array<{ key: string; label: string; value: string; ariaLabel?: string }>;
};

type LivePlanClientProps = {
  initialSnapshot: PlanSnapshot;
  planId: string;
  session?: SessionSSOT | null;
  nextObjective?: NextObjectiveMeta | null;
  watchlist?: WatchlistRailItem[] | null;
  coach?: CoachSnapshot | null;
};

type StatusToken = "connected" | "connecting" | "disconnected";

type StatusBuckets = {
  ws: StatusToken;
  price: StatusToken;
};

export default function LivePlanClient({
  initialSnapshot,
  planId,
  session: sessionProp = null,
  nextObjective: nextObjectiveProp = null,
  watchlist: watchlistProp = null,
  coach: coachProp = null,
}: LivePlanClientProps) {
  const router = useRouter();
  const planState = useStore((state) => state.plan);
  const planLayersState = useStore((state) => state.planLayers);
  const applyEvent = useStore((state) => state.applyEvent);
  const hydratePlan = useStore((state) => state.hydratePlan);
  const setPlanLayers = useStore((state) => state.setPlanLayers);
  const setSession = useStore((state) => state.setSession);
  const setConnection = useStore((state) => state.setConnection);
  const connectionState = useStore((state) => state.connection);
  const storeSession = useStore((state) => state.session);
  const coachPulse = useStore((state) => state.coach);
  const barsBySymbol = useStore((state) => state.barsBySymbol);
  const { items: watchlistItems } = useWatchlist();

  const initialLayers = React.useMemo(() => extractInitialLayers(initialSnapshot), [initialSnapshot]);
  const plan = planState ?? initialSnapshot.plan;
  const layers = planLayersState ?? initialLayers;

  const planConnectionStatus = connectionState.plan;
  const barsConnectionStatus = connectionState.bars;

  const [followLive, setFollowLive] = React.useState(true);
  const [streamingEnabled] = React.useState(true);
  const [supportVisible, setSupportVisible] = React.useState(false);
  const [timeframe, setTimeframe] = React.useState(() => normalizeTimeframeFromPlan(initialSnapshot.plan));
  const [nowTick, setNowTick] = React.useState(Date.now());
  const [lastBarTime, setLastBarTime] = React.useState<number | null>(null);
  const [devMode, setDevMode] = React.useState(() => process.env.NEXT_PUBLIC_DEVTOOLS === "1");
  const [priceRefreshToken, setPriceRefreshToken] = React.useState(0);
  const [lastPlanHeartbeat, setLastPlanHeartbeat] = React.useState<number | null>(() => Date.now());
  const [symbolDraft, setSymbolDraft] = React.useState(() => sanitizeSymbolToken(plan.symbol ?? ""));
  const [symbolSubmitting, setSymbolSubmitting] = React.useState(false);
  const [coachCollapsed, setCoachCollapsed] = React.useState(false);

  const lastPriceRefreshRef = React.useRef(0);
  const lastDeltaAtRef = React.useRef(Date.now());
  const replanPendingRef = React.useRef(false);
  const symbolRequestRef = React.useRef(0);

  const activePlanId = plan.plan_id || planId;
  const theme = "dark" as const;

  React.useEffect(() => {
    let disposed = false;
    if (disposed) return () => undefined;
    if (typeof window === "undefined") return () => undefined;
    hydratePlan(initialSnapshot.plan);
    setPlanLayers(initialLayers ?? null);
    const sessionBlock = initialSnapshot.plan?.session_state ?? null;
    if (sessionBlock) {
      setSession(sessionBlock);
    }
    return () => {
      disposed = true;
    };
  }, [hydratePlan, initialSnapshot.plan, initialLayers, setPlanLayers, setSession]);

  const requestPriceRefresh = React.useCallback(() => {
    const now = Date.now();
    if (now - lastPriceRefreshRef.current < 4000) return;
    lastPriceRefreshRef.current = now;
    setPriceRefreshToken((token) => token + 1);
  }, []);

  const handleToggleSupporting = React.useCallback(() => {
    setSupportVisible((prev) => !prev);
  }, []);

  const handleSymbolInputChange = React.useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setSymbolDraft(sanitizeSymbolToken(event.target.value));
  }, []);

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

  const markPlanHeartbeat = React.useCallback(() => {
    const now = Date.now();
    lastDeltaAtRef.current = now;
    setLastPlanHeartbeat(now);
  }, []);

  React.useEffect(() => {
    const timer = window.setInterval(() => setNowTick(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get("dev") === "1") {
        setDevMode(true);
      }
    } catch {
      // ignore
    }
  }, []);

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
        setTimeframe((prev) => prev || normalizeTimeframeFromPlan(payload.plan));
        setLastPlanHeartbeat(Date.now());
        lastDeltaAtRef.current = Date.now();
        requestPriceRefresh();
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("[LivePlanClient] refreshPlanSnapshot", error);
        }
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
    if (!streamingEnabled) return;
    const unsubscribe = wsMux.onEvent((event) => {
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
  }, [applyEvent, markPlanHeartbeat, plan.plan_id, activePlanId, queueReplan, streamingEnabled]);

  const primaryLevels = React.useMemo(() => extractPrimaryLevels(layers), [layers]);

  const resolutionMs = React.useMemo(() => timeframeToMs(timeframe), [timeframe]);
  const dataAgeSeconds = React.useMemo(() => {
    if (!lastBarTime) return null;
    return Math.max(0, (nowTick - lastBarTime) / 1000);
  }, [lastBarTime, nowTick]);

  const planHeartbeatAgeSeconds = React.useMemo(() => {
    if (!lastPlanHeartbeat) return null;
    return Math.max(0, (nowTick - lastPlanHeartbeat) / 1000);
  }, [lastPlanHeartbeat, nowTick]);

  const planAsOfLabel = React.useMemo(() => {
    const asOf = plan.session_state?.as_of ?? layers?.as_of ?? null;
    if (!asOf) return null;
    const date = new Date(asOf);
    if (Number.isNaN(date.getTime())) return null;
    return new Intl.DateTimeFormat(undefined, {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }, [plan.session_state?.as_of, layers?.as_of]);

  const sessionData = React.useMemo<SessionSSOT | null>(() => {
    if (sessionProp) return sessionProp;
    const status = plan.session_state?.status ?? storeSession?.status;
    const asOf = plan.session_state?.as_of ?? storeSession?.as_of ?? null;
    if (!status || !asOf) return null;
    const tz = storeSession?.tz ?? (plan.session_state as { tz?: string | null } | undefined)?.tz ?? "UTC";
    const nextOpen = plan.session_state?.next_open ?? storeSession?.next_open ?? null;
    return {
      status,
      tz: tz || "UTC",
      as_of: asOf,
      next_open: nextOpen ?? null,
    };
  }, [plan.session_state, sessionProp, storeSession]);

  const sessionBanner = React.useMemo(() => {
    if (!sessionData) return null;
    const tz = sessionData.tz?.toUpperCase() || "UTC";
    return `${sessionData.status.toUpperCase()} • ${tz} • ${safeTime(sessionData.as_of)}`;
  }, [sessionData]);

  const objectiveMeta = React.useMemo<NextObjectiveMeta | null>(() => {
    if (nextObjectiveProp) return nextObjectiveProp;
    const meta = layers?.meta;
    if (!meta || typeof meta !== "object" || !("next_objective" in meta)) return null;
    const candidate = (meta as Record<string, unknown>).next_objective;
    if (!candidate || typeof candidate !== "object") return null;
    return candidate as NextObjectiveMeta;
  }, [layers?.meta, nextObjectiveProp]);

  const confluenceModel = React.useMemo(() => {
    const meta = (layers?.meta ?? {}) as Record<string, unknown>;
    const planObj = (plan ?? {}) as Record<string, unknown>;
    const rawComponents = meta.confluence_components ?? planObj.confluence_components ?? {};
    const components =
      rawComponents && typeof rawComponents === "object" ? (rawComponents as Record<string, unknown>) : {};
    const normalize = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : 0);
    const evidence = planObj.evidence && typeof planObj.evidence === "object" ? (planObj.evidence as Record<string, unknown>) : null;
    const whyCandidate =
      typeof meta.confluence_why === "string"
        ? meta.confluence_why
        : typeof planObj.confluence_why === "string"
          ? planObj.confluence_why
          : typeof evidence?.why === "string"
            ? (evidence.why as string)
            : null;
    const confidenceCandidate =
      typeof planObj.confidence === "number" && Number.isFinite(planObj.confidence)
        ? (planObj.confidence as number)
        : typeof meta.confidence === "number" && Number.isFinite(meta.confidence)
          ? (meta.confidence as number)
          : null;
    return {
      atr: normalize(components.atr),
      vwap: normalize(components.vwap),
      emas: normalize(components.emas),
      orderflow: normalize(components.orderflow),
      liquidity: normalize(components.liquidity),
      why: typeof whyCandidate === "string" && whyCandidate.trim().length ? whyCandidate.trim() : null,
      confidence: confidenceCandidate,
      dataAge: lastBarTime ? new Date(lastBarTime) : null,
    };
  }, [layers?.meta, plan, lastBarTime]);

  const watchlistRailItems = React.useMemo<WatchlistRailItem[]>(() => {
    if (watchlistProp?.length) return watchlistProp;
    if (!watchlistItems?.length) return [];
    return watchlistItems
      .map<WatchlistRailItem | null>((item) => {
        if (!item.plan_id || !item.symbol) return null;
        const bias = extractBias(item.meta);
        return {
          planId: item.plan_id,
          symbol: item.symbol,
          actionableSoon: item.actionable_soon,
          entryDistancePct: item.entry_distance_pct,
          barsToTrigger: item.bars_to_trigger,
          bias,
        };
      })
      .filter((item): item is WatchlistRailItem => item !== null);
  }, [watchlistItems, watchlistProp]);

  const coachNoteContent = React.useMemo(() => {
    if (coachProp?.text) {
      return {
        note: {
          text: coachProp.text,
          goal: "neutral" as const,
          progressPct: normalizeProgressPct(coachProp.progressPct),
          updatedAt: Date.now(),
        },
        metrics: coachProp.metrics ?? [],
      };
    }
    const diff = coachPulse.diff;
    if (!diff) return null;
    const derivedText = deriveCoachText(diff);
    if (!derivedText) return null;
    const progress = normalizeProgressPct(diff.objective_progress?.progress);
    return {
      note: {
        text: derivedText,
        goal: "neutral" as const,
        progressPct: progress,
        updatedAt: Date.now(),
      },
      metrics: buildCoachMetrics(diff),
    };
  }, [coachProp, coachPulse.diff]);

  const planMetrics = React.useMemo(() => buildPlanMetrics(plan), [plan]);
  const additionalCoachMetrics = React.useMemo(
    () => coachNoteContent?.metrics ?? [],
    [coachNoteContent?.metrics],
  );
  const coachNote = React.useMemo(() => {
    if (coachNoteContent?.note) {
      return coachNoteContent.note;
    }
    return {
      text: "Awaiting guidance…",
      goal: "neutral" as const,
      progressPct: 0,
      updatedAt: Date.now(),
    };
  }, [coachNoteContent?.note]);

  const nextActionText = React.useMemo(() => coachNote.text, [coachNote.text]);
  const coachMetrics = React.useMemo(() => {
    const metrics = [...planMetrics, ...additionalCoachMetrics];
    const trimmed = nextActionText.trim();
    if (trimmed) {
      metrics.unshift({
        key: "next-action",
        label: "Next Action",
        value: truncateMetric(trimmed),
        ariaLabel: `Next action ${trimmed}`,
      });
    }
    return metrics;
  }, [planMetrics, additionalCoachMetrics, nextActionText]);

  React.useEffect(() => {
    setSymbolDraft(sanitizeSymbolToken(plan.symbol ?? ""));
  }, [plan.symbol]);

  const statusTokens = React.useMemo<StatusBuckets>(() => {
    const ws: StatusToken =
      planConnectionStatus === "connected"
        ? planHeartbeatAgeSeconds != null && planHeartbeatAgeSeconds > 60
          ? "disconnected"
          : "connected"
        : planConnectionStatus === "connecting"
          ? "connecting"
          : "disconnected";
    const price: StatusToken =
      !streamingEnabled || barsConnectionStatus === "error"
        ? "disconnected"
        : barsConnectionStatus === "connected"
          ? dataAgeSeconds != null && resolutionMs != null && dataAgeSeconds > Math.max(60, (resolutionMs / 1000) * 6)
            ? "disconnected"
            : "connected"
          : "connecting";
    return { ws, price };
  }, [planConnectionStatus, planHeartbeatAgeSeconds, streamingEnabled, barsConnectionStatus, dataAgeSeconds, resolutionMs]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const container = document.querySelector<HTMLElement>('[data-testid="chart-container"]');
    if (!container) return;
    const buttons = Array.from(container.querySelectorAll("button"));
    buttons.forEach((button) => {
      const text = button.textContent?.trim().toLowerCase();
      if (!text) return;
      if (text === "replay" || text === "stop replay") {
        button.dataset.testid = "replay-btn";
      }
      if (text === "follow live") {
        button.dataset.testid = "follow-live-btn";
      }
      if (text === "5m") {
        button.dataset.testid = "timeframe-btn-5m";
      }
    });
  }, [followLive, timeframe]);

  const handleSelectTimeframe = React.useCallback(
    (value: string) => {
      setTimeframe(value);
      requestPriceRefresh();
    },
    [requestPriceRefresh],
  );

  const STATUS_COLOR_CLASS: Record<StatusToken, string> = {
    connected: "bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.6)]",
    connecting: "bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.5)]",
    disconnected: "bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.55)]",
  };

  const STATUS_LABEL: Record<StatusToken, string> = {
    connected: "Live",
    connecting: "Connecting",
    disconnected: "Offline",
  };

  const statusItems: Array<{ label: string; status: StatusToken }> = [
    { label: "Stream", status: statusTokens.ws },
    { label: "Data", status: statusTokens.price },
  ];

  const headerContent = (
    <>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-lg font-semibold uppercase tracking-[0.35em] text-emerald-300">Fancy Trader</span>
          <SessionChip session={sessionData} />
          {planAsOfLabel ? (
            <span className="text-[11px] uppercase tracking-[0.22em] text-neutral-400">
              Last update {planAsOfLabel}
            </span>
          ) : null}
        </div>
        <ObjectiveProgress meta={objectiveMeta ?? undefined} />
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-3 text-[10px] uppercase tracking-[0.24em] text-neutral-400">
          {statusItems.map((item) => (
            <span key={item.label} className="flex items-center gap-2">
              <span
                className={clsx("inline-flex h-2.5 w-2.5 rounded-full", STATUS_COLOR_CLASS[item.status])}
                aria-hidden="true"
              />
              <span className="text-neutral-300">{item.label}</span>
              <span className="text-neutral-500">{STATUS_LABEL[item.status]}</span>
            </span>
          ))}
          {sessionBanner ? <span className="text-neutral-500">{sessionBanner}</span> : null}
        </div>
        <form className="flex w-full max-w-xs items-center gap-2 rounded-full border border-neutral-800/60 bg-neutral-950/60 px-3 py-2" onSubmit={handleSymbolSubmit}>
          <label className="sr-only" htmlFor="find-setup-input">
            Find setup
          </label>
          <input
            id="find-setup-input"
            type="text"
            inputMode="text"
            autoComplete="off"
            maxLength={6}
            placeholder="Find setup (SYM)"
            value={symbolDraft}
            onChange={handleSymbolInputChange}
            className="w-full bg-transparent text-[11px] font-semibold uppercase tracking-[0.28em] text-neutral-100 outline-none placeholder:text-neutral-500"
          />
          <button
            type="submit"
            disabled={symbolSubmitting || !symbolDraft}
            className={clsx(
              "rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
              symbolSubmitting || !symbolDraft
                ? "border border-neutral-800/60 bg-neutral-900/40 text-neutral-500"
                : "border border-emerald-500/60 bg-emerald-500/15 text-emerald-100 hover:border-emerald-400",
            )}
          >
            {symbolSubmitting ? "…" : "Find"}
          </button>
        </form>
      </div>
    </>
  );

  const coachContent = (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-[0.24em] text-neutral-400">Coach Guidance</span>
        <button
          type="button"
          onClick={() => setCoachCollapsed((prev) => !prev)}
          className="rounded-full border border-neutral-800/60 bg-neutral-900/70 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.22em] text-neutral-300 transition hover:border-emerald-400 hover:text-emerald-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
          aria-pressed={coachCollapsed}
        >
          {coachCollapsed ? "Expand" : "Collapse"}
        </button>
      </div>
      {coachCollapsed ? (
        <div className="rounded-xl border border-neutral-800/60 bg-neutral-950/50 px-3 py-2 text-[12px] leading-relaxed text-neutral-200">
          {nextActionText}
        </div>
      ) : (
        <>
          <CoachNote note={coachNote} metrics={coachMetrics} live />
          <ConfluenceStrength model={confluenceModel} />
        </>
      )}
    </div>
  );

  const chartNode = (
    <PlanChartPanel
      plan={plan}
      layers={layers}
      primaryLevels={primaryLevels}
      supportingVisible={supportVisible}
      onToggleSupporting={handleToggleSupporting}
      followLive={followLive}
      streamingEnabled={streamingEnabled}
      onSetFollowLive={setFollowLive}
      timeframe={timeframe}
      timeframeOptions={TIMEFRAME_OPTIONS}
      onSelectTimeframe={handleSelectTimeframe}
      onLastBarTimeChange={setLastBarTime}
      onReplayStateChange={(state) => {
        if (state === "playing") {
          setFollowLive(false);
        }
      }}
      devMode={devMode}
      theme={theme}
      priceRefreshToken={priceRefreshToken}
      highlightLevelId={null}
      hiddenLevelIds={[]}
      liveBars={streamingBars}
    />
  );

  const footerContent = (
    <>
      <span>Live Plan Console</span>
      <span className="text-neutral-500">{BUILD_SHA ? `Build ${BUILD_SHA.slice(0, 7)}` : "Build n/a"}</span>
    </>
  );

  React.useEffect(() => {
    if (!watchlistRailItems.length) return;
    // Preload navigation for watchlist items.
    watchlistRailItems.slice(0, 5).forEach((item) => {
      router.prefetch(`/plan/${encodeURIComponent(item.planId)}`);
    });
  }, [watchlistRailItems, router]);

  return (
    <PlanShell
      header={headerContent}
      leftRail={<WatchlistRail items={watchlistRailItems} />}
      coach={coachContent}
      chart={chartNode}
      footer={footerContent}
    />
  );
}

function extractBias(meta: Record<string, unknown> | undefined): WatchlistRailItem["bias"] {
  if (!meta) return null;
  const bias = meta.bias;
  if (bias === "long" || bias === "short") {
    return bias;
  }
  if (typeof bias === "string") {
    const token = bias.toLowerCase();
    if (token.includes("long")) return "long";
    if (token.includes("short")) return "short";
  }
  return null;
}

function normalizeProgressPct(value: number | null | undefined): number {
  if (!Number.isFinite(value ?? NaN)) return 0;
  const numeric = value as number;
  if (numeric > 1) return Math.max(0, Math.min(100, Math.round(numeric)));
  return Math.max(0, Math.min(100, Math.round(numeric * 100)));
}

function deriveCoachText(diff: NonNullable<ReturnType<typeof useStore>["coach"]["diff"]>): string | null {
  if (diff?.next_action && diff.next_action.trim()) {
    return diff.next_action.trim();
  }
  if (diff?.waiting_for) {
    return diff.waiting_for.replace(/[_-]/g, " ");
  }
  if (diff?.risk_cue) {
    return diff.risk_cue;
  }
  return null;
}

function buildCoachMetrics(
  diff: NonNullable<ReturnType<typeof useStore>["coach"]["diff"]>,
): Array<{ key: string; label: string; value: string; ariaLabel?: string }> {
  const metrics: Array<{ key: string; label: string; value: string; ariaLabel?: string }> = [];
  const objective = diff.objective_progress;
  if (objective?.entry_distance_pct != null && Number.isFinite(objective.entry_distance_pct)) {
    metrics.push({
      key: "entry-distance",
      label: "Entry Δ",
      value: `${Math.round(objective.entry_distance_pct * 10) / 10}%`,
    });
  }
  const confluence = diff.confluence_delta;
  if (confluence?.mtf != null && Number.isFinite(confluence.mtf)) {
    metrics.push({
      key: "mtf",
      label: "MTF",
      value: `${confluence.mtf >= 0 ? "+" : ""}${confluence.mtf}`,
    });
  }
  if (confluence?.vwap_side) {
    metrics.push({
      key: "vwap",
      label: "VWAP",
      value: confluence.vwap_side === "above" ? "Above" : "Below",
    });
  }
  return metrics;
}

function normalizeTimeframeFromPlan(plan: PlanSnapshot["plan"]): string {
  const raw = (plan.charts_params as Record<string, unknown> | undefined)?.interval ?? plan.chart_timeframe ?? "5";
  return normalizeTimeframe(raw);
}

function normalizeTimeframe(token: unknown): string {
  const value = typeof token === "string" ? token.trim() : "";
  if (!value) return "5";
  const lower = value.toLowerCase();
  if (lower.endsWith("m")) {
    return String(Number.parseInt(lower.replace("m", ""), 10) || 5);
  }
  if (lower.endsWith("h")) {
    const minutes = (Number.parseInt(lower.replace("h", ""), 10) || 1) * 60;
    return String(minutes);
  }
  if (lower === "d" || lower === "1d") return "1D";
  if (lower === "w" || lower === "1w") return "1W";
  return value.toUpperCase();
}

function timeframeToMs(resolution: string): number | null {
  const token = resolution.toUpperCase();
  if (token === "1D" || token === "D") return 24 * 60 * 60 * 1000;
  if (token === "1W" || token === "W") return 7 * 24 * 60 * 60 * 1000;
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

function sanitizeSymbolToken(value: string | null | undefined): string {
  if (!value) return "";
  return value.replace(/[^A-Za-z0-9]/g, "").toUpperCase().slice(0, 6);
}

function getNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function formatPrice(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function truncateMetric(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length <= 42) return trimmed;
  return `${trimmed.slice(0, 39)}…`;
}

function buildPlanMetrics(
  plan: PlanSnapshot["plan"],
): Array<{ key: string; label: string; value: string; ariaLabel?: string }> {
  const metrics: Array<{ key: string; label: string; value: string; ariaLabel?: string }> = [];

  const entry =
    getNumber((plan as Record<string, unknown>).entry) ??
    getNumber((plan.structured_plan as { entry?: { level?: unknown } } | null | undefined)?.entry?.level);
  if (entry != null) {
    metrics.push({
      key: "entry",
      label: "Entry",
      value: formatPrice(entry),
    });
  }

  const stop =
    getNumber((plan as Record<string, unknown>).stop) ??
    getNumber((plan.structured_plan as { stop?: unknown } | null | undefined)?.stop);
  if (stop != null) {
    metrics.push({
      key: "stop",
      label: "Stop",
      value: formatPrice(stop),
    });
  }

  const rawTargets = Array.isArray(plan.targets) ? plan.targets : [];
  const firstTarget = rawTargets.map((value) => getNumber(value)).find((value): value is number => value != null);
  if (firstTarget != null) {
    metrics.push({
      key: "tp",
      label: "TP",
      value: formatPrice(firstTarget),
    });
  }

  return metrics;
}
