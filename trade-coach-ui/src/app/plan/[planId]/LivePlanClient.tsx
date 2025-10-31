"use client";

import clsx from "clsx";
import * as React from "react";
import WebviewShell from "@/components/webview/WebviewShell";
import PlanPanel from "@/components/webview/PlanPanel";
import PlanHeader from "@/components/PlanHeader";
import PlanChartPanel from "@/components/PlanChartPanel";
import { usePlanSocket } from "@/lib/hooks/usePlanSocket";
import { usePlanLayers } from "@/lib/hooks/usePlanLayers";
import { extractPrimaryLevels, extractSupportingLevels } from "@/lib/utils/layers";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanDeltaEvent, PlanLayers, PlanSnapshot } from "@/lib/types";
import { PUBLIC_UI_BASE_URL, WS_BASE_URL, BUILD_SHA } from "@/lib/env";

const TIMEFRAME_OPTIONS = [
  { value: '1', label: '1m' },
  { value: '3', label: '3m' },
  { value: '5', label: '5m' },
  { value: '15', label: '15m' },
  { value: '60', label: '1h' },
  { value: '240', label: '4h' },
  { value: '1D', label: '1D' },
];

type LivePlanClientProps = {
  initialSnapshot: PlanSnapshot;
  planId: string;
};

type StatusTone = "green" | "yellow" | "red";

export default function LivePlanClient({ initialSnapshot, planId }: LivePlanClientProps) {
  const [snapshot, setSnapshot] = React.useState(initialSnapshot);
  const [plan, setPlan] = React.useState(initialSnapshot.plan);
  const [followLive, setFollowLive] = React.useState(true);
  const [streamingEnabled, setStreamingEnabled] = React.useState(true);
  const [supportVisible, setSupportVisible] = React.useState(() => extractSupportVisible(initialSnapshot.plan));
  const [highlightedLevel, setHighlightedLevel] = React.useState<SupportingLevel | null>(null);
  const [theme, setTheme] = React.useState<'dark' | 'light'>('dark');
  const [layerSeed, setLayerSeed] = React.useState<PlanLayers | null>(() => extractInitialLayers(initialSnapshot));
  const [timeframe, setTimeframe] = React.useState(() => normalizeTimeframeFromPlan(initialSnapshot.plan));
  const [nowTick, setNowTick] = React.useState(Date.now());
  const [lastBarTime, setLastBarTime] = React.useState<number | null>(null);
  const [devMode, setDevMode] = React.useState(() => process.env.NEXT_PUBLIC_DEVTOOLS === "1");
  const [priceRefreshToken, setPriceRefreshToken] = React.useState(0);
  const [lastPlanHeartbeat, setLastPlanHeartbeat] = React.useState<number | null>(() => Date.now());
  const [noteEpoch, setNoteEpoch] = React.useState(0);
  const [noteAnimating, setNoteAnimating] = React.useState(false);
  const lastPriceRefreshRef = React.useRef(0);
  const lastDeltaAtRef = React.useRef(Date.now());
  const prefersReducedMotion = usePrefersReducedMotion();

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
          setPlan((prev) => mergePlanWithDelta(prev, delta));
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
      [requestPriceRefresh, markPlanHeartbeat],
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

  const planStatusColor = React.useMemo<StatusTone>(() => {
    if (socketStatus === "disconnected") return "red";
    if (planHeartbeatAgeSeconds == null) return "yellow";
    if (planHeartbeatAgeSeconds <= 15) return "green";
    if (planHeartbeatAgeSeconds <= 60) return "yellow";
    return "red";
  }, [socketStatus, planHeartbeatAgeSeconds]);

  const dataStatusColor = React.useMemo<StatusTone>(() => {
    if (!streamingEnabled) return "red";
    if (dataAgeSeconds == null || resolutionSeconds == null) return "yellow";
    if (dataAgeSeconds <= resolutionSeconds * 2) return "green";
    if (dataAgeSeconds <= resolutionSeconds * 6) return "yellow";
    return "red";
  }, [streamingEnabled, dataAgeSeconds, resolutionSeconds]);

  const sessionBanner = plan.session_state?.banner ?? null;
  const riskBanner =
    Array.isArray(plan.warnings) && plan.warnings.length ? String(plan.warnings[0]) : plan.session_state?.message ?? null;

  const planTargets = React.useMemo(() => {
    const rawTargets = Array.isArray(plan.targets) ? plan.targets : [];
    return rawTargets.filter((value): value is number => typeof value === "number");
  }, [plan.targets]);

  const coachNote = React.useMemo(() => {
    const fromPlan = typeof plan.notes === "string" ? plan.notes.trim() : "";
    const sessionState = plan.session_state as Record<string, unknown> | null | undefined;
    const sessionNote =
      sessionState && typeof sessionState["coach_note"] === "string"
        ? (sessionState["coach_note"] as string).trim()
        : "";
    const fallbackMessage =
      sessionState && typeof sessionState["message"] === "string" ? (sessionState["message"] as string).trim() : "";
    return fromPlan || sessionNote || fallbackMessage || "";
  }, [plan]);

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

  const lastPrice = React.useMemo(() => {
    const details = (plan.details ?? {}) as Record<string, unknown>;
    return (
      getNumber((plan as Record<string, unknown>).last_price) ??
      getNumber(details.last) ??
      getNumber((plan as Record<string, unknown>).mark) ??
      null
    );
  }, [plan]);

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

  React.useEffect(() => {
    if (typeof window === "undefined") return undefined;
    const interval = window.setInterval(() => {
      if (!streamingEnabledRef.current) return;
      const now = Date.now();
      if (now - lastDeltaAtRef.current >= 5000) {
        requestPriceRefresh();
      }
    }, 5000);
    return () => {
      window.clearInterval(interval);
    };
  }, [requestPriceRefresh]);

  React.useEffect(() => {
    setNoteEpoch((prev) => prev + 1);
  }, [coachNote]);

  React.useEffect(() => {
    if (prefersReducedMotion) {
      setNoteAnimating(false);
      return;
    }
    setNoteAnimating(true);
    const timer = window.setTimeout(() => {
      setNoteAnimating(false);
    }, 450);
    return () => {
      window.clearTimeout(timer);
    };
  }, [noteEpoch, prefersReducedMotion]);

  const planUiLink = React.useMemo(() => extractUiPlanLink(snapshot), [snapshot]);

  const statusStrip = (
    <div className="flex flex-col gap-3 px-4 py-4 sm:px-6 lg:px-8" role="region" aria-label="Fancy Trader status">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.25em] text-neutral-400">
          <span className="text-sm font-semibold tracking-[0.35em] text-emerald-300">Fancy Trader</span>
          <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[0.68rem] text-emerald-200">
            {plan.symbol?.toUpperCase() ?? "—"}
          </span>
          <span className="rounded-full border border-neutral-700/50 px-2 py-0.5 text-[0.68rem] text-neutral-300">
            {timeframeLabel}
          </span>
          {sessionBanner ? (
            <span className="rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-[0.68rem] text-sky-200">
              {sessionBanner.toUpperCase()}
            </span>
          ) : null}
          {riskBanner ? (
            <span className="rounded-full border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-[0.68rem] text-amber-200">
              {riskBanner}
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[0.68rem] uppercase tracking-[0.25em] text-neutral-300">
          <span className="flex items-center gap-2">
            Plan
            <StatusDot color={planStatusColor} label={`Plan status ${planStatusColor}`} />
          </span>
          <span className="flex items-center gap-2">
            Data
            <StatusDot color={dataStatusColor} label={`Data status ${dataStatusColor}`} />
          </span>
          <span className="text-neutral-200">
            Last:&nbsp;
            <span className="font-semibold text-white">{lastPriceLabel}</span>
          </span>
        </div>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        {confluenceTokens.length ? (
          <div className="flex items-center gap-2 text-[0.62rem] uppercase tracking-[0.35em] text-neutral-400">
            <span className="text-neutral-500">Confluence</span>
            <div className="flex items-center gap-2">
              {confluenceTokens.map((token, index) => (
                <span key={`${token}-${index}`} className="flex items-center gap-1">
                  <span className="block h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden />
                  <span className="text-neutral-300">{token}</span>
                </span>
              ))}
            </div>
          </div>
        ) : null}
        <div
          className={clsx(
            "flex min-h-[2.5rem] max-w-full items-center gap-2 overflow-hidden rounded-2xl border px-3 py-2 text-sm leading-tight text-neutral-100 shadow-sm sm:max-w-[380px]",
            streamingEnabled ? "border-emerald-500/40 bg-emerald-500/10" : "border-neutral-700/60 bg-neutral-900/70",
          )}
        >
          <span className="text-[0.62rem] uppercase tracking-[0.3em] text-emerald-200">Coach note</span>
          <div className="relative flex-1 overflow-hidden">
            <div
              key={noteEpoch}
              className={clsx(
                "truncate text-neutral-200",
                prefersReducedMotion
                  ? ""
                  : noteAnimating
                    ? "translate-y-0 opacity-100 transition-transform duration-300 ease-out"
                    : "translate-y-0 opacity-100",
              )}
            >
              {coachNote || "Coach note pending"}
            </div>
          </div>
        </div>
      </div>
      <div className="flex justify-end pt-1">
        <button
          type="button"
          onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
          className={clsx(
            "rounded-full px-3 py-1 text-xs uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
            theme === "light"
              ? "border border-slate-300 bg-white text-slate-700 hover:border-emerald-400 hover:text-emerald-600"
              : "border border-neutral-700 bg-neutral-900 text-neutral-200 hover:border-emerald-400 hover:text-emerald-200",
          )}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
        >
          {theme === "light" ? "Dark Mode" : "Light Mode"}
        </button>
      </div>
    </div>
  );

  const planPanel = (
    <div className="flex h-full flex-col gap-4">
      <PlanHeader planId={plan.plan_id} uiUrl={planUiLink} theme={theme} />
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
          setFollowLive(false);
        }
      }}
      theme={theme}
      devMode={!!devMode}
      priceRefreshToken={priceRefreshToken}
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

function extractUiPlanLink(snapshot: PlanSnapshot): string | undefined {
  const uiBlock = (snapshot as unknown as { ui?: Record<string, unknown> }).ui;
  const link = uiBlock?.plan_link;
  if (typeof link === 'string' && link.trim()) {
    return link;
  }
  const planId = snapshot.plan?.plan_id;
  if (!planId) return undefined;
  return `${PUBLIC_UI_BASE_URL}/plan/${encodeURIComponent(planId)}`;
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

function usePrefersReducedMotion(): boolean {
  const [prefers, setPrefers] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefers(media.matches);
    update();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", update);
      return () => {
        media.removeEventListener("change", update);
      };
    }
    media.addListener(update);
    return () => {
      media.removeListener(update);
    };
  }, []);

  return prefers;
}
