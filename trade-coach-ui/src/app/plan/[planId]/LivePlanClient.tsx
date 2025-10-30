'use client';

import * as React from 'react';
import WebviewShell from '@/components/webview/WebviewShell';
import StatusStrip, { type StatusToken } from '@/components/webview/StatusStrip';
import PlanPanel from '@/components/webview/PlanPanel';
import PlanHeader from '@/components/PlanHeader';
import PlanChartPanel from '@/components/PlanChartPanel';
import { usePlanSocket } from '@/lib/hooks/usePlanSocket';
import { usePlanLayers } from '@/lib/hooks/usePlanLayers';
import { extractPrimaryLevels, extractSupportingLevels } from '@/lib/utils/layers';
import type { SupportingLevel } from '@/lib/chart';
import type { PlanDeltaEvent, PlanLayers, PlanSnapshot } from '@/lib/types';
import { WS_BASE_URL } from '@/lib/env';

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
  const devMode = process.env.NEXT_PUBLIC_DEVTOOLS === '1';

  const activePlanId = plan.plan_id || planId;

  React.useEffect(() => {
    const interval = window.setInterval(() => setNowTick(Date.now()), 1000);
    return () => window.clearInterval(interval);
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
        if (!event || typeof event !== 'object') return;
        const type = typeof event.t === 'string' ? event.t : null;
        if (!type) return;
        if (type === 'plan_delta') {
          const delta = event as unknown as PlanDeltaEvent;
          setPlan((prev) => mergePlanWithDelta(prev, delta));
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
          return;
        }
        if (type === 'plan_state') {
          return;
        }
      },
      [],
    ),
  );

  const primaryLevels = React.useMemo(() => extractPrimaryLevels(layers), [layers]);
  const supportingLevels = React.useMemo(() => extractSupportingLevels(layers), [layers]);

  const resolutionMs = React.useMemo(() => timeframeToMs(timeframe), [timeframe]);
  const dataAgeSeconds = React.useMemo(() => {
    if (!lastBarTime) return null;
    return Math.max(0, (nowTick - lastBarTime) / 1000);
  }, [lastBarTime, nowTick]);

  const priceStatus: StatusToken = React.useMemo(() => {
    if (!streamingEnabled) return 'disconnected';
    if (socketStatus === 'disconnected') return 'disconnected';
    if (!lastBarTime || dataAgeSeconds == null || resolutionMs == null) return 'connecting';
    const threshold = Math.max((resolutionMs / 1000) * 2, 30);
    return dataAgeSeconds <= threshold ? 'connected' : 'connecting';
  }, [streamingEnabled, socketStatus, lastBarTime, dataAgeSeconds, resolutionMs]);

  const wsStatus = React.useMemo<StatusToken>(() => {
    if (socketStatus === 'connected') return 'connected';
    if (socketStatus === 'connecting') return 'connecting';
    return 'disconnected';
  }, [socketStatus]);

  const sessionBanner = plan.session_state?.banner ?? null;
  const riskBanner =
    Array.isArray(plan.warnings) && plan.warnings.length ? String(plan.warnings[0]) : plan.session_state?.message ?? null;

  const planTargets = React.useMemo(() => {
    const rawTargets = Array.isArray(plan.targets) ? plan.targets : [];
    return rawTargets.filter((value): value is number => typeof value === 'number');
  }, [plan.targets]);

  const handleSetFollowLive = React.useCallback((value: boolean) => {
    setFollowLive(value);
  }, []);

  const handleToggleStreaming = React.useCallback(() => {
    setStreamingEnabled((prev) => !prev);
  }, []);

  const handleToggleSupporting = React.useCallback(() => {
    setSupportVisible((prev) => !prev);
  }, []);

  const handleSelectTimeframe = React.useCallback((value: string) => {
    setTimeframe(value);
  }, []);

  const handleSelectLevel = React.useCallback((level: SupportingLevel | null) => {
    setHighlightedLevel(level);
  }, []);

  React.useEffect(() => {
    setTimeframe((current) => current || normalizeTimeframeFromPlan(plan));
  }, [plan]);

  const statusStrip = (
    <StatusStrip
      theme={theme}
      wsStatus={wsStatus}
      priceStatus={priceStatus}
      dataAgeSeconds={dataAgeSeconds}
      riskBanner={riskBanner}
      sessionBanner={sessionBanner}
      onToggleTheme={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}
    />
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
      onLastBarTimeChange={setLastBarTime}
      onReplayStateChange={(state) => {
        if (state === 'playing') {
          setFollowLive(false);
        }
      }}
      theme={theme}
      devMode={!!devMode}
    />
  );

  const planUiLink = React.useMemo(() => extractUiPlanLink(snapshot), [snapshot]);

  const debugPanel = devMode ? (
    <div className="fixed bottom-4 left-4 z-50 rounded-lg border border-neutral-800 bg-neutral-950/85 px-4 py-3 text-xs text-neutral-200 shadow-lg">
      <div className="font-semibold uppercase tracking-[0.2em] text-neutral-400">Dev Stats</div>
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
  return typeof link === 'string' ? link : undefined;
}
