'use client';

import * as React from 'react';
import ResponsiveShell from '@/components/ResponsiveShell';
import StatusBanner from '@/components/StatusBanner';
import PlanSummaryCard from '@/components/PlanSummaryCard';
import ChartContainer from '@/components/webview/ChartContainer';
import ActionDock from '@/components/ActionDock';
import type { PlanDeltaEvent, PlanSnapshot } from '@/lib/types';
import { WS_BASE_URL } from '@/lib/env';
import { usePlanSocket } from '@/lib/hooks/usePlanSocket';
import { useChartUrl } from '@/lib/hooks/useChartUrl';
import { useLatency } from '@/lib/hooks/useLatency';

type LivePlanClientProps = {
  initialSnapshot: PlanSnapshot;
  planId: string;
  symbol?: string;
};

export default function LivePlanClient({ initialSnapshot, planId, symbol }: LivePlanClientProps) {
  const [plan, setPlan] = React.useState(initialSnapshot.plan);
  const [followLive, setFollowLive] = React.useState(true);
  const [streamingEnabled, setStreamingEnabled] = React.useState(true);
  const [supportVisible, setSupportVisible] = React.useState(() => extractSupportVisible(initialSnapshot.plan));
  const [lastUpdateAt, setLastUpdateAt] = React.useState<number | undefined>(() =>
    parseTimestamp(initialSnapshot.plan.session_state?.as_of),
  );

  const chartWrapperRef = React.useRef<HTMLDivElement | null>(null);
  const streamingEnabledRef = React.useRef(streamingEnabled);

  React.useEffect(() => {
    streamingEnabledRef.current = streamingEnabled;
  }, [streamingEnabled]);

  React.useEffect(() => {
    setPlan(initialSnapshot.plan);
    setLastUpdateAt(parseTimestamp(initialSnapshot.plan.session_state?.as_of));
  }, [initialSnapshot]);

  const upperSymbol = React.useMemo(() => (symbol ?? plan.symbol ?? '').toUpperCase(), [symbol, plan.symbol]);
  // Cast to satisfy PlanLike (allows undefined but not null in nested fields)
  const chartUrl = useChartUrl(plan as any);
  const latency = useLatency(streamingEnabled ? lastUpdateAt : undefined);

  const handlePlanDelta = React.useCallback((payload: unknown) => {
    if (!streamingEnabledRef.current) return;
    if (!payload || typeof payload !== 'object') return;
    const event = payload as PlanDeltaEvent;
    if (event.t !== 'plan_delta') return;
    setPlan((prev) => mergePlanWithDelta(prev, event));
    const parsed = parseTimestamp(event.changes.timestamp);
    setLastUpdateAt(parsed ?? Date.now());
  }, []);

  const planSocketUrl = React.useMemo(() => `${WS_BASE_URL}/ws/plans/${encodeURIComponent(planId)}`, [planId]);
  const socketStatus = usePlanSocket(planSocketUrl, handlePlanDelta);

  const session = normalizeSession(plan.session_state?.status);
  const asOfText = React.useMemo(() => formatAsOf(plan.session_state?.as_of), [plan.session_state?.as_of]);
  const summaryPlan = React.useMemo(() => buildSummaryPlan(plan), [plan]);
  const streamingInfo = React.useMemo(
    () => ({
      connected: streamingEnabled && socketStatus === 'connected',
      latencyMs: streamingEnabled ? latency.latencyMs : undefined,
    }),
    [streamingEnabled, socketStatus, latency.latencyMs],
  );

  const syncChartSupport = React.useCallback(
    (visible: boolean) => {
      if (typeof document === 'undefined') return false;
      const wrapper = chartWrapperRef.current;
      if (!wrapper) return false;
      const toggleBtn = wrapper.querySelector('button[aria-pressed]') as HTMLButtonElement | null;
      if (!toggleBtn) return false;
      const current = toggleBtn.getAttribute('aria-pressed') === 'true';
      if (current === visible) return true;
      toggleBtn.click();
      return true;
    },
    [],
  );

  const handleToggleSupport = React.useCallback(() => {
    const desired = !supportVisible;
    const synced = syncChartSupport(desired);
    if (!synced) {
      setSupportVisible(desired);
    }
  }, [supportVisible, syncChartSupport]);

  const handleSupportToggled = React.useCallback((visible: boolean) => {
    setSupportVisible(visible);
  }, []);

  const handleToggleStreaming = React.useCallback(() => {
    setStreamingEnabled((prev) => !prev);
  }, []);

  const handleToggleFollowLive = React.useCallback(() => {
    setFollowLive((prev) => !prev);
  }, []);

  const pageTitle = React.useMemo(() => {
    if (upperSymbol) return `${upperSymbol} Plan`;
    return 'Plan';
  }, [upperSymbol]);

  const nextStep = (plan as Record<string, unknown>).next_step as string | undefined;
  const notes = plan.notes ?? null;
  const warnings = Array.isArray(plan.warnings) ? plan.warnings : [];

  return (
    <ResponsiveShell title={pageTitle}>
      <StatusBanner
        session={session}
        asOfText={asOfText}
        message={plan.session_state?.banner ?? undefined}
        streaming={streamingInfo}
      />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.2fr)_minmax(0,1fr)] xl:gap-6">
        <div className="flex flex-col gap-4 xl:gap-6">
          <div ref={chartWrapperRef}>
            <ChartContainer
              chartUrl={chartUrl ?? '/tv'}
              overlays={plan.charts?.params ?? plan.charts_params ?? undefined}
              onSupportToggled={handleSupportToggled}
            />
          </div>
          <PlanContextPanel notes={notes} warnings={warnings} nextStep={nextStep} followLive={followLive} />
        </div>
        <div className="flex flex-col gap-4 xl:gap-6">
          <PlanSummaryCard plan={summaryPlan} />
          {plan.expected_duration?.label ? (
            <section className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm shadow-[var(--elev-1)]">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">Expected Duration</h3>
              <p className="mt-2 font-medium">{plan.expected_duration.label}</p>
              {plan.expected_duration.basis ? (
                <p className="mt-1 text-[var(--muted)]">
                  Basis: {plan.expected_duration.basis.join(', ')}
                </p>
              ) : null}
            </section>
          ) : null}
        </div>
      </div>
      <ActionDock
        streaming={streamingEnabled}
        followLive={followLive}
        supportVisible={supportVisible}
        onToggleStreaming={handleToggleStreaming}
        onToggleFollowLive={handleToggleFollowLive}
        onToggleSupport={handleToggleSupport}
      />
    </ResponsiveShell>
  );
}

function PlanContextPanel({
  notes,
  warnings,
  nextStep,
  followLive,
}: {
  notes: string | null;
  warnings: string[];
  nextStep?: string;
  followLive: boolean;
}) {
  const hasContent = Boolean(notes) || warnings.length > 0 || Boolean(nextStep);
  if (!hasContent) {
    return (
      <section className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm text-[var(--muted)] shadow-[var(--elev-1)]">
        No additional plan context provided.
      </section>
    );
  }
  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm shadow-[var(--elev-1)]">
      <div className="space-y-4">
        {nextStep ? (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">Next Step</h3>
            <p className="mt-1 text-sm">{nextStep}</p>
          </div>
        ) : null}
        {notes ? (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">Coach Notes</h3>
            <p className="mt-1 leading-relaxed">{notes}</p>
          </div>
        ) : null}
        {warnings.length ? (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">Warnings</h3>
            <ul className="mt-1 space-y-1">
              {warnings.map((warning, index) => (
                <li
                  key={`${index}-${warning}`}
                  className="rounded-lg bg-[var(--danger)]/10 px-3 py-2 text-sm text-[var(--danger)]"
                >
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <div className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--muted)]">
          <span>Follow live updates</span>
          <span className="font-semibold text-[var(--text)]">{followLive ? 'Enabled' : 'Paused'}</span>
        </div>
      </div>
    </section>
  );
}

function normalizeSession(status?: string | null): 'open' | 'closed' | 'pre' {
  const value = (status ?? '').toLowerCase();
  if (value.includes('open')) return 'open';
  if (value.includes('pre')) return 'pre';
  return 'closed';
}

function formatAsOf(asOf?: string | null): string {
  if (!asOf) return 'As of unavailable';
  const date = new Date(asOf);
  if (Number.isNaN(date.getTime())) return asOf;
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function parseTimestamp(value?: string | number | null): number | undefined {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined;
  }
  if (!value) return undefined;
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) return undefined;
  return ms;
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

function buildSummaryPlan(plan: PlanSnapshot['plan']): Record<string, unknown> {
  const detailsSource = (plan.details ?? {}) as Record<string, unknown>;
  const details: Record<string, unknown> = { ...detailsSource };

  if (plan.entry != null && details.entry == null) {
    details.entry = plan.entry;
  }
  if (plan.stop != null && details.stop == null) {
    details.stop = plan.stop;
  }
  if ((plan as Record<string, unknown>).last_price != null) {
    details.last = (plan as Record<string, unknown>).last_price;
  }
  if (plan.expected_duration?.label && details.horizon == null) {
    details.horizon = plan.expected_duration.label;
  }

  const summary: Record<string, unknown> = { ...plan, details };

  const riskReward = (plan as Record<string, unknown>).riskReward ?? plan.rr_to_t1 ?? null;
  if (riskReward != null) {
    summary.riskReward = riskReward;
  }

  const structure =
    (plan as Record<string, unknown>).structure ??
    (plan.structured_plan ? (plan.structured_plan.invalid ? 'INVALID' : 'INTACT') : null);
  if (structure) {
    summary.structure = structure;
  }

  if (Array.isArray(plan.targets)) {
    summary.targets = plan.targets.map((target: unknown, index: number) => {
      if (target && typeof target === 'object') return target;
      if (typeof target === 'number') {
        return { label: `TP${index + 1}`, price: target };
      }
      return target;
    });
  }

  return summary;
}
