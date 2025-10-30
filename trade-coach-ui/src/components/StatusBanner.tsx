'use client';

import * as React from 'react';
import { ClockIcon } from '@heroicons/react/24/outline';

type Streaming = { connected: boolean; latencyMs?: number };

export function StatusBanner({
  session,
  asOfText,
  message,
  streaming,
}: {
  session: 'open' | 'closed' | 'pre';
  asOfText: string;
  message?: string;
  streaming?: Streaming;
}) {
  const bg =
    session === 'open'
      ? 'bg-green-500/10'
      : session === 'pre'
      ? 'bg-yellow-500/10'
      : 'bg-slate-500/10';
  const dot =
    session === 'open'
      ? 'bg-[var(--accent)]'
      : session === 'pre'
      ? 'bg-[var(--warn)]'
      : 'bg-[var(--muted)]';

  const latency = streaming?.latencyMs ?? undefined;
  const level =
    latency == null ? 'idle' : latency < 250 ? 'good' : latency < 1000 ? 'warn' : 'bad';
  const latencyColor =
    level === 'good'
      ? 'text-[var(--accent)]'
      : level === 'warn'
      ? 'text-[var(--warn)]'
      : 'text-[var(--danger)]';

  return (
    <div
      role="status"
      aria-live="polite"
      className={`mb-3 rounded-lg border border-[var(--border)] ${bg} px-3 py-2 flex items-center justify-between`}
    >
      <div className="flex items-center gap-2">
        <span className={`h-2.5 w-2.5 rounded-full ${dot}`} aria-hidden="true" />
        <span className="text-sm">
          {session === 'open' ? 'Market Open' : session === 'pre' ? 'Pre-Market' : 'Market Closed'}
          {' • '}
          <span className="inline-flex items-center gap-1 text-[var(--muted)]">
            <ClockIcon className="h-4 w-4" aria-hidden />
            {asOfText}
          </span>
          {message ? (
            <>
              {' '}
              • <span className="text-[var(--muted)]">{message}</span>
            </>
          ) : null}
        </span>
      </div>
      <div className="text-sm">
        {streaming?.connected ? (
          <span className={latencyColor}>Live{latency != null ? ` • ${latency}ms` : ''}</span>
        ) : (
          <span className="text-[var(--muted)]">Live: idle</span>
        )}
      </div>
    </div>
  );
}

export default StatusBanner;
