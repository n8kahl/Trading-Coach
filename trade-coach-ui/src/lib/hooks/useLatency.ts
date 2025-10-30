'use client';

import * as React from 'react';

/**
 * Calculates a simple UI latency from a server timestamp (ms).
 * If no timestamp, returns idle (no fabrication).
 */
export function useLatency(lastUpdateAt?: number) {
  const [now, setNow] = React.useState<number>(() => Date.now());
  React.useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  if (!lastUpdateAt) return { latencyMs: undefined, level: 'idle' as const };
  const latencyMs = Math.max(0, now - lastUpdateAt);
  const level = latencyMs < 250 ? 'good' : latencyMs < 1000 ? 'warn' : 'bad';
  return { latencyMs, level };
}
