import { WS_BASE } from './hosts';

export function planSocket(planId: string): WebSocket {
  const url = `${WS_BASE}/ws/plans/${encodeURIComponent(planId)}`;
  return new WebSocket(url);
}

