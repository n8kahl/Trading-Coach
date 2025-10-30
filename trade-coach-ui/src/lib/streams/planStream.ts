"use client";

import { makeBackoff } from "@/lib/hooks/useBackoff";

export type SocketStatus = "connecting" | "connected" | "disconnected";

type MessageListener = (payload: unknown) => void;
type StatusListener = (status: SocketStatus) => void;
type HeartbeatListener = (timestamp: number) => void;

const RECONNECT_CAP_MS = 15_000;
const HEARTBEAT_INTERVAL_MS = 20_000;
const STALE_HEARTBEAT_MS = 45_000;

class PlanStream {
  private ws: WebSocket | null = null;
  private status: SocketStatus = "connecting";
  private readonly messageListeners = new Set<MessageListener>();
  private readonly statusListeners = new Set<StatusListener>();
  private readonly heartbeatListeners = new Set<HeartbeatListener>();
  private backoff = makeBackoff(6, 750);
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private teardownTimer: number | null = null;
  private lastHeartbeat = 0;

  constructor(private readonly planId: string, private readonly url: string) {}

  subscribe(listener: MessageListener): () => void {
    this.messageListeners.add(listener);
    this.connect();
    return () => {
      this.messageListeners.delete(listener);
      if (this.messageListeners.size === 0) {
        this.scheduleTeardown();
      }
    };
  }

  onStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.status);
    return () => {
      this.statusListeners.delete(listener);
    };
  }

  onHeartbeat(listener: HeartbeatListener): () => void {
    this.heartbeatListeners.add(listener);
    if (this.lastHeartbeat) {
      listener(this.lastHeartbeat);
    }
    return () => {
      this.heartbeatListeners.delete(listener);
    };
  }

  getStatus(): SocketStatus {
    return this.status;
  }

  getLastHeartbeat(): number {
    return this.lastHeartbeat;
  }

  private connect(): void {
    if (typeof window === "undefined") return;
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.teardownTimer != null) {
      window.clearTimeout(this.teardownTimer);
      this.teardownTimer = null;
    }
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    try {
      this.updateStatus("connecting");
      this.ws = new WebSocket(this.url);
    } catch (error) {
      console.warn("[plan-stream] create socket failed", error);
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.backoff = makeBackoff(6, 750);
      this.lastHeartbeat = Date.now();
      this.emitHeartbeat();
      this.updateStatus("connected");
      try {
        this.ws?.send(
          JSON.stringify({
            type: "subscribe",
            planId: this.planId,
          }),
        );
      } catch (error) {
        console.warn("[plan-stream] subscribe send failed", error);
      }
      this.startHeartbeat();
    };
    this.ws.onmessage = (event) => this.handleMessage(event);
    this.ws.onerror = () => this.handleDisconnect();
    this.ws.onclose = () => this.handleDisconnect();
  }

  private handleMessage(event: MessageEvent): void {
    this.lastHeartbeat = Date.now();
    this.emitHeartbeat();
    let payload: unknown;
    try {
      payload = JSON.parse(typeof event.data === "string" ? event.data : "{}");
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[plan-stream] parse event failed", error);
      }
      return;
    }
    if (payload && typeof payload === "object" && (payload as { type?: string }).type === "ping") {
      this.send({ type: "pong", planId: this.planId, ts: Date.now() });
      return;
    }
    this.messageListeners.forEach((listener) => {
      try {
        listener(payload);
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[plan-stream] listener failed", error);
        }
      }
    });
  }

  private handleDisconnect(): void {
    this.stopHeartbeat();
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
    const now = Date.now();
    if (now - this.lastHeartbeat > STALE_HEARTBEAT_MS) {
      this.lastHeartbeat = now;
      this.emitHeartbeat();
    }
    this.updateStatus("disconnected");
    if (this.messageListeners.size > 0) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (typeof window === "undefined") return;
    if (this.reconnectTimer != null) return;
    const delay = Math.min(this.backoff(), RECONNECT_CAP_MS);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (this.messageListeners.size > 0) {
        this.connect();
      }
    }, delay);
  }

  private scheduleTeardown(): void {
    if (typeof window === "undefined") return;
    if (this.teardownTimer != null) return;
    this.teardownTimer = window.setTimeout(() => {
      this.teardownTimer = null;
      if (this.messageListeners.size === 0) {
        this.stopHeartbeat();
        if (this.ws) {
          this.ws.close();
          this.ws = null;
        }
        this.updateStatus("disconnected");
      }
    }, 2_500);
  }

  private startHeartbeat(): void {
    if (typeof window === "undefined") return;
    this.stopHeartbeat();
    this.heartbeatTimer = window.setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      this.send({ type: "ping", planId: this.planId, ts: Date.now() });
    }, HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer != null) {
      window.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private send(payload: Record<string, unknown>): void {
    try {
      this.ws?.send(JSON.stringify(payload));
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[plan-stream] send failed", error);
      }
    }
  }

  private updateStatus(next: SocketStatus): void {
    if (this.status === next) return;
    this.status = next;
    this.statusListeners.forEach((listener) => {
      try {
        listener(next);
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[plan-stream] status listener failed", error);
        }
      }
    });
  }

  private emitHeartbeat(): void {
    const ts = this.lastHeartbeat || Date.now();
    this.heartbeatListeners.forEach((listener) => {
      try {
        listener(ts);
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[plan-stream] heartbeat listener failed", error);
        }
      }
    });
  }
}

const STREAMS = new Map<string, PlanStream>();

export function getPlanStream(planId: string, url: string): PlanStream {
  const key = `${planId}::${url}`;
  const existing = STREAMS.get(key);
  if (existing) return existing;
  const stream = new PlanStream(planId, url);
  STREAMS.set(key, stream);
  return stream;
}
