"use client";

import { API_KEY_HEADER, WS_BASE_URL } from "@/lib/env";

type ConnectionKind = "plan" | "bars" | "coach";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

export type BarsEvt = {
  t: "bars";
  symbol: string;
  bars: Array<{ t: number; o: number; h: number; l: number; c: number; v: number }>;
};

export type PlanDelta = {
  t: "plan_delta";
  plan_id: string;
  version: number;
  fields: Record<string, unknown>;
};

export type CoachPulse = {
  t: "coach_pulse";
  plan_id: string;
  ts: string;
  diff: {
    next_action?: string;
    waiting_for?: string;
    risk_cue?: string;
    confluence_delta?: { mtf?: number; vwap_side?: "above" | "below" };
    objective_progress?: { progress?: number; entry_distance_pct?: number };
  };
  session?: { status: "open" | "closed"; as_of: string; tz?: string; next_open?: string | null };
};

export type Event = BarsEvt | PlanDelta | CoachPulse;

type EventListener = (event: Event) => void;
type StateListener = (state: ConnectionState) => void;

const MAX_CHANNEL_BUFFER = 5;
const FRAME_EVENT_LIMIT = 12;
const SOCKET_IDLE_CLOSE_MS = 90_000;
const BACKOFF_MAX_MS = 30_000;

function nowIso(): string {
  return new Date().toISOString();
}

function makeBackoff(maxMs: number): { next(): number; reset(): void } {
  let attempt = 0;
  return {
    next() {
      const base = Math.min(maxMs, Math.pow(2, attempt) * 750);
      attempt = Math.min(attempt + 1, 10);
      const jitter = Math.random() * 250;
      return Math.round(base + jitter);
    },
    reset() {
      attempt = 0;
    },
  };
}

const SUPPORTS_WS = typeof window !== "undefined" && typeof window.WebSocket !== "undefined";

class Channel {
  readonly key: string;
  private socket: WebSocket | null = null;
  private state: ConnectionState = "idle";
  private reconnectTimer: number | null = null;
  private idleTimer: number | null = null;
  private readonly backoff = makeBackoff(BACKOFF_MAX_MS);
  private refCount = 0;
  private readonly pending: Event[] = [];
  private lastActivity = Date.now();

  constructor(
    private readonly mux: WsMultiplexer,
    readonly kind: ConnectionKind,
    readonly target: string,
  ) {
    this.key = `${kind}:${target}`;
  }

  addRef(): void {
    this.refCount += 1;
    if (this.refCount === 1) {
      this.ensureConnected();
    }
  }

  release(): void {
    if (this.refCount === 0) return;
    this.refCount -= 1;
    if (this.refCount === 0) {
      this.scheduleTeardown();
    }
  }

  queue(event: Event): void {
    if (this.pending.length >= MAX_CHANNEL_BUFFER) {
      this.pending.length = 0;
    }
    this.pending.push(event);
    this.mux.enqueueChannel(this);
  }

  shiftPending(): Event | undefined {
    return this.pending.shift();
  }

  hasPending(): boolean {
    return this.pending.length > 0;
  }

  setState(next: ConnectionState): void {
    if (this.state === next) return;
    this.state = next;
    this.mux.notifyState(this.kind, this.target, next);
  }

  private ensureConnected(): void {
    if (!SUPPORTS_WS) {
      this.setState("error");
      return;
    }
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.connect();
  }

  private connect(): void {
    const url = this.mux.buildUrl(this.kind, this.target);
    if (!url) {
      this.setState("error");
      return;
    }
    try {
      this.setState("connecting");
      const protocols = this.kind === "coach" ? this.mux.getProtocols() : undefined;
      const socket = protocols && protocols.length ? new WebSocket(url, protocols) : new WebSocket(url);
      socket.onopen = () => this.handleOpen();
      socket.onmessage = (event) => this.handleMessage(event);
      socket.onerror = () => this.handleError();
      socket.onclose = () => this.handleClose();
      this.socket = socket;
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[wsMux] socket connect failed", { kind: this.kind, target: this.target, error });
      }
      this.scheduleReconnect();
      this.setState("error");
    }
  }

  private handleOpen(): void {
    this.setState("connected");
    this.backoff.reset();
    this.markActivity();
    if (this.kind === "plan") {
      this.safeSend({ type: "subscribe", planId: this.target });
    }
    if (this.kind === "coach") {
      // Immediately request a ping so heartbeat loop can stabilise.
      this.safeSend({ t: "pong", ts: nowIso() });
    }
    this.armIdleTimer();
  }

  private handleMessage(event: MessageEvent<unknown>): void {
    this.markActivity();
    this.armIdleTimer();
    const data = event.data;
    if (typeof data !== "string") {
      if (data instanceof Blob) {
        data
          .text()
          .then((text) => this.processPayload(text))
          .catch((error) => {
            if (process.env.NODE_ENV !== "production") {
              console.warn("[wsMux] blob parse failed", error);
            }
          });
      }
      return;
    }
    this.processPayload(data);
  }

  private processPayload(raw: string): void {
    let payload: unknown;
    try {
      payload = JSON.parse(raw);
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[wsMux] json parse failed", error);
      }
      return;
    }
    if (!payload || typeof payload !== "object") return;

    const token = (payload as { t?: string; type?: string }).t || (payload as { type?: string }).type || "";
    const lower = typeof token === "string" ? token.toLowerCase() : "";
    if (lower === "ping") {
      this.safeSend({ t: "pong", ts: nowIso() });
      return;
    }

    if (this.kind === "plan") {
      const evt = normalizePlanEvent(payload);
      if (evt) {
        this.queue(evt);
      }
      return;
    }
    if (this.kind === "bars") {
      const evt = normalizeBarsEvent(payload);
      if (evt) {
        this.queue(evt);
      }
      return;
    }
    if (this.kind === "coach") {
      const evt = normalizeCoachPulse(payload);
      if (evt) {
        this.queue(evt);
      }
    }
  }

  private handleError(): void {
    this.setState("error");
  }

  private handleClose(): void {
    this.clearSocket();
    if (this.refCount > 0) {
      this.scheduleReconnect();
    } else {
      this.setState("idle");
    }
  }

  private scheduleReconnect(): void {
    if (typeof window === "undefined") return;
    if (this.reconnectTimer != null) return;
    const delay = this.backoff.next();
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (this.refCount > 0) {
        this.connect();
      } else {
        this.setState("idle");
      }
    }, delay);
  }

  private scheduleTeardown(): void {
    this.clearIdleTimer();
    if (this.socket) {
      try {
        this.socket.close();
      } catch {
        // ignore
      }
      this.clearSocket();
    }
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.setState("idle");
  }

  private clearSocket(): void {
    if (this.socket) {
      this.socket.onopen = null;
      this.socket.onmessage = null;
      this.socket.onerror = null;
      this.socket.onclose = null;
    }
    this.socket = null;
  }

  private safeSend(payload: Record<string, unknown>): void {
    try {
      this.socket?.send(JSON.stringify(payload));
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[wsMux] send failed", error);
      }
    }
  }

  private markActivity(): void {
    this.lastActivity = Date.now();
  }

  private armIdleTimer(): void {
    if (typeof window === "undefined") return;
    this.clearIdleTimer();
    this.idleTimer = window.setTimeout(() => {
      const elapsed = Date.now() - this.lastActivity;
      if (elapsed >= SOCKET_IDLE_CLOSE_MS) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[wsMux] idle socket closing", { kind: this.kind, target: this.target, elapsed });
        }
        try {
          this.socket?.close(4000, "idle");
        } catch {
          // ignore
        }
      } else {
        this.armIdleTimer();
      }
    }, SOCKET_IDLE_CLOSE_MS);
  }

  private clearIdleTimer(): void {
    if (this.idleTimer != null) {
      window.clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function normalizeBarsEvent(payload: unknown): BarsEvt | null {
  if (!payload || typeof payload !== "object") return null;
  const envelope = payload as { symbol?: unknown; event?: unknown };
  const symbol = typeof envelope.symbol === "string" ? envelope.symbol : "";
  const event = (envelope.event || payload) as Record<string, unknown>;
  const typeToken = typeof event.t === "string" ? event.t.toLowerCase() : "";
  if (!symbol) return null;

  if (typeToken === "bars" && Array.isArray(event.bars)) {
    const bars = event.bars
      .map((entry) => {
        if (!entry || typeof entry !== "object") return null;
        const record = entry as Record<string, unknown>;
        const tVal = toNumber(record.t);
        const oVal = toNumber(record.o);
        const hVal = toNumber(record.h);
        const lVal = toNumber(record.l);
        const cVal = toNumber(record.c);
        const vVal = toNumber(record.v);
        if (tVal == null || oVal == null || hVal == null || lVal == null || cVal == null || vVal == null) {
          return null;
        }
        return { t: tVal, o: oVal, h: hVal, l: lVal, c: cVal, v: vVal };
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
    if (!bars.length) return null;
    return { t: "bars", symbol, bars };
  }

  if (typeToken === "bar" || typeToken === "tick") {
    const tVal = toNumber(event.ts ?? event.t ?? event.time ?? Date.now());
    const priceCandidate = toNumber(event.price ?? event.p ?? event.close ?? event.c);
    const open =
      typeToken === "tick"
        ? priceCandidate
        : toNumber(event.open ?? event.o ?? priceCandidate ?? event.close ?? event.c);
    const close = priceCandidate ?? open;
    const high = toNumber(event.high ?? event.h ?? close ?? open);
    const low = toNumber(event.low ?? event.l ?? close ?? open);
    const volume = toNumber(event.volume ?? event.v ?? 0) ?? 0;
    if (tVal == null || open == null || high == null || low == null || close == null) return null;
    return {
      t: "bars",
      symbol,
      bars: [{ t: tVal, o: open, h: high, l: low, c: close, v: volume }],
    };
  }
  return null;
}

function normalizePlanEvent(payload: unknown): PlanDelta | null {
  if (!payload || typeof payload !== "object") return null;
  const envelope = payload as { plan_id?: unknown; event?: unknown };
  const planId = typeof envelope.plan_id === "string" ? envelope.plan_id : undefined;
  const event = (envelope.event || payload) as Record<string, unknown>;
  const typeToken = typeof event.t === "string" ? event.t.toLowerCase() : "";
  const resolvedPlanId = planId || (typeof event.plan_id === "string" ? event.plan_id : undefined);
  const version = toNumber(event.version) ?? 0;

  if (!resolvedPlanId) return null;
  if (typeToken === "plan_delta") {
    const fields = (event.changes && typeof event.changes === "object" ? (event.changes as Record<string, unknown>) : {}) ?? {};
    return { t: "plan_delta", plan_id: resolvedPlanId, version, fields };
  }
  if (typeToken === "plan_full" || typeToken === "plan_full_snapshot") {
    const payloadBlock =
      (event.payload && typeof event.payload === "object" ? (event.payload as Record<string, unknown>) : null) ||
      (event.snapshot && typeof event.snapshot === "object" ? (event.snapshot as Record<string, unknown>) : null);
    if (!payloadBlock) return null;
    return {
      t: "plan_delta",
      plan_id: resolvedPlanId,
      version,
      fields: { __snapshot: payloadBlock },
    };
  }
  return null;
}

function normalizeCoachPulse(payload: unknown): CoachPulse | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  const token = typeof record.t === "string" ? record.t.toLowerCase() : "";
  if (token !== "coach_pulse") return null;
  const planId = typeof record.plan_id === "string" ? record.plan_id : null;
  const ts = typeof record.ts === "string" ? record.ts : nowIso();
  const diff = (record.diff && typeof record.diff === "object" ? (record.diff as CoachPulse["diff"]) : {}) ?? {};
  const session = record.session && typeof record.session === "object" ? (record.session as CoachPulse["session"]) : undefined;
  if (!planId) return null;
  return {
    t: "coach_pulse",
    plan_id: planId,
    ts,
    diff,
    session,
  };
}

class WsMultiplexer {
  private readonly channels = new Map<string, Channel>();
  private readonly eventListeners = new Set<EventListener>();
  private readonly stateListeners = new Map<string, Set<StateListener>>();
  private readonly pendingChannels: Channel[] = [];
  private readonly pendingKeys = new Set<string>();
  private frameHandle: number | null = null;

  onEvent(listener: EventListener): () => void {
    this.eventListeners.add(listener);
    return () => {
      this.eventListeners.delete(listener);
    };
  }

  onState(kind: ConnectionKind, target: string, listener: StateListener): () => void {
    const key = `${kind}:${target}`;
    const bucket = this.stateListeners.get(key) ?? new Set<StateListener>();
    bucket.add(listener);
    this.stateListeners.set(key, bucket);
    return () => {
      const existing = this.stateListeners.get(key);
      if (!existing) return;
      existing.delete(listener);
      if (existing.size === 0) {
        this.stateListeners.delete(key);
      }
    };
  }

  connectPlan(planId: string): () => void {
    const channel = this.resolveChannel("plan", planId);
    channel.addRef();
    return () => channel.release();
  }

  connectBars(symbol: string): () => void {
    const token = symbol.toUpperCase();
    const channel = this.resolveChannel("bars", token);
    channel.addRef();
    return () => channel.release();
  }

  connectCoach(planId: string): () => void {
    const channel = this.resolveChannel("coach", planId);
    channel.addRef();
    return () => channel.release();
  }

  enqueueChannel(channel: Channel): void {
    if (!this.pendingKeys.has(channel.key)) {
      this.pendingKeys.add(channel.key);
      this.pendingChannels.push(channel);
    }
    this.scheduleFrame();
  }

  notifyState(kind: ConnectionKind, target: string, state: ConnectionState): void {
    const key = `${kind}:${target}`;
    const listeners = this.stateListeners.get(key);
    if (!listeners || listeners.size === 0) return;
    listeners.forEach((listener) => {
      try {
        listener(state);
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[wsMux] state listener failed", error);
        }
      }
    });
  }

  buildUrl(kind: ConnectionKind, target: string): string | null {
    const base = WS_BASE_URL || "";
    if (!base) return null;
    if (kind === "plan") {
      return `${base}/ws/plans/${encodeURIComponent(target)}`;
    }
    if (kind === "coach") {
      return `${base}/ws/coach/${encodeURIComponent(target)}`;
    }
    return `${base}/stream/${encodeURIComponent(target)}`;
  }

  getProtocols(): string[] {
    if (!API_KEY_HEADER) return [];
    return [`Bearer ${API_KEY_HEADER}`];
  }

  private resolveChannel(kind: ConnectionKind, target: string): Channel {
    const key = `${kind}:${target}`;
    const existing = this.channels.get(key);
    if (existing) return existing;
    const channel = new Channel(this, kind, target);
    this.channels.set(key, channel);
    return channel;
  }

  private scheduleFrame(): void {
    if (typeof window === "undefined") {
      this.flushQueue();
      return;
    }
    if (this.frameHandle != null) return;
    this.frameHandle = window.requestAnimationFrame(() => {
      this.frameHandle = null;
      this.flushQueue();
    });
  }

  private flushQueue(): void {
    let processed = 0;
    while (processed < FRAME_EVENT_LIMIT && this.pendingChannels.length > 0) {
      const channel = this.pendingChannels.shift();
      if (!channel) continue;
      this.pendingKeys.delete(channel.key);
      const nextEvent = channel.shiftPending();
      if (!nextEvent) {
        continue;
      }
      this.emitEvent(nextEvent);
      processed += 1;
      if (channel.hasPending()) {
        this.pendingChannels.push(channel);
        this.pendingKeys.add(channel.key);
      }
    }
    if (this.pendingChannels.length > 0) {
      this.scheduleFrame();
    }
  }

  private emitEvent(event: Event): void {
    this.eventListeners.forEach((listener) => {
      try {
        listener(event);
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[wsMux] listener error", error);
        }
      }
    });
  }
}

export const wsMux = new WsMultiplexer();

export type { ConnectionKind };
