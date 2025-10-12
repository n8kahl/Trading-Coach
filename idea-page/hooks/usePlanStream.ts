import { useEffect, useRef, useState } from "react";

export type PlanStreamState = {
  status: "intact" | "at_risk" | "invalidated" | "reversal";
  rrToT1: number | null;
  note: string | null;
  nextStep: string | null;
  breach: string | null;
  timestamp: string | null;
  lastPrice: number | null;
  version: number | null;
  marketPhase: string | null;
  marketNote: string | null;
};

type UsePlanStreamOptions<TPlanPayload> = {
  symbol: string | null | undefined;
  planId: string | null | undefined;
  initialState: PlanStreamState;
  onPlanFull?: (payload: TPlanPayload) => void;
};

const STATUS_FALLBACK: PlanStreamState = {
  status: "intact",
  rrToT1: null,
  note: "Plan intact. Risk profile unchanged.",
  nextStep: "hold_plan",
  breach: null,
  timestamp: null,
  lastPrice: null,
  version: null,
  marketPhase: null,
  marketNote: null,
};

function resolveBaseUrl(): string | null {
  if (typeof window === "undefined") return null;
  return window.location.origin;
}

function parseEnvelope(raw: string): { symbol?: string; event?: any } | null {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function normalizeState(current: PlanStreamState, changes: any): PlanStreamState {
  if (!changes || typeof changes !== "object") {
    return current;
  }
  return {
    status: (changes.status as PlanStreamState["status"]) || current.status,
    rrToT1: typeof changes.rr_to_t1 === "number" ? changes.rr_to_t1 : current.rrToT1,
    note: typeof changes.note === "string" ? changes.note : current.note,
    nextStep: typeof changes.next_step === "string" ? changes.next_step : current.nextStep,
    breach: typeof changes.breach === "string" ? changes.breach : current.breach,
    timestamp: typeof changes.timestamp === "string" ? changes.timestamp : current.timestamp,
    lastPrice: typeof changes.last_price === "number" ? changes.last_price : current.lastPrice,
    version: typeof changes.version === "number" ? changes.version : current.version,
    marketPhase: current.marketPhase,
    marketNote: current.marketNote,
  };
}

export function usePlanStream<TPlanPayload>({
  symbol,
  planId,
  initialState,
  onPlanFull,
}: UsePlanStreamOptions<TPlanPayload>) {
  const [planState, setPlanState] = useState<PlanStreamState>(initialState ?? STATUS_FALLBACK);
  const baseUrlRef = useRef<string | null>(resolveBaseUrl());

  useEffect(() => {
    const next = initialState ?? STATUS_FALLBACK;
    setPlanState((prev) => ({
      ...prev,
      status: next.status,
      rrToT1: next.rrToT1,
      note: next.note,
      nextStep: next.nextStep,
      breach: next.breach,
      timestamp: next.timestamp,
      lastPrice: next.lastPrice ?? prev.lastPrice,
      version: next.version ?? prev.version,
      marketPhase: next.marketPhase ?? prev.marketPhase,
      marketNote: next.marketNote ?? prev.marketNote,
    }));
  }, [
    initialState.status,
    initialState.rrToT1,
    initialState.note,
    initialState.nextStep,
    initialState.version,
    initialState.marketPhase,
    initialState.marketNote,
    initialState.timestamp,
    initialState.lastPrice,
  ]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return () => {};
    }
    const sym = (symbol || "").trim();
    const pid = (planId || "").trim();
    if (!sym || !pid) {
      return () => {};
    }

    let active = true;
    let ws: WebSocket | null = null;
    let es: EventSource | null = null;

    const applyPlanDelta = (event: any) => {
      if (!event || event.plan_id !== pid) return;
      setPlanState((prev) => normalizeState(prev, { ...event.changes, version: event.version }));
    };

    const applyPlanState = (event: any) => {
      const plans = Array.isArray(event?.plans) ? event.plans : [];
      const match = plans.find((item) => item.plan_id === pid);
      if (!match) return;
      setPlanState((prev) =>
        normalizeState(prev, {
          status: match.status,
          rr_to_t1: match.rr_to_t1,
          note: match.note,
          next_step: match.next_step,
          timestamp: match.timestamp,
          version: match.version,
        }),
      );
    };

    const applyPlanFull = (event: any) => {
      const payloadPlanId =
        event?.plan_id ||
        (typeof event?.payload === "object" && event?.payload?.plan && event?.payload?.plan?.plan_id);
      if (payloadPlanId !== pid) {
        return;
      }
      const payload = event?.payload;
      if (payload?.plan) {
        setPlanState((prev) => {
          const rrValue = typeof payload.plan.rr_to_t1 === "number" ? payload.plan.rr_to_t1 : prev.rrToT1;
          const version =
            typeof payload.plan.version === "number" ? payload.plan.version : prev.version;
          return {
            ...prev,
            status: "intact",
            rrToT1: rrValue,
            note: "Plan intact. Risk profile unchanged.",
            nextStep: "hold_plan",
            breach: null,
            timestamp: new Date().toISOString(),
            lastPrice: prev.lastPrice,
            version,
          };
        });
      }
      if (payload && typeof onPlanFull === "function") {
        onPlanFull(payload);
      }
    };

    const applyTick = (event: any) => {
      const price =
        typeof event?.p === "number"
          ? event.p
          : typeof event?.close === "number"
            ? event.close
            : null;
      if (price === null) return;
      setPlanState((prev) => ({
        ...prev,
        lastPrice: price,
      }));
    };

    const applyMarketStatus = (event: any) => {
      const phase = typeof event?.phase === "string" ? event.phase : null;
      const note = typeof event?.note === "string" ? event.note : null;
      setPlanState((prev) => ({
        ...prev,
        marketPhase: phase ?? prev.marketPhase,
        marketNote: note ?? prev.marketNote,
      }));
    };

    const handleMessage = (raw: string) => {
      if (!active) return;
      const envelope = parseEnvelope(raw);
      if (!envelope?.event) return;
      const event = envelope.event;
      switch (event.t) {
        case "plan_delta":
          applyPlanDelta(event);
          break;
        case "plan_state":
          applyPlanState(event);
          break;
        case "plan_full":
          applyPlanFull(event);
          break;
        case "tick":
        case "bar":
          applyTick(event);
          break;
        case "market_status":
          applyMarketStatus(event);
          break;
        default:
          break;
      }
    };

    const connectSse = () => {
      if (!baseUrlRef.current) return;
      es = new EventSource(`${baseUrlRef.current}/stream/${encodeURIComponent(sym)}`);
      es.onmessage = (ev) => handleMessage(ev.data);
      es.onerror = () => {
        if (es) {
          es.close();
        }
      };
    };

    const connectWebSocket = () => {
      const baseUrl = baseUrlRef.current;
      if (!baseUrl) {
        connectSse();
        return;
      }
      const protocol = baseUrl.startsWith("https") ? "wss" : "ws";
      const url = `${protocol}://${window.location.host}/stream/${encodeURIComponent(sym)}`;
      ws = new WebSocket(url);
      ws.onmessage = (event) => handleMessage(event.data);
      ws.onerror = () => {
        if (ws) {
          ws.close();
        }
      };
      ws.onclose = () => {
        if (!active) return;
        if (ws) {
          ws = null;
        }
        if (!es) {
          connectSse();
        }
      };
    };

    connectWebSocket();

    return () => {
      active = false;
      if (ws) {
        ws.close();
      }
      if (es) {
        es.close();
      }
    };
  }, [symbol, planId, onPlanFull]);

  return { planState, setPlanState };
}
