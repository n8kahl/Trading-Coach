export type PlanRealtimeEventType = "tp_hit" | "stop_hit" | "scale" | "reload" | "plan_invalid" | "note_update";

export type PlanRealtimeEvent<T = unknown> = {
  type: PlanRealtimeEventType;
  planId: string;
  payload?: T;
  at?: number;
};

type Listener = (event: PlanRealtimeEvent) => void;

const listeners = new Map<string, Set<Listener>>();

export function emitPlanEvent(event: PlanRealtimeEvent): void {
  const bucket = listeners.get(event.planId);
  if (!bucket || bucket.size === 0) return;
  bucket.forEach((listener) => {
    try {
      listener(event);
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.error("[plan/events] listener failure", error);
      }
    }
  });
}

export function subscribePlanEvents(planId: string, listener: Listener): () => void {
  if (!listeners.has(planId)) {
    listeners.set(planId, new Set());
  }
  const bucket = listeners.get(planId)!;
  bucket.add(listener);
  return () => {
    bucket.delete(listener);
    if (bucket.size === 0) {
      listeners.delete(planId);
    }
  };
}
