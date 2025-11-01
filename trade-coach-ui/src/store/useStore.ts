"use client";

import { create } from "zustand";
import type { BarsEvt, CoachPulse, Event, PlanDelta, ConnectionKind, ConnectionState } from "@/lib/wsMux";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";

type SessionInfo = CoachPulse["session"] & { status: string } | null;

export type WatchlistItem = {
  plan_id: string;
  symbol: string;
  style?: string | null;
  plan_url: string;
  chart_url: string | null;
  actionable_soon: boolean | null;
  entry_distance_pct: number | null;
  entry_distance_atr: number | null;
  bars_to_trigger: number | null;
  meta?: Record<string, unknown>;
  raw: Record<string, unknown>;
};

type WatchlistStatus = "idle" | "loading" | "ready" | "error";

type WatchlistSlice = {
  watchlist: {
    items: WatchlistItem[];
    status: WatchlistStatus;
    lastUpdated: number | null;
    error: string | null;
  };
  setWatchlist(items: WatchlistItem[]): void;
  setWatchlistStatus(status: WatchlistStatus, error?: string | null): void;
};

type WsStatsEntry = {
  uptimeMs: number;
  lastConnectedAt: number | null;
  reconnects: number;
  everConnected: boolean;
};

type ObservabilitySlice = {
  wsStats: Record<ConnectionKind, WsStatsEntry>;
  resetWsStats(kind?: ConnectionKind): void;
};

type SessionSlice = {
  session: SessionInfo;
  setSession(session: SessionInfo): void;
};

type BarsSlice = {
  barsBySymbol: Record<string, BarsEvt["bars"]>;
  lastBarAt: Record<string, number>;
  updateBars(event: BarsEvt): void;
};

type PlanSlice = {
  plan: PlanSnapshot["plan"] | null;
  planId: string | null;
  planVersion: number;
  lastPlanEventAt: number | null;
  hydratePlan(plan: PlanSnapshot["plan"] | null): void;
  applyPlanDelta(delta: PlanDelta): void;
};

type OverlaySlice = {
  planLayers: PlanLayers | null;
  setPlanLayers(layers: PlanLayers | null): void;
  patchObjective(meta: Record<string, unknown> | null | undefined): void;
};

type CoachSlice = {
  coach: {
    planId: string | null;
    lastPulseTs: string | null;
    diff: CoachPulse["diff"] | null;
    timeline: Array<{ ts: string; diff: CoachPulse["diff"] }>;
  };
  applyCoachPulse(pulse: CoachPulse): void;
};

type UiSlice = {
  connection: Record<"plan" | "bars" | "coach", ConnectionState>;
  setConnection(kind: "plan" | "bars" | "coach", state: ConnectionState): void;
};

type StoreState = SessionSlice &
  BarsSlice &
  PlanSlice &
  OverlaySlice &
  CoachSlice &
  UiSlice &
  WatchlistSlice &
  ObservabilitySlice & {
    applyEvent(event: Event): void;
    bootstrap(snapshot: PlanSnapshot, layers: PlanLayers | null): void;
    reset(): void;
  };

const MAX_BAR_STORE = 2_000;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function mergeBars(existing: BarsEvt["bars"] | undefined, incoming: BarsEvt["bars"]): BarsEvt["bars"] {
  if (!existing || existing.length === 0) {
    return incoming.slice(-MAX_BAR_STORE);
  }
  const merged = new Map<number, BarsEvt["bars"][number]>();
  for (const bar of existing) {
    merged.set(bar.t, bar);
  }
  for (const bar of incoming) {
    merged.set(bar.t, bar);
  }
  return Array.from(merged.values())
    .sort((a, b) => a.t - b.t)
    .slice(-MAX_BAR_STORE);
}

function mergeField(prevValue: unknown, nextValue: unknown): { changed: boolean; value: unknown } {
  if (isPlainObject(prevValue) && isPlainObject(nextValue)) {
    const result: Record<string, unknown> = { ...prevValue };
    let changed = false;
    for (const [key, value] of Object.entries(nextValue)) {
      const current = result[key];
      if (isPlainObject(current) && isPlainObject(value)) {
        const nested = mergeField(current, value);
        if (nested.changed) {
          result[key] = nested.value;
          changed = true;
        }
        continue;
      }
      if (!Object.is(current, value)) {
        result[key] = value;
        changed = true;
      }
    }
    return { changed, value: changed ? result : prevValue };
  }
  if (!Object.is(prevValue, nextValue)) {
    return { changed: true, value: nextValue };
  }
  return { changed: false, value: prevValue };
}

export const useStore = create<StoreState>((set, get) => ({
  session: null,
  barsBySymbol: {},
  lastBarAt: {},
  plan: null,
    planId: null,
    planVersion: 0,
    lastPlanEventAt: null,
    planLayers: null,
  coach: {
    planId: null,
    lastPulseTs: null,
    diff: null,
    timeline: [],
  },
  connection: {
    plan: "idle",
    bars: "idle",
    coach: "idle",
  },
  watchlist: {
    items: [],
    status: "idle",
    lastUpdated: null,
    error: null,
  },
  wsStats: {
    plan: { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false },
    bars: { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false },
    coach: { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false },
  },

  setSession(session) {
    set({ session });
  },

  updateBars(event) {
    if (!event.symbol) return;
    set((state) => {
      const nextBars = mergeBars(state.barsBySymbol[event.symbol], event.bars);
      const previous = state.barsBySymbol[event.symbol];
      if (previous && previous.length === nextBars.length) {
        const lastPrev = previous[previous.length - 1];
        const lastNext = nextBars[nextBars.length - 1];
        if (lastPrev && lastNext && lastPrev.t === lastNext.t && lastPrev.c === lastNext.c) {
          return state;
        }
      }
      return {
        barsBySymbol: {
          ...state.barsBySymbol,
          [event.symbol]: nextBars,
        },
        lastBarAt: {
          ...state.lastBarAt,
          [event.symbol]: nextBars.length ? nextBars[nextBars.length - 1].t : state.lastBarAt[event.symbol],
        },
      };
    });
  },

  setWatchlist(items) {
    set((state) => ({
      watchlist: {
        ...state.watchlist,
        items,
        status: "ready",
        lastUpdated: Date.now(),
        error: null,
      },
    }));
  },

  setWatchlistStatus(status, error = null) {
    set((state) => ({
      watchlist: {
        ...state.watchlist,
        status,
        error,
        lastUpdated: status === "ready" ? Date.now() : state.watchlist.lastUpdated,
      },
    }));
  },

  hydratePlan(plan) {
    if (!plan) {
      set({
        plan: null,
        planId: null,
        planVersion: 0,
      });
      return;
    }
    set({
      plan,
      planId: plan.plan_id ?? null,
      planVersion: typeof plan.version === "number" ? plan.version : 0,
      lastPlanEventAt: Date.now(),
    });
  },

  applyPlanDelta(delta) {
    const current = get().plan;
    if (!current || !delta.plan_id || current.plan_id !== delta.plan_id) {
      return;
    }
    const snapshotBlock = delta.fields.__snapshot;
    if (isPlainObject(snapshotBlock)) {
      const planPayload = (snapshotBlock.plan && isPlainObject(snapshotBlock.plan)
        ? (snapshotBlock.plan as PlanSnapshot["plan"])
        : (snapshotBlock as PlanSnapshot["plan"])) ?? null;
      if (planPayload) {
        set({
          plan: planPayload,
          planVersion: typeof planPayload.version === "number" ? planPayload.version : delta.version ?? get().planVersion,
          lastPlanEventAt: Date.now(),
          session: planPayload.session_state ?? get().session,
        });
      }
      return;
    }

    const fieldEntries = Object.entries(delta.fields).filter(([key]) => key !== "__snapshot");
    if (fieldEntries.length === 0) {
      return;
    }
    set((state) => {
      if (!state.plan) return state;
      let changed = false;
      const nextPlan: Record<string, unknown> = { ...state.plan };
      let sessionUpdate: SessionInfo | null = null;
      for (const [key, value] of fieldEntries) {
        const currentValue = (nextPlan as Record<string, unknown>)[key];
        const merged = mergeField(currentValue, value);
        if (merged.changed) {
          nextPlan[key] = merged.value;
          changed = true;
        }
        if (key === "session_state") {
          sessionUpdate = isPlainObject(value) ? (value as SessionInfo) : null;
        }
      }
      if (!changed) {
        if (sessionUpdate) {
          return {
            ...state,
            session: sessionUpdate,
          };
        }
        return state;
      }
      const nextVersion = delta.version && delta.version > state.planVersion ? delta.version : state.planVersion;
      return {
        ...state,
        plan: nextPlan as PlanSnapshot["plan"],
        planVersion: nextVersion,
        lastPlanEventAt: Date.now(),
        session: sessionUpdate ?? state.session,
      };
    });
  },

  setPlanLayers(layers) {
    set({ planLayers: layers });
  },

  patchObjective(meta) {
    if (!meta) return;
    set((state) => {
      if (!state.planLayers) return state;
      const existing = state.planLayers.meta ?? {};
      const currentObjective = isPlainObject(existing.next_objective) ? existing.next_objective : {};
      const mergedEntries = mergeField(currentObjective, meta);
      if (!mergedEntries.changed) {
        return state;
      }
      const nextMeta = { ...existing, next_objective: mergedEntries.value };
      return {
        ...state,
        planLayers: {
          ...state.planLayers,
          meta: nextMeta,
        },
      };
    });
  },

  applyCoachPulse(pulse) {
    if (pulse.diff?.objective_progress) {
      const progressPayload = pulse.diff.objective_progress;
      const metaPatch: Record<string, unknown> = {};
      if (typeof progressPayload.progress === "number") {
        metaPatch.progress = progressPayload.progress;
      }
      if (typeof progressPayload.entry_distance_pct === "number") {
        metaPatch.entry_distance_pct = progressPayload.entry_distance_pct;
      }
      if (Object.keys(metaPatch).length > 0) {
        get().patchObjective(metaPatch);
      }
    }
    set((state) => {
      const samePlan = state.coach.planId === pulse.plan_id;
      let timeline = state.coach.timeline;
      if (pulse.diff) {
        const base = samePlan ? timeline.slice(-19) : [];
        timeline = [...base, { ts: pulse.ts, diff: pulse.diff }];
      } else if (!samePlan) {
        timeline = [];
      }
      const updates: Partial<StoreState> = {
        coach: {
          planId: pulse.plan_id,
          lastPulseTs: pulse.ts,
          diff: pulse.diff,
          timeline,
        },
      };
      if (pulse.session) {
        updates.session = { ...pulse.session };
      }
      return updates;
    });
  },

  setConnection(kind, stateValue) {
    set((state) => {
      if (state.connection[kind] === stateValue) return state;
      const now = Date.now();
      const currentStats = state.wsStats[kind];
      let { uptimeMs, lastConnectedAt, reconnects, everConnected } = currentStats;
      if (stateValue === "connected") {
        if (lastConnectedAt == null) {
          if (everConnected) {
            reconnects += 1;
          }
          lastConnectedAt = now;
          everConnected = true;
        }
      } else if (lastConnectedAt != null) {
        uptimeMs += now - lastConnectedAt;
        lastConnectedAt = null;
      }
      return {
        connection: {
          ...state.connection,
          [kind]: stateValue,
        },
        wsStats: {
          ...state.wsStats,
          [kind]: {
            uptimeMs,
            lastConnectedAt,
            reconnects,
            everConnected,
          },
        },
      };
    });
  },

  resetWsStats(kind?: ConnectionKind) {
    const base: WsStatsEntry = { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false };
    if (kind) {
      set((state) => ({
        wsStats: {
          ...state.wsStats,
          [kind]: { ...base },
        },
      }));
      return;
    }
    set({
      wsStats: {
        plan: { ...base },
        bars: { ...base },
        coach: { ...base },
      },
    });
  },

  applyEvent(event) {
    if (event.t === "bars") {
      get().updateBars(event);
      return;
    }
    if (event.t === "plan_delta") {
      get().applyPlanDelta(event);
      return;
    }
    if (event.t === "coach_pulse") {
      get().applyCoachPulse(event);
    }
  },

  bootstrap(snapshot, layers) {
    const planBlock = snapshot.plan ?? null;
    const session = planBlock?.session_state ?? null;
    set({
      plan: planBlock,
      planId: planBlock?.plan_id ?? null,
      planVersion: typeof planBlock?.version === "number" ? planBlock.version : 0,
      session: session ? { ...session } : null,
      planLayers: layers,
      lastPlanEventAt: Date.now(),
      coach: {
        planId: planBlock?.plan_id ?? null,
        lastPulseTs: null,
        diff: null,
        timeline: [],
      },
    });
  },

  reset() {
    set({
      session: null,
      barsBySymbol: {},
      lastBarAt: {},
      plan: null,
      planId: null,
      planVersion: 0,
      lastPlanEventAt: null,
      planLayers: null,
      coach: {
        planId: null,
        lastPulseTs: null,
        diff: null,
        timeline: [],
      },
      connection: {
        plan: "idle",
        bars: "idle",
        coach: "idle",
      },
      watchlist: {
        items: [],
        status: "idle",
        lastUpdated: null,
        error: null,
      },
      wsStats: {
        plan: { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false },
        bars: { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false },
        coach: { uptimeMs: 0, lastConnectedAt: null, reconnects: 0, everConnected: false },
      },
    });
  },
}));

export type Store = ReturnType<typeof useStore.getState>;
