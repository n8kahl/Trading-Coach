import { describe, expect, it } from "vitest";
import type { PlanSnapshot } from "@/lib/types";
import { deriveCoachMessage, deriveProgressPct, type CoachGoal } from "@/lib/plan/coach";
import { fitVisibleRangeForTrade } from "@/lib/plan/chart";
import { emitPlanEvent, subscribePlanEvents } from "@/lib/plan/events";

function makePlan(overrides: Partial<PlanSnapshot["plan"]> = {}): PlanSnapshot["plan"] {
  return {
    plan_id: "PLAN-1",
    symbol: "AAPL",
    entry: 100,
    stop: 98,
    targets: [105, 110],
    target_meta: [
      { label: "TP1", price: 105 },
      { label: "TP2", price: 110 },
    ],
    ...overrides,
  } as PlanSnapshot["plan"];
}

describe("deriveProgressPct", () => {
  it("computes progress for long plans", () => {
    const pct = deriveProgressPct({ price: 102, nextTarget: 105, stop: 98 });
    expect(pct).toBeCloseTo(((102 - 98) / (105 - 98)) * 100);
  });

  it("computes progress for short plans", () => {
    const pct = deriveProgressPct({ price: 94, nextTarget: 92, stop: 98 });
    expect(pct).toBeCloseTo(((98 - 94) / (98 - 92)) * 100);
  });

  it("guards against invalid input", () => {
    expect(deriveProgressPct({ price: null, nextTarget: 105, stop: 98 })).toBe(0);
    expect(deriveProgressPct({ price: 102, nextTarget: 0, stop: 0 })).toBe(0);
  });
});

describe("deriveCoachMessage", () => {
  it("returns awaiting state when price missing", () => {
    const note = deriveCoachMessage({ plan: makePlan(), price: null, now: 1 });
    expect(note.goal).toBe<CoachGoal>("neutral");
    expect(note.text).toContain("Awaiting tick");
    expect(note.progressPct).toBe(0);
    expect(note.updatedAt).toBe(1);
  });

  it("identifies approach to target", () => {
    const note = deriveCoachMessage({ plan: makePlan(), price: 104.6, now: 2 });
    expect(note.goal).toBe<CoachGoal>("approach_tp");
    expect(note.text).toContain("TP1");
    expect(note.text).toContain("Δ");
  });

  it("marks target hit when price reaches level", () => {
    const note = deriveCoachMessage({ plan: makePlan(), price: 105, now: 3 });
    expect(note.goal).toBe<CoachGoal>("tp_hit");
    expect(note.text).toContain("TP1 reached");
    expect(note.text).toContain("105.00");
  });

  it("signals stop proximity", () => {
    const note = deriveCoachMessage({ plan: makePlan(), price: 98, now: 4 });
    expect(note.goal).toBe<CoachGoal>("approach_stop");
    expect(note.text).toContain("stop");
  });
});

describe("fitVisibleRangeForTrade", () => {
  it("returns padded range from bars and levels", () => {
    const range = fitVisibleRangeForTrade({
      bars: [
        { time: 1 as never, open: 100, high: 102, low: 99, close: 101 },
        { time: 2 as never, open: 101, high: 106, low: 100, close: 105 },
      ],
      entry: 100,
      stop: 98,
      targets: [110],
    });
    expect(range).not.toBeNull();
    if (!range) return;
    expect(range.min).toBeLessThan(98);
    expect(range.max).toBeGreaterThan(110);
  });

  it("handles empty bar array with levels", () => {
    const range = fitVisibleRangeForTrade({
      bars: [],
      entry: 100,
      stop: 98,
      targets: [110],
    });
    expect(range).not.toBeNull();
    if (!range) return;
    expect(range.min).toBeLessThan(98);
    expect(range.max).toBeGreaterThan(110);
  });

  it("returns null without data", () => {
    const range = fitVisibleRangeForTrade({ bars: [] });
    expect(range).toBeNull();
  });
});

describe("plan event bus", () => {
  it("delivers events to subscribers", () => {
    const events: string[] = [];
    const unsubscribe = subscribePlanEvents("PLAN-1", (event) => {
      events.push(event.type);
    });
    emitPlanEvent({ type: "tp_hit", planId: "PLAN-1" });
    emitPlanEvent({ type: "stop_hit", planId: "PLAN-2" });
    unsubscribe();
    emitPlanEvent({ type: "scale", planId: "PLAN-1" });
    expect(events).toEqual(["tp_hit"]);
  });
});
