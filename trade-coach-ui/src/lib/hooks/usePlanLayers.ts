import { useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "@/lib/env";
import type { PlanLayers } from "@/lib/types";

type Status = "idle" | "loading" | "ready" | "error";

function coerceNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function sanitizePlanLayers(raw: unknown, fallbackPlanId?: string | null): PlanLayers | null {
  if (!raw || typeof raw !== "object") return null;
  const block = raw as Record<string, unknown>;
  const levels = Array.isArray(block.levels)
    ? block.levels
        .map((entry) => {
          if (!entry || typeof entry !== "object") return null;
          const price = coerceNumber((entry as Record<string, unknown>).price);
          if (price == null) return null;
          const labelRaw = (entry as Record<string, unknown>).label;
          const kindRaw = (entry as Record<string, unknown>).kind;
          return {
            price,
            label: typeof labelRaw === "string" ? labelRaw : null,
            kind: typeof kindRaw === "string" ? kindRaw : null,
          };
        })
        .filter((item): item is NonNullable<typeof item> => item !== null)
    : [];
  const zones = Array.isArray(block.zones)
    ? block.zones
        .map((entry) => {
          if (!entry || typeof entry !== "object") return null;
          const zoneRecord = entry as Record<string, unknown>;
          const high = coerceNumber(zoneRecord.high);
          const low = coerceNumber(zoneRecord.low);
          if (high == null && low == null) return null;
          const label = typeof zoneRecord.label === "string" ? zoneRecord.label : null;
          const kind = typeof zoneRecord.kind === "string" ? zoneRecord.kind : null;
          return { high, low, label, kind };
        })
        .filter((item): item is NonNullable<typeof item> => item !== null)
    : [];
  const annotations = Array.isArray(block.annotations) ? (block.annotations as Array<Record<string, unknown>>) : undefined;
  const meta = block.meta && typeof block.meta === "object" ? (block.meta as Record<string, unknown>) : undefined;
  const interval = typeof block.interval === "string" ? block.interval : null;
  const asOf = typeof block.as_of === "string" ? block.as_of : null;
  const planningContext = typeof block.planning_context === "string" ? block.planning_context : null;
  const planId = typeof block.plan_id === "string" ? block.plan_id : fallbackPlanId;
  const precision = coerceNumber(block.precision);
  const symbol = typeof block.symbol === "string" ? block.symbol : undefined;
  return {
    plan_id: planId ?? undefined,
    symbol,
    interval,
    as_of: asOf,
    planning_context: planningContext,
    precision: precision ?? undefined,
    levels,
    zones,
    annotations,
    meta,
  };
}

export function usePlanLayers(planId: string | null | undefined, initial: PlanLayers | null | undefined) {
  const [layers, setLayers] = useState<PlanLayers | null>(() => sanitizePlanLayers(initial, planId));
  const [status, setStatus] = useState<Status>(initial ? "ready" : "idle");
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    setLayers(sanitizePlanLayers(initial, planId));
    if (initial) {
      setStatus("ready");
      setError(null);
    }
  }, [initial, planId]);

  useEffect(() => {
    if (!planId) {
      setLayers(null);
      setStatus("idle");
      setError(null);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    async function load() {
      try {
        setStatus((prev) => (prev === "ready" && layers ? "ready" : "loading"));
        setError(null);
        const qs = new URLSearchParams({ plan_id: planId });
        const response = await fetch(`${API_BASE_URL}/api/v1/gpt/chart-layers?${qs.toString()}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`chart-layers failed (${response.status})`);
        }
        const payload = await response.json();
        if (!cancelled) {
          setLayers(sanitizePlanLayers(payload, planId));
          setStatus("ready");
        }
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) return;
        setError(err instanceof Error ? err : new Error("Failed to load chart layers"));
        setStatus("error");
      }
    }
    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [planId, reloadToken]);

  const reload = () => setReloadToken((token) => token + 1);

  return useMemo(
    () => ({
      layers,
      status,
      error,
      reload,
    }),
    [layers, status, error],
  );
}

export type UsePlanLayersReturn = ReturnType<typeof usePlanLayers>;
