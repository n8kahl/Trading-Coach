import { useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "@/lib/env";

type PlanLike = {
  charts?: { interactive?: string | null; params?: Record<string, unknown> | null };
  chart_url?: string | null;
  charts_params?: Record<string, unknown> | null;
};

const VALID_HOST = /^https:\/\/trading-coach-production\.up\.railway\.app\/tv/;

export function useChartUrl(plan: PlanLike | null | undefined): string | null {
  const direct = useMemo(
    () => plan?.charts?.interactive ?? plan?.chart_url ?? null,
    [plan?.charts?.interactive, plan?.chart_url],
  );
  const [url, setUrl] = useState<string | null>(direct);
  const paramsKey = useMemo(
    () => JSON.stringify(plan?.charts?.params ?? plan?.charts_params ?? null),
    [plan?.charts?.params, plan?.charts_params],
  );

  useEffect(() => {
    if (direct) {
      setUrl(direct);
      return;
    }
    const params = plan?.charts?.params ?? plan?.charts_params ?? null;
    if (!params || typeof params !== "object" || !(params as Record<string, unknown>).symbol) {
      setUrl(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/gpt/chart-url`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(params),
        });
        const data = (await response.json()) as { interactive?: string | null };
        const candidate = data?.interactive ?? null;
        if (!cancelled && candidate && VALID_HOST.test(candidate)) {
          setUrl(candidate);
        } else if (!cancelled) {
          setUrl(null);
        }
      } catch {
        if (!cancelled) setUrl(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [direct, paramsKey, plan]);

  return url;
}

