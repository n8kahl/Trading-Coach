"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";

import ChartEmbed from "@/components/ChartEmbed";
import ConfluencePanel from "@/components/ConfluencePanel";
import ContractsTable from "@/components/ContractsTable";
import EducationRail from "@/components/EducationRail";
import IdeaHeader from "@/components/IdeaHeader";
import LiveCoaching from "@/components/LiveCoaching";
import Playbook from "@/components/Playbook";
import Provenance from "@/components/Provenance";
import QuickPlan from "@/components/QuickPlan";
import SanityBanner from "@/components/SanityBanner";
import NextStepCard from "@/components/NextStepCard";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchIdea } from "@/lib/api";
import { usePlanStream } from "@/hooks/usePlanStream";
import type { TIdeaSnapshot } from "@/lib/types";

type IdeaSnapshotClientProps = {
  initialData: TIdeaSnapshot;
  planId: string;
  version?: string | number | null;
};

const fetcher = async (url: string) => {
  const planId = url.split("?")[0];
  const version = url.includes("?v=") ? url.split("?v=")[1] : undefined;
  const parsedVersion = version ? Number(version) || version : undefined;
  return fetchIdea(planId, parsedVersion);
};

export default function IdeaSnapshotClient({ initialData, planId, version }: IdeaSnapshotClientProps) {
  const queryKey = version ? `${planId}?v=${version}` : planId;
  const { data, isValidating, error, mutate } = useSWR<TIdeaSnapshot>(queryKey, fetcher, {
    fallbackData: initialData,
    refreshInterval: 20000,
    revalidateOnFocus: true,
  });

  const [highlightedRows, setHighlightedRows] = useState<Set<string>>(new Set());
  const previousSnapshot = useRef<TIdeaSnapshot | null>(initialData);

  const activePlan = data?.plan ?? initialData.plan;
  const baselinePlanState = useMemo(
    () => ({
      status: "intact" as const,
      rrToT1: typeof activePlan?.rr_to_t1 === "number" ? activePlan.rr_to_t1 : null,
      note: "Plan intact. Risk profile unchanged.",
      nextStep: "hold_plan" as const,
      breach: null,
      timestamp: null,
      lastPrice: null,
      version: typeof activePlan?.version === "number" ? activePlan.version : null,
    }),
    [activePlan?.rr_to_t1, activePlan?.version],
  );

  const handlePlanFull = useCallback(
    (payload: TIdeaSnapshot) => {
      mutate(payload, { revalidate: false });
    },
    [mutate],
  );

  const { planState, setPlanState } = usePlanStream<TIdeaSnapshot>({
    symbol: activePlan?.symbol,
    planId,
    initialState: baselinePlanState,
    onPlanFull: handlePlanFull,
  });

  useEffect(() => {
    if (!data?.options?.table) {
      previousSnapshot.current = data ?? previousSnapshot.current;
      return;
    }
    const previous = previousSnapshot.current?.options?.table;
    if (!previous) {
      previousSnapshot.current = data;
      return;
    }
    const changes = new Set<string>();
    data.options.table.slice(0, 6).forEach((row) => {
      const baseline = previous.find((item) => item.label === row.label);
      if (!baseline) return;
      if (
        baseline.price !== row.price ||
        baseline.bid !== row.bid ||
        baseline.ask !== row.ask ||
        baseline.iv !== row.iv ||
        baseline.liquidity_score !== row.liquidity_score
      ) {
        changes.add(row.label);
      }
    });
    if (changes.size > 0) {
      setHighlightedRows(changes);
      const timer = setTimeout(() => setHighlightedRows(new Set()), 1600);
      return () => clearTimeout(timer);
    }
    previousSnapshot.current = data;
  }, [data]);

  const hasOptions = Boolean(data?.options?.table && data.options.table.length > 0);

  const handleKeepPlan = useCallback(() => {
    setPlanState((prev) => ({
      ...prev,
      status: "intact",
      nextStep: "hold_plan",
      breach: null,
      note: "Continuing with current plan.",
      timestamp: new Date().toISOString(),
    }));
  }, [setPlanState]);

  const handleApplyUpdate = useCallback(() => {
    void mutate();
  }, [mutate]);

  if (error) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-16">
        <Card className="border-destructive/40 bg-destructive/10">
          <CardContent className="py-10 text-center">
            <p className="text-lg font-semibold text-destructive">Unable to load idea snapshot.</p>
            <p className="mt-2 text-sm text-muted-foreground">Please retry or contact support if the issue persists.</p>
            <button
              onClick={() => mutate()}
              className="mt-6 inline-flex items-center rounded-md bg-destructive px-4 py-2 text-sm font-medium text-destructive-foreground transition hover:bg-destructive/90"
            >
              Retry
            </button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-12">
        <div className="grid gap-6">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-40 w-full" />
          <div className="grid gap-6 lg:grid-cols-[2fr,3fr]">
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 pb-20 pt-10">
      <IdeaHeader idea={data} isRefreshing={isValidating} />
      <SanityBanner plan={data.plan} />
      <QuickPlan idea={data} />
      <NextStepCard planState={planState} onKeepPlan={handleKeepPlan} onApplyUpdate={handleApplyUpdate} disabled={isValidating} />
      <section className="grid gap-6 xl:grid-cols-[3fr,2fr]">
        <ChartEmbed chartUrl={data.chart_url} />
        <LiveCoaching idea={data} />
      </section>
      <section className="grid gap-6 xl:grid-cols-[3fr,2fr]">
        {hasOptions ? (
          <ContractsTable idea={data} highlightedRows={highlightedRows} onRefresh={() => mutate()} />
        ) : (
          <Card className="min-h-[320px]">
            <CardContent className="flex h-full flex-col items-center justify-center text-center text-muted-foreground">
              <p className="text-sm font-medium">Options contracts unavailable</p>
              <p className="mt-2 text-xs leading-relaxed">Once server-side options data is ready, it will appear here automatically.</p>
            </CardContent>
          </Card>
        )}
        <ConfluencePanel idea={data} />
      </section>
      <section className="grid gap-6 xl:grid-cols-[3fr,2fr]">
        <Playbook idea={data} />
        <EducationRail idea={data} />
      </section>
      <Provenance idea={data} />
    </div>
  );
}
