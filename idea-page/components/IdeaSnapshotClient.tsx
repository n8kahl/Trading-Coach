"use client";

import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";

import ChartEmbed from "@/components/ChartEmbed";
import ConfluencePanel from "@/components/ConfluencePanel";
import ContractsTable from "@/components/ContractsTable";
import EducationRail from "@/components/EducationRail";
import IdeaHeader from "@/components/IdeaHeader";
import Playbook from "@/components/Playbook";
import Provenance from "@/components/Provenance";
import QuickPlan from "@/components/QuickPlan";
import SanityBanner from "@/components/SanityBanner";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchIdea } from "@/lib/api";
import type { TIdeaSnapshot } from "@/lib/types";
import { cn } from "@/lib/utils";

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

  useEffect(() => {
    if (!data?.options?.table || !initialData?.options?.table) return;
    const changed = new Set<string>();
    data.options.table.slice(0, 6).forEach((row) => {
      const baseline = initialData.options?.table?.find((item) => item.label === row.label);
      if (!baseline) return;
      if (
        baseline.price !== row.price ||
        baseline.bid !== row.bid ||
        baseline.ask !== row.ask ||
        baseline.iv !== row.iv ||
        baseline.liquidity_score !== row.liquidity_score
      ) {
        changed.add(row.label);
      }
    });
    if (changed.size > 0) {
      setHighlightedRows(changed);
      const timer = setTimeout(() => setHighlightedRows(new Set()), 1500);
      return () => clearTimeout(timer);
    }
  }, [data?.options?.table, initialData?.options?.table]);

  const hasOptions = Boolean(data?.options?.table && data.options.table.length > 0);

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

  const plan = data.plan;
  const confidence = plan.confidence;
  const snapshot = data;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 pb-20 pt-10">
      <IdeaHeader idea={snapshot} isRefreshing={isValidating} />
      <SanityBanner plan={snapshot.plan} />
      <QuickPlan idea={snapshot} />
      <section className="grid gap-6 lg:grid-cols-[3fr,2fr]">
        <ChartEmbed chartUrl={snapshot.chart_url} />
        {hasOptions ? (
          <ContractsTable idea={snapshot} highlightedRows={highlightedRows} onRefresh={() => mutate()} />
        ) : (
          <Card className="min-h-[320px]">
            <CardContent className="flex h-full flex-col items-center justify-center text-center text-muted-foreground">
              <p className="text-sm font-medium">Options contracts unavailable</p>
              <p className="mt-2 text-xs leading-relaxed">Once server-side options data is ready, it will appear here automatically.</p>
            </CardContent>
          </Card>
        )}
      </section>
      <section className="grid gap-6 lg:grid-cols-[3fr,2fr]">
        <ConfluencePanel idea={snapshot} />
        <EducationRail idea={snapshot} />
      </section>
      <section className="grid gap-6 lg:grid-cols-[3fr,2fr]">
        <Playbook idea={snapshot} />
        <Provenance idea={snapshot} />
      </section>
    </div>
  );
}
