import { notFound, redirect } from "next/navigation";
import type { Metadata } from "next";
import { fetchPlanSnapshot } from "@/lib/api";
import type { PlanSnapshot } from "@/lib/types";
import LivePlanClient from "./LivePlanClient";

type PlanPageProps = {
  params: Promise<{ planId: string }>;
};

export async function generateMetadata({ params }: PlanPageProps): Promise<Metadata> {
  const { planId } = await params;
  let plan: PlanSnapshot | null = null;
  try {
    const idOrUrl = planId;
    if (idOrUrl.startsWith('http')) {
      try {
        const url = new URL(idOrUrl);
        const pid = url.searchParams.get('plan_id');
        if (pid) {
          // Redirect so the URL is canonical in the browser
          redirect(`/plan/${encodeURIComponent(pid)}`);
        }
      } catch {}
    }
    plan = await fetchPlanSnapshot(idOrUrl);
  } catch {
    // ignore
  }

  const symbol = plan?.plan?.symbol ?? "Plan";
  const style = plan?.plan?.style ?? plan?.plan?.structured_plan?.style ?? "";
  const title = `${symbol.toUpperCase()} · ${style ? `${style} ` : ""}Plan`;
  const description = `Live trading console for ${symbol.toUpperCase()} (${style || "latest"} style).`;

  return {
    title,
    description,
  };
}

export default async function PlanPage({ params }: PlanPageProps) {
  const { planId } = await params;
  const idOrUrl = planId;
  if (idOrUrl.startsWith('http')) {
    try {
      const url = new URL(idOrUrl);
      const pid = url.searchParams.get('plan_id');
      if (pid) redirect(`/plan/${encodeURIComponent(pid)}`);
    } catch {
      // fall through
    }
  }
  let snapshot: PlanSnapshot;
  try {
    snapshot = await fetchPlanSnapshot(idOrUrl);
  } catch (error) {
    if (process.env.NODE_ENV !== "production") {
      console.error(error);
    }
    notFound();
  }

  if (!snapshot?.plan?.plan_id) {
    notFound();
  }

  return (
    <LivePlanClient
      initialSnapshot={snapshot}
      planId={snapshot.plan.plan_id}
      symbol={snapshot.plan.symbol}
    />
  );
}
