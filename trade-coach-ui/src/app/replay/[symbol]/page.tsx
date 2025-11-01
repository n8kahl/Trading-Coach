import type { Metadata } from "next";
import ReplayClient from "./ReplayClient";
import { fetchSimulatedPlan } from "@/lib/api";

type ReplayPageProps = {
  params: { symbol: string };
};

export async function generateMetadata({ params }: ReplayPageProps): Promise<Metadata> {
  const symbol = params.symbol;
  return {
    title: `${symbol.toUpperCase()} · Simulated Dojo`,
    description: `Simulated live market dojo for ${symbol.toUpperCase()}.`,
  };
}

export default async function ReplayPage({ params }: ReplayPageProps) {
  const symbol = params.symbol;
  const snapshot = await fetchSimulatedPlan(symbol).catch(() => null);

  return <ReplayClient symbol={symbol} initialSnapshot={snapshot} />;
}
