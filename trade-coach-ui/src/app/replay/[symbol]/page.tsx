import type { Metadata } from 'next';
import ReplayClient from './ReplayClient';
import { fetchPlanForSymbol } from '@/lib/api';

type ReplayPageProps = {
  params: Promise<{ symbol: string }>;
};

export async function generateMetadata({ params }: ReplayPageProps): Promise<Metadata> {
  const { symbol } = await params;
  return {
    title: `${symbol.toUpperCase()} · Market Replay`,
    description: `Market Replay with Live and Scenario plans for ${symbol.toUpperCase()}.`,
  };
}

export default async function ReplayPage({ params }: ReplayPageProps) {
  const { symbol } = await params;
  const initialSnapshot = await fetchPlanForSymbol(symbol).catch(() => null);
  const latestPlanId = initialSnapshot?.plan?.plan_id ?? null;

  return (
    <ReplayClient symbol={symbol} initialLivePlanId={latestPlanId} initialSnapshot={initialSnapshot} />
  );
}
