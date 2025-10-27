import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import PlanDetail from '../PlanDetail';
import type { PlanSnapshot, StructuredPlan } from '@/lib/types';

describe('PlanDetail', () => {
  const basePlan: PlanSnapshot['plan'] = {
    plan_id: 'PLAN-1',
    symbol: 'AAPL',
    entry: 100,
    stop: 98,
    targets: [103, 105],
    rr_to_t1: 1.6,
    confidence: 0.72,
    session_state: {
      status: 'open',
      banner: 'Market open',
      as_of: '2025-01-01T14:30:00Z',
    },
    target_meta: [
      {
        label: 'TP1',
        price: 103,
        prob_touch_calibrated: 0.62,
        em_fraction: 0.45,
        snap_tag: 'VAH',
      },
      {
        label: 'TP2',
        price: 105,
        prob_touch: 0.5,
      },
    ],
    badges: [
      { label: 'Power Hour Continuation', kind: 'strategy' },
      { label: 'Power Hour', kind: 'strategy' },
      { label: 'Intraday', kind: 'style' },
    ],
    expected_duration: {
      minutes: 95,
      label: 'intraday ~1–2h',
      basis: ['ATR', 'Distance'],
      inputs: { interval: '5m' },
    },
    strategy_profile: {
      name: 'Power Hour Continuation',
      trigger: ['Late session expansion'],
      invalidation: 'Close below VWAP',
      management: 'Trim into prior high/low liquidity; trail remainder.',
      reload: 'Avoid re-entries after 3:45pm ET.',
      runner: 'Tighten to 0.6× ATR once TP1 hits.',
      badges: ['Power Hour'],
    },
    risk_block: {
      rr_to_tp1: 1.6,
      atr_stop_multiple: 1.2,
    },
    execution_rules: {
      trigger: 'Break above VAH',
    },
    options_contracts: [],
    rejected_contracts: [],
    source_paths: {
      entry: 'geometry_engine',
      stop: 'geometry_engine',
      tp1: 'geometry_engine',
    },
  } as PlanSnapshot['plan'];

  const structured: StructuredPlan = {
    plan_id: 'PLAN-1',
    symbol: 'AAPL',
    style: 'intraday',
    direction: 'long',
    entry: { type: 'limit', level: 100 },
    stop: 98,
    targets: [103, 105],
    runner: null,
    chart_url: null,
    as_of: '2025-01-01T14:30:00Z',
    badges: basePlan.badges,
    strategy_profile: basePlan.strategy_profile,
    expected_duration: basePlan.expected_duration,
  };

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ plan_id: 'PLAN-1', levels: [], zones: [], annotations: [] }),
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders badges in given order', async () => {
    render(<PlanDetail plan={basePlan} structured={structured} planId={basePlan.plan_id} />);

    const badges = await screen.findAllByText(/Power Hour/i);
    expect(badges[0]).toHaveTextContent('Power Hour Continuation');
    expect(badges[1]).toHaveTextContent('Power Hour');
    expect(screen.getByText('Expected Duration')).toBeInTheDocument();
    expect(screen.getByText('intraday ~1–2h')).toBeInTheDocument();
    expect(screen.getByText(/~95 min/i)).toBeInTheDocument();
  });

  it('shows warning banner when overlays respond with 409', async () => {
    vi.unstubAllGlobals();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: async () => ({ message: 'Plan overlays are stale.' }),
    }));

    render(<PlanDetail plan={basePlan} structured={structured} planId={basePlan.plan_id} />);

    await waitFor(() => {
      expect(screen.getByText(/overlays are stale/i)).toBeInTheDocument();
    });
  });
});
