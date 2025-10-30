import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import ScanTable from '../ScanTable';

describe('ScanTable', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('preserves server ordering and renders chart link', async () => {
    const rows = [
      {
        plan_id: 'PLAN-1',
        rank: 1,
        symbol: 'AAPL',
        bias: 'long',
        confidence: 0.7,
        setup: 'power_hour_trend',
        plan: {
          expected_duration: {
            minutes: 78,
          },
        },
        charts: {
          params: {
            symbol: 'AAPL',
            interval: '5m',
            direction: 'long',
            entry: 100,
            stop: 98,
            tp: '103,105',
          },
        },
      },
    ];

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ interactive: 'https://trading-coach-production.up.railway.app/tv?plan_id=PLAN-1' }),
    }));

    render(<ScanTable rows={rows} />);

    await waitFor(() => {
      expect(screen.getByText('Open chart')).toBeInTheDocument();
    });

    const rankCell = screen.getByText('1');
    expect(rankCell).toBeInTheDocument();
    const chartLink = screen.getByText('Open chart');
    expect(chartLink).toHaveAttribute('href', 'https://trading-coach-production.up.railway.app/tv?plan_id=PLAN-1');
    expect(screen.getByText('~78m')).toBeInTheDocument();
  });

  it('hides chart link when server returns non-canonical URL', async () => {
    const rows = [
      {
        plan_id: 'PLAN-3',
        rank: 3,
        symbol: 'TSLA',
        charts: {
          params: {
            symbol: 'TSLA',
            interval: '15m',
          },
        },
      },
    ];

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ interactive: 'https://evil.example.com/tv?plan_id=PLAN-3' }),
    }));

    render(<ScanTable rows={rows} />);

    await waitFor(() => {
      expect(screen.getByText('No link')).toBeInTheDocument();
    });
  });

  it('uses structured plan duration when plan is missing it', () => {
    const rows = [
      {
        plan_id: 'PLAN-2',
        rank: 2,
        symbol: 'MSFT',
        setup: 'ema_stack',
        structured_plan: {
          expected_duration: {
            minutes: 42,
          },
        },
      },
    ];

    render(<ScanTable rows={rows} />);

    expect(screen.getByText('~42m')).toBeInTheDocument();
  });
});
