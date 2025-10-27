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
      json: async () => ({ interactive: 'https://example.com/chart' }),
    }));

    render(<ScanTable rows={rows} />);

    await waitFor(() => {
      expect(screen.getByText('Open chart')).toBeInTheDocument();
    });

    const rankCell = screen.getByText('1');
    expect(rankCell).toBeInTheDocument();
    const chartLink = screen.getByText('Open chart');
    expect(chartLink).toHaveAttribute('href', 'https://example.com/chart');
    expect(screen.getByText('~78m')).toBeInTheDocument();
  });
});
