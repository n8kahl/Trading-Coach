import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { usePriceSeries } from "../usePriceSeries";

describe("usePriceSeries", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("loads price bars and merges subsequent reloads", async () => {
    const firstPayload = {
      s: "ok",
      t: [1000, 1010],
      o: [1, 2],
      h: [2, 3],
      l: [0.5, 1.5],
      c: [1.5, 2.5],
      v: [100, 120],
    };
    const secondPayload = {
      s: "ok",
      t: [1010, 1020],
      o: [2, 3],
      h: [3, 4],
      l: [1.5, 2.5],
      c: [2.6, 3.6],
      v: [130, 140],
    };

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => firstPayload,
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => secondPayload,
      } as Response);

    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { result, rerender } = renderHook(({ token }: { token: number }) => usePriceSeries("SPY", "1", ["SPY", token]), {
      initialProps: { token: 0 },
    });

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.bars).toHaveLength(2);
    expect(result.current.bars[1]).toMatchObject({ time: 1010, close: 2.5 });

    act(() => {
      rerender({ token: 1 });
    });

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.bars).toHaveLength(3);
    const mergedTimes = result.current.bars.map((bar) => Number(bar.time));
    expect(mergedTimes).toEqual([1000, 1010, 1020]);
    expect(result.current.bars[1]).toMatchObject({ time: 1010, close: 2.6 });
  });
});
