import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";

const { makeBackoffMock } = vi.hoisted(() => ({
  makeBackoffMock: vi.fn(() => vi.fn(() => 1000)),
}));

vi.mock("../useBackoff", () => ({
  makeBackoff: makeBackoffMock,
}));

import { useSymbolStream } from "../useSymbolStream";

class MockEventSource {
  static instances: MockEventSource[] = [];
  public onopen: (() => void) | null = null;
  public onmessage: ((event: { data: string }) => void) | null = null;
  public onerror: (() => void) | null = null;
  public closed = false;

  constructor(public url: string) {
    MockEventSource.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  triggerOpen() {
    this.onopen?.();
  }

  triggerMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  triggerError() {
    this.onerror?.();
  }

  static reset() {
    MockEventSource.instances = [];
  }
}

describe("useSymbolStream", () => {
  const originalEventSource = globalThis.EventSource;

  beforeEach(() => {
    vi.useFakeTimers();
    makeBackoffMock.mockClear();
    globalThis.EventSource = MockEventSource as unknown as typeof EventSource;
  });

  afterEach(() => {
    vi.useRealTimers();
    globalThis.EventSource = originalEventSource;
    MockEventSource.reset();
    vi.restoreAllMocks();
  });

  it("replays ticks and reconnects with exponential backoff", () => {
    const timeoutSpy = vi.spyOn(window, "setTimeout");
    const onTick = vi.fn();
    const { result, unmount } = renderHook(() => useSymbolStream("https://example/stream", onTick));

    expect(result.current).toBe("connecting");
    expect(makeBackoffMock).toHaveBeenCalledTimes(1);

    const source = MockEventSource.instances[0];
    act(() => {
      source.triggerOpen();
    });

    expect(result.current).toBe("connected");
    expect(makeBackoffMock.mock.calls.length).toBeGreaterThanOrEqual(2);

    act(() => {
      source.triggerMessage({ t: "tick", p: 123.45 });
    });

    expect(onTick).toHaveBeenCalledWith({ t: "tick", p: 123.45 });

    act(() => {
      source.triggerError();
    });

    expect(result.current).toBe("disconnected");
    expect(timeoutSpy).toHaveBeenLastCalledWith(expect.any(Function), 1000);
    expect(source.closed).toBe(true);

    act(() => {
      vi.runOnlyPendingTimers();
    });

    expect(MockEventSource.instances).toHaveLength(2);
    unmount();
  });
});
