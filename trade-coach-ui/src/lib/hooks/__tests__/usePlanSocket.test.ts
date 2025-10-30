import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";

const { makeBackoffMock } = vi.hoisted(() => ({
  makeBackoffMock: vi.fn(() => vi.fn(() => 1000)),
}));

vi.mock("../useBackoff", () => ({
  makeBackoff: makeBackoffMock,
}));

import { usePlanSocket } from "../usePlanSocket";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  public onopen: (() => void) | null = null;
  public onmessage: ((event: { data: string }) => void) | null = null;
  public onerror: (() => void) | null = null;
  public onclose: (() => void) | null = null;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  close() {
    this.onclose?.();
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
    MockWebSocket.instances = [];
  }
}

describe("usePlanSocket", () => {
  const originalWebSocket = globalThis.WebSocket;

  beforeEach(() => {
    vi.useFakeTimers();
    makeBackoffMock.mockClear();
    globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket;
  });

  afterEach(() => {
    vi.useRealTimers();
    globalThis.WebSocket = originalWebSocket;
    MockWebSocket.reset();
    vi.restoreAllMocks();
  });

  it("streams plan deltas and retries with backoff on failure", () => {
    const timeoutSpy = vi.spyOn(window, "setTimeout");
    const onDelta = vi.fn();
    const { result, unmount } = renderHook(() => usePlanSocket("ws://example", onDelta));

    expect(result.current).toBe("connecting");
    expect(makeBackoffMock).toHaveBeenCalledTimes(1);

    const socket = MockWebSocket.instances[0];
    act(() => {
      socket.triggerOpen();
    });

    expect(result.current).toBe("connected");
    expect(makeBackoffMock.mock.calls.length).toBeGreaterThanOrEqual(2);

    act(() => {
      socket.triggerMessage({ t: "plan_delta", changes: { status: "open" } });
    });

    expect(onDelta).toHaveBeenCalledWith({ t: "plan_delta", changes: { status: "open" } });

    act(() => {
      socket.triggerError();
    });

    expect(result.current).toBe("disconnected");
    expect(timeoutSpy).toHaveBeenLastCalledWith(expect.any(Function), 1000);
    act(() => {
      vi.runOnlyPendingTimers();
    });

    expect(MockWebSocket.instances).toHaveLength(2);
    unmount();
  });
});
