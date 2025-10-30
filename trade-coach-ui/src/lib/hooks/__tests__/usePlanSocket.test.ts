import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";

vi.mock("@/lib/streams/planStream", () => {
  type SocketStatus = "connecting" | "connected" | "disconnected";

  class MockPlanStream {
    static instances: MockPlanStream[] = [];
    public status: SocketStatus = "connecting";
    public heartbeat = Date.now();
    private readonly listeners = new Set<(payload: unknown) => void>();
    private readonly statusListeners = new Set<(status: SocketStatus) => void>();
    private readonly heartbeatListeners = new Set<(ts: number) => void>();

    constructor() {
      MockPlanStream.instances.push(this);
    }

    subscribe(listener: (payload: unknown) => void) {
      this.listeners.add(listener);
      return () => {
        this.listeners.delete(listener);
      };
    }

    onStatus(listener: (status: SocketStatus) => void) {
      this.statusListeners.add(listener);
      listener(this.status);
      return () => {
        this.statusListeners.delete(listener);
      };
    }

    onHeartbeat(listener: (ts: number) => void) {
      this.heartbeatListeners.add(listener);
      listener(this.heartbeat);
      return () => {
        this.heartbeatListeners.delete(listener);
      };
    }

    getStatus() {
      return this.status;
    }

    getLastHeartbeat() {
      return this.heartbeat;
    }

    emit(payload: unknown) {
      this.listeners.forEach((listener) => listener(payload));
    }

    setStatus(status: SocketStatus) {
      this.status = status;
      this.statusListeners.forEach((listener) => listener(status));
    }

    setHeartbeat(ts: number) {
      this.heartbeat = ts;
      this.heartbeatListeners.forEach((listener) => listener(ts));
    }

    static reset() {
      MockPlanStream.instances = [];
    }
  }

  return {
    getPlanStream: vi.fn(() => new MockPlanStream()),
    MockPlanStream,
  };
});

import { MockPlanStream } from "@/lib/streams/planStream";
import { usePlanSocket } from "../usePlanSocket";

describe("usePlanSocket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockPlanStream.reset();
  });

  afterEach(() => {
    vi.useRealTimers();
    MockPlanStream.reset();
    vi.restoreAllMocks();
  });

  it("subscribes to plan stream events and updates status", () => {
    const onDelta = vi.fn();
    const { result, unmount } = renderHook(() => usePlanSocket("ws://example", "PLAN-1", onDelta));

    expect(result.current).toBe("connecting");
    expect(MockPlanStream.instances).toHaveLength(1);
    const stream = MockPlanStream.instances[0];

    act(() => {
      stream.setStatus("connected");
    });

    expect(result.current).toBe("connected");

    act(() => {
      stream.emit({ t: "plan_delta", changes: { status: "open" } });
    });

    expect(onDelta).toHaveBeenCalledWith({ t: "plan_delta", changes: { status: "open" } });

    act(() => {
      stream.setHeartbeat(Date.now() - 40_000);
      vi.advanceTimersByTime(5_000);
    });

    expect(result.current).toBe("connecting");

    act(() => {
      stream.setStatus("disconnected");
    });

    expect(result.current).toBe("disconnected");
    unmount();
  });
});
