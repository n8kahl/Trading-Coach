import React from "react";
import { afterEach, describe, expect, test } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { act } from "@testing-library/react";
import ObjectiveProgress from "../ObjectiveProgress";
import { useStore } from "@/store/useStore";

describe("ObjectiveProgress", () => {
  afterEach(() => {
    cleanup();
    act(() => {
      useStore.getState().reset();
    });
  });

  test("returns null when no objective present", () => {
    act(() => {
      useStore.setState({
        plan: null,
        planLayers: null,
      });
    });
    const { container } = render(<ObjectiveProgress />);
    expect(container.firstChild).toBeNull();
  });

  test("renders objective details when available", () => {
    act(() => {
      useStore.setState({
        plan: {
          plan_id: "alpha",
          symbol: "AAPL",
          session_state: {
            status: "open",
            banner: "",
            as_of: "2025-01-02T14:30:00Z",
            next_open: null,
          },
          details: {},
        } as any,
        planLayers: {
          plan_id: "alpha",
          symbol: "AAPL",
          interval: "5m",
          as_of: "2025-01-02T14:30:00Z",
          planning_context: "live",
          precision: 2,
          levels: [],
          zones: [],
          meta: {
            next_objective: {
              state: "arming",
              why: ["MTF:+3/5"],
              objective_price: 430.25,
              band: { low: 430.1, high: 430.4 },
              timeframe: "5m",
              progress: 0.6,
            },
            _next_objective_internal: {
              _tick_size: 0.05,
              _last_price: 430.35,
            },
          },
        },
      });
    });

    render(<ObjectiveProgress />);

    expect(screen.getByText(/Progress 60%/i)).toBeInTheDocument();
    expect(screen.getByText(/Band/)).toBeInTheDocument();
  });
});
