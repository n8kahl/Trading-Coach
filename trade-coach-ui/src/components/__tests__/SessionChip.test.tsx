import React from "react";
import { afterEach, describe, expect, test } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import SessionChip from "../SessionChip";
import { useStore } from "@/store/useStore";
import { act } from "@testing-library/react";

describe("SessionChip", () => {
  afterEach(() => {
    cleanup();
    act(() => {
      useStore.getState().reset();
    });
  });

  test("returns null when session is missing", () => {
    const { container } = render(<SessionChip />);
    expect(container.firstChild).toBeNull();
  });

  test("renders session label and timing", () => {
    act(() => {
      useStore.setState({
        session: {
          status: "open",
          as_of: "2025-01-02T14:30:00Z",
          next_open: null,
          tz: "America/New_York",
        },
      });
    });

    render(<SessionChip />);

    expect(screen.getByText("RTH")).toBeInTheDocument();
    expect(screen.getByText(/As of/i)).toBeInTheDocument();
  });
});
