import { describe, expect, it } from "vitest";
import { parseSupportingLevels, parseTargets, parseUiState } from "@/lib/chart";

describe("parseTargets", () => {
  it("extracts numeric targets from arrays and strings", () => {
    expect(parseTargets([1, "2.5", null, "bad"])).toEqual([1, 2.5]);
    expect(parseTargets("1.25, 2,invalid, , 4.01")).toEqual([1.25, 2, 4.01]);
  });

  it("returns empty array for nullish or invalid input", () => {
    expect(parseTargets(null)).toEqual([]);
    expect(parseTargets({})).toEqual([]);
  });
});

describe("parseSupportingLevels", () => {
  it("parses pipe-delimited entries and applies default labels", () => {
    expect(parseSupportingLevels("123.4|VWAP; 120 |")).toEqual([
      { price: 123.4, label: "VWAP" },
      { price: 120, label: "Level" },
    ]);
  });

  it("drops entries without a numeric price", () => {
    expect(parseSupportingLevels("not-a-number|Text;")).toEqual([]);
  });
});

describe("parseUiState", () => {
  it("falls back to defaults when the payload is missing or invalid", () => {
    expect(parseUiState(null)).toEqual({ session: "live", confidence: 0, style: "unknown" });
    expect(parseUiState("{invalid json")).toEqual({ session: "live", confidence: 0, style: "unknown" });
  });

  it("normalizes tokens and scales confidence values above 1", () => {
    expect(parseUiState(JSON.stringify({ session: "PREMKT", confidence: 82, style: "Swing" }))).toEqual({
      session: "premkt",
      confidence: 0.82,
      style: "swing",
    });
  });

  it("clamps confidence to the [0,1] range", () => {
    expect(parseUiState(JSON.stringify({ confidence: -5, style: "unknown" }))).toEqual({
      session: "live",
      confidence: 0,
      style: "unknown",
    });
  });
});

