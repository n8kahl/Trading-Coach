import { ensureCanonicalChartUrl, isCanonicalChartUrl } from "../chartUrl";

describe("chartUrl", () => {
  it("accepts canonical /chart URLs", () => {
    const url = "https://trading-coach-production.up.railway.app/chart?plan_id=ABC";
    expect(isCanonicalChartUrl(url)).toBe(true);
    expect(ensureCanonicalChartUrl(url)).toBe(url);
  });

  it("rejects non-https URLs", () => {
    const url = "http://trading-coach-production.up.railway.app/chart";
    expect(isCanonicalChartUrl(url)).toBe(false);
    expect(ensureCanonicalChartUrl(url)).toBeNull();
  });

  it("rejects non-canonical hosts", () => {
    const url = "https://example.com/chart";
    expect(isCanonicalChartUrl(url)).toBe(false);
    expect(ensureCanonicalChartUrl(url)).toBeNull();
  });

  it("rejects non-chart paths", () => {
    const url = "https://trading-coach-production.up.railway.app/other";
    expect(isCanonicalChartUrl(url)).toBe(false);
    expect(ensureCanonicalChartUrl(url)).toBeNull();
  });
});
