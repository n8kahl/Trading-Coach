import { ensureCanonicalChartUrl, isCanonicalChartUrl } from "../chartUrl";

describe("chartUrl", () => {
  it("accepts canonical /tv URLs", () => {
    const url = "https://trading-coach-production.up.railway.app/tv?plan_id=ABC";
    expect(isCanonicalChartUrl(url)).toBe(true);
    expect(ensureCanonicalChartUrl(url)).toBe(url);
  });

  it("rejects non-https URLs", () => {
    const url = "http://trading-coach-production.up.railway.app/tv";
    expect(isCanonicalChartUrl(url)).toBe(false);
    expect(ensureCanonicalChartUrl(url)).toBeNull();
  });

  it("rejects non-canonical hosts", () => {
    const url = "https://example.com/tv";
    expect(isCanonicalChartUrl(url)).toBe(false);
    expect(ensureCanonicalChartUrl(url)).toBeNull();
  });

  it("rejects non-tv paths", () => {
    const url = "https://trading-coach-production.up.railway.app/other";
    expect(isCanonicalChartUrl(url)).toBe(false);
    expect(ensureCanonicalChartUrl(url)).toBeNull();
  });
});
