const DEFAULT_API = process.env.NEXT_PUBLIC_API_BASE_URL ?? process.env.TRADE_COACH_API_BASE_URL ?? "http://localhost:8000";

export const API_BASE_URL = DEFAULT_API.replace(/\/$/, "");

const explicitWsBase = process.env.NEXT_PUBLIC_WS_BASE_URL ?? process.env.TRADE_COACH_WS_BASE_URL;

export const WS_BASE_URL = explicitWsBase
  ? explicitWsBase.replace(/\/$/, "")
  : API_BASE_URL.replace(/^http/i, (match) => (match.toLowerCase() === "https" ? "wss" : "ws"));

export const API_KEY_HEADER = process.env.NEXT_PUBLIC_BACKEND_API_KEY ?? process.env.TRADE_COACH_API_KEY ?? "";

const DEFAULT_PUBLIC_BASE =
  process.env.NEXT_PUBLIC_PUBLIC_BASE_URL ?? process.env.PUBLIC_BASE_URL ?? process.env.NEXT_PUBLIC_BASE_URL ?? API_BASE_URL;

export const PUBLIC_BASE_URL = DEFAULT_PUBLIC_BASE.replace(/\/$/, "");

const DEFAULT_UI_BASE =
  process.env.NEXT_PUBLIC_UI_BASE_URL ??
  process.env.PUBLIC_UI_BASE_URL ??
  process.env.NEXT_PUBLIC_UI_HOST ??
  process.env.PUBLIC_UI_HOST ??
  PUBLIC_BASE_URL;

export const PUBLIC_UI_BASE_URL = DEFAULT_UI_BASE.replace(/\/$/, "");

export const BUILD_SHA =
  (process.env.NEXT_PUBLIC_BUILD_SHA as string | undefined) ||
  (process.env.VERCEL_GIT_COMMIT_SHA as string | undefined) ||
  (process.env.RAILWAY_GIT_COMMIT_SHA as string | undefined) ||
  (process.env.GIT_COMMIT as string | undefined) ||
  "";

export function withAuthHeaders(headers: HeadersInit = {}): HeadersInit {
  if (!API_KEY_HEADER) {
    return headers;
  }
  return {
    ...headers,
    Authorization: `Bearer ${API_KEY_HEADER}`,
  };
}

// Scenario Plans UI ships without a feature flag.
