import { API_BASE as ENV_API_BASE } from './hosts';

let override: string | null = null;

export function setRuntimeApiBase(url?: string) {
  override = url ? url.replace(/\/+$/, '') : null;
}

export function getApiBase() {
  return override ?? ENV_API_BASE;
}

