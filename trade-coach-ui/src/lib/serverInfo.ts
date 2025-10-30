import { API_BASE } from './hosts';

export async function fetchServerInfo() {
  try {
    const res = await fetch(`${API_BASE}/version`, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

