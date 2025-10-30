import { describe, it, expect } from 'vitest';
import { parsePlanIdFromMaybeUrl } from '../links';

describe('parsePlanIdFromMaybeUrl', () => {
  it('passes through raw id', () => {
    const id = 'SPY-ABC-123';
    expect(parsePlanIdFromMaybeUrl(id)).toBe(id);
  });

  it('parses plan_id from full tv url', () => {
    const url = 'https://api.example.com/tv?symbol=SPY&plan_id=SPY-ABC-123&interval=5m';
    expect(parsePlanIdFromMaybeUrl(url)).toBe('SPY-ABC-123');
  });

  it('returns null for malformed url', () => {
    expect(parsePlanIdFromMaybeUrl('http://%zz')).toBeNull();
  });
});

