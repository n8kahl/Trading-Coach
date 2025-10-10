import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function safeJsonStringify(value: unknown, space = 2) {
  try {
    return JSON.stringify(value, null, space);
  } catch (error) {
    return "{\"error\": \"Unable to stringify\"}";
  }
}
