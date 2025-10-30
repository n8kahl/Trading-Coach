export function makeBackoff(max = 5, base = 500) {
  let n = 0;
  return () => Math.min(base * 2 ** n++ + Math.random() * 250, 8000);
}

