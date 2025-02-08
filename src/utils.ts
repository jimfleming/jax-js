/** @file Generic programming utilities with no dependencies on library code. */

export function unzip2<T, U>(pairs: [T, U][]): [T[], U[]] {
  const lst1: T[] = [];
  const lst2: U[] = [];
  for (const [x, y] of pairs) {
    lst1.push(x);
    lst2.push(y);
  }
  return [lst1, lst2];
}

export function zip<T, U>(xs: T[], ys: U[]): [T, U][] {
  return xs.map((x, i) => [x, ys[i]]);
}

/** Check if two objects are deep equal. */
export function deepEqual(a: any, b: any): boolean {
  if (a === b) {
    return true;
  }
  if (typeof a !== "object" || typeof b !== "object") {
    return false;
  }
  if (a === null || b === null) {
    return false;
  }
  if (Object.keys(a).length !== Object.keys(b).length) {
    return false;
  }
  for (const key of Object.keys(a)) {
    if (!deepEqual(a[key], b[key])) {
      return false;
    }
  }
  return true;
}

export function range(start: number, stop: number, step: number = 1): number[] {
  const result = [];
  for (let i = start; i < stop; i += step) {
    result.push(i);
  }
  return result;
}
