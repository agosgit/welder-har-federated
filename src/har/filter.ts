// lfilter y[n] = (b*x - a*y) rekursif (a[0] = 1)
export function lfilter(b: number[], a: number[], x: number[]): number[] {
  const y = new Array(x.length).fill(0);
  for (let n = 0; n < x.length; n++) {
    let acc = 0;
    for (let i = 0; i < b.length; i++) if (n - i >= 0) acc += b[i] * x[n - i];
    for (let j = 1; j < a.length; j++) if (n - j >= 0) acc -= a[j] * y[n - j];
    y[n] = acc / a[0];
  }
  return y;
}

// filtfilt sederhana (forward lalu reverse) — cocok untuk 1 window statis
export function filtfilt(b: number[], a: number[], x: number[]): number[] {
  const forward = lfilter(b, a, x);
  const rev = forward.slice().reverse();
  const backward = lfilter(b, a, rev).reverse();
  return backward;
}
