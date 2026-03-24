export const mean = (v:number[]) => v.reduce((s,x)=>s+x,0)/v.length;
export const std = (v:number[]) => {
  const m = mean(v);
  return Math.sqrt(v.reduce((s,x)=>s+(x-m)*(x-m),0)/v.length);
};
export const percentile = (v:number[], p:number) => {
  const a = v.slice().sort((x,y)=>x-y);
  const idx = (p/100) * (a.length-1);
  const i0 = Math.floor(idx), i1 = Math.ceil(idx);
  if (i0===i1) return a[i0];
  const t = idx - i0;
  return a[i0]*(1-t) + a[i1]*t;
};
export const iqr = (v:number[]) => percentile(v,75)-percentile(v,25);
export const median = (v:number[]) => percentile(v,50);
export const mad = (v:number[]) => {
  const m = mean(v);
  return mean(v.map(x=>Math.abs(x-m)));
};
export const energy = (v:number[]) => v.reduce((s,x)=>s+x*x,0)/v.length;

// histogram entropy
export function entropy(v:number[], bins=20): number {
  const min = Math.min(...v), max = Math.max(...v);
  const w = (max-min) || 1;
  const edges = [...Array(bins+1)].map((_,i)=>min + (i*w)/bins);
  const h = new Array(bins).fill(0);
  for (const x of v) {
    let k = Math.min(bins-1, Math.max(0, Math.floor(((x-min)/w)*bins)));
    h[k] += 1;
  }
  const total = v.length || 1;
  let ent = 0;
  for (const c of h) {
    if (!c) continue;
    const p = c/total;
    ent -= p * Math.log(p + 1e-12);
  }
  return ent;
}

// DFT magnitude (N=100 kecil; O(N^2) masih ok)
export function dftMag(v:number[]): { mag:number[]; freqs:number[] } {
  const N = v.length, mag = new Array(Math.floor(N/2)).fill(0);
  const freqs = mag.map((_,k)=> k * (50/N)); // FS=50, set di caller kalau mau
  for (let k=0;k<mag.length;k++) {
    let re=0, im=0;
    for (let n=0;n<N;n++) {
      const phi = -2*Math.PI*k*n/N;
      re += v[n]*Math.cos(phi);
      im += v[n]*Math.sin(phi);
    }
    mag[k] = Math.hypot(re,im);
  }
  return {mag, freqs};
}

// DCT-II sederhana
export function dct(x: number[]) {
  const N = x.length;
  const out = new Array<number>(N).fill(0);
  const factor0 = Math.sqrt(1 / N);
  const factor  = Math.sqrt(2 / N);
  for (let k = 0; k < N; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += x[n] * Math.cos(Math.PI * (n + 0.5) * k / N);
    }
    out[k] = (k === 0 ? factor0 : factor) * sum;
  }
  return out;
}
