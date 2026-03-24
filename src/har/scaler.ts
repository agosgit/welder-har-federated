export type Scaler = { mean:number[]; scale:number[] };
export function applyScaler(vec:number[], s:Scaler) {
  const y = new Array(vec.length);
  for (let i=0;i<vec.length;i++) {
    const mu = s.mean[i] ?? 0;
    const sc = (s.scale[i] ?? 1) || 1;
    y[i] = (vec[i] - mu) / sc;
  }
  return y;
}
