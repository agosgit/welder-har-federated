// features.ts
import { BUTTER_A, BUTTER_B } from './constants';
import { filtfilt } from './filter';
import { mean, std, iqr, percentile, mad, energy, entropy, median, dftMag, dct } from './math';

export type Window = {
  accel_x: number[]; accel_y: number[]; accel_z: number[];
  gyro_x:  number[]; gyro_y:  number[]; gyro_z:  number[];
  mag_x:   number[]; mag_y:   number[]; mag_z:   number[];
  label?: string; // simpan label di sini saja (jangan dimasukkan ke fitur)
};

export type ExtractResult = { [featureName: string]: number };
export type ExtractOpts   = { preFiltered?: boolean }; // <-- penting

function oneSignalFeatures(
  raw: number[],
  prefix: string,
  out: ExtractResult,
  preFiltered: boolean
) {
  // kalau data sudah difilter (mis. *_filtered dari CSV), pakai langsung
  const f = preFiltered ? raw : filtfilt(BUTTER_B, BUTTER_A, raw);

  out[`${prefix}_filtered_mean`]    = mean(f);
  out[`${prefix}_filtered_std`]     = std(f);
  out[`${prefix}_filtered_iqr`]     = iqr(f);

  // kurtosis (excess)
  const m  = mean(f);
  const s2 = f.reduce((s, x) => s + (x - m) ** 2, 0) / f.length || 1e-12;
  const k  = f.reduce((s, x) => s + (x - m) ** 4, 0) / f.length / (s2 * s2) - 3;
  out[`${prefix}_filtered_kurtosis`] = k;

  out[`${prefix}_filtered_p10`]     = percentile(f, 10);
  out[`${prefix}_filtered_p90`]     = percentile(f, 90);
  out[`${prefix}_filtered_mad`]     = mad(f);
  out[`${prefix}_filtered_min`]     = Math.min(...f);
  out[`${prefix}_filtered_max`]     = Math.max(...f);
  out[`${prefix}_filtered_median`]  = median(f);
  out[`${prefix}_filtered_energy`]  = energy(f);
  out[`${prefix}_filtered_entropy`] = entropy(f);

  // FFT features
  const { mag, freqs } = dftMag(f); // pastikan dftMag mengembalikan { mag, freqs }
  const sumMag = mag.reduce((s, x) => s + x, 0) || 1e-12;
  let kmax = 0;
  for (let i = 1; i < mag.length; i++) if (mag[i] > mag[kmax]) kmax = i;

  out[`${prefix}_filtered_dom_freq`]      = freqs[kmax];
  out[`${prefix}_filtered_spec_entropy`]  = entropy(mag, Math.min(20, mag.length));

  let num = 0;
  for (let i = 0; i < mag.length; i++) num += freqs[i] * mag[i];
  out[`${prefix}_filtered_spec_centroid`] = num / sumMag;

  // DCT
  const dc = dct(f);
  out[`${prefix}_filtered_dct_mean`] = mean(dc);
  out[`${prefix}_filtered_dct_std`]  = std(dc);
}

export function extractWindowFeatures(win: Window, opts: ExtractOpts = {}): ExtractResult {
  const { preFiltered = false } = opts;
  const out: ExtractResult = {};

  // 9 kanal
  oneSignalFeatures(win.accel_x, 'accel_x', out, preFiltered);
  oneSignalFeatures(win.accel_y, 'accel_y', out, preFiltered);
  oneSignalFeatures(win.accel_z, 'accel_z', out, preFiltered);
  oneSignalFeatures(win.gyro_x,  'gyro_x',  out, preFiltered);
  oneSignalFeatures(win.gyro_y,  'gyro_y',  out, preFiltered);
  oneSignalFeatures(win.gyro_z,  'gyro_z',  out, preFiltered);
  oneSignalFeatures(win.mag_x,   'mag_x',   out, preFiltered);
  oneSignalFeatures(win.mag_y,   'mag_y',   out, preFiltered);
  oneSignalFeatures(win.mag_z,   'mag_z',   out, preFiltered);

  // korelasi antar sumbu accelerometer (pakai data yang sudah difilter sekali)
  const fx = preFiltered ? win.accel_x : filtfilt(BUTTER_B, BUTTER_A, win.accel_x);
  const fy = preFiltered ? win.accel_y : filtfilt(BUTTER_B, BUTTER_A, win.accel_y);
  const fz = preFiltered ? win.accel_z : filtfilt(BUTTER_B, BUTTER_A, win.accel_z);

  const corr = (a: number[], b: number[]) => {
    const ma = mean(a), mb = mean(b);
    const da = a.map(v => v - ma);
    const db = b.map(v => v - mb);
    const num = da.reduce((s, v, i) => s + v * db[i], 0);
    const sa  = Math.sqrt(da.reduce((s, v) => s + v*v, 0)) || 1e-12;
    const sb  = Math.sqrt(db.reduce((s, v) => s + v*v, 0)) || 1e-12;
    return num / (sa * sb);
  };

  out['corr_xy'] = corr(fx, fy);
  out['corr_yz'] = corr(fy, fz);
  out['corr_xz'] = corr(fx, fz);

  return out;
}
