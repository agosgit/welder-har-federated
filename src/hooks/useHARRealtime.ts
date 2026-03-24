import { useEffect, useRef, useState } from 'react';
import { Platform, Alert, ToastAndroid } from 'react-native';
import { WINDOW_FS, WINDOW_SIZE, STEP_SIZE } from '../har/constants';
import { predictWindow, loadSession } from '../har/pipeline';

export type GetSampleFn = () => {
  ax:number; ay:number; az:number;
  gx:number; gy:number; gz:number;
  mx:number; my:number; mz:number;
};

const notify = (t: string, m?: string) =>
  Platform.OS === 'android'
    ? ToastAndroid.show((m ?? t).toString(), ToastAndroid.SHORT)
    : Alert.alert(t, m);

/**
 * Hook yang mengelola buffering → windowing → inferensi.
 * Ia *hanya* membaca snapshot dari getSample() sehingga sampling terpisah.
 */
export function useHARRealtime(
  enabled: boolean,
  rateHz: number,
  getSample: GetSampleFn
) {
  const [ready, setReady] = useState(false);
  const [predId, setPredId] = useState<number | null>(null);

  // ring buffer 9 channel
  const B = useRef({
    accel_x: [] as number[], accel_y: [] as number[], accel_z: [] as number[],
    gyro_x:  [] as number[], gyro_y:  [] as number[], gyro_z:  [] as number[],
    mag_x:   [] as number[], mag_y:   [] as number[], mag_z:   [] as number[],
  });
  const sinceInfer = useRef(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // load model sekali
  useEffect(() => {
    (async () => {
      try { await loadSession(); setReady(true); }
      catch (e:any) { setReady(false); notify('Gagal load model', String(e?.message ?? e)); }
    })();
  }, []);

  // loop inference terpisah dari sampling
  useEffect(() => {
    const canRun = enabled && ready && rateHz === WINDOW_FS;
    if (!canRun) { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } return; }

    // reset buffer
    B.current = {
      accel_x: [], accel_y: [], accel_z: [],
      gyro_x:  [], gyro_y:  [], gyro_z:  [],
      mag_x:   [], mag_y:   [], mag_z:   [],
    };
    sinceInfer.current = 0;

    timerRef.current = setInterval(async () => {
      const s = getSample();

      // push
      B.current.accel_x.push(s.ax); B.current.accel_y.push(s.ay); B.current.accel_z.push(s.az);
      B.current.gyro_x .push(s.gx); B.current.gyro_y .push(s.gy); B.current.gyro_z .push(s.gz);
      B.current.mag_x  .push(s.mx); B.current.mag_y  .push(s.my); B.current.mag_z  .push(s.mz);

      // trim
      if (B.current.accel_x.length > WINDOW_SIZE) {
        (['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','mag_x','mag_y','mag_z'] as const)
          .forEach(k => B.current[k].splice(0, B.current[k].length - WINDOW_SIZE));
      }

      if (B.current.accel_x.length >= WINDOW_SIZE) {
        sinceInfer.current += 1;
        if (sinceInfer.current >= STEP_SIZE) {
          sinceInfer.current = 0;
          const start = B.current.accel_x.length - WINDOW_SIZE;
          const win = {
            accel_x: B.current.accel_x.slice(start),
            accel_y: B.current.accel_y.slice(start),
            accel_z: B.current.accel_z.slice(start),
            gyro_x:  B.current.gyro_x.slice(start),
            gyro_y:  B.current.gyro_y.slice(start),
            gyro_z:  B.current.gyro_z.slice(start),
            mag_x:   B.current.mag_x.slice(start),
            mag_y:   B.current.mag_y.slice(start),
            mag_z:   B.current.mag_z.slice(start),
          };
          try {
            const { classId } = await predictWindow(win);
            setPredId(classId);
          } catch (e:any) {
            notify('Infer error', String(e?.message ?? e));
          }
        }
      }
    }, 1000 / WINDOW_FS);

    return () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  }, [enabled, ready, rateHz, getSample]);

  return { modelReady: ready, predId };
}
