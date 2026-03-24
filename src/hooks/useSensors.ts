import { useEffect, useRef, useState } from 'react';
import {
  accelerometer, gyroscope, magnetometer,
  setUpdateIntervalForType, SensorTypes, SensorData
} from 'react-native-sensors';

export type RateOption = { hz: number; ms: number };
export type Vec3 = { x: number; y: number; z: number };
export type HzState = { acc: number; gyr: number; mag: number };

export type Sample = {
  ax: number; ay: number; az: number;
  gx: number; gy: number; gz: number;
  mx: number; my: number; mz: number;
};

/**
 * Hook untuk mengurus sampling 3 sensor dan mengekspose:
 *  - nilai tampilan (accDisp/gyrDisp/magDisp)
 *  - estimasi Hz tiap sensor (low-pass)
 *  - getLastSample(): ambil snapshot sample terbaru (untuk dipakai predictor)
 * Interval sampling diatur dari prop `rate`.
 */
export function useSensors(rate: RateOption) {
  const [hz, setHz] = useState<HzState>({ acc: 0, gyr: 0, mag: 0 });
  const [accDisp, setAccDisp] = useState<Vec3>({ x: 0, y: 0, z: 0 });
  const [gyrDisp, setGyrDisp] = useState<Vec3>({ x: 0, y: 0, z: 0 });
  const [magDisp, setMagDisp] = useState<Vec3>({ x: 0, y: 0, z: 0 });

  const lastTs = { acc: useRef(0), gyr: useRef(0), mag: useRef(0) };
  const lastAccRef = useRef<SensorData | null>(null);
  const lastGyrRef = useRef<SensorData | null>(null);
  const lastMagRef = useRef<SensorData | null>(null);

  useEffect(() => {
    setUpdateIntervalForType(SensorTypes.accelerometer, rate.ms);
    setUpdateIntervalForType(SensorTypes.gyroscope,     rate.ms);
    setUpdateIntervalForType(SensorTypes.magnetometer,  rate.ms);
  }, [rate.ms]);

  useEffect(() => {
    const updHz = (key: 'acc' | 'gyr' | 'mag', now: number) => {
      const last = lastTs[key].current;
      const dt = now - last;
      if (last && dt >= 1) {
        const inst = 1000 / dt;
        setHz(prev => ({ ...prev, [key]: +(0.8 * prev[key] + 0.2 * inst).toFixed(1) }));
      }
      lastTs[key].current = now;
    };

    const subAcc = accelerometer.subscribe(v => { lastAccRef.current = v; setAccDisp(v as any); updHz('acc', Date.now()); });
    const subGyr = gyroscope    .subscribe(v => { lastGyrRef.current = v; setGyrDisp(v as any); updHz('gyr', Date.now()); });
    const subMag = magnetometer .subscribe(v => { lastMagRef.current = v; setMagDisp(v as any); updHz('mag', Date.now()); });

    return () => { subAcc.unsubscribe(); subGyr.unsubscribe(); subMag.unsubscribe(); };
  }, []);

  const getLastSample = (): Sample => ({
    ax: lastAccRef.current?.x ?? 0,
    ay: lastAccRef.current?.y ?? 0,
    az: lastAccRef.current?.z ?? 0,
    gx: lastGyrRef.current?.x ?? 0,
    gy: lastGyrRef.current?.y ?? 0,
    gz: lastGyrRef.current?.z ?? 0,
    mx: lastMagRef.current?.x ?? 0,
    my: lastMagRef.current?.y ?? 0,
    mz: lastMagRef.current?.z ?? 0,
  });

  return { hz, accDisp, gyrDisp, magDisp, getLastSample };
}
