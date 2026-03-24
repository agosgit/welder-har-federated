//src//har//sampler.ts

import {
  accelerometer,
  gyroscope,
  magnetometer,
  setUpdateIntervalForType,
  SensorTypes,
  SensorData,
} from 'react-native-sensors';

export type Window = {
  accel_x: number[]; accel_y: number[]; accel_z: number[];
  gyro_x:  number[]; gyro_y:  number[]; gyro_z:  number[];
  mag_x:   number[]; mag_y:   number[]; mag_z:   number[];
};

type Latest = { acc: { x: number; y: number; z: number }, gyr: { x: number; y: number; z: number }, mag: { x: number; y: number; z: number } };
type Hz = { acc: number; gyr: number; mag: number };

type Opts = {
  fs: number;              // Hz target sampler
  windowSec: number;       // durasi window
  overlap: number;         // 0..1, contoh 0.5
  onWindow: (win: Window) => void | Promise<void>;
  onHz?: (hz: Hz) => void;
  onLatest?: (latest: Latest) => void;
};

export class RealtimeSampler {
  private opts: Opts;
  private timer: ReturnType<typeof setInterval> | null = null;
  private since = 0;
  private readonly size: number;
  private readonly step: number;

  private lastAcc: SensorData | null = null;
  private lastGyr: SensorData | null = null;
  private lastMag: SensorData | null = null;

  private hz: Hz = { acc: 0, gyr: 0, mag: 0 };
  private lastTs = { acc: 0, gyr: 0, mag: 0 };

  private B: Window = {
    accel_x: [], accel_y: [], accel_z: [],
    gyro_x:  [], gyro_y:  [], gyro_z:  [],
    mag_x:   [], mag_y:   [], mag_z:   [],
  };

  private subs: Array<{ unsubscribe: () => void }> = [];

  constructor(opts: Opts) {
    this.opts = opts;
    this.size = Math.max(1, Math.round(opts.fs * opts.windowSec));
    this.step = Math.max(1, Math.round(this.size * (1 - opts.overlap)));
  }

  start() {
    if (this.timer) return;

    // set interval sensor sesuai fs
    const ms = Math.max(5, Math.round(1000 / this.opts.fs));
    setUpdateIntervalForType(SensorTypes.accelerometer, ms);
    setUpdateIntervalForType(SensorTypes.gyroscope, ms);
    setUpdateIntervalForType(SensorTypes.magnetometer, ms);

    // subscribe sensor, hitung Hz, simpan latest
    const updHz = (key: keyof Hz, now: number) => {
      const last = this.lastTs[key];
      const dt = now - last;
      if (last && dt > 0) {
        const inst = 1000 / dt;
        this.hz[key] = +(0.8 * this.hz[key] + 0.2 * inst).toFixed(1);
        this.opts.onHz?.(this.hz);
      }
      this.lastTs[key] = now;
    };

    this.subs.push(
      accelerometer.subscribe((v) => { this.lastAcc = v; updHz('acc', Date.now()); this.pushLatest(); }),
      gyroscope.subscribe((v) => { this.lastGyr = v; updHz('gyr', Date.now()); this.pushLatest(); }),
      magnetometer.subscribe((v) => { this.lastMag = v; updHz('mag', Date.now()); this.pushLatest(); }),
    );

    // sampler detik-nyata berdasarkan fs
    this.B = {
      accel_x: [], accel_y: [], accel_z: [],
      gyro_x:  [], gyro_y:  [], gyro_z:  [],
      mag_x:   [], mag_y:   [], mag_z:   [],
    };
    this.since = 0;

    this.timer = setInterval(() => this.tick(), ms);
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    this.subs.forEach((s) => s.unsubscribe());
    this.subs = [];
  }

  private pushLatest() {
    this.opts.onLatest?.({
      acc: { x: this.lastAcc?.x ?? 0, y: this.lastAcc?.y ?? 0, z: this.lastAcc?.z ?? 0 },
      gyr: { x: this.lastGyr?.x ?? 0, y: this.lastGyr?.y ?? 0, z: this.lastGyr?.z ?? 0 },
      mag: { x: this.lastMag?.x ?? 0, y: this.lastMag?.y ?? 0, z: this.lastMag?.z ?? 0 },
    });
  }

  private tick() {
    const ax = this.lastAcc?.x ?? 0, ay = this.lastAcc?.y ?? 0, az = this.lastAcc?.z ?? 0;
    const gx = this.lastGyr?.x ?? 0, gy = this.lastGyr?.y ?? 0, gz = this.lastGyr?.z ?? 0;
    const mx = this.lastMag?.x ?? 0, my = this.lastMag?.y ?? 0, mz = this.lastMag?.z ?? 0;

    const K = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','mag_x','mag_y','mag_z'] as const;
    this.B.accel_x.push(ax); this.B.accel_y.push(ay); this.B.accel_z.push(az);
    this.B.gyro_x.push(gx);  this.B.gyro_y.push(gy);  this.B.gyro_z.push(gz);
    this.B.mag_x.push(mx);   this.B.mag_y.push(my);   this.B.mag_z.push(mz);

    // jaga panjang (ring buffer)
    if (this.B.accel_x.length > this.size) {
      K.forEach((k) => this.B[k].splice(0, this.B[k].length - this.size));
    }

    // cukup 1 window?
    if (this.B.accel_x.length >= this.size) {
      this.since += 1;
      if (this.since >= this.step) {
        this.since = 0;
        const start = this.B.accel_x.length - this.size;
        const win: Window = {
          accel_x: this.B.accel_x.slice(start),
          accel_y: this.B.accel_y.slice(start),
          accel_z: this.B.accel_z.slice(start),
          gyro_x:  this.B.gyro_x.slice(start),
          gyro_y:  this.B.gyro_y.slice(start),
          gyro_z:  this.B.gyro_z.slice(start),
          mag_x:   this.B.mag_x.slice(start),
          mag_y:   this.B.mag_y.slice(start),
          mag_z:   this.B.mag_z.slice(start),
        };
        this.opts.onWindow(win);
      }
    }
  }
}
