// App.tsx
import 'react-native-gesture-handler';
import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, Platform, Alert, ToastAndroid, Pressable,
  FlatList, RefreshControl, Switch, TextInput, ScrollView 
} from 'react-native';
import { NavigationContainer, DefaultTheme, useFocusEffect } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Picker } from '@react-native-picker/picker';
import RNFS from 'react-native-fs';
import Share from 'react-native-share';
import { NativeModules, NativeEventEmitter } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import {
  accelerometer, gyroscope, magnetometer,
  setUpdateIntervalForType, SensorTypes, type SensorData
} from 'react-native-sensors';
import { pick as pickDoc, types } from '@react-native-documents/picker';

// === Sampler (realtime window) & Predictor (ONNX)
import { RealtimeSampler, type Window as MLWindow } from './src/har/sampler';
// PAKAI salah satu import ini sesuai file kamu:
// import { loadSession, predictWindow } from './src/har/predictor';
import { loadSession, predictWindow, benchmarkModel } from './src/har/pipeline';

// === LABELS (urutan persis LabelEncoder)
import labelsJson from './assets/ml/labels.json';

import FeedbackSection from './src/components/FeedbackSection';

import DashboardScreen from "./src/screens/DashboardScreen";


import LiveSensorCharts from './src/components/LiveSensorCharts';

// import { performance } from 'react-native-performance';
import performance from 'react-native-performance';

const RAW_LABELS: string[] = (labelsJson as any)?.classes ?? [];
const toPretty = (s: string) => s.replace(/_/g, ' ').trim().replace(/\b\w/g, c => c.toUpperCase());
const CLASS_NAMES: string[] = (RAW_LABELS.length ? RAW_LABELS : [
  'berdiri_tidak_aktif', 'berdiri_aktif', 'duduk_tidak_aktif',
  'duduk_aktif', 'berbaring_tidak_aktif', 'berbaring_aktif',
]).map(toPretty);

// === Native recorder (untuk simpan CSV)
type SensorRecorderType = {
  startRecording(rateHz: number, durationSec: number, label: string): Promise<string>;
  stopRecording(): Promise<void>;
  shareLast?: () => Promise<string>;
  sharePath?: (path: string) => Promise<string>;
};
const { SensorRecorder } = NativeModules as { SensorRecorder: SensorRecorderType };

type Label =
  | 'berdiri_tidak_aktif' | 'berdiri_aktif'
  | 'duduk_tidak_aktif' | 'duduk_aktif'
  | 'berbaring_tidak_aktif' | 'berbaring_aktif';

const RATES = [{ hz: 20, ms: 50 }, { hz: 25, ms: 40 }, { hz: 50, ms: 20 }, { hz: 100, ms: 10 }] as const;
const DURATIONS = [5, 6, 8, 10, 12, 120] as const; // capture default 6s

const HAR_DIR = Platform.OS === 'android'
  ? `${RNFS.ExternalDirectoryPath}/har`
  : `${RNFS.DocumentDirectoryPath}/har`;

const appTheme = {
  ...DefaultTheme,
  colors: { ...DefaultTheme.colors, background: '#F7F9FB', primary: '#006D77', card: '#FFFFFF', text: '#0F172A', border: '#E2E8F0' }
};

// ---------- Helpers & UI ----------
const fmt = (n: number) => Number.isFinite(n) ? n.toFixed(3) : '-';
function Pill({ text, tone = 'default' }: { text: string; tone?: 'default' | 'good' | 'warn' }) {
  const bg = tone === 'good' ? '#E7F9ED' : tone === 'warn' ? '#FFF4E5' : '#EAF2FF';
  const fg = tone === 'good' ? '#0B7A3B' : tone === 'warn' ? '#9A5B00' : '#1C4ED8';
  return <View style={[styles.pill, { backgroundColor: bg }]}><Text style={[styles.pillText, { color: fg }]}>{text}</Text></View>;
}
function Card({ title, footer, children }: { title: string; footer?: React.ReactNode; children: React.ReactNode }) {
  return (
    <View style={styles.card}>
      <View style={styles.cardHead}>
        <Text style={styles.cardTitle}>{title}</Text>
        {footer}
      </View>
      {children}
    </View>
  );
}
function PrimaryButton({ title, onPress, disabled }: { title: string; onPress: () => void; disabled?: boolean }) {
  return (
    <Pressable onPress={disabled ? undefined : onPress}
      style={({ pressed }) => [styles.btnPrimary, disabled && styles.btnDisabled, pressed && !disabled && { opacity: 0.8 }]} >
      <Text style={styles.btnPrimaryText}>{title}</Text>
    </Pressable>
  );
}
function GhostButton({ title, onPress, disabled }: { title: string; onPress: () => void; disabled?: boolean }) {
  return (
    <Pressable onPress={disabled ? undefined : onPress}
      style={({ pressed }) => [styles.btnGhost, disabled && styles.btnGhostDisabled, pressed && !disabled && { opacity: 0.6 }]} >
      <Text style={styles.btnGhostText}>{title}</Text>
    </Pressable>
  );
}

let lastInferToast = 0;

// ======================================================
// ===============   TAB 1: SAMPLING   ==================
// (rekam CSV saja + HUD)
// ======================================================
function SamplingScreen() {
  const [running, setRunning] = useState(false);
  const [label, setLabel] = useState<Label>('berdiri_tidak_aktif');
  const [rate, setRate] = useState<typeof RATES[number]>(RATES[2]); // 50Hz
  const [durationSec, setDurationSec] = useState<typeof DURATIONS[number]>(10);
  const [count, setCount] = useState(0);
  const hzRef = useRef({ acc: 0, gyr: 0, mag: 0 });
  const accDispRef = useRef({ x: 0, y: 0, z: 0 });
  const gyrDispRef = useRef({ x: 0, y: 0, z: 0 });
  const magDispRef = useRef({ x: 0, y: 0, z: 0 });
const [hz, setHz] = useState({ acc: 0, gyr: 0, mag: 0 });
const [accDisp, setAccDisp] = useState({ x: 0, y: 0, z: 0 });
const [gyrDisp, setGyrDisp] = useState({ x: 0, y: 0, z: 0 });
const [magDisp, setMagDisp] = useState({ x: 0, y: 0, z: 0 });


  const [lastPath, setLastPath] = useState<string | null>(null);
  const [validation, setValidation] = useState<{ expected: number; actual: number; durationSec: number; effectiveHz: number; deviationPct: number } | null>(null);
  const [manualMode, setManualMode] = useState(false);

  useFocusEffect(React.useCallback(() => {
    const sampler = new RealtimeSampler({
      fs: rate.hz, windowSec: 1, overlap: 0,
      onWindow: () => { },
      onHz: (h) => {
        hzRef.current = h;
      },

      onLatest: (s) => {
        accDispRef.current = s.acc;
        gyrDispRef.current = s.gyr;
        magDispRef.current = s.mag;
      },

    });
    sampler.start();
    return () => sampler.stop();
  }, [rate.hz]));

  useEffect(() => {
    const emitter = new NativeEventEmitter(NativeModules.SensorRecorder);
    const sub1 = emitter.addListener('SensorRecorderProgress', (e: any) => { setCount(e.samples ?? 0); setLastPath(e.path ?? null); });
    const sub2 = emitter.addListener('SensorRecorderComplete', (e: any) => {
      setRunning(false);
      const actual = e.samples ?? 0;
      const expected = e.expected ?? rate.hz * durationSec;
      const dur = Math.max(0.001, e.elapsedSec ?? durationSec);
      setValidation({ expected, actual, durationSec: +dur.toFixed(2), effectiveHz: +(actual / dur).toFixed(2), deviationPct: +(((actual - expected) / Math.max(1, expected)) * 100).toFixed(2) });
    });
    return () => { sub1.remove(); sub2.remove(); };
  }, [rate.hz, durationSec]);
useEffect(() => {
  const interval = setInterval(() => {
    setHz({ ...hzRef.current });
    setAccDisp({ ...accDispRef.current });
    setGyrDisp({ ...gyrDispRef.current });
    setMagDisp({ ...magDispRef.current });
  }, 150); // update UI tiap 150ms (6–7 FPS)

  return () => clearInterval(interval);
}, []);

  const start = async () => {
    if (running) return;
    setValidation(null);
    setCount(0);
    setRunning(true);

    try {
      await RNFS.mkdir(HAR_DIR);

      if (manualMode) {
        // ⚙️ mode manual — kirim -1 ke native biar jalan terus
        const path = await SensorRecorder.startRecording(rate.hz, -1, label);
        setLastPath(path);
      } else {
        // mode otomatis — tetap pakai durasi
        const path = await SensorRecorder.startRecording(rate.hz, durationSec, label);
        setLastPath(path);
      }
    } catch (e: any) {
      setRunning(false);
      try {
        ToastAndroid.show(String(e?.message ?? e), ToastAndroid.LONG);
      } catch {
        Alert.alert('Start failed', String(e?.message ?? e));
      }
    }
  };

  const stop = async () => { try { await SensorRecorder.stopRecording(); } catch { } };
  const canShare = !!lastPath && !running && !!validation;

  return (
    <ScrollView style={styles.screen} contentContainerStyle={{ paddingBottom: 100 }}>
      <View style={styles.headerRow}><Text style={styles.title}>Sampling</Text><Pill text={running ? 'Recording' : 'Idle'} tone={running ? 'good' : 'default'} /></View>
      <View style={styles.grid2}>
        <Card title="Rekam CSV">
          <View style={styles.controlsRow}>
            <View style={styles.pickerBox}><Text style={styles.label}>Rate</Text>
              <Picker enabled={!running} selectedValue={rate.hz} onValueChange={(v) => setRate(RATES.find(r => r.hz === v)!)} >
                {RATES.map(r => <Picker.Item key={r.hz} label={`${r.hz} Hz`} value={r.hz} />)}
              </Picker>
            </View>
            <View style={styles.pickerBox}>
              <Text style={styles.label}>Mode</Text>
              <Picker
                enabled={!running}
                selectedValue={manualMode ? 'manual' : 'auto'}
                onValueChange={(v) => setManualMode(v === 'manual')}
              >
                <Picker.Item label="Auto (berdasarkan durasi)" value="auto" />
                <Picker.Item label="Manual (start/stop bebas)" value="manual" />
              </Picker>
            </View>

            {!manualMode && (
              <View style={styles.pickerBox}>
                <Text style={styles.label}>Durasi</Text>
                <Picker
                  enabled={!running}
                  selectedValue={durationSec}
                  onValueChange={(v) => setDurationSec(v)}
                >
                  {DURATIONS.map((d) => (
                    <Picker.Item key={d} label={`${d} s`} value={d} />
                  ))}
                </Picker>
              </View>
            )}

            <View style={styles.pickerBoxFull}><Text style={styles.label}>Label Aktivitas</Text>
              <Picker enabled={!running} selectedValue={label} onValueChange={(v) => setLabel(v)}>
                {(['berdiri_tidak_aktif', 'berdiri_aktif', 'duduk_tidak_aktif', 'duduk_aktif', 'berbaring_tidak_aktif', 'berbaring_aktif'] as Label[])
                  .map(l => <Picker.Item key={l} label={toPretty(l)} value={l} />)}
              </Picker>
            </View>
          </View>

          <View style={styles.actionsRow}>
            <PrimaryButton title={running ? 'Stop' : `Start (${durationSec}s)`} onPress={running ? stop : start} />
            <GhostButton title="Share CSV" disabled={!canShare} onPress={async () => {
              try {
                if (!canShare) { ToastAndroid.show('Belum ada file', ToastAndroid.SHORT); return; }
                if (SensorRecorder.shareLast) await SensorRecorder.shareLast();
                else if (lastPath) await Share.open({ url: 'file://' + lastPath, type: 'text/csv', failOnCancel: false });
              } catch (e: any) {
                const msg = String(e?.message ?? e);
                if (!msg.includes('User did not share')) {
                  try { ToastAndroid.show('Share gagal: ' + msg, ToastAndroid.LONG); } catch { Alert.alert('Share gagal', msg); }
                }
              }
            }} />
          </View>

          <Text style={styles.metaText}>Samples: {count} @ {rate.hz}Hz</Text>
          {validation && (
            <View style={styles.validationBox}>
              <Text style={styles.validationText}>
                Actual {validation.actual} vs Expected {validation.expected}{'\n'}
                Durasi {validation.durationSec}s · Effective {validation.effectiveHz} Hz · Δ {validation.deviationPct}%
              </Text>
            </View>
          )}
        </Card>

        <Card title="Live Sensors">
          <View style={styles.cardsRow}>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Accelerometer · {hz.acc.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(accDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(accDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(accDisp.z)}</Text>
            </View>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Gyroscope · {hz.gyr.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(gyrDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(gyrDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(gyrDisp.z)}</Text>
            </View>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Magnetometer · {hz.mag.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(magDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(magDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(magDisp.z)}</Text>
            </View>
          </View>
        </Card>
        <Card title="Live Sensors">
          <LiveSensorCharts
            hz={hz}
            accDisp={accDisp}
            gyrDisp={gyrDisp}
            magDisp={magDisp}
          />
        </Card>

      </View>
    </ScrollView>
  );
}

// ======================================================
// ===========   TAB 2: CAPTURE + PREDIK    =============
// (ambil sample N detik → selesai → predik semua window)
// ======================================================
function CapturePredictScreen() {
  const FS = 50;
  const WIN = 2 * FS;    // 2s
  const STEP = WIN / 2;  // 50% overlap

  const [modelReady, setModelReady] = useState(false);
  const [durationSec, setDurationSec] = useState<typeof DURATIONS[number]>(6);
  const [capturing, setCapturing] = useState(false);
  const [progress, setProgress] = useState(0);

  const [hz, setHz] = useState({ acc: 0, gyr: 0, mag: 0 });
  const [accDisp, setAccDisp] = useState({ x: 0, y: 0, z: 0 });
  const [gyrDisp, setGyrDisp] = useState({ x: 0, y: 0, z: 0 });
  const [magDisp, setMagDisp] = useState({ x: 0, y: 0, z: 0 });

  const [winResults, setWinResults] = useState<Array<{ idx: number; label: string; conf: number }>>([]);
  const [majority, setMajority] = useState<string>('-');
  const [avgProbs, setAvgProbs] = useState<number[] | null>(null);
  const [lastCsvPath, setLastCsvPath] = useState<string | null>(null);

  // last sensor samples
  const lastAccRef = useRef<SensorData | null>(null);
  const lastGyrRef = useRef<SensorData | null>(null);
  const lastMagRef = useRef<SensorData | null>(null);
  const lastTs = { acc: useRef(0), gyr: useRef(0), mag: useRef(0) };
const accHUDRef = useRef({ x: 0, y: 0, z: 0 });
const gyrHUDRef = useRef({ x: 0, y: 0, z: 0 });
const magHUDRef = useRef({ x: 0, y: 0, z: 0 });
const hzHUDRef = useRef({ acc: 0, gyr: 0, mag: 0 });

useEffect(() => {
  const t = setInterval(() => {
    setAccDisp({ ...accHUDRef.current });
    setGyrDisp({ ...gyrHUDRef.current });
    setMagDisp({ ...magHUDRef.current });
  }, 150); // 6–7 FPS

  return () => clearInterval(t);
}, []);

  useEffect(() => {
    (async () => {
      try { await loadSession(); setModelReady(true); }
      catch (e: any) { setModelReady(false); try { ToastAndroid.show(String(e?.message ?? e), ToastAndroid.LONG); } catch { Alert.alert('Load model gagal', String(e?.message ?? e)); } }
    })();
  }, []);
useEffect(() => {
  const t = setInterval(() => {
    setHz({ ...hzHUDRef.current });
  }, 200);
  return () => clearInterval(t);
}, []);

  // HUD sensor live (selalu 50Hz)
  useFocusEffect(React.useCallback(() => {
    setUpdateIntervalForType(SensorTypes.accelerometer, 1000 / FS);
    setUpdateIntervalForType(SensorTypes.gyroscope, 1000 / FS);
    setUpdateIntervalForType(SensorTypes.magnetometer, 1000 / FS);

    const updHz = (key: 'acc' | 'gyr' | 'mag', now: number) => {
      const last = lastTs[key].current; const dt = now - last;
      if (last && dt >= 1) {
        const inst = 1000 / dt;
        hzHUDRef.current[key] = +(0.8 * hzHUDRef.current[key] + 0.2 * inst).toFixed(1);
      }
      lastTs[key].current = now;
    };
const subAcc = accelerometer.subscribe(v => { 
   lastAccRef.current = v; 
   accHUDRef.current = v;
   updHz('acc', Date.now());
});
const subGyr = gyroscope.subscribe(v => { 
   lastGyrRef.current = v; 
   gyrHUDRef.current = v;
   updHz('gyr', Date.now());
});
const subMag = magnetometer.subscribe(v => { 
   lastMagRef.current = v; 
   magHUDRef.current = v;
   updHz('mag', Date.now());
});

    return () => { subAcc.unsubscribe(); subGyr.unsubscribe(); subMag.unsubscribe(); };
  }, []));

  const startCapture = async () => {
    if (!modelReady || capturing) return;

    // buffers
    const A = { x: [] as number[], y: [] as number[], z: [] as number[] };
    const G = { x: [] as number[], y: [] as number[], z: [] as number[] };
    const M = { x: [] as number[], y: [] as number[], z: [] as number[] };

    const target = FS * durationSec;
    setCapturing(true); setProgress(0); setWinResults([]); setMajority('-'); setAvgProbs(null); setLastCsvPath(null);

    // tick 50Hz, ambil snapshot terbaru
    let count = 0;
    const timer = setInterval(() => {
      const ax = lastAccRef.current?.x ?? 0, ay = lastAccRef.current?.y ?? 0, az = lastAccRef.current?.z ?? 0;
      const gx = lastGyrRef.current?.x ?? 0, gy = lastGyrRef.current?.y ?? 0, gz = lastGyrRef.current?.z ?? 0;
      const mx = lastMagRef.current?.x ?? 0, my = lastMagRef.current?.y ?? 0, mz = lastMagRef.current?.z ?? 0;

      A.x.push(ax); A.y.push(ay); A.z.push(az);
      G.x.push(gx); G.y.push(gy); G.z.push(gz);
      M.x.push(mx); M.y.push(my); M.z.push(mz);

      count++; setProgress(count / target);
      if (count >= target) {
        clearInterval(timer);
        void afterCapture(A, G, M);
      }
    }, 1000 / FS);
  };

  const afterCapture = async (
    A: { x: number[]; y: number[]; z: number[] },
    G: { x: number[]; y: number[]; z: number[] },
    M: { x: number[]; y: number[]; z: number[] },
  ) => {
    try {
      // simpan CSV (opsional)
      await RNFS.mkdir(HAR_DIR);
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      const path = `${HAR_DIR}/capture_${ts}.csv`;
      const header = 'ax,ay,az,gx,gy,gz,mx,my,mz\n';
      const lines = A.x.map((_, i) => [A.x[i], A.y[i], A.z[i], G.x[i], G.y[i], G.z[i], M.x[i], M.y[i], M.z[i]].join(','));
      await RNFS.writeFile(path, header + lines.join('\n'), 'utf8');
      setLastCsvPath(path);

      // sliding windows → predict
      const N = A.x.length;
      const windows: MLWindow[] = [];
      for (let start = 0; start + WIN <= N; start += STEP) {
        windows.push({
          accel_x: A.x.slice(start, start + WIN),
          accel_y: A.y.slice(start, start + WIN),
          accel_z: A.z.slice(start, start + WIN),
          gyro_x: G.x.slice(start, start + WIN),
          gyro_y: G.y.slice(start, start + WIN),
          gyro_z: G.z.slice(start, start + WIN),
          mag_x: M.x.slice(start, start + WIN),
          mag_y: M.y.slice(start, start + WIN),
          mag_z: M.z.slice(start, start + WIN),
        });
      }

      const votes = new Array(CLASS_NAMES.length).fill(0);
      const results: Array<{ idx: number; label: string; conf: number }> = [];
      let sumProbs: number[] | null = null;

      for (let i = 0; i < windows.length; i++) {
        // @ts-ignore  (predictWindow bisa mengembalikan probs)
        const { classId, conf, probs } = await predictWindow(windows[i]);
        const idx = (classId >= 0 && classId < CLASS_NAMES.length) ? classId : Math.max(0, Math.min(classId - 1, CLASS_NAMES.length - 1));
        votes[idx] += 1;
        results.push({ idx: i + 1, label: CLASS_NAMES[idx], conf: Number((conf ?? 0) as number) });

        if (Array.isArray(probs) && probs.length) {
          if (!sumProbs) sumProbs = new Array(probs.length).fill(0);
          for (let k = 0; k < probs.length; k++) sumProbs[k] += probs[k] ?? 0;
        }
      }

      setWinResults(results);

      // majority
      let best = 0;
      for (let i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
      const pct = windows.length ? (votes[best] / windows.length) * 100 : 0;
      setMajority(`${CLASS_NAMES[best]}  (${pct.toFixed(0)}% votes)`);

      // avg probs
      if (sumProbs && windows.length) {
        const avg = sumProbs.map(v => v / windows.length);
        setAvgProbs(avg);
      } else {
        setAvgProbs(null);
      }
    } catch (e: any) {
      try { ToastAndroid.show(String(e?.message ?? e), ToastAndroid.LONG); } catch { Alert.alert('Capture error', String(e?.message ?? e)); }
    } finally {
      setCapturing(false);
      setProgress(0);
    }
  };

  return (
    <View style={styles.screen}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Capture → Predik</Text>
        <Pill text={modelReady ? (capturing ? 'Capturing…' : 'Ready') : 'Model…'} tone={modelReady ? (capturing ? 'warn' : 'good') : 'warn'} />
      </View>

      <View style={styles.grid2}>
        <Card title="Ambil Sampel Lalu Predik">
          <View style={styles.controlsRow}>
            <View style={styles.pickerBox}>
              <Text style={styles.label}>Durasi Capture</Text>
              <Picker enabled={!capturing} selectedValue={durationSec} onValueChange={(v) => setDurationSec(v)}>
                {DURATIONS.map(d => <Picker.Item key={d} label={`${d} s`} value={d} />)}
              </Picker>
            </View>
            <View style={[styles.validationBox, { flex: 1 }]}>
              <Text style={styles.validationText}>Progress: {(progress * 100).toFixed(0)}%</Text>
            </View>
          </View>

          <View style={styles.actionsRow}>
            <PrimaryButton title={capturing ? 'Capturing…' : `Start (${durationSec}s @50Hz)`} disabled={!modelReady || capturing} onPress={startCapture} />
            <GhostButton title="Save CSV" disabled={!lastCsvPath} onPress={async () => {
              try {
                if (!lastCsvPath) return;
                await Share.open({ url: 'file://' + lastCsvPath, type: 'text/csv', failOnCancel: false });
              } catch (e: any) {
                if (!String(e?.message || '').includes('User did not share')) {
                  try { ToastAndroid.show('Share gagal', ToastAndroid.LONG); } catch { }
                }
              }
            }} />
          </View>

          <View style={[styles.validationBox, { marginTop: 8 }]}>
            <Text style={[styles.validationText, { fontWeight: '700' }]}>Hasil Mayoritas</Text>
            <Text style={[styles.validationText, { marginTop: 4 }]}>{majority}</Text>
          </View>

          {winResults.length > 0 && (
            <View style={[styles.validationBox, { marginTop: 8 }]}>
              <Text style={[styles.validationText, { fontWeight: '700' }]}>Prediksi per Window (2s)</Text>
              {winResults.map(r => (
                <View key={r.idx} style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 2 }}>
                  <Text style={{ color: '#0F172A' }}>W{r.idx}</Text>
                  <Text style={{ color: '#0F172A' }}>{r.label}</Text>
                  <Text style={{ color: '#334155' }}>{(r.conf * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>
          )}

          {avgProbs && (
            <View style={[styles.validationBox, { marginTop: 8 }]}>
              <Text style={[styles.validationText, { fontWeight: '700' }]}>Rata-rata Probabilitas</Text>
              {CLASS_NAMES.map((name, i) => (
                <View key={i} style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 2 }}>
                  <Text style={{ color: '#0F172A' }}>{name}</Text>
                  <Text style={{ color: '#334155' }}>{((avgProbs[i] ?? 0) * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>
          )}
        </Card>

        <Card title="Live Sensors (HUD)">
          <View style={styles.cardsRow}>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Accelerometer · {hz.acc.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(accDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(accDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(accDisp.z)}</Text>
            </View>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Gyroscope · {hz.gyr.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(gyrDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(gyrDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(gyrDisp.z)}</Text>
            </View>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Magnetometer · {hz.mag.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(magDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(magDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(magDisp.z)}</Text>
            </View>
          </View>
        </Card>
      </View>
    </View>
  );
}

// ======================================================
// ===============   TAB 3: PREDIKSI   ==================
// (realtime continuous)
// ======================================================
function PredictScreen() {
  const [modelReady, setModelReady] = useState(false);
  const [inferOn, setInferOn] = useState(true);
  const [rate, setRate] = useState<typeof RATES[number]>(RATES[2]); // 50Hz
  const [hz, setHz] = useState({ acc: 0, gyr: 0, mag: 0 });
  const [accDisp, setAccDisp] = useState({ x: 0, y: 0, z: 0 });
  const [gyrDisp, setGyrDisp] = useState({ x: 0, y: 0, z: 0 });
  const [magDisp, setMagDisp] = useState({ x: 0, y: 0, z: 0 });
  const [predText, setPredText] = useState<string>('-');
  const [probs, setProbs] = useState<number[] | null>(null);
  const samplerRef = useRef<RealtimeSampler | null>(null);
  const [currentWindow, setCurrentWindow] = useState<MLWindow | null>(null);
  const accRef = useRef({x:0,y:0,z:0});
  const gyrRef = useRef({x:0,y:0,z:0});
  const magRef = useRef({x:0,y:0,z:0});
  const hzRef = useRef({ acc: 0, gyr: 0, mag: 0 });
  const [latency, setLatency] = useState({
    buffering: 0,
    preprocessing: 0,
    inference: 0,
    decision: 0,
    total: 0,
  });


  useEffect(() => {
    (async () => {
      try { await loadSession(); setModelReady(true); }
      catch (e: any) { setModelReady(false); try { ToastAndroid.show(String(e?.message ?? e), ToastAndroid.LONG); } catch { Alert.alert('Load model gagal', String(e?.message ?? e)); } }
    })();
  }, []);
  const runBenchmark = async () => {
    try {
      const { mean, sd, iterations } = await benchmarkModel(30);
      Alert.alert('Benchmark', `CNN-LSTM: ${mean.toFixed(1)} ± ${sd.toFixed(1)} ms (${iterations} iter)`);
    } catch (e: any) {
      Alert.alert('Error', String(e?.message ?? e));
    }
  };
useEffect(() => {
   const t = setInterval(() => {
      setAccDisp({...accRef.current});
      setGyrDisp({...gyrRef.current});
      setMagDisp({...magRef.current});
   }, 150);
   return () => clearInterval(t);
}, []);
useEffect(() => {
   const t = setInterval(() => {
      setHz({ ...hzRef.current });  // update UI 5 FPS
   }, 200);
   return () => clearInterval(t);
}, []);

  useEffect(() => {
    if (!(modelReady && inferOn && rate.hz === 50)) {
      samplerRef.current?.stop(); samplerRef.current = null; setPredText('-'); setProbs(null); return;
    }
    samplerRef.current?.stop();
    samplerRef.current = new RealtimeSampler({
      fs: 50, windowSec: 2, overlap: 0.5,
      onWindow: async (win: MLWindow) => {
  try {
    const bufferingMs = 1000; // window 2s overlap 50%

    // ================= PREPROCESSING =================
    const tPre0 = performance.now();

    const processed = win; // kalau belum ada preprocessing

    const preprocessingMs = performance.now() - tPre0;

    // ================= INFERENCE =================
    const tInf0 = performance.now();

    const { classId, conf, probs } = await predictWindow(processed);

    const inferenceMs = performance.now() - tInf0;

    // ================= DECISION =================
    const tDec0 = performance.now();

    const idx =
      (classId >= 0 && classId < CLASS_NAMES.length)
        ? classId
        : Math.max(0, Math.min(classId - 1, CLASS_NAMES.length - 1));

    const decisionMs = performance.now() - tDec0;

    const totalMs =
      bufferingMs +
      preprocessingMs +
      inferenceMs +
      decisionMs;

    setLatency({
      buffering: bufferingMs,
      preprocessing: preprocessingMs,
      inference: inferenceMs,
      decision: decisionMs,
      total: totalMs,
    });

    setPredText(`${CLASS_NAMES[idx]} (${(conf * 100).toFixed(1)}%)`);
    setProbs(probs);

  } catch (e) {
    setPredText('-');
    setProbs(null);
  }
},



      onHz: (h) => {
    hzRef.current = h;  // simpan di ref, tidak memicu re-render
},

          onLatest: (s) => {
      accRef.current = s.acc;
      gyrRef.current = s.gyr;
      magRef.current = s.mag;
    },

    });
    samplerRef.current.start();
    return () => { samplerRef.current?.stop(); samplerRef.current = null; };
  }, [modelReady, inferOn, rate.hz]);

  const probRows = (() => {
    if (!probs || probs.length === 0) return [];
    const pairs = CLASS_NAMES.map((name, i) => ({ name, p: probs[i] ?? 0, i }));
    pairs.sort((a, b) => b.p - a.p);
    return pairs;
  })();

  return (
    <View style={styles.screen}>
      <View style={styles.headerRow}><Text style={styles.title}>Prediksi</Text><Pill text={modelReady ? 'Model Ready' : 'Model…'} tone={modelReady ? 'good' : 'warn'} /></View>
      <View style={styles.grid2}>
        <Card
          title="Realtime Inference"
          footer={<View style={{ flexDirection: 'row', alignItems: 'center' }}><Text style={{ color: '#334155', marginRight: 8 }}>Aktif</Text><Switch value={inferOn} onValueChange={setInferOn} disabled={!modelReady} /></View>}
        >
          {rate.hz !== 50 && <Text style={{ color: '#9A5B00', backgroundColor: '#FFF4E5', padding: 8, borderRadius: 8, marginBottom: 8 }}>Disarankan 50 Hz agar prediksi akurat.</Text>}
          <View style={styles.controlsRow}>
            <View style={styles.pickerBox}><Text style={styles.label}>Rate</Text>
              <Picker selectedValue={rate.hz} onValueChange={(v) => setRate(RATES.find(r => r.hz === v)!)} >
                {RATES.map(r => <Picker.Item key={r.hz} label={`${r.hz} Hz`} value={r.hz} />)}
              </Picker>
            </View>
          </View>

          <View style={styles.actionsRow}>
            <PrimaryButton title="Benchmark" onPress={runBenchmark} disabled={!modelReady} />
          </View>

          <View style={styles.validationBox}>
            <Text style={[styles.validationText, { fontWeight: '700' }]}>Prediksi</Text>
            <Text style={[styles.validationText, { marginTop: 4 }]}>{inferOn && modelReady && rate.hz === 50 ? (predText || '-') : '— nonaktif —'}</Text>
            {probRows.length > 0 && (
              <View style={{ marginTop: 8 }}>
                {probRows.map((row) => (
                  <View key={row.i} style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 2 }}>
                    <Text style={{ color: '#0F172A' }}>{row.name}</Text>
                    <Text style={{ color: '#334155' }}>{(row.p * 100).toFixed(1)}%</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        </Card>
        <FeedbackSection currentWindowData={currentWindow} />
        <Card title="Model Performance">
          <View style={{ marginTop: 8 }}>
            <Text>Model: SVM</Text>
            <Text>Buffering: {latency.buffering.toFixed(1)} ms</Text>
            <Text>Preprocessing: {latency.preprocessing.toFixed(2)} ms</Text>
            <Text>Inference: {latency.inference.toFixed(2)} ms</Text>
            <Text>Decision: {latency.decision.toFixed(2)} ms</Text>
            <Text style={{ fontWeight: 'bold' }}>
              Total: {latency.total.toFixed(2)} ms
            </Text>
          </View>
        </Card>

        <Card title="Live Sensors">
          <View style={styles.cardsRow}>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Accelerometer · {hz.acc.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(accDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(accDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(accDisp.z)}</Text>
            </View>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Gyroscope · {hz.gyr.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(gyrDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(gyrDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(gyrDisp.z)}</Text>
            </View>
            <View style={styles.sensorCol}>
              <Text style={styles.cardKicker}>Magnetometer · {hz.mag.toFixed(1)} Hz</Text>
              <Text style={styles.cardVal}>x: {fmt(magDisp.x)}</Text><Text style={styles.cardVal}>y: {fmt(magDisp.y)}</Text><Text style={styles.cardVal}>z: {fmt(magDisp.z)}</Text>
            </View>
          </View>
        </Card>
      </View>
    </View>
  );
}
function CsvPredictScreen() {
  const [modelReady, setModelReady] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [rowsOut, setRowsOut] = useState<{ idx: number; startRow: number; endRow: number; label: string; conf?: number }[]>([]);

  const FS = 50;             // sesuai training
  const WIN = 2 * FS;        // 2 detik => 100 baris
  const STEP = WIN / 2;      // 50% overlap

  useEffect(() => {
    (async () => {
      try { await loadSession(); setModelReady(true); }
      catch (e: any) { setModelReady(false); toast('Load model gagal: ' + String(e?.message ?? e)); }
    })();
  }, []);

  const toast = (m: string) => { try { ToastAndroid.show(m, ToastAndroid.LONG); } catch { Alert.alert('Info', m); } };

  // normalisasi angka lokal: "43.143.751" -> "43.143751", koma -> titik
  const toNum = (s: string) => {
    let v = String(s).trim();
    if (!v) return NaN;
    v = v.replace(/,/g, '.');
    const firstDot = v.indexOf('.');
    if (firstDot >= 0) {
      const head = v.slice(0, firstDot + 1);
      const tail = v.slice(firstDot + 1).replace(/\./g, '');
      v = head + tail;
    }
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  };

  const pick = (headers: string[], aliases: string[]) => {
    const lower = headers.map(h => h.toLowerCase());
    for (const a of aliases) {
      const i = lower.indexOf(a.toLowerCase());
      if (i >= 0) return i;
    }
    return -1;
  };

  const parseText = (text: string) => {
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (!lines.length) throw new Error('File kosong.');
    const sep = lines[0].includes(';') ? ';' : ',';

    const headRaw = lines[0].split(sep).map(x => x.trim());
    const hasHeader = headRaw.some(h => /accel_|gyro_|mag_|label|seconds_elapsed|time|no/i.test(h));

    let headers: string[] = [];
    let startLine = 0;
    if (hasHeader) { headers = headRaw; startLine = 1; }
    else {
      headers = ['no', 'time', 'seconds_elapsed', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'label'];
    }

    const idx_ax = pick(headers, ['accel_x_filtered', 'accel_x', 'ax', 'a_x']);
    const idx_ay = pick(headers, ['accel_y_filtered', 'accel_y', 'ay', 'a_y']);
    const idx_az = pick(headers, ['accel_z_filtered', 'accel_z', 'az', 'a_z']);
    const idx_gx = pick(headers, ['gyro_x_filtered', 'gyro_x', 'gyr_x', 'gx']);
    const idx_gy = pick(headers, ['gyro_y_filtered', 'gyro_y', 'gyr_y', 'gy']);
    const idx_gz = pick(headers, ['gyro_z_filtered', 'gyro_z', 'gyr_z', 'gz']);
    const idx_mx = pick(headers, ['mag_x_filtered', 'mag_x', 'mx']);
    const idx_my = pick(headers, ['mag_y_filtered', 'mag_y', 'my']);
    const idx_mz = pick(headers, ['mag_z_filtered', 'mag_z', 'mz']);

    const need = [idx_ax, idx_ay, idx_az, idx_gx, idx_gy, idx_gz, idx_mx, idx_my, idx_mz];
    if (need.some(i => i < 0)) throw new Error('Kolom sensor tidak lengkap. Perlu accel_*, gyro_*, mag_* (boleh _filtered).');

    const ax: number[] = [], ay: number[] = [], az: number[] = [];
    const gx: number[] = [], gy: number[] = [], gz: number[] = [];
    const mx: number[] = [], my: number[] = [], mz: number[] = [];
    for (let li = startLine; li < lines.length; li++) {
      const parts = lines[li].split(sep).map(x => x.trim());
      if (!parts.length) continue;
      const get = (i: number) => toNum(parts[i] ?? '');
      ax.push(get(idx_ax)); ay.push(get(idx_ay)); az.push(get(idx_az));
      gx.push(get(idx_gx)); gy.push(get(idx_gy)); gz.push(get(idx_gz));
      mx.push(get(idx_mx)); my.push(get(idx_my)); mz.push(get(idx_mz));
    }
    const N = ax.length;
    if (!N) throw new Error('Tidak ada data numerik yang bisa dibaca.');
    return { ax, ay, az, gx, gy, gz, mx, my, mz, N };
  };

  const runPredictFromText = async (text: string) => {
    if (!modelReady) { toast('Model belum siap'); return; }
    setMsg(null);
    setRowsOut([]);
    try {
      setRunning(true);
      const s = parseText(text);
      if (s.N < WIN) { setMsg(`Minimal ${WIN} baris (2 detik @50Hz). File berisi ${s.N} baris.`); return; }

      const out: { idx: number; startRow: number; endRow: number; label: string; conf?: number }[] = [];
      let w = 0;
      for (let start = 0; start + WIN <= s.N; start += STEP) {
        const end = start + WIN;
        const win = {
          accel_x: s.ax.slice(start, end),
          accel_y: s.ay.slice(start, end),
          accel_z: s.az.slice(start, end),
          gyro_x: s.gx.slice(start, end),
          gyro_y: s.gy.slice(start, end),
          gyro_z: s.gz.slice(start, end),
          mag_x: s.mx.slice(start, end),
          mag_y: s.my.slice(start, end),
          mag_z: s.mz.slice(start, end),
        };
        const { classId, conf } = await predictWindow(win as any);
        const idx = classId >= 0 ? classId : 0;
        out.push({ idx: w++, startRow: start + 1, endRow: end, label: CLASS_NAMES[idx] ?? `Class ${classId}`, conf });
      }
      setRowsOut(out);
      setMsg(`Selesai. ${out.length} window diprediksi.`);
    } catch (e: any) {
      setMsg(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };
  const isPickerCancel = (e: any) =>
    e?.code === 'DOCUMENT_PICKER_CANCELED' ||   // e.g. Android
    e?.code === 'DOCUMENT_PICKER_CANCELLED' ||  // beberapa varian
    /cancel/i.test(String(e?.message || ''));
  const pickFile = async () => {
    try {
      const res = await pickDoc({
        type: [types.plainText, 'text/csv', 'application/csv', 'application/vnd.ms-excel', types.allFiles],
        allowMultiSelection: false,
        copyTo: 'cachesDirectory',
      });

      const doc = Array.isArray(res) ? res[0] : res;
      const path = (doc as any).fileCopyUri || doc.uri;
      setFileName(doc.name ?? 'selected.csv');

      const text = await RNFS.readFile(path.replace('file://', ''), 'utf8');
      await runPredictFromText(text);
    } catch (e: any) {
      if (isPickerCancel(e)) return;          // ✅ ganti isCancel
      setMsg('Gagal baca file: ' + String(e?.message ?? e));
    }
  };



  return (
    <View style={styles.screen}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Prediksi CSV</Text>
        <Pill text={modelReady ? 'Model Ready' : 'Model…'} tone={modelReady ? 'good' : 'warn'} />
      </View>

      <View style={styles.grid2}>
        <Card title="Pilih File CSV">
          <View style={styles.actionsRow}>
            <PrimaryButton title={running ? 'Processing…' : 'Pilih CSV'} onPress={running ? () => { } : pickFile} disabled={!modelReady || running} />
            {fileName && <Text style={{ color: '#334155' }}>{fileName}</Text>}
          </View>
          {msg && <Text style={{ marginTop: 8, color: '#334155' }}>{msg}</Text>}
        </Card>

        {rowsOut.length > 0 && (
          <Card title="Hasil Per Window">
            <FlatList
              data={rowsOut}
              keyExtractor={(it) => String(it.idx)}
              ItemSeparatorComponent={() => <View style={{ height: 6 }} />}
              renderItem={({ item }) => (
                <View style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 }}>
                  <Text style={{ color: '#0F172A' }}>#{item.idx + 1} • rows {item.startRow}–{item.endRow}</Text>
                  <Text style={{ color: '#334155' }}>
                    {item.label}{typeof item.conf === 'number' ? `  (${(item.conf * 100).toFixed(1)}%)` : ''}
                  </Text>
                </View>
              )}
            />
          </Card>
        )}
      </View>
    </View>
  );
}

// ======================================================
// ===============   TAB 4: FILES    ====================
// ======================================================
type FileItem = { name: string; path: string; size: number; mtime?: Date };
function FilesScreen() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const load = async () => {
    try {
      await RNFS.mkdir(HAR_DIR);
      const list = await RNFS.readDir(HAR_DIR);
      const items: FileItem[] = list
        .filter((x) => x.isFile() && x.name.endsWith('.csv'))
        .map((x) => ({ name: x.name, path: x.path, size: x.size ?? 0, mtime: x.mtime ? new Date(x.mtime) : undefined }))
        .sort((a, b) => (b.mtime?.getTime() || 0) - (a.mtime?.getTime() || 0));
      setFiles(items);
    } catch (e: any) {
      try { ToastAndroid.show(String(e?.message ?? e), ToastAndroid.LONG); } catch { Alert.alert('Gagal membaca folder', String(e?.message ?? e)); }
    }
  };
  useEffect(() => { load(); }, []);

  const onShare = async (f: FileItem) => {
    try {
      if ((NativeModules as any).SensorRecorder?.sharePath) await (NativeModules as any).SensorRecorder.sharePath(f.path);
      else await Share.open({ url: 'file://' + f.path, type: 'text/csv', failOnCancel: false });
    } catch (e: any) {
      if (!String(e?.message || '').includes('User did not share')) {
        try { ToastAndroid.show('Share gagal: ' + String(e?.message ?? e), ToastAndroid.LONG); } catch { Alert.alert('Share gagal', String(e?.message ?? e)); }
      }
    }
  };
  const onDelete = async (f: FileItem) => {
    try { await RNFS.unlink(f.path); await load(); try { ToastAndroid.show('Deleted: ' + f.name, ToastAndroid.SHORT); } catch { } }
    catch (e: any) { try { ToastAndroid.show(String(e?.message ?? e), ToastAndroid.LONG); } catch { Alert.alert('Delete gagal', String(e?.message ?? e)); } }
  };

  return (
    <View style={styles.screen}>
      <Text style={styles.title}>Files</Text>
      <FlatList
        data={files}
        keyExtractor={(it) => it.path}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await load(); setRefreshing(false); }} />}
        ItemSeparatorComponent={() => <View style={{ height: 10 }} />}
        renderItem={({ item }) => (
          <View style={styles.fileRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fileName}>{item.name}</Text>
              <Text style={styles.fileMeta}>{(item.size / 1024).toFixed(1)} KB · {item.mtime ? item.mtime.toLocaleString() : '-'}</Text>
            </View>
            <GhostButton title="Share" onPress={() => onShare(item)} />
            <GhostButton title="Delete" onPress={() => onDelete(item)} />
          </View>
        )}
        ListEmptyComponent={<Text style={{ color: '#64748B' }}>Belum ada file rekaman.</Text>}
        contentContainerStyle={{ paddingVertical: 8, gap: 10 }}
      />
    </View>
  );
}


// === TAB 3: INPUT ROWS (paste baris CSV langsung) ===
function RowPredictScreen() {
  const [modelReady, setModelReady] = useState(false);
  const [raw, setRaw] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [rowsOut, setRowsOut] = useState<{ idx: number; startRow: number; endRow: number; label: string; conf?: number }[]>([]);
  const [msg, setMsg] = useState<string | null>(null);

  const FS = 50;            // training rate
  const WIN = 2 * FS;       // 2 detik => 100 baris
  const STEP = WIN / 2;     // 50% overlap

  useEffect(() => {
    (async () => {
      try { await loadSession(); setModelReady(true); }
      catch (e: any) { setModelReady(false); toast('Load model gagal: ' + String(e?.message ?? e)); }
    })();
  }, []);

  const toast = (m: string) => { try { ToastAndroid.show(m, ToastAndroid.LONG); } catch { Alert.alert('Info', m); } };

  // angka "lokal": ganti koma -> titik; kalau ada banyak titik (43.143.753) jadikan 43.143753
  const toNum = (s: string) => {
    let v = String(s).trim();
    if (!v) return NaN;
    v = v.replace(/,/g, '.');                 // koma -> titik
    const firstDot = v.indexOf('.');
    if (firstDot >= 0) {
      const head = v.slice(0, firstDot + 1);
      const tail = v.slice(firstDot + 1).replace(/\./g, ''); // hapus titik sisanya (anggap ribuan)
      v = head + tail;
    }
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  };

  const colIndex = (headers: string[], aliases: string[]) => {
    const lower = headers.map(h => h.toLowerCase());
    for (const a of aliases) {
      const i = lower.indexOf(a.toLowerCase());
      if (i >= 0) return i;
    }
    return -1;
  };

  const parse = (text: string) => {
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (!lines.length) throw new Error('Tidak ada baris.');
    const sep = lines[0].includes(';') ? ';' : ',';

    // cek header
    const headRaw = lines[0].split(sep).map(x => x.trim());
    const hasHeader = headRaw.some(h =>
      /accel_|gyro_|mag_|label|seconds_elapsed|time|no/i.test(h)
    );

    let headers: string[] = [];
    let startLine = 0;
    if (hasHeader) {
      headers = headRaw;
      startLine = 1;
    } else {
      // fallback: pakai urutan default dari contohmu
      headers = [
        'no', 'time', 'seconds_elapsed',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z', 'label'
      ];
    }

    const idx_ax = colIndex(headers, ['accel_x_filtered', 'accel_x', 'ax', 'a_x']);
    const idx_ay = colIndex(headers, ['accel_y_filtered', 'accel_y', 'ay', 'a_y']);
    const idx_az = colIndex(headers, ['accel_z_filtered', 'accel_z', 'az', 'a_z']);
    const idx_gx = colIndex(headers, ['gyro_x_filtered', 'gyro_x', 'gyr_x', 'gx']);
    const idx_gy = colIndex(headers, ['gyro_y_filtered', 'gyro_y', 'gyr_y', 'gy']);
    const idx_gz = colIndex(headers, ['gyro_z_filtered', 'gyro_z', 'gyr_z', 'gz']);
    const idx_mx = colIndex(headers, ['mag_x_filtered', 'mag_x', 'mx']);
    const idx_my = colIndex(headers, ['mag_y_filtered', 'mag_y', 'my']);
    const idx_mz = colIndex(headers, ['mag_z_filtered', 'mag_z', 'mz']);

    const need = [idx_ax, idx_ay, idx_az, idx_gx, idx_gy, idx_gz, idx_mx, idx_my, idx_mz];
    if (need.some(i => i < 0)) {
      throw new Error('Kolom sensor tidak lengkap. Perlu accel_*, gyro_*, mag_* (boleh _filtered).');
    }

    const ax: number[] = [], ay: number[] = [], az: number[] = [];
    const gx: number[] = [], gy: number[] = [], gz: number[] = [];
    const mx: number[] = [], my: number[] = [], mz: number[] = [];
    for (let li = startLine; li < lines.length; li++) {
      const parts = lines[li].split(sep).map(x => x.trim());
      if (!parts.length) continue;
      const get = (i: number) => toNum(parts[i] ?? '');
      ax.push(get(idx_ax)); ay.push(get(idx_ay)); az.push(get(idx_az));
      gx.push(get(idx_gx)); gy.push(get(idx_gy)); gz.push(get(idx_gz));
      mx.push(get(idx_mx)); my.push(get(idx_my)); mz.push(get(idx_mz));
    }
    const N = ax.length;
    if (!N) throw new Error('Tidak ada data numerik yang bisa dibaca.');
    return { ax, ay, az, gx, gy, gz, mx, my, mz, N, startLine };
  };

  const run = async () => {
    if (!modelReady) { toast('Model belum siap'); return; }
    setMsg(null);
    setRowsOut([]);
    try {
      setRunning(true);
      const s = parse(raw);
      if (s.N < WIN) {
        setMsg(`Butuh minimal ${WIN} baris (2 detik @50Hz). Sekarang hanya ${s.N} baris.`);
        return;
      }
      const out: { idx: number; startRow: number; endRow: number; label: string; conf?: number }[] = [];
      let w = 0;
      for (let start = 0; start + WIN <= s.N; start += STEP) {
        const end = start + WIN;
        const win: MLWindow = {
          accel_x: s.ax.slice(start, end),
          accel_y: s.ay.slice(start, end),
          accel_z: s.az.slice(start, end),
          gyro_x: s.gx.slice(start, end),
          gyro_y: s.gy.slice(start, end),
          gyro_z: s.gz.slice(start, end),
          mag_x: s.mx.slice(start, end),
          mag_y: s.my.slice(start, end),
          mag_z: s.mz.slice(start, end),
        };
        const { classId, conf } = await predictWindow(win);
        const idx = (classId >= 0 && classId < CLASS_NAMES.length)
          ? classId
          : Math.max(0, Math.min(classId - 1, CLASS_NAMES.length - 1));
        out.push({
          idx: w++,
          startRow: start + 1,            // +1 biar 1-based
          endRow: end,
          label: CLASS_NAMES[idx],
          conf,
        });
      }
      setRowsOut(out);
      setMsg(`Selesai. ${out.length} window diprediksi.`);
    } catch (e: any) {
      setMsg(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <View style={styles.screen}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Input Rows</Text>
        <Pill text={modelReady ? 'Model Ready' : 'Model…'} tone={modelReady ? 'good' : 'warn'} />
      </View>

      <View style={styles.grid2}>
        <Card title="Paste baris CSV (pakai ; atau ,)">
          <TextInput
            multiline
            value={raw}
            onChangeText={setRaw}
            placeholder={
              'no;time;seconds_elapsed;accel_x;accel_y;accel_z;gyro_x;gyro_y;gyro_z;mag_x;mag_y;mag_z;label\n' +
              '0;...;0.113647;0.073315;0.140247;0.06725;-0.191675;-0.180263;0.1804;139.125;-20.643.751;214.125;Duduk Tidak Aktif\n' +
              '... (≥ 100 baris total)'
            }
            style={{
              minHeight: 160, borderWidth: 1, borderColor: '#E2E8F0',
              borderRadius: 12, padding: 10, textAlignVertical: 'top', backgroundColor: '#fff'
            }}
          />

          <View style={[styles.actionsRow, { marginTop: 10 }]}>
            <PrimaryButton title={running ? 'Processing…' : 'Prediksi Rows'} onPress={running ? () => { } : run} disabled={!raw.trim() || running || !modelReady} />
            <GhostButton title="Clear" onPress={() => { setRaw(''); setRowsOut([]); setMsg(null); }} />
          </View>

          {msg && <Text style={{ marginTop: 8, color: '#334155' }}>{msg}</Text>}
        </Card>

        {rowsOut.length > 0 && (
          <Card title="Hasil Per Window">
            <FlatList
              data={rowsOut}
              keyExtractor={(it) => String(it.idx)}
              ItemSeparatorComponent={() => <View style={{ height: 6 }} />}
              renderItem={({ item }) => (
                <View style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 }}>
                  <Text style={{ color: '#0F172A' }}>#{item.idx + 1} • rows {item.startRow}–{item.endRow}</Text>
                  <Text style={{ color: '#334155' }}>
                    {item.label}{typeof item.conf === 'number' ? `  (${(item.conf * 100).toFixed(1)}%)` : ''}
                  </Text>
                </View>
              )}
            />
          </Card>
        )}
      </View>
    </View>
  );
}

// ======================================================
// ===============     NAVIGATOR     ====================
// ======================================================
const Tab = createBottomTabNavigator();
export default function App() {
  return (
    <NavigationContainer theme={appTheme}>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarActiveTintColor: appTheme.colors.primary,
          tabBarStyle: { backgroundColor: '#FFFFFF', borderTopColor: '#E2E8F0', height: 58, paddingBottom: 8 },
          tabBarLabelStyle: { fontSize: 12 },
          tabBarIcon: ({ color, size, focused }) => {
            const icon =
              route.name === 'Dashboard' ? (focused ? 'stats-chart' : 'stats-chart-outline') :
                route.name === 'Sampling' ? (focused ? 'pulse' : 'pulse-outline') :
                  route.name === 'Capture' ? (focused ? 'recording' : 'recording-outline') :
                    route.name === 'Prediksi' ? (focused ? 'analytics' : 'analytics-outline') :
                      route.name === 'Input Rows' ? (focused ? 'clipboard' : 'clipboard-outline') :
                        route.name === 'Prediksi CSV' ? (focused ? 'document-text' : 'document-text-outline') :
                          (focused ? 'folder' : 'folder-outline');
            return <Ionicons name={icon} size={size} color={color} />;
          },
        })}
      >
        <Tab.Screen name="Dashboard" component={DashboardScreen} />
        <Tab.Screen name="Sampling" component={SamplingScreen} />
        <Tab.Screen name="Capture" component={CapturePredictScreen} />
        <Tab.Screen name="Prediksi" component={PredictScreen} />
        <Tab.Screen name="Input Rows" component={RowPredictScreen} />
        <Tab.Screen name="Prediksi CSV" component={CsvPredictScreen} />
        <Tab.Screen name="Files" component={FilesScreen} />
      </Tab.Navigator>

    </NavigationContainer>
  );
}

// ============== Styles ==============
const styles = StyleSheet.create({
  screen: { flex: 1, padding: 16 },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  title: { fontSize: 20, fontWeight: '700', color: '#0F172A' },
  grid2: { gap: 12 },
  card: { backgroundColor: '#FFFFFF', borderRadius: 16, padding: 14, borderWidth: 1, borderColor: '#E2E8F0' },
  cardHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  cardTitle: { fontSize: 16, fontWeight: '700', color: '#0F172A' },
  cardKicker: { fontWeight: '600', marginBottom: 6, color: '#334155' },
  sensorCol: { flex: 1, gap: 2, paddingRight: 8 },
  cardsRow: { flexDirection: 'row', gap: 12 },
  label: { fontSize: 12, color: '#64748B', marginLeft: 8, marginTop: 6 },
  pickerBox: { flex: 1, borderWidth: 1, borderColor: '#E2E8F0', borderRadius: 12, backgroundColor: '#fff' },
  pickerBoxFull: { borderWidth: 1, borderColor: '#E2E8F0', borderRadius: 12, backgroundColor: '#fff', marginTop: 8 },
  actionsRow: { flexDirection: 'row', gap: 10, marginTop: 10, alignItems: 'center' },
  btnPrimary: { backgroundColor: '#006D77', paddingVertical: 12, paddingHorizontal: 18, borderRadius: 12 },
  btnPrimaryText: { color: '#fff', fontWeight: '700' },
  btnDisabled: { opacity: 0.45 },
  btnGhost: { borderWidth: 1, borderColor: '#CBD5E1', paddingVertical: 10, paddingHorizontal: 14, borderRadius: 12, backgroundColor: '#fff' },
  btnGhostText: { color: '#0F172A', fontWeight: '600' },
  btnGhostDisabled: { opacity: 0.45 },
  pill: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  pillText: { fontSize: 12, fontWeight: '700' },
  metaText: { marginTop: 6, color: '#475569' },
  validationBox: { padding: 10, borderWidth: 1, borderRadius: 12, borderColor: '#E2E8F0', backgroundColor: '#F8FAFC', marginTop: 8 },
  validationText: { fontSize: 12, color: '#334155' },
  fileRow: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#fff', borderRadius: 14, borderWidth: 1, borderColor: '#E2E8F0', padding: 12 },
  fileName: { fontWeight: '700', color: '#0F172A' },
  fileMeta: { color: '#64748B', fontSize: 12 },
  cardVal: { fontVariant: ['tabular-nums'], color: '#0F172A' },
  controlsRow: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginTop: 8, marginBottom: 8 },
});
