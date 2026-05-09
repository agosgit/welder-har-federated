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
import { SafeAreaView } from "react-native-safe-area-context";

// === Sampler (realtime window) & Predictor (ONNX)
import { RealtimeSampler, type Window as MLWindow } from './src/har/sampler';
// PAKAI salah satu import ini sesuai file kamu:
// import { loadSession, predictWindow } from './src/har/predictor';
import { loadSession, predictWindow, benchmarkModel } from './src/har/pipeline';

import labelsJson from './assets/ml/labels.json';

import FeedbackSection from './src/components/FeedbackSection';

import DashboardScreen from "./src/screens/DashboardScreen";

import LiveSensorCharts from './src/components/LiveSensorCharts';

// import { performance } from 'react-native-performance';
import performance from 'react-native-performance';
import { getLocalSampleCount } from './src/har/localDataset';
import { trainLocalModelNative, exportLocalModelWeights } from './src/har/localTrainer';
import { requestGlobalModel, getCurrentModelVersion, sendLocalModel, setNetworkStatsListener, predictViaCloud } from './src/har/flClient';
import { ensureFLAssets } from "./src/har/bootstrapFLAssets";

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
    <SafeAreaView style={styles.safeArea}>
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
    </SafeAreaView>
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
  const accRef = useRef({ x: 0, y: 0, z: 0 });
  const gyrRef = useRef({ x: 0, y: 0, z: 0 });
  const magRef = useRef({ x: 0, y: 0, z: 0 });
  const hzRef = useRef({ acc: 0, gyr: 0, mag: 0 });
  const [latency, setLatency] = useState({
    buffering: 0,
    preprocessing: 0,
    inference: 0,
    decision: 0,
    total: 0,
  });
  const [localSampleCount, setLocalSampleCount] = useState(0);
  const [lastTrainAt, setLastTrainAt] = useState<string | null>(null);
  const [trainingLocal, setTrainingLocal] = useState(false);
  const [syncingModel, setSyncingModel] = useState(false);
  // Tambahkan state ini
  const [netLatency, setNetLatency] = useState({
    upload: 0,
    download: 0,
    load: 0
  });
  const [isCloudMode, setIsCloudMode] = useState(false);
  const [cloudLatency, setCloudLatency] = useState({
    server: 0,
    network: 0
  });
  // Pasang listener saat layar dimuat
  useEffect(() => {
    setNetworkStatsListener((stats) => {
      setNetLatency(prev => ({
        ...prev,
        download: stats.downloadMs,
        load: stats.loadMs
      }));
    });
  }, []);
  useEffect(() => {
    (async () => {
      try {
        await ensureFLAssets();
        await loadSession();
        setModelReady(true);
      } catch (e: any) {
        setModelReady(false);
        Alert.alert("Init gagal", String(e?.message ?? e));
      }
    })();
  }, []);

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
      setAccDisp({ ...accRef.current });
      setGyrDisp({ ...gyrRef.current });
      setMagDisp({ ...magRef.current });
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
    // ================= VALIDASI =================
    if (!(modelReady && inferOn && rate.hz === 50)) {
      samplerRef.current?.stop();
      samplerRef.current = null;

      setPredText("-");
      setProbs(null);

      return;
    }

    // stop sampler sebelumnya
    samplerRef.current?.stop();

    // ================= SETTINGS =================
    const fs = 50;
    const windowSec = 2;
    const overlap = 0.75;

    // latency akibat window overlap
    const bufferingMs =
      windowSec * (1 - overlap) * 1000;

    // guard supaya inferensi tidak bertumpuk
    let inferBusy = false;

    // ================= SAMPLER =================
    samplerRef.current = new RealtimeSampler({
      fs,
      windowSec,
      overlap,

      // ==================================================
      // WINDOW CALLBACK
      // ==================================================
      onWindow: async (win: MLWindow) => {

        // skip jika inferensi sebelumnya belum selesai
        if (inferBusy) return;

        inferBusy = true;

        try {
          setCurrentWindow(win);

          // ================= PREPROCESSING =================
          const tPre0 = performance.now();

          // preprocessing bisa ditambahkan di sini:
          // - normalization
          // - filtering
          // - FFT
          // - feature extraction

          const processed = win;

          const preprocessingMs =
            performance.now() - tPre0;

          // ================= INFERENCE =================
          const tInf0 = performance.now();

          const {
            classId,
            conf,
            probs
          } = await predictWindow(processed);

          const inferenceMs =
            performance.now() - tInf0;

          // ================= DECISION =================
          const tDec0 = performance.now();

          // validasi index class
          if (
            classId < 0 ||
            classId >= CLASS_NAMES.length
          ) {
            throw new Error(
              `Invalid classId: ${classId}`
            );
          }

          const idx = classId;

          const decisionMs =
            performance.now() - tDec0;

          // ================= LATENCY =================

          // latency komputasi
          const computeMs =
            preprocessingMs +
            inferenceMs +
            decisionMs;

          // total latency realtime
          const totalMs =
            bufferingMs +
            computeMs;

          setLatency({
            buffering: bufferingMs,
            preprocessing: preprocessingMs,
            inference: inferenceMs,
            decision: decisionMs,
            total: totalMs,
          });

          // ================= OUTPUT =================
          setPredText(
            `${CLASS_NAMES[idx]} (${(conf * 100).toFixed(1)}%)`
          );

          setProbs(probs);

        } catch (e) {
          console.error("Inference error:", e);

          setPredText("-");
          setProbs(null);

        } finally {
          inferBusy = false;
        }
      },

      // ==================================================
      // REALTIME HZ
      // ==================================================
      onHz: (h) => {
        // simpan ke ref tanpa re-render
        hzRef.current = h;
      },

      // ==================================================
      // REALTIME SENSOR
      // ==================================================
      onLatest: (s) => {
        accRef.current = s.acc;
        gyrRef.current = s.gyr;
        magRef.current = s.mag;
      },
    });

    // ================= START =================
    samplerRef.current.start();

    // ================= CLEANUP =================
    return () => {
      samplerRef.current?.stop();
      samplerRef.current = null;
    };

  }, [modelReady, inferOn, rate.hz]);

  const refreshLocalStats = async () => {
    try {
      const count = await getLocalSampleCount();
      setLocalSampleCount(count);
    } catch (err) {
      console.warn("⚠️ Gagal membaca jumlah sample lokal:", err);
    }
  };

  useEffect(() => {
    refreshLocalStats();
  }, []);
  const probRows = (() => {
    if (!probs || probs.length === 0) return [];
    const pairs = CLASS_NAMES.map((name, i) => ({ name, p: probs[i] ?? 0, i }));
    pairs.sort((a, b) => b.p - a.p);
    return pairs;
  })();

  const handleLocalTrain = async () => {
    try {
      setTrainingLocal(true);

      // 1. train lokal
      const result = await trainLocalModelNative(2, 4);

      if (!result.success) {
        Alert.alert(
          "Training belum jalan",
          result.message ?? `Sample lokal belum cukup. Sekarang baru ${result.sampleCount} sample.`
        );
        return;
      }

      // 2. export flattened weights dari trainer TFLite
      const exported = await exportLocalModelWeights();

      if (!exported.success || !Array.isArray(exported.weights) || exported.weights.length === 0) {
        Alert.alert("Export gagal", "Weights model lokal tidak berhasil diekspor.");
        return;
      }

      // 🟢 TANGKAP UPLOAD LATENCY DI SINI
      const uploadTimeMs = await sendLocalModel(exported.weights) as number;
      
      setNetLatency(prev => ({ ...prev, upload: uploadTimeMs }));

      setLastTrainAt(result.trainedAt ?? new Date().toISOString());
      await refreshLocalStats();

      Alert.alert(
        "Local training selesai",
        `Samples: ${result.sampleCount}\nLoss: ${result.lastLoss ?? "-"}\nWeights sent: ${exported.length}`
      );
    } catch (err: any) {
      Alert.alert("Local training gagal", String(err?.message ?? err));
    } finally {
      setTrainingLocal(false);
    }
  };

  const handleSyncGlobalModel = async () => {
    try {
      setSyncingModel(true);
      await requestGlobalModel();
      Alert.alert("Sync model", "Permintaan model global sudah dikirim. Jika server punya versi baru, model ONNX akan diunduh.");
    } catch (err: any) {
      Alert.alert("Sync gagal", String(err?.message ?? err));
    } finally {
      setSyncingModel(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        style={styles.screen}
        contentContainerStyle={{ paddingBottom: 100 }}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.headerRow}>
          <Text style={styles.title}>Prediksi</Text>
          <Pill
            text={modelReady ? 'Model Ready' : 'Model…'}
            tone={modelReady ? 'good' : 'warn'}
          />
        </View>

        <View style={styles.grid2}>
          <Card
            title="Realtime Inference"
            footer={
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Text style={{ color: '#334155', marginRight: 8 }}>Aktif</Text>
                <Switch
                  value={inferOn}
                  onValueChange={setInferOn}
                  disabled={!modelReady}
                />
              </View>
            }
          >
            {rate.hz !== 50 && (
              <Text
                style={{
                  color: '#9A5B00',
                  backgroundColor: '#FFF4E5',
                  padding: 8,
                  borderRadius: 8,
                  marginBottom: 8,
                }}
              >
                Disarankan 50 Hz agar prediksi akurat.
              </Text>
            )}

            <View style={styles.controlsRow}>
              <View style={styles.pickerBox}>
                <Text style={styles.label}>Rate</Text>
                <Picker
                  selectedValue={rate.hz}
                  onValueChange={(v) => setRate(RATES.find(r => r.hz === v)!)}
                >
                  {RATES.map(r => (
                    <Picker.Item key={r.hz} label={`${r.hz} Hz`} value={r.hz} />
                  ))}
                </Picker>
              </View>
            </View>

            <View style={styles.actionsRow}>
              <PrimaryButton
                title="Benchmark"
                onPress={runBenchmark}
                disabled={!modelReady}
              />
            </View>

            <View style={styles.validationBox}>
              <Text style={[styles.validationText, { fontWeight: '700' }]}>
                Prediksi
              </Text>
              <Text style={[styles.validationText, { marginTop: 4 }]}>
                {inferOn && modelReady && rate.hz === 50 ? (predText || '-') : '— nonaktif —'}
              </Text>

              {probRows.length > 0 && (
                <View style={{ marginTop: 8 }}>
                  {probRows.map((row) => (
                    <View
                      key={row.i}
                      style={{
                        flexDirection: 'row',
                        justifyContent: 'space-between',
                        paddingVertical: 2,
                      }}
                    >
                      <Text style={{ color: '#0F172A' }}>{row.name}</Text>
                      <Text style={{ color: '#334155' }}>
                        {(row.p * 100).toFixed(1)}%
                      </Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          </Card>

          <FeedbackSection
            currentWindowData={currentWindow}
            onSaved={refreshLocalStats}
          />
          <Card title="Federated Learning">
            <View style={{ marginTop: 8 }}>
              <Text style={styles.validationText}>
                Model version: {getCurrentModelVersion() ?? "-"}
              </Text>
              <Text style={styles.validationText}>
                Local samples: {localSampleCount}
              </Text>
              <Text style={styles.validationText}>
                Last local train: {lastTrainAt ?? "-"}
              </Text>
            </View>

            <View style={styles.actionsRow}>
              <PrimaryButton
                title={trainingLocal ? "Training..." : "Train Local Model"}
                onPress={handleLocalTrain}
                disabled={trainingLocal}
              />
              <GhostButton
                title={syncingModel ? "Syncing..." : "Sync Global Model"}
                onPress={handleSyncGlobalModel}
                disabled={syncingModel}
              />
            </View>
          </Card>

          <Card title="Model Performance">
            <View style={{ marginTop: 8 }}>
              <Text style={styles.cardKicker}>Realtime Inference</Text>
              <Text>Buffering: {latency.buffering.toFixed(1)} ms</Text>
              <Text>Preprocessing: {latency.preprocessing.toFixed(2)} ms</Text>
              <Text>Inference: {latency.inference.toFixed(2)} ms</Text>
              <Text>Decision: {latency.decision.toFixed(2)} ms</Text>
              <Text style={{ fontWeight: 'bold', marginBottom: 12 }}>
                Total: {latency.total.toFixed(2)} ms
              </Text>

              <View style={{ height: 1, backgroundColor: '#E2E8F0', marginVertical: 8 }} />

              <Text style={styles.cardKicker}>Network & FL Sync</Text>
              <Text>Upload Weights: {netLatency.upload > 0 ? `${netLatency.upload.toFixed(2)} ms` : '-'}</Text>
              <Text>Download Model: {netLatency.download > 0 ? `${netLatency.download.toFixed(2)} ms` : '-'}</Text>
              <Text>Load to Memory: {netLatency.load > 0 ? `${netLatency.load.toFixed(2)} ms` : '-'}</Text>
            </View>
          </Card>

          <Card title="Live Sensors">
            <View style={styles.cardsRow}>
              <View style={styles.sensorCol}>
                <Text style={styles.cardKicker}>
                  Accelerometer · {hz.acc.toFixed(1)} Hz
                </Text>
                <Text style={styles.cardVal}>x: {fmt(accDisp.x)}</Text>
                <Text style={styles.cardVal}>y: {fmt(accDisp.y)}</Text>
                <Text style={styles.cardVal}>z: {fmt(accDisp.z)}</Text>
              </View>

              <View style={styles.sensorCol}>
                <Text style={styles.cardKicker}>
                  Gyroscope · {hz.gyr.toFixed(1)} Hz
                </Text>
                <Text style={styles.cardVal}>x: {fmt(gyrDisp.x)}</Text>
                <Text style={styles.cardVal}>y: {fmt(gyrDisp.y)}</Text>
                <Text style={styles.cardVal}>z: {fmt(gyrDisp.z)}</Text>
              </View>

              <View style={styles.sensorCol}>
                <Text style={styles.cardKicker}>
                  Magnetometer · {hz.mag.toFixed(1)} Hz
                </Text>
                <Text style={styles.cardVal}>x: {fmt(magDisp.x)}</Text>
                <Text style={styles.cardVal}>y: {fmt(magDisp.y)}</Text>
                <Text style={styles.cardVal}>z: {fmt(magDisp.z)}</Text>
              </View>
            </View>
          </Card>
        </View>
      </ScrollView>
    </SafeAreaView>
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
    <SafeAreaView style={styles.safeArea}>
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
    </SafeAreaView>
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
              // route.name === 'Dashboard' ? (focused ? 'stats-chart' : 'stats-chart-outline') :
              route.name === 'Sampling' ? (focused ? 'pulse' : 'pulse-outline') :
                // route.name === 'Capture' ? (focused ? 'recording' : 'recording-outline') :
                route.name === 'Prediksi' ? (focused ? 'analytics' : 'analytics-outline') :
                  // route.name === 'Input Rows' ? (focused ? 'clipboard' : 'clipboard-outline') :
                  // route.name === 'Prediksi CSV' ? (focused ? 'document-text' : 'document-text-outline') :
                  (focused ? 'folder' : 'folder-outline');
            return <Ionicons name={icon} size={size} color={color} />;
          },
        })}
      >
        {/* <Tab.Screen name="Dashboard" component={DashboardScreen} /> */}
        <Tab.Screen name="Sampling" component={SamplingScreen} />
        {/* <Tab.Screen name="Capture" component={CapturePredictScreen} /> */}
        <Tab.Screen name="Prediksi" component={PredictScreen} />
        {/* <Tab.Screen name="Input Rows" component={RowPredictScreen} /> */}
        {/* <Tab.Screen name="Prediksi CSV" component={CsvPredictScreen} /> */}
        <Tab.Screen name="Files" component={FilesScreen} />
      </Tab.Navigator>

    </NavigationContainer>
  );
}

// ============== Styles ==============
const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F7F9FB',
  },
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
