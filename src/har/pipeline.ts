// // src/har/pipeline.ts

// import { InferenceSession, Tensor } from 'onnxruntime-react-native';
// import RNFS from 'react-native-fs';
// import { Platform } from 'react-native';
// import { type Window } from './features'; // cukup type-nya saja
// import { sendLocalModel } from './flClient'; 
// import performance from 'react-native-performance';
// // lstm_har_model2
// // ----- KONFIG -----
// // const MODEL_FILE = 'cnn_har_model2.onnx';
// // const MODEL_FILE = 'cnn_lstm_har_model2.onnx';
// const MODEL_FILE = 'lstm_har_model2.onnx';

// let session: InferenceSession | null = null;
// let currentModel: InferenceSession | null = null;

// async function ensureSession() {
//   if (session) return session!;
//   if (currentModel) return currentModel!;
//   let uri = MODEL_FILE;

//   if (Platform.OS === 'android') {
//     const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;
//     try {
//       if (await RNFS.exists(dest)) {
//         await RNFS.unlink(dest); // hapus file lama
//       }
//       await RNFS.copyFileAssets(MODEL_FILE, dest);
//       console.log('✅ Copied model to:', dest);
//       uri = 'file://' + dest;
//     } catch {
//       uri = `asset:///${MODEL_FILE}`;
//     }
//   }

//   session = await InferenceSession.create(uri);
//   console.log('📦 Loaded model from:', uri);
//   console.log('🧠 Input names:', session.inputNames);
//   console.log('🎯 Output names:', session.outputNames);
//   return session!;
// }

// export async function loadSession() {
//   return ensureSession();
// }

// // 🔹 CNN: input raw [1, 100, 9]
// export async function predictWindow(win: Window): Promise<{ classId: number; conf: number; probs: number[] }> {
//   const sess = await ensureSession();

//   // 1️⃣ gabungkan data sensor
//   const rawData = [
//     ...win.accel_x, ...win.accel_y, ...win.accel_z,
//     ...win.gyro_x,  ...win.gyro_y,  ...win.gyro_z,
//     ...win.mag_x,   ...win.mag_y,   ...win.mag_z,
//   ];

//   // 2️⃣ pastikan total data = 100×9 = 900
//   if (rawData.length !== 900) {
//     throw new Error(`Invalid window shape: got ${rawData.length}, expected 900 (100x9)`);
//   }

//   // 3️⃣ buat tensor [1, 100, 9]
//   const tensor = new Tensor('float32', Float32Array.from(rawData), [1, 100, 9]);

//   // 4️⃣ jalankan inference
//   const out = await sess.run({ input: tensor });
//   const firstOutputName = sess.outputNames[0];
//   const probT = out[firstOutputName];
//   const probs = Array.from(probT.data as Float32Array);


//   // 🔸 Kirim hasil probabilitas ke server federated (asinkron) 
//   try {
//     await sendLocalModel(probs);
//     console.log('📡 Sent probabilities to FL server');
//   } catch (err) {
//     console.warn('⚠️ Failed to send weights to FL server:', err);
//   }

//   // 5️⃣ ambil prediksi tertinggi
//   let imax = 0, pmax = probs[0];
//   for (let i = 1; i < probs.length; i++) {
//     if (probs[i] > pmax) {
//       pmax = probs[i];
//       imax = i;
//     }
//   }

//   return { classId: imax, conf: pmax, probs };
// }

// // opsional: benchmark model
// export async function benchmarkModel(iterations = 30) {
//   const sess = await ensureSession();

//   const dummyData = new Float32Array(100 * 9).fill(0);
//   const tensor = new Tensor('float32', dummyData, [1, 100, 9]);

//   const times: number[] = [];
//   for (let i = 0; i < iterations; i++) {
//     const t0 = performance.now();
//     await sess.run({ input: tensor });
//     const t1 = performance.now();
//     times.push(t1 - t0);
//   }

//   const mean = times.reduce((a, b) => a + b, 0) / times.length;
//   const sd = Math.sqrt(times.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) / times.length);
//   return { mean, sd, iterations };
// }
// export async function getModelWeights(): Promise<number[]> {
//   await ensureSession();
//   // Sementara: kirim dummy float array untuk uji federated learning
//   const dummyWeights = Array.from({ length: 100 }, () => Math.random());
//   return dummyWeights;
// }

// export async function reloadOnnxModel(path?: string) {
//   const modelPath = path ?? `${RNFS.DocumentDirectoryPath}/cnn_lstm_har_model2.onnx`;
//   console.log("🧠 Memuat ulang model dari:", modelPath);

//   // Buat ulang session ONNX baru
//   const newSession = await InferenceSession.create(modelPath);
//   session = newSession; // 🔥 ganti session aktif yang dipakai untuk prediksi
//   currentModel = newSession;

//   console.log("✅ Model ONNX berhasil dimuat ulang dan session aktif diperbarui");
//   console.log("🧩 Aktifkan model versi:", modelPath.split("_v").pop()?.split(".")[0]);

// }

// export function getCurrentModel() {
//   return currentModel;
// }
  


//src//har//pipeline.ts

import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';

// === fitur, scaler ===
import featureOrderJson from '../../assets/ml/feature_order.json';
import scalerParams from '../../assets/ml/scaler.json';
import { extractWindowFeatures, type Window, type ExtractResult } from './features';
import { applyScaler } from './scaler';
import performance from 'react-native-performance';
// ----- KONFIG -----
// const MODEL_FILE = 'lgbm_har_model3.onnx'; // ganti jika namamu beda (mis. 'lgbm_har_model2.onnx')
const MODEL_FILE = 'svm_har_model4.onnx'; // ganti jika namamu beda (mis. 'lgbm_har_model2.onnx')

const FEATURE_ORDER: string[] = (featureOrderJson as any).feature_order;
const SCALER = scalerParams as { mean: number[]; scale: number[] };

let session: InferenceSession | null = null;

async function ensureSession() {
  if (session) return session!;
  let uri = MODEL_FILE;

  if (Platform.OS === 'android') {
    const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;
    try {
      if (!(await RNFS.exists(dest))) await RNFS.copyFileAssets(MODEL_FILE, dest);
      uri = 'file://' + dest;
    } catch {
      uri = `asset:///${MODEL_FILE}`;
    }
  }
  session = await InferenceSession.create(uri);
  return session!;
}

export async function loadSession() {
  return ensureSession();
}

// Kembalikan: classId, conf (max prob), dan probs[] untuk semua kelas
export async function predictWindow(win: Window): Promise<{ classId:number; conf:number; probs:number[] }> {
  const sess = await ensureSession();

  // 1) fitur -> urutan training
  const feats = extractWindowFeatures(win);
  const x = FEATURE_ORDER.map(k => feats[k] ?? 0);

  // 2) cek & scale (harus sama dgn training)
  if (x.length !== SCALER.mean.length || x.length !== SCALER.scale.length) {
    throw new Error(`Feature length mismatch: x=${x.length}, mean=${SCALER.mean.length}, scale=${SCALER.scale.length}`);
  }
  const xNorm = applyScaler(x, SCALER);

  // 3) feed ke ONNX
  const tensor = new Tensor('float32', Float32Array.from(xNorm), [1, xNorm.length]);
  const out = await sess.run({ input: tensor }, ['probabilities']); // <- nama fix
  const probT = out['probabilities'];
  const probs = Array.from(probT.data as Float32Array);

  // 4) argmax
  let imax = 0, pmax = probs[0];
  for (let i = 1; i < probs.length; i++) if (probs[i] > pmax) { pmax = probs[i]; imax = i; }

  return { classId: imax, conf: pmax, probs };
}
export async function benchmarkModel(iterations = 30) {
  const sess = await ensureSession();

  // Contoh window dummy (isi nol), bisa ganti dengan window nyata dari sampler
  const dummyWin: Window = {
    accel_x: Array(100).fill(0),
    accel_y: Array(100).fill(0),
    accel_z: Array(100).fill(0),
    gyro_x:  Array(100).fill(0),
    gyro_y:  Array(100).fill(0),
    gyro_z:  Array(100).fill(0),
    mag_x:   Array(100).fill(0),
    mag_y:   Array(100).fill(0),
    mag_z:   Array(100).fill(0),
  };

  const feats = FEATURE_ORDER.map(() => Math.random()); // fitur dummy
  const xNorm = applyScaler(feats, SCALER);
  const tensor = new Tensor('float32', Float32Array.from(xNorm), [1, xNorm.length]);

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    await sess.run({ input: tensor }, ['probabilities']);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const mean =
    times.reduce((a, b) => a + b, 0) / times.length;
  const sd = Math.sqrt(
    times.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) /
    times.length
  );

  return { mean, sd, iterations };
}