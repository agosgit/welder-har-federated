// src/har/pipeline.ts // deep learning
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';
import { type Window } from './features';
import performance from 'react-native-performance';

// const MODEL_FILE = 'cnn_lstm_har_model_mobile.onnx';
const MODEL_FILE = 'cnn_lstm_har_model_mobile.onnx';

let session: InferenceSession | null = null;
let currentModel: InferenceSession | null = null;
const CLIPPING_THRESHOLD = 3.0;

async function ensureSession() {
  if (session) return session;
  if (currentModel) return currentModel;

  let uri = MODEL_FILE;

  if (Platform.OS === 'android') {
    const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;

    try {
      const exists = await RNFS.exists(dest);
      if (!exists) {
        await RNFS.copyFileAssets(MODEL_FILE, dest);
        console.log('✅ Copied model to:', dest);
      } else {
        console.log('✅ ONNX already exists at:', dest);
      }

      uri = 'file://' + dest;
    } catch (err) {
      console.error('❌ Failed copying ONNX asset:', err);
      throw err;
    }
  }

  session = await InferenceSession.create(uri);
  currentModel = session;

  console.log('📦 Loaded model from:', uri);
  console.log('🧠 Input names:', session.inputNames);
  console.log('🎯 Output names:', session.outputNames);

  return session;
}

export async function loadSession() {
  return ensureSession();
}
// Tambahkan di bagian atas file (di bawah let session)
let cachedScaler: { mean: number[], scale: number[] } | null = null;

async function getScalerValues() {
  if (cachedScaler) return cachedScaler; // Ambil dari cache jika sudah ada

  const path = `${RNFS.DocumentDirectoryPath}/models/scaler_nomag.json`;
  try {
    const content = await RNFS.readFile(path, 'utf8');
    const scaler = JSON.parse(content);
    cachedScaler = {
      mean: scaler.mean,
      scale: scaler.scale
    };
    return cachedScaler;
  } catch (err) {
    console.error("Gagal membaca scaler:", err);
    return null;
  }
}


export async function predictWindow(
  win: Window
): Promise<{ classId: number; conf: number; probs: number[] }> {
  const sess = await ensureSession();
  const scaler = await getScalerValues();
  if (!scaler) {
    throw new Error("Scaler not loaded");
  }

  console.log("📊 Scaler Mean Accel_X:", scaler.mean[0], "| Scale:", scaler.scale[0]);

  if (!scaler) throw new Error("Scaler data tidak ditemukan!");

  // 1. Deteksi otomatis apakah scaler dilatih dengan 6 atau 9 fitur
  const numFeatures = scaler.mean.length;
  if (numFeatures !== 6 && numFeatures !== 9) {
    throw new Error(`Scaler memiliki ${numFeatures} fitur, tidak sesuai ekspektasi (6 atau 9).`);
  }

  const interleavedData = [];
  
  for (let i = 0; i < 100; i++) {
    // Ambil data mentah (default 9 fitur)
    let rawRow = [
      win.accel_x[i], win.accel_y[i], win.accel_z[i],
      win.gyro_x[i],  win.gyro_y[i],  win.gyro_z[i]
      // win.mag_x[i],   win.mag_y[i],   win.mag_z[i]
    ];

    // Jika model Python HANYA dilatih dengan 6 fitur, potong array-nya
    if (numFeatures === 6) {
      rawRow = rawRow.slice(0, 6);
    }

    const normalizedRow = rawRow.map((val, idx) => {
      // Safety net untuk NaN/Undefined dari sensor
      if (val === undefined || isNaN(val)) {
        return 0; 
      }
      if (idx < 3) {
        val = val / 9.80665;
      }

      // OPTIONAL: Buka komentar di bawah ini JIKA dataset Python menggunakan satuan "g" bukan "m/s^2"
      // if (idx < 3) { // Hanya index 0, 1, 2 (Accelerometer)
      //   val = val / 9.80665; 
      // }

      // Normalisasi & Clipping
      const zScore = (val - scaler.mean[idx]) / scaler.scale[idx];
      return Math.max(-CLIPPING_THRESHOLD, Math.min(CLIPPING_THRESHOLD, zScore));
    });

    // Debugging krusial untuk baris pertama saja
    if (i === 0) {
      console.log(`[DEBUG] Raw Row 0:`, rawRow);
      console.log(`[DEBUG] Norm Row 0:`, normalizedRow);
    }

    interleavedData.push(...normalizedRow);
  }

  // 2. Shape tensor disesuaikan dinamis dengan jumlah fitur scaler
  const tensor = new Tensor("float32", Float32Array.from(interleavedData), [1, 100, numFeatures]);
  
  const inputName = sess.inputNames[0];
  const outputName = sess.outputNames[0];

  const out = await sess.run({ [inputName]: tensor });

  const probT = out[outputName];
  const probs = Array.from(probT.data as Float32Array);

  let imax = 0;
  let pmax = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > pmax) {
      pmax = probs[i];
      imax = i;
    }
  }

  return { classId: imax, conf: pmax, probs };
}
export async function benchmarkModel(iterations = 30) {
  const sess = await ensureSession();

  // Update dummy data menjadi 9 fitur
  const dummyData = new Float32Array(100 * 6).fill(0);
  const tensor = new Tensor("float32", dummyData, [1, 100, 6]);

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    await sess.run({ input: tensor }); // Pastikan nama input sesuai (default biasanya 'input' atau 'x')
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const sd = Math.sqrt(
    times.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) / times.length
  );

  return { mean, sd, iterations };
}

export async function reloadOnnxModel(path?: string) {
  const modelPath = path ?? `${RNFS.DocumentDirectoryPath}/cnn_lstm_har_model_mobile.onnx`;
  console.log("🧠 Reloading model from:", modelPath);

  const normalizedPath = modelPath.startsWith("file://") ? modelPath : `file://${modelPath}`;
  const newSession = await InferenceSession.create(normalizedPath);

  session = newSession;
  currentModel = newSession;

  console.log("✅ Model reloaded successfully");
}

export function getCurrentModel() {
  return currentModel;
}

// export async function predictWindow(
//   win: Window
// ): Promise<{ classId: number; conf: number; probs: number[] }> {
//   const sess = await ensureSession();
//   const scaler = await getScalerValues();

//   if (!scaler) throw new Error("Scaler data tidak ditemukan!");

//   const interleavedData = [];
  
//   // Looping sebanyak 100 timesteps
//   for (let i = 0; i < 100; i++) {
//     // 1. Ambil baris data mentah (HANYA 6 FITUR, Magnetometer dihapus)
//     const rawRow = [
//       win.accel_x[i], win.accel_y[i], win.accel_z[i],
//       win.gyro_x[i],  win.gyro_y[i],  win.gyro_z[i],
//       win.mag_x[i],   win.mag_y[i],   win.mag_z[i]
//     ];
    
//     // 2. Normalisasi & Clipping yang BENAR
//     const normalizedRow = rawRow.map((val, idx) => {
//       // Hitung Z-Score (Normalisasi) DULU menggunakan scaler dari Python
//       const zScore = (val - scaler.mean[idx]) / scaler.scale[idx];
      
//       // Baru lakukan Clipping (Safety Net) pada hasil normalisasinya
//       return Math.max(-CLIPPING_THRESHOLD, Math.min(CLIPPING_THRESHOLD, zScore));
//     });

//     if (i === 0) console.log("Hasil Normalisasi Pertama (9 Fitur):", normalizedRow);

//     // 3. Masukkan ke array utama
//     interleavedData.push(...normalizedRow);
//   }

//   // Sekarang data sudah [1, 100, 6] 
//   const tensor = new Tensor("float32", Float32Array.from(interleavedData), [1, 100, 9]);
  
//   const inputName = sess.inputNames[0];
//   const outputName = sess.outputNames[0];

//   const out = await sess.run({ [inputName]: tensor });

//   const probT = out[outputName];
//   const probs = Array.from(probT.data as Float32Array);

//   let imax = 0;
//   let pmax = probs[0];
//   for (let i = 1; i < probs.length; i++) {
//     if (probs[i] > pmax) {
//       pmax = probs[i];
//       imax = i;
//     }
//   }

//   return { classId: imax, conf: pmax, probs };
// }

// export async function predictWindow(
//   win: Window
// ): Promise<{ classId: number; conf: number; probs: number[] }> {
//   // const sess = await ensureSession();

//   // const rawData = [
//   //   ...win.accel_x, ...win.accel_y, ...win.accel_z,
//   //   ...win.gyro_x,  ...win.gyro_y,  ...win.gyro_z,
//   //   ...win.mag_x,   ...win.mag_y,   ...win.mag_z,
//   // ];

//   // if (rawData.length !== 900) {
//   //   throw new Error(`Invalid window shape: got ${rawData.length}, expected 900`);
//   // }

//   // const tensor = new Tensor("float32", Float32Array.from(rawData), [1, 100, 9]);

//   // const inputName = sess.inputNames[0];
//   // const outputName = sess.outputNames[0];

//   // const out = await sess.run({ [inputName]: tensor });
//   const sess = await ensureSession();
//   const scaler = await getScalerValues();

//   if (!scaler) throw new Error("Scaler data tidak ditemukan!");

//   const interleavedData = [];
  
//   // Looping sebanyak 100 timesteps
//   for (let i = 0; i < 100; i++) {
//     // 1. Ambil baris data mentah (9 fitur)
//     const rawRow = [
//       win.accel_x[i], win.accel_y[i], win.accel_z[i],
//       win.gyro_x[i],  win.gyro_y[i],  win.gyro_z[i],
//       win.mag_x[i],   win.mag_y[i],   win.mag_z[i]
      
//     ];
    
//     // 2. Normalisasi setiap fitur menggunakan mean dan scale dari Python
//     const normalizedRow = rawRow.map((val, idx) => {
      
//       return (val - scaler.mean[idx]) / scaler.scale[idx];
//     });
//     if (i === 0) console.log("Hasil Normalisasi Pertama:", normalizedRow);

//     // 3. Masukkan ke array utama
//     interleavedData.push(...normalizedRow);
//     console.log("nialai win.accel_z[0] : " ,win.accel_z[0]);
//   }

//   // Sekarang data sudah [1, 100, 9] dengan urutan dan skala yang benar
//   const tensor = new Tensor("float32", Float32Array.from(interleavedData), [1, 100, 9]);
  
//   const inputName = sess.inputNames[0];
//   const outputName = sess.outputNames[0];

//   const out = await sess.run({ [inputName]: tensor });

//   const probT = out[outputName];
//   const probs = Array.from(probT.data as Float32Array);

//   let imax = 0;
//   let pmax = probs[0];
//   for (let i = 1; i < probs.length; i++) {
//     if (probs[i] > pmax) {
//       pmax = probs[i];
//       imax = i;
//     }
//   }

//   return { classId: imax, conf: pmax, probs };
// }

// export async function predictWindow(
//   win: Window
// ): Promise<{ classId: number; conf: number; probs: number[] }> {
//   const sess = await ensureSession();

//   const rawData = [
//     ...win.accel_x, ...win.accel_y, ...win.accel_z,
//     ...win.gyro_x,  ...win.gyro_y,  ...win.gyro_z,
//     ...win.mag_x,   ...win.mag_y,   ...win.mag_z,
//   ];

//   if (rawData.length !== 900) {
//     throw new Error(`Invalid window shape: got ${rawData.length}, expected 900`);
//   }

//   const tensor = new Tensor("float32", Float32Array.from(rawData), [1, 100, 9]);

//   const inputName = sess.inputNames[0];
//   const outputName = sess.outputNames[0];

//   const out = await sess.run({ [inputName]: tensor });
  

//   const probT = out[outputName];
//   const probs = Array.from(probT.data as Float32Array);

//   let imax = 0;
//   let pmax = probs[0];
//   for (let i = 1; i < probs.length; i++) {
//     if (probs[i] > pmax) {
//       pmax = probs[i];
//       imax = i;
//     }
//   }

//   return { classId: imax, conf: pmax, probs };
// }

// export async function benchmarkModel(iterations = 30) {
//   const sess = await ensureSession();

//   const dummyData = new Float32Array(100 * 9).fill(0);
//   const tensor = new Tensor("float32", dummyData, [1, 100, 9]);

//   const times: number[] = [];
//   for (let i = 0; i < iterations; i++) {
//     const t0 = performance.now();
//     await sess.run({ input: tensor });
//     const t1 = performance.now();
//     times.push(t1 - t0);
//   }

//   const mean = times.reduce((a, b) => a + b, 0) / times.length;
//   const sd = Math.sqrt(
//     times.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) / times.length
//   );

//   return { mean, sd, iterations };
// }





//src//har//pipeline.ts

// import { InferenceSession, Tensor } from 'onnxruntime-react-native';
// import RNFS from 'react-native-fs';
// import { Platform } from 'react-native';

// // === fitur, scaler ===
// import featureOrderJson from '../../assets/ml/feature_order.json';
// import scalerParams from '../../assets/ml/scaler.json';
// import { extractWindowFeatures, type Window, type ExtractResult } from './features';
// import { applyScaler } from './scaler';
// import performance from 'react-native-performance';
// // ----- KONFIG -----
// // const MODEL_FILE = 'lgbm_har_model3.onnx'; // ganti jika namamu beda (mis. 'lgbm_har_model2.onnx')
// const MODEL_FILE = 'svm_har_model4.onnx'; // ganti jika namamu beda (mis. 'lgbm_har_model2.onnx')

// const FEATURE_ORDER: string[] = (featureOrderJson as any).feature_order;
// const SCALER = scalerParams as { mean: number[]; scale: number[] };

// let session: InferenceSession | null = null;

// async function ensureSession() {
//   if (session) return session!;
//   let uri = MODEL_FILE;

//   if (Platform.OS === 'android') {
//     const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;
//     try {
//       if (!(await RNFS.exists(dest))) await RNFS.copyFileAssets(MODEL_FILE, dest);
//       uri = 'file://' + dest;
//     } catch {
//       uri = `asset:///${MODEL_FILE}`;
//     }
//   }
//   session = await InferenceSession.create(uri);
//   return session!;
// }

// export async function loadSession() {
//   return ensureSession();
// }

// // Kembalikan: classId, conf (max prob), dan probs[] untuk semua kelas
// export async function predictWindow(win: Window): Promise<{ classId:number; conf:number; probs:number[] }> {
//   const sess = await ensureSession();

//   // 1) fitur -> urutan training
//   const feats = extractWindowFeatures(win);
//   const x = FEATURE_ORDER.map(k => feats[k] ?? 0);

//   // 2) cek & scale (harus sama dgn training)
//   if (x.length !== SCALER.mean.length || x.length !== SCALER.scale.length) {
//     throw new Error(`Feature length mismatch: x=${x.length}, mean=${SCALER.mean.length}, scale=${SCALER.scale.length}`);
//   }
//   const xNorm = applyScaler(x, SCALER);

//   // 3) feed ke ONNX
//   const tensor = new Tensor('float32', Float32Array.from(xNorm), [1, xNorm.length]);
//   const out = await sess.run({ input: tensor }, ['probabilities']); // <- nama fix
//   const probT = out['probabilities'];
//   const probs = Array.from(probT.data as Float32Array);

//   // 4) argmax
//   let imax = 0, pmax = probs[0];
//   for (let i = 1; i < probs.length; i++) if (probs[i] > pmax) { pmax = probs[i]; imax = i; }

//   return { classId: imax, conf: pmax, probs };
// }
// export async function benchmarkModel(iterations = 30) {
//   const sess = await ensureSession();

//   // Contoh window dummy (isi nol), bisa ganti dengan window nyata dari sampler
//   const dummyWin: Window = {
//     accel_x: Array(100).fill(0),
//     accel_y: Array(100).fill(0),
//     accel_z: Array(100).fill(0),
//     gyro_x:  Array(100).fill(0),
//     gyro_y:  Array(100).fill(0),
//     gyro_z:  Array(100).fill(0),
//     mag_x:   Array(100).fill(0),
//     mag_y:   Array(100).fill(0),
//     mag_z:   Array(100).fill(0),
//   };

//   const feats = FEATURE_ORDER.map(() => Math.random()); // fitur dummy
//   const xNorm = applyScaler(feats, SCALER);
//   const tensor = new Tensor('float32', Float32Array.from(xNorm), [1, xNorm.length]);

//   const times: number[] = [];
//   for (let i = 0; i < iterations; i++) {
//     const t0 = performance.now();
//     await sess.run({ input: tensor }, ['probabilities']);
//     const t1 = performance.now();
//     times.push(t1 - t0);
//   }

//   const mean =
//     times.reduce((a, b) => a + b, 0) / times.length;
//   const sd = Math.sqrt(
//     times.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) /
//     times.length
//   );

//   return { mean, sd, iterations };
// }