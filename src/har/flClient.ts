// flClient.ts
import io from "socket.io-client";
import RNFS from "react-native-fs";
import performance from 'react-native-performance';

import { reloadOnnxModel } from "./pipeline";

const SERVER_URL = "http://10.60.157.150:5000";
const socket = io(SERVER_URL, { transports: ["websocket"] });

let currentModelVersion: number | null = null;
let modelUpdating = false;
// 1. Tambahkan tipe dan variabel untuk listener di bagian atas
type NetworkStatsCallback = (stats: { downloadMs: number; loadMs: number }) => void;
let networkStatsListener: NetworkStatsCallback | null = null;

// 2. Fungsi untuk dipanggil oleh App.tsx agar bisa mendengarkan update
export function setNetworkStatsListener(callback: NetworkStatsCallback) {
  networkStatsListener = callback;
}

socket.on("connect", () => {
  console.log("🌐 Connected to FL server");
});

socket.on("disconnect", () => {
  console.log("❌ Disconnected from FL server");
});

socket.on("connect_error", (err) => {
  console.warn("⚠️ Socket connect error:", err?.message ?? err);
});

socket.on("fl_status", (data) => {
  console.log("📣 FL status:", data);
});

socket.on("fl_error", (data) => {
  console.warn("⚠️ FL error:", data);
});

socket.on("global_model", async (data) => {
  try {
    // server memberi tahu ada model ONNX global baru
    if (data.updated && !modelUpdating) {
      modelUpdating = true;

      const version = data.version ?? Date.now();
      currentModelVersion = version;

      const modelUrl = `${SERVER_URL}/model/latest?v=${version}`;
      const destPath = `${RNFS.DocumentDirectoryPath}/cnn_lstm_har_model_v${version}.onnx`;

      console.log(`✅ Global model updated (v${version}), downloading from ${modelUrl}`);

      if (await RNFS.exists(destPath)) {
        try {
          await RNFS.unlink(destPath);
          console.log("🗑️ Old version removed:", destPath);
        } catch (err) {
          console.warn("⚠️ Failed removing old ONNX:", err);
        }
      }

      // ==========================================
      // ⏱️ MULAI STOPWATCH DOWNLOAD LATENCY
      // ==========================================
      const t0_download = performance.now();

      const download = await RNFS.downloadFile({
        fromUrl: modelUrl,
        toFile: destPath,
      }).promise;

      if (download.statusCode === 200) {
        // ==========================================
        // ⏱️ HENTIKAN STOPWATCH DOWNLOAD LATENCY
        // ==========================================
        const t1_download = performance.now();
        const downloadLatency = t1_download - t0_download;
        
        console.log(`⏱️ Download Latency (Terima Model): ${downloadLatency.toFixed(2)} ms`);
        console.log(`✅ Model downloaded: ${destPath}`);

        // --- Opsional: Ukur juga waktu Load ONNX ke Memori ---
        const t0_load = performance.now();
        await reloadOnnxModel(destPath);
        const loadLatency = performance.now() - t0_load;
        console.log(`⏱️ Load ONNX Latency: ${loadLatency.toFixed(2)} ms`);
        // -----------------------------------------------------
        // 3. 🟢 PANGGIL LISTENER DI SINI untuk mengirim data ke UI
        if (networkStatsListener) {
          networkStatsListener({ 
            downloadMs: downloadLatency, 
            loadMs: loadLatency 
          });
        }

        console.log(`🧠 Model v${version} loaded into ONNX pipeline`);
      } else {
        console.warn("⚠️ Failed downloading ONNX model:", download.statusCode);
      }

      modelUpdating = false;
      return;
    }

    // fallback kalau server hanya mengirim metadata versi
    if (typeof data.version === "number") {
      currentModelVersion = data.version;
      console.log(`📌 Current global model version from server: ${data.version}`);
    }

    // legacy payload: weights metadata saja
    if (data.weights) {
      console.log("📥 Received global weights metadata:", data.weights.length);
    }
  } catch (err) {
    modelUpdating = false;
    console.error("❌ Error handling global_model event:", err);
  }
});

export function getCurrentModelVersion() {
  return currentModelVersion;
}

export function isModelUpdating() {
  return modelUpdating;
}

export async function requestGlobalModel() {
  console.log("📡 Requesting global model...");
  socket.emit("request_model");
}

export async function sendLocalModel(weights: number[]) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now(); // Mulai Stopwatch Upload

    // Dengarkan konfirmasi dari server
    socket.once("weights_received", () => {
      const t1 = performance.now(); // Hentikan Stopwatch Upload
      const uploadLatency = t1 - t0;
      console.log(`⏱️ Upload Latency (Kirim Bobot): ${uploadLatency.toFixed(2)} ms`);
      
      // Opsional: Simpan ke state/database untuk dianalisis nanti
      resolve(uploadLatency); 
    });

    // Kirim bobot ke server
    socket.emit("send_weights", { weights });
    
    // Timeout safety jika server mati
    setTimeout(() => reject("Timeout sending weights"), 10000);
  });
}

// ====================================================
// CLOUD INFERENCE (EVALUASI ABLASI)
// ====================================================
export async function predictViaCloud(windowData: number[][]) {
  const url = `${SERVER_URL}/predict_cloud`;
  
  // ⏱️ Stopwatch mulai (Total Latency Jaringan bolak-balik + Server)
  const t0 = performance.now();

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ window: windowData }),
    });

    if (!response.ok) {
      throw new Error(`HTTP Error! Status: ${response.status}`);
    }

    const result = await response.json();
    
    // ⏱️ Stopwatch berhenti
    const t1 = performance.now();
    const totalCloudLatencyMs = t1 - t0;
    
    // Hitung waktu jaringan murni (Total waktu dikurangi waktu mikir server)
    const networkLatencyMs = totalCloudLatencyMs - result.server_inference_ms;

    console.log(`☁️ Cloud Total: ${totalCloudLatencyMs.toFixed(2)} ms`);
    console.log(`☁️ Server Compute: ${result.server_inference_ms.toFixed(2)} ms`);
    console.log(`☁️ Network RTT: ${networkLatencyMs.toFixed(2)} ms`);

    // Kembalikan semua metrik agar bisa ditampilkan di layar App.tsx
    return {
      success: true,
      classId: result.class_id,
      cloudTotalMs: totalCloudLatencyMs,
      serverMs: result.server_inference_ms,
      networkMs: networkLatencyMs
    };
  } catch (error) {
    console.error("❌ Cloud prediction failed:", error);
    return {
      success: false,
      classId: -1,
      cloudTotalMs: 0,
      serverMs: 0,
      networkMs: 0
    };
  }
}














// export async function sendLocalModel(weights: number[]) {
//   if (!Array.isArray(weights) || weights.length === 0) {
//     throw new Error("Local model weights kosong.");
//   }

//   socket.emit("send_weights", { weights });
//   console.log(`📤 Local trained weights sent to server (len=${weights.length})`);
// }







// import io from "socket.io-client";
// import { Alert } from "react-native";
// // import { getModelWeights } from "../har/pipeline";
// import RNFS from "react-native-fs";
// import { reloadOnnxModel } from "../har/pipeline"; 
// const SERVER_URL = "http://192.168.68.213:5000"; // ganti IP sesuai server kamu
// const socket = io(SERVER_URL, { transports: ["websocket"] });

// let currentModelVersion: number | null = null;
// let modelUpdating = false;

// socket.on("connect", () => {
//   console.log("🌐 Connected to FL server");
// });

// socket.on("disconnect", () => {
//   console.log("❌ Disconnected from FL server");
// });

// socket.on("global_model", async (data) => {
//   // ---- Saat server mengirim sinyal bahwa model baru tersedia ----
//   if (data.updated && !modelUpdating) {
//     modelUpdating = true;

//     const version = data.version ?? Date.now(); // ambil versi model dari server
//     const modelUrl = `${SERVER_URL}/model/latest?v=${version}`; // tambahkan versi agar tidak kena cache
//     const destPath = `${RNFS.DocumentDirectoryPath}/cnn_lstm_har_model_v${version}.onnx`;

//     console.log(`✅ Model global updated (v${version}), downloading from ${modelUrl}`);

//     // Hapus model lama dengan nama yang sama jika ada
//     if (await RNFS.exists(destPath)) {
//       try {
//         await RNFS.unlink(destPath);
//         console.log("🗑️ Old version removed");
//       } catch (err) {
//         if (err instanceof Error) {
//           console.warn("⚠️ Gagal menghapus model lama:", err.message);
//         } else {
//           console.warn("⚠️ Gagal menghapus model lama (unknown error):", err);
//         }
//       }

//     }

//     // Download model baru langsung dari server
//     const download = await RNFS.downloadFile({
//       fromUrl: modelUrl,
//       toFile: destPath,
//     }).promise;

//     if (download.statusCode === 200) {
//       console.log(`✅ Model baru diunduh (v${version}):`, destPath);
//       await reloadOnnxModel(destPath); // reload ONNX model ke pipeline
//       console.log(`🧠 Model v${version} dimuat ke pipeline`);
//     } else {
//       console.warn("⚠️ Gagal mengunduh model, status:", download.statusCode);
//     }

//     modelUpdating = false;
//     return; // selesai di sini, tidak lanjut ke bawah
//   }

//   // ---- Saat server mengirim bobot model terbaru ----
//   if (data.weights) {
//     // await updateLocalModel(data.weights);
//     modelUpdating = false;
//     console.log("📥 Received and applied new model weights:", data.weights.length);
//   }
// });


// // ====== FUNGSI EKSPOR UNTUK CLIENT ======
// export function getCurrentModelVersion() {
//   return currentModelVersion;
// }

// export async function requestGlobalModel() {
//   console.log("📡 Requesting global model...");
//   socket.emit("request_model");
// }

// export async function sendLocalModel(weights: number[]) {
//   socket.emit("send_weights", { weights });
//   console.log("📤 Local trained weights sent to server");
// }

// export function sendFeedback(windowData: any, label: string) {
//   try {
//     socket.emit("send_feedback", { window: windowData, label });
//     console.log("📨 Feedback sent:", label);
//   } catch (err) {
//     console.error("⚠️ Failed to send feedback:", err);
//   }
// }
