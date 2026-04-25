// flClient.ts
import io from "socket.io-client";
import RNFS from "react-native-fs";
import { reloadOnnxModel } from "./pipeline";

const SERVER_URL = "http://192.168.1.9:5000";
const socket = io(SERVER_URL, { transports: ["websocket"] });

let currentModelVersion: number | null = null;
let modelUpdating = false;

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

      const download = await RNFS.downloadFile({
        fromUrl: modelUrl,
        toFile: destPath,
      }).promise;

      if (download.statusCode === 200) {
        console.log(`✅ Model downloaded: ${destPath}`);
        await reloadOnnxModel(destPath);
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
  if (!Array.isArray(weights) || weights.length === 0) {
    throw new Error("Local model weights kosong.");
  }

  socket.emit("send_weights", { weights });
  console.log(`📤 Local trained weights sent to server (len=${weights.length})`);
}







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
