// // flClient.ts
// import io from "socket.io-client";
// import { Alert } from "react-native";
// // import { getModelWeights } from "../har/pipeline";
// import RNFS from "react-native-fs";
// import { getModelWeights, reloadOnnxModel } from "../har/pipeline"; 
// const SERVER_URL = "http://192.168.1.20:5000"; // ganti IP sesuai server kamu
// const socket = io(SERVER_URL, { transports: ["websocket"] });

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

// export async function requestGlobalModel() {
//   console.log("📡 Requesting global model...");
//   socket.emit("request_model");
// }

// export async function sendLocalModel(weights?: number[]) {
//   const finalWeights = weights ?? (await getModelWeights());
//   socket.emit("send_weights", { weights: finalWeights });
//   console.log("📤 Local weights sent to server");
// }

// export function sendFeedback(windowData: any, label: string) {
//   try {
//     socket.emit("send_feedback", { window: windowData, label });
//     console.log("📨 Feedback sent:", label);
//   } catch (err) {
//     console.error("⚠️ Failed to send feedback:", err);
//   }
// }

// async function updateLocalModel(weights: number[]) {
//   try {
//     console.log("🔄 Updating local model...");
//     const modelUrl = `${SERVER_URL}/model/latest`;
//     const destPath = `${RNFS.DocumentDirectoryPath}/cnn_lstm_har_model2.onnx`;

//     // Hapus model lama (jika ada)
//     // Hapus model lama (jika ada)
//     if (await RNFS.exists(destPath)) {
//       try {
//         await RNFS.unlink(destPath);
//         console.log("🗑️ Old model removed");
//       } catch (err) {
//         if (err instanceof Error) {
//           console.error("❌ Error saat memperbarui model:", err.message);
//         } else {
//           console.error("❌ Error saat memperbarui model:", err);
//         }
//       }
//     } else {
//       console.log("ℹ️ Tidak ada model lama untuk dihapus");
//     }


//     // Download model baru dari server
//     const download = await RNFS.downloadFile({
//       fromUrl: modelUrl,
//       toFile: destPath,
//     }).promise;

//     if (download.statusCode === 200) {
//       console.log("✅ Model baru diunduh:", destPath);
//       // Alert.alert("Model Diperbarui", "Model terbaru telah diunduh dan siap digunakan 🎉");

//       // Opsional: panggil pipeline untuk reload ONNX model
//       await reloadOnnxModel(destPath);
//     } else {
//       console.warn("⚠️ Gagal mengunduh model, status:", download.statusCode);
//     }
//   } catch (err) {
//     console.error("❌ Error saat memperbarui model:", err);
//   }
// }
