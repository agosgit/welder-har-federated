import RNFS from "react-native-fs";
import { Platform } from "react-native";

const MODELS_DIR = `${RNFS.DocumentDirectoryPath}/models`;
const ONNX_DEST = `${RNFS.DocumentDirectoryPath}/cnn_lstm_har_model2.onnx`;
const SCALER_DEST = `${RNFS.DocumentDirectoryPath}/models/scaler.json`;

export async function ensureFLAssets() {
  if (Platform.OS !== "android") return;

  const existsDir = await RNFS.exists(MODELS_DIR);
  if (!existsDir) {
    await RNFS.mkdir(MODELS_DIR);
  }

  const tfliteDest = `${MODELS_DIR}/trainable_har_model.tflite`;
  const labelMapDest = `${MODELS_DIR}/label_map.json`;
  const SCALER_DEST = `${RNFS.DocumentDirectoryPath}/models/scaler.json`;

  if (!(await RNFS.exists(tfliteDest))) {
    await RNFS.copyFileAssets("trainable_har_model.tflite", tfliteDest);
    console.log("✅ Copied trainable_har_model.tflite");
  }

  if (!(await RNFS.exists(labelMapDest))) {
    await RNFS.copyFileAssets("label_map.json", labelMapDest);
    console.log("✅ Copied label_map.json");
  }

  if (!(await RNFS.exists(ONNX_DEST))) {
    await RNFS.copyFileAssets("cnn_lstm_har_model2.onnx", ONNX_DEST);
    console.log("✅ Copied cnn_lstm_har_model2.onnx");
  }

  if (!(await RNFS.exists(SCALER_DEST))) {
    await RNFS.copyFileAssets("scaler.json", SCALER_DEST);
    console.log("✅ Copied scaler.json");
}
}