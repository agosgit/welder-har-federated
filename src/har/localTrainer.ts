import { NativeModules } from "react-native";
import RNFS from "react-native-fs";

type NativeTrainResult = {
  success: boolean;
  message?: string;
  sampleCount: number;
  numClasses?: number;
  lastLoss?: number;
  trainedAt: string;
};

type NativeInferResult = {
  pred: number;
  probs: number[];
};

type ExportWeightsResult = {
  success: boolean;
  length: number;
  weights: number[];
};

const { FLTrainer } = NativeModules;

const DATASET_PATH = `${RNFS.DocumentDirectoryPath}/fl_data/local_samples.json`;
const TRAINING_MODEL_PATH = `${RNFS.DocumentDirectoryPath}/models/trainable_har_model.tflite`;
const LABEL_MAP_PATH = `${RNFS.DocumentDirectoryPath}/models/label_map_nomag.json`;

async function ensureDirs() {
  const dirs = [
    `${RNFS.DocumentDirectoryPath}/fl_data`,
    `${RNFS.DocumentDirectoryPath}/models`,
  ];

  for (const dir of dirs) {
    const exists = await RNFS.exists(dir);
    if (!exists) {
      await RNFS.mkdir(dir);
    }
  }
}

export async function trainLocalModelNative(
  epochs = 1,
  batchSize = 4
): Promise<NativeTrainResult> {
  await ensureDirs();

  const requiredPaths = [
    DATASET_PATH,
    TRAINING_MODEL_PATH,
    LABEL_MAP_PATH,
  ];

  for (const path of requiredPaths) {
    const exists = await RNFS.exists(path);
    if (!exists) {
      throw new Error(`File wajib belum ada: ${path}`);
    }
  }

  const result = await FLTrainer.trainLocalModel(
    DATASET_PATH,
    TRAINING_MODEL_PATH,
    LABEL_MAP_PATH,
    epochs,
    batchSize
  );

  return result as NativeTrainResult;
}

export async function inferLocalTrainableModel(windowData: any): Promise<NativeInferResult> {
  await ensureDirs();

  const exists = await RNFS.exists(TRAINING_MODEL_PATH);
  if (!exists) {
    throw new Error(`Model training TFLite belum ada: ${TRAINING_MODEL_PATH}`);
  }

  const result = await FLTrainer.inferLocalModel(
    windowData,
    TRAINING_MODEL_PATH
  );

  return result as NativeInferResult;
}

export async function exportLocalModelWeights(): Promise<ExportWeightsResult> {
  await ensureDirs();

  const exists = await RNFS.exists(TRAINING_MODEL_PATH);
  if (!exists) {
    throw new Error(`Model training TFLite belum ada: ${TRAINING_MODEL_PATH}`);
  }

  const result = await FLTrainer.exportModelWeights(TRAINING_MODEL_PATH);
  return result as ExportWeightsResult;
}

export function getTrainingModelPath() {
  return TRAINING_MODEL_PATH;
}

export function getDatasetPath() {
  return DATASET_PATH;
}

export function getLabelMapPath() {
  return LABEL_MAP_PATH;
}