//localDataset.ts
import RNFS from "react-native-fs";
import { LocalSample, MLWindow } from "./flTypes";

const DATASET_DIR = `${RNFS.DocumentDirectoryPath}/fl_data`;
const DATASET_FILE = `${DATASET_DIR}/local_samples.json`;

async function ensureDatasetFile() {
  const existsDir = await RNFS.exists(DATASET_DIR);
  if (!existsDir) {
    await RNFS.mkdir(DATASET_DIR);
  }

  const existsFile = await RNFS.exists(DATASET_FILE);
  if (!existsFile) {
    await RNFS.writeFile(DATASET_FILE, JSON.stringify([], null, 2), "utf8");
  }
}

export async function loadLocalSamples(): Promise<LocalSample[]> {
  await ensureDatasetFile();
  const raw = await RNFS.readFile(DATASET_FILE, "utf8");
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export async function saveLocalSamples(samples: LocalSample[]) {
  await ensureDatasetFile();
  await RNFS.writeFile(DATASET_FILE, JSON.stringify(samples, null, 2), "utf8");
}

export async function addLocalSample(
  window: MLWindow,
  label: string,
  modelVersion?: number | null
) {
  const samples = await loadLocalSamples();

  const newSample: LocalSample = {
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    label,
    window,
    createdAt: new Date().toISOString(),
    modelVersion: modelVersion ?? null,
  };

  samples.push(newSample);
  await saveLocalSamples(samples);

  return newSample;
}

export async function getLocalSampleCount(): Promise<number> {
  const samples = await loadLocalSamples();
  return samples.length;
}

export async function clearLocalSamples() {
  await saveLocalSamples([]);
}

export async function getRecentLocalSamples(limit = 20): Promise<LocalSample[]> {
  const samples = await loadLocalSamples();
  return samples.slice(-limit).reverse();
}