export type MLWindow = {
  accel_x: number[];
  accel_y: number[];
  accel_z: number[];
  gyro_x: number[];
  gyro_y: number[];
  gyro_z: number[];
  mag_x: number[];
  mag_y: number[];
  mag_z: number[];
};

export type LocalSample = {
  id: string;
  label: string;
  window: MLWindow;
  createdAt: string;
  modelVersion?: number | null;
};

export type LocalTrainResult = {
  success: boolean;
  weights: number[];
  sampleCount: number;
  trainedAt: string;
  modelVersion?: number | null;
};