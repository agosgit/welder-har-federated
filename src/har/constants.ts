export const FS = 50;            // Hz
export const WIN = 100;          // 2s @ 50Hz
export const STEP = 50;          // 50% overlap

// Koefisien Butterworth 4th, lowpass 5 Hz @ 50 Hz (scipy.signal.butter)
export const BUTTER_B = [0.00482434, 0.01929737, 0.02894606, 0.01929737, 0.00482434];
export const BUTTER_A = [1.00000000, -2.36951301, 2.31398841, -1.05466541, 0.18737949];

// Window setup untuk inference: 2 detik @ 50Hz (must match training)
export const WINDOW_FS   = 50;
export const WINDOW_SEC  = 2;
export const WINDOW_SIZE = WINDOW_FS * WINDOW_SEC;   // 100
export const STEP_SIZE   = WINDOW_SIZE / 2;          // 50 (50% overlap)
