import socketio
import numpy as np
import uvicorn
import json
import os
import asyncio
from datetime import datetime

import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import FileResponse
import time
from fastapi import Request

@tf.keras.utils.register_keras_serializable()
class CompatibleLSTM(tf.keras.layers.LSTM):
    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)
        return cls(**config)

def load_keras_model_compat(path: str):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "CompatibleLSTM": CompatibleLSTM,
            "LSTM": CompatibleLSTM,
        },
        compile=False
    )

# ====================================================
# KONFIGURASI
# ====================================================
HOST = "0.0.0.0"
PORT = 5000

MODEL_DIR = "models"
MODEL_VERSION_FILE = "model_version.txt"
WEIGHTS_VECTOR_FILE = "weights.json"
FL_LOG_FILE = "fl_round_log.json"

# model dasar dari training Python
BASE_KERAS_MODEL_PATH = "model_cnn_lstm_trained.h5"

# model global yang akan terus diperbarui server
GLOBAL_KERAS_MODEL_PATH = "global_model.keras"

MIN_CLIENT_UPDATES = 2
ONNX_OPSET = 13

SCALER_PATH = "models/scaler_nomag.json" 

if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "r") as f:
        scaler_dict = json.load(f)
        scaler_mean = np.array(scaler_dict["mean"], dtype=np.float32)
        scaler_scale = np.array(scaler_dict["scale"], dtype=np.float32)
    print("✅ Scaler loaded for Cloud Inference")
else:
    print("⚠️ Scaler JSON tidak ditemukan!")
    # Nilai default (fallback) agar tidak crash
    scaler_mean = np.zeros(6, dtype=np.float32) 
    scaler_scale = np.ones(6, dtype=np.float32)

os.makedirs(MODEL_DIR, exist_ok=True)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
fastapi_app = FastAPI()
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

client_updates = []
updates_lock = asyncio.Lock()

# ====================================================
# HELPER: VERSION
# ====================================================
def load_version() -> int:
    if os.path.exists(MODEL_VERSION_FILE):
        with open(MODEL_VERSION_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except Exception:
                return 0
    return 0

def save_version(version: int):
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write(str(version))

current_version = load_version()

# ====================================================
# HELPER: KERAS MODEL <-> VECTOR
# ====================================================
def flatten_trainable_variables(model: tf.keras.Model) -> np.ndarray:
    parts = []
    for var in model.trainable_variables:
        arr = var.numpy().astype(np.float32).ravel()
        parts.append(arr)
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)

def assign_flat_vector_to_model(model: tf.keras.Model, flat_vector: np.ndarray):
    offset = 0
    for var in model.trainable_variables:
        shape = var.shape
        numel = int(np.prod(shape))
        chunk = flat_vector[offset:offset + numel]
        if chunk.size != numel:
            raise ValueError(
                f"Vector size mismatch when rebuilding Keras weights for '{var.name}': "
                f"need {numel}, got {chunk.size}"
            )
        reshaped = chunk.reshape(shape)
        var.assign(reshaped)
        offset += numel

    if offset != len(flat_vector):
        raise ValueError(
            f"Unused values in flat_vector: used {offset}, total {len(flat_vector)}"
        )

def save_weights_vector(flat_vector: np.ndarray):
    with open(WEIGHTS_VECTOR_FILE, "w") as f:
        json.dump([flat_vector.tolist()], f)

def append_round_log(log_entry: dict):
    logs = []
    if os.path.exists(FL_LOG_FILE):
        with open(FL_LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(log_entry)
    with open(FL_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def export_onnx_from_keras(model: tf.keras.Model, version: int) -> str:
    try:
        import tf2onnx
    except ImportError as e:
        raise RuntimeError(
            "tf2onnx belum terinstall. Install dulu: pip install tf2onnx"
        ) from e

    onnx_name = f"cnn_lstm_har_model_mobile_v{version}.onnx"
    onnx_path = os.path.join(MODEL_DIR, onnx_name)

    # patch kompatibilitas untuk Keras 3 + tf2onnx
    if not hasattr(model, "output_names") or model.output_names is None:
        try:
            model.output_names = [out.name.split(":")[0] for out in model.outputs]
        except Exception:
            model.output_names = ["output"]

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    concrete_shape = [None if i == 0 else d for i, d in enumerate(input_shape)]
    spec = (tf.TensorSpec(concrete_shape, tf.float32, name="input"),)

    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=ONNX_OPSET,
        output_path=onnx_path
    )
    return onnx_path

# ====================================================
# INIT GLOBAL MODEL
# ====================================================
if os.path.exists(GLOBAL_KERAS_MODEL_PATH):
    print(f"📦 Loading global Keras model from {GLOBAL_KERAS_MODEL_PATH}")
    global_model = load_keras_model_compat(GLOBAL_KERAS_MODEL_PATH)
elif os.path.exists(BASE_KERAS_MODEL_PATH):
    print(f"📦 Loading base Keras model from {BASE_KERAS_MODEL_PATH}")
    global_model = load_keras_model_compat(BASE_KERAS_MODEL_PATH)
    global_model.save(GLOBAL_KERAS_MODEL_PATH)
else:
    raise FileNotFoundError(
        f"Tidak ditemukan model dasar Keras. "
        f"Cek {GLOBAL_KERAS_MODEL_PATH} atau {BASE_KERAS_MODEL_PATH}"
    )

global_model_vector = flatten_trainable_variables(global_model)
expected_vector_len = len(global_model_vector)
save_weights_vector(global_model_vector)

existing_onnx = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")]
if not existing_onnx:
    export_path = export_onnx_from_keras(global_model, current_version)
    print(f"✅ Initial ONNX exported: {export_path}")

print(
    f"✅ Global Keras model ready | version={current_version} | "
    f"vector_len={expected_vector_len}"
)

# ====================================================
# SOCKET EVENTS
# ====================================================
@sio.event
async def connect(sid, environ):
    print(f"🟢 Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"🔴 Client disconnected: {sid}")

@sio.event
async def request_model(sid):
    await sio.emit(
        "global_model",
        {
            "updated": True,
            "version": current_version,
            "vector_len": expected_vector_len,
        },
        to=sid
    )
    print(f"📤 Sent model metadata to {sid} | version={current_version}")

@sio.event
async def send_weights(sid, data):
    global global_model, global_model_vector, current_version, expected_vector_len

    if "weights" not in data:
        print(f"⚠️ No 'weights' field from {sid}")
        return

    try:
        weights = np.array(data["weights"], dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Invalid weights payload from {sid}: {e}")
        return

    if weights.ndim != 1:
        print(f"⚠️ Invalid weights ndim from {sid}: {weights.ndim}")
        return

    if len(weights) != expected_vector_len:
        print(
            f"⚠️ Shape mismatch from {sid}: got {len(weights)}, "
            f"expected {expected_vector_len}"
        )
        await sio.emit(
            "fl_error",
            {
                "message": (
                    f"weights length mismatch: got {len(weights)}, "
                    f"expected {expected_vector_len}"
                )
            },
            to=sid
        )
        return

    async with updates_lock:
        client_updates.append(weights)
        print(
            f"📦 Received weights from {sid} (len={len(weights)}) | "
            f"buffered={len(client_updates)}"
        )
        await sio.emit("weights_received", {"status": "ok"}, to=sid)
        
        print(f"📦 Received weights from {sid} (len={len(weights)}) | buffered={len(client_updates)}")

        if len(client_updates) < MIN_CLIENT_UPDATES:
            await sio.emit(
                "fl_status",
                {
                    "message": (
                        f"Update buffered ({len(client_updates)}/{MIN_CLIENT_UPDATES})"
                    )
                },
                to=sid
            )
            return

        # ====================================================
        # FEDERATED AVERAGING
        # ====================================================
        stacked = np.stack(client_updates, axis=0)
        aggregated = np.mean(stacked, axis=0).astype(np.float32)
        client_updates.clear()

        # pasang vector hasil agregasi ke model Keras
        assign_flat_vector_to_model(global_model, aggregated)

        # simpan model global baru
        global_model.save(GLOBAL_KERAS_MODEL_PATH)

        # simpan vector juga
        save_weights_vector(aggregated)

        # bump version
        current_version += 1
        save_version(current_version)

        # export ONNX baru untuk mobile inference
        onnx_path = export_onnx_from_keras(global_model, current_version)

        # update in-memory
        global_model_vector = aggregated
        expected_vector_len = len(global_model_vector)

        log_entry = {
            "version": current_version,
            "timestamp": datetime.now().isoformat(),
            "num_client_updates": int(stacked.shape[0]),
            "vector_len": int(expected_vector_len),
            "onnx_path": onnx_path,
        }
        append_round_log(log_entry)

        print(f"🧮 Aggregation complete | version={current_version}")
        print(f"✅ New Keras global model saved: {GLOBAL_KERAS_MODEL_PATH}")
        print(f"✅ New ONNX exported: {onnx_path}")

        await sio.emit(
            "global_model",
            {
                "updated": True,
                "version": current_version,
                "vector_len": expected_vector_len,
            }
        )
        print(f"📢 Broadcasted global model v{current_version} to all clients")

# ====================================================
# FASTAPI ROUTES
# ====================================================
@fastapi_app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": current_version,
        "vector_len": expected_vector_len,
        "buffered_updates": len(client_updates),
    }

@fastapi_app.get("/model/latest")
async def get_latest_model():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")]
    if not files:
        return {"error": "No ONNX model available"}

    def extract_version(name: str) -> int:
        parts = name.split("_v")
        if len(parts) > 1:
            tail = parts[-1].split(".")[0]
            if tail.isdigit():
                return int(tail)
        return -1

    files.sort(key=extract_version)
    latest_file = files[-1]
    latest_path = os.path.join(MODEL_DIR, latest_file)

    print(f"📤 Serving latest model: {latest_file}")
    return FileResponse(
        latest_path,
        filename=latest_file,
        media_type="application/octet-stream"
    )

@fastapi_app.post("/predict_cloud")
async def predict_cloud(request: Request):
    data = await request.json()
    
    # window_data memiliki shape (1, 100, 6)
    window_data = np.array(data["window"], dtype=np.float32) 
    
    # ⏱️ Stopwatch mulai (Waktu komputasi murni di server)
    t0 = time.time()
    
    # 1. KONVERSI GRAVITASI (Hanya untuk fitur Accel_x, Accel_y, Accel_z)
    # [:, :, 0:3] artinya: semua batch, semua 100 timestep, ambil fitur ke 0 sampai 2
    window_data[:, :, 0:3] = window_data[:, :, 0:3] / 9.80665
    
    # 2. NORMALISASI Z-SCORE
    # Mengurangi dengan mean dan membagi dengan scale
    # NumPy otomatis mencocokkan dimensi (1, 100, 6) dengan (6,) secara pintar
    x_norm = (window_data - scaler_mean) / scaler_scale
    
    # Opsional: Lakukan Clipping agar sama persis dengan di HP
    x_norm = np.clip(x_norm, -3.0, 3.0)
    
    # 3. PREDIKSI
    probs = global_model.predict(x_norm, verbose=0)
    class_id = int(np.argmax(probs, axis=1)[0])
    
    t1 = time.time()
    server_inference_ms = (t1 - t0) * 1000

    return {
        "class_id": class_id,
        "server_inference_ms": server_inference_ms
    }

# ====================================================
# START SERVER
# ====================================================
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)