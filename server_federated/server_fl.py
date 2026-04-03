# server_fl.py
import socketio
import numpy as np
import uvicorn
import json
import os
import asyncio
import torch
from datetime import datetime
from ModelHAR import ModelHAR
import threading
from fastapi import FastAPI
from fastapi.responses import FileResponse

# ====== KONFIGURASI DASAR ======
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

WEIGHTS_FILE = "weights.json"
FEEDBACK_FILE = "feedback_buffer.json"
MODEL_DIR = "models"
MODEL_VERSION_FILE = "model_version.txt"
LOG_FILE = "retrain_log.json"

RETRAIN_INTERVAL = 30        # retrain tiap 1 menit
MAX_BUFFER_SIZE = 300         # simpan max 300 feedback terakhir
MIN_FEEDBACK_TO_TRAIN = 10    # mulai retrain kalau ada >=20 data
feedback_lock = threading.Lock()
os.makedirs(MODEL_DIR, exist_ok=True)

# ====== MUAT MODEL GLOBAL AWAL ======
if os.path.exists(WEIGHTS_FILE):
    with open(WEIGHTS_FILE, "r") as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            global_model = np.array(data[0], dtype=np.float32)
        else:
            global_model = np.zeros(100, dtype=np.float32)
    print(f"📦 Loaded global model from {WEIGHTS_FILE} (len={len(global_model)})")
else:
    global_model = np.zeros(100, dtype=np.float32)
    print("⚠️ No weights.json found — using zeros")

client_updates = []

# ====================================================
# EVENT HANDLERS SOCKET.IO
# ====================================================
@sio.event
async def connect(sid, environ):
    print(f"🟢 Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"🔴 Client disconnected: {sid}")

@sio.event
async def send_weights(sid, data):
    """Terima bobot model dari client"""
    global global_model
    weights = np.array(data["weights"], dtype=np.float32)
    client_updates.append(weights)
    print(f"📦 Received weights from {sid} (len={len(weights)})")

    # agregasi sederhana
    if len(client_updates) >= 2:
        global_model = np.mean(client_updates, axis=0)
        client_updates.clear()
        with open(WEIGHTS_FILE, "w") as f:
            json.dump([global_model.tolist()], f)
        print(f"🧮 Updated global model → saved to {WEIGHTS_FILE}")
        await sio.emit("global_model", {"weights": global_model.tolist()})

@sio.event
async def request_model(sid):
    """Client minta model global"""
    await sio.emit("global_model", {"weights": global_model.tolist()}, to=sid)
    print(f"📤 Sent global model to {sid}")

@sio.event
async def send_feedback(sid, data):
    """Terima feedback window sensor dari klien"""
    window = data.get("window")
    label = data.get("label")
    if not window or not label:
        print(f"⚠️ Invalid feedback from {sid}")
        return

    buf = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            try:
                buf = json.load(f)
            except:
                buf = []

    # simpan data baru
    buf.append({"window": window, "label": label})
    if len(buf) > MAX_BUFFER_SIZE:
        buf = buf[-MAX_BUFFER_SIZE:]  # buang data lama

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(buf, f)

    print(f"💾 Feedback received from {sid}: {label} (buffer={len(buf)})")

# ====================================================
# LOOP RETRAIN OTOMATIS
# ====================================================
async def retrain_loop():
    await asyncio.sleep(10)
    while True:
        try:
            if not os.path.exists(FEEDBACK_FILE):
                await asyncio.sleep(RETRAIN_INTERVAL)
                continue

            with open(FEEDBACK_FILE) as f:
                buf = json.load(f)

            if len(buf) >= MIN_FEEDBACK_TO_TRAIN:
                print(f"🔁 Starting retrain ({len(buf)} feedback samples)...")

                # ----- Persiapkan data -----
                X, y_labels = [], []
                for b in buf:
                    w = b["window"]
                    sample = np.column_stack([
                        w["accel_x"], w["accel_y"], w["accel_z"],
                        w["gyro_x"],  w["gyro_y"],  w["gyro_z"],
                        w["mag_x"],   w["mag_y"],   w["mag_z"]
                    ])
                    X.append(sample)
                    y_labels.append(b["label"])

                X = np.array(X, dtype=np.float32)
                X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-8)

                label_map = {
                    "Duduk Aktif": 0,
                    "Berbaring Aktif": 1,
                    "Berdiri Aktif": 2,
                    "Berdiri Tidak Aktif": 3,
                    "Duduk Tidak Aktif": 4,
                    "Berbaring Tidak Aktif": 5,
                }
                y = np.array([label_map.get(l, 0) for l in y_labels], dtype=np.int64)

                # ----- Load dan latih model -----
                model = ModelHAR()
                if os.path.exists("global_model.pt"):
                    model.load_state_dict(torch.load("global_model.pt", weights_only=True))

                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
                criterion = torch.nn.CrossEntropyLoss()

                X_tensor = torch.tensor(X)
                y_tensor = torch.tensor(y)
                batch_size = 4

                for epoch in range(10):
                    perm = torch.randperm(len(X_tensor))
                    total_loss = 0
                    for i in range(0, len(X_tensor), batch_size):
                        idx = perm[i:i + batch_size]
                        batch_x, batch_y = X_tensor[idx], y_tensor[idx]
                        optimizer.zero_grad()
                        out = model(batch_x)
                        loss = criterion(out, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    print(f"🧠 Retrain Epoch {epoch + 1}, Avg Loss={total_loss / len(X_tensor):.4f}")

                # ----- Simpan model baru -----
                version = 1
                if os.path.exists(MODEL_VERSION_FILE):
                    with open(MODEL_VERSION_FILE) as f:
                        version = int(f.read().strip()) + 1

                versioned_name = f"cnn_lstm_har_model_v{version}.onnx"
                model_path = os.path.join(MODEL_DIR, versioned_name)

                torch.save(model.state_dict(), "global_model.pt")

                dummy_input = torch.randn(1, 100, 9)
                torch.onnx.export(
                    model,
                    dummy_input,
                    model_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print(f"✅ Model updated → {versioned_name}")

                with open(MODEL_VERSION_FILE, "w") as f:
                    f.write(str(version))

                # ----- Simpan log retrain -----
                log_entry = {
                    "version": version,
                    "timestamp": datetime.now().isoformat(),
                    "feedback_used": len(y),
                    "final_loss": float(total_loss / len(X_tensor))
                }
                logs = []
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE) as f:
                        try:
                            logs = json.load(f)
                        except:
                            logs = []
                logs.append(log_entry)
                with open(LOG_FILE, "w") as f:
                    json.dump(logs, f, indent=2)

                # kosongkan buffer setelah retrain
                buf.clear()
                with open(FEEDBACK_FILE, "w") as f:
                    json.dump(buf, f)

                await sio.emit("global_model", {"updated": True, "version": version})
                print(f"📢 Model v{version} broadcasted to all clients")

        except Exception as e:
            print(f"⚠️ Retrain loop error: {e}")

        await asyncio.sleep(RETRAIN_INTERVAL)

# ====================================================
# FASTAPI UNTUK DOWNLOAD MODEL
# ====================================================
fastapi_app = FastAPI()
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

@fastapi_app.get("/model/latest")
async def get_latest_model():
    """Kirim model ONNX terbaru berdasarkan versi tertinggi"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")]
    if not files:
        fallback_path = "cnn_lstm_har_model2.onnx"
        if os.path.exists(fallback_path):
            print(f"📤 Serving fallback model: {fallback_path}")
            return FileResponse(fallback_path, filename=os.path.basename(fallback_path),
                                media_type="application/octet-stream")
        return {"error": "No model available"}

    def extract_version(name: str) -> int:
        parts = name.split("_v")
        if len(parts) > 1 and parts[-1].split(".")[0].isdigit():
            return int(parts[-1].split(".")[0])
        return 0

    files.sort(key=extract_version)
    latest_file = files[-1]
    latest_path = os.path.join(MODEL_DIR, latest_file)
    print(f"📤 Serving latest model: {latest_file}")
    return FileResponse(latest_path, filename=latest_file, media_type="application/octet-stream")

# ====================================================
# JALANKAN SERVER
# ====================================================
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=5000)

def start_retrain():
    asyncio.run(retrain_loop())

if __name__ == "__main__":
    threading.Thread(target=start_retrain, daemon=True).start()
    start_server()
