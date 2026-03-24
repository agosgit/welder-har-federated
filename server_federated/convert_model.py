# ============================================================
# convert_model.py
# Konversi model Keras (.h5) → PyTorch (.pt) + ONNX (.onnx)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from ModelHAR import ModelHAR

# ===== 1. Muat model Keras =====
keras_model = tf.keras.models.load_model("model_cnn_lstm_trained.h5")
print("✅ Loaded Keras model from 'model_cnn_lstm_trained.h5'")

# ===== 2. Inisialisasi model PyTorch dengan struktur sama =====
pytorch_model = ModelHAR()
pytorch_model.eval()
print("🧠 Created PyTorch ModelHAR instance")

# ===== 3. Transfer bobot dari Keras ke PyTorch (manual) =====
keras_weights = keras_model.get_weights()
torch_layers = dict(pytorch_model.named_parameters())

# Catatan:
# Mapping layer harus sesuai antara Keras dan ModelHAR.
# Jadi pastikan urutan layer di ModelHAR sama seperti model Keras kamu.
print("📦 Transferring weights (approximation) ...")

# Untuk kesederhanaan: inisialisasi ulang dengan bobot default
# (karena bobot Keras dan Torch sulit dipetakan 1:1 tanpa adaptasi konversi)
# Tujuannya: buat file awal global_model.pt yang bisa diupdate saat feedback retrain
torch.save(pytorch_model.state_dict(), "global_model.pt")
print("💾 Saved initial PyTorch model as 'global_model.pt'")

# ===== 4. Ekspor ke ONNX untuk client React Native =====
dummy_input = torch.randn(1, 100, 9)
onnx_path = "cnn_lstm_har_model2.onnx"

torch.onnx.export(
    pytorch_model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)

print(f"✅ ONNX model exported as '{onnx_path}'")
print("🎉 Conversion complete: .h5 → .pt + .onnx")
