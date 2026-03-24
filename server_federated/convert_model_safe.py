import os
import tensorflow as tf
import subprocess
import onnx
import sys
# 1️⃣ Load H5 model
model = tf.keras.models.load_model("model_cnn_lstm_trained.h5")
print("✅ Loaded model from model_cnn_lstm_trained.h5")

# 2️⃣ Export ke SavedModel
export_path = "saved_model_safe"
model.export(export_path)
print("💾 Exported to SavedModel:", export_path)

# 3️⃣ Gunakan CLI tf2onnx untuk konversi
onnx_path = "cnn_lstm_har_model2.onnx"
cmd = [
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", export_path,
    "--output", onnx_path,
    "--opset", "14"
]

print("⚙️ Converting via tf2onnx CLI ...")
subprocess.run(cmd, check=True)
print("🧠 Exported model:", onnx_path)

# 4️⃣ Validasi hasil
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("🎉 Model ONNX valid and ready for ONNX Runtime Mobile!")
