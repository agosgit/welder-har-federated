import tensorflow as tf
import tf2onnx
import onnx

# === 1️⃣ Load model dari file H5 ===
model = tf.keras.models.load_model("model_cnn_lstm_trained.h5")
print("✅ Loaded model from model_cnn_lstm_trained.h5")

# === 2️⃣ Beri nama input & output agar tf2onnx tidak error ===
# (karena Keras 3 tidak punya 'output_names' lagi)
model.inputs[0]._name = "input"
model.outputs[0]._name = "output"

# === 3️⃣ Tentukan input signature ===
spec = (tf.TensorSpec((1, 100, 9), tf.float32, name="input"),)

# === 4️⃣ Konversi ke ONNX ===
onnx_path = "cnn_lstm_har_model2.onnx"
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=14,
    output_path=onnx_path
)
print(f"🧠 Exported model to {onnx_path}")

# === 5️⃣ Validasi hasil ===
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("🎉 Model ONNX valid and ready for ONNX Runtime Mobile!")
