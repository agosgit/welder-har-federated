import tensorflow as tf
import tf2onnx

# === Load model Keras ===
model = tf.keras.models.load_model("model_cnn_lstm_trained.h5")
print("✅ Loaded model from model_cnn_lstm_trained.h5")

# === Tentukan input shape ===
dummy_input = tf.random.normal([1, 100, 9])
spec = (tf.TensorSpec(dummy_input.shape, tf.float32, name="input"),)

# === Konversi ke ONNX ===
model_proto, external_tensor_storage = tf2onnx.convert.from_function(
    tf.function(model),
    input_signature=spec,
    opset=14,
    output_path="cnn_lstm_har_model2.onnx"
)

print("🎉 Successfully exported model → cnn_lstm_har_model2.onnx")
