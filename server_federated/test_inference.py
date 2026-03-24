import onnxruntime as ort
import numpy as np

# 🧠 Load model
sess = ort.InferenceSession("cnn_lstm_har_model2.onnx")

# Cek nama input/output
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# 📈 Contoh data IMU simulasi (accel + gyro + mag)
# Misal kita simulasi jalan kaki
# Biasanya accel sekitar 9.8 di sumbu Z, kecil di X dan Y
imu_data = np.zeros((100, 9), dtype=np.float32)
for i in range(100):
    imu_data[i, 0] = np.sin(i / 10) * 0.5      # accel_x
    imu_data[i, 1] = np.cos(i / 10) * 0.5      # accel_y
    imu_data[i, 2] = 9.8 + np.sin(i / 5) * 0.1 # accel_z
    imu_data[i, 3:6] = np.random.normal(0, 0.05, 3)  # gyro
    imu_data[i, 6:9] = np.random.normal(0, 0.02, 3)  # mag

# Tambahkan dimensi batch
x = np.expand_dims(imu_data, axis=0)

# 🔮 Jalankan inferensi
y = sess.run([output_name], {input_name: x})[0]

# 📤 Tampilkan hasil
print("\n✅ Inference sukses!")
print("Output shape:", y.shape)
print("Prediksi probabilitas:", np.round(y, 4))
print("Kelas prediksi:", np.argmax(y))
