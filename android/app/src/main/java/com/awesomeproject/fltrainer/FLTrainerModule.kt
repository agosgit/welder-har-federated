package com.awesomeproject.fltrainer

import com.facebook.react.bridge.*
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.time.Instant
import kotlin.math.min

class FLTrainerModule(private val reactCtx: ReactApplicationContext)
  : ReactContextBaseJavaModule(reactCtx) {

  override fun getName(): String = "FLTrainer"

  companion object {
    private const val WINDOW_SIZE = 100
    private const val NUM_FEATURES = 9
  }

  @ReactMethod
  fun trainLocalModel(
    datasetPath: String,
    tfliteModelPath: String,
    labelMapPath: String,
    epochs: Int,
    batchSize: Int,
    promise: Promise
  ) {
    try {
      val datasetFile = File(datasetPath)
      if (!datasetFile.exists()) {
        promise.reject("E_DATASET", "Dataset not found: $datasetPath")
        return
      }

      val modelFile = File(tfliteModelPath)
      if (!modelFile.exists()) {
        promise.reject("E_MODEL", "TFLite model not found: $tfliteModelPath")
        return
      }

      val labelMapFile = File(labelMapPath)
      if (!labelMapFile.exists()) {
        promise.reject("E_LABEL_MAP", "Label map not found: $labelMapPath")
        return
      }

      val classToIndex = loadClassToIndex(labelMapFile)
      val numClasses = classToIndex.size

      val raw = datasetFile.readText()
      val arr = JSONArray(raw)
      val sampleCount = arr.length()

      if (sampleCount == 0) {
        val emptyResult = Arguments.createMap().apply {
          putBoolean("success", false)
          putString("message", "No local samples")
          putInt("sampleCount", 0)
          putString("trainedAt", Instant.now().toString())
        }
        promise.resolve(emptyResult)
        return
      }

      val xAll = Array(sampleCount) { Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) } }
      val yAll = Array(sampleCount) { FloatArray(numClasses) }

      for (i in 0 until sampleCount) {
        val item = arr.getJSONObject(i)
        val label = item.getString("label")
        val windowObj = item.getJSONObject("window")

        val clsIdx = classToIndex[label]
        if (clsIdx == null) {
          promise.reject("E_UNKNOWN_LABEL", "Label '$label' tidak ada di label_map.json")
          return
        }

        xAll[i] = windowToTimeMajor(windowObj)
        yAll[i][clsIdx] = 1f
      }

      val options = Interpreter.Options()
      val interpreter = Interpreter(loadModelFile(modelFile), options)

      var lastLoss = Float.NaN
      val usedBatchSize = if (batchSize <= 0) 4 else batchSize
      val usedEpochs = if (epochs <= 0) 1 else epochs

      // Ambil shape output "loss" dari signature train
      val lossTensor = interpreter.getOutputTensorFromSignature("loss", "train")
      val lossSize = lossTensor.numElements()
      if (lossSize <= 0) {
        interpreter.close()
        promise.reject("E_TRAIN", "Invalid loss tensor size from train signature")
        return
      }

      for (epoch in 0 until usedEpochs) {
        var start = 0
        while (start < sampleCount) {
          val end = min(start + usedBatchSize, sampleCount)
          val currentBatch = end - start

          val xBatch = Array(currentBatch) { Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) } }
          val yBatch = Array(currentBatch) { FloatArray(numClasses) }

          for (j in 0 until currentBatch) {
            xBatch[j] = xAll[start + j]
            yBatch[j] = yAll[start + j]
          }

          lastLoss = runTrainSignature(interpreter, xBatch, yBatch, lossSize)
          start = end
        }
      }

      val result = Arguments.createMap().apply {
        putBoolean("success", true)
        putString("message", "Local training completed")
        putInt("sampleCount", sampleCount)
        putInt("numClasses", numClasses)
        putDouble("lastLoss", lastLoss.toDouble())
        putString("trainedAt", Instant.now().toString())
      }

      interpreter.close()
      promise.resolve(result)

    } catch (e: Exception) {
      promise.reject("E_TRAIN", e)
    }
  }

  @ReactMethod
  fun inferLocalModel(
    windowReadableMap: ReadableMap,
    tfliteModelPath: String,
    promise: Promise
  ) {
    try {
      val modelFile = File(tfliteModelPath)
      if (!modelFile.exists()) {
        promise.reject("E_MODEL", "TFLite model not found: $tfliteModelPath")
        return
      }

      val x = Array(1) { windowReadableMapToTimeMajor(windowReadableMap) }

      val interpreter = Interpreter(loadModelFile(modelFile), Interpreter.Options())

      val predTensor = interpreter.getOutputTensorFromSignature("pred", "infer")
      val probsTensor = interpreter.getOutputTensorFromSignature("probs", "infer")

      val pred = IntArray(predTensor.numElements())
      val probs = FloatArray(probsTensor.numElements())

      val inputs = HashMap<String, Any>()
      inputs["x"] = x

      val outputs = HashMap<String, Any>()
      outputs["pred"] = pred
      outputs["probs"] = probs

      interpreter.runSignature(inputs, outputs, "infer")

      val result = Arguments.createMap().apply {
        putInt("pred", if (pred.isNotEmpty()) pred[0] else -1)

        val probsArray = Arguments.createArray()
        for (v in probs) {
          probsArray.pushDouble(v.toDouble())
        }
        putArray("probs", probsArray)
      }

      interpreter.close()
      promise.resolve(result)
    } catch (e: Exception) {
      promise.reject("E_INFER", e)
    }
  }

  @ReactMethod
  fun exportModelWeights(
    tfliteModelPath: String,
    promise: Promise
  ) {
    try {
      val modelFile = File(tfliteModelPath)
      if (!modelFile.exists()) {
        promise.reject("E_MODEL", "TFLite model not found: $tfliteModelPath")
        return
      }

      val interpreter = Interpreter(loadModelFile(modelFile), Interpreter.Options())

      val weightsTensor = interpreter.getOutputTensorFromSignature("weights", "export_weights")
      val weightCount = weightsTensor.numElements()

      if (weightCount <= 0) {
        interpreter.close()
        promise.reject("E_EXPORT_WEIGHTS", "Invalid weights tensor size")
        return
      }

      val weights = FloatArray(weightCount)

      val inputs = HashMap<String, Any>()
      inputs["token"] = floatArrayOf(0f)

      val outputs = HashMap<String, Any>()
      outputs["weights"] = weights

      interpreter.runSignature(inputs, outputs, "export_weights")
        
      val weightsArray = Arguments.createArray()
      for (v in weights) {
        weightsArray.pushDouble(v.toDouble())
      }

      val result = Arguments.createMap().apply {
        putBoolean("success", true)
        putInt("length", weightCount)
        putArray("weights", weightsArray)
      }

      interpreter.close()
      promise.resolve(result)
    } catch (e: Exception) {
      promise.reject("E_EXPORT_WEIGHTS", e)
    }
  }

  private fun loadModelFile(file: File): MappedByteBuffer {
    FileInputStream(file).use { input ->
      val channel = input.channel
      return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    }
  }

  private fun loadClassToIndex(file: File): Map<String, Int> {
    val obj = JSONObject(file.readText())
    val classToIndexObj = obj.getJSONObject("class_to_index")

    val map = mutableMapOf<String, Int>()
    val keys = classToIndexObj.keys()
    while (keys.hasNext()) {
      val key = keys.next()
      map[key] = classToIndexObj.getInt(key)
    }
    return map
  }

  private fun runTrainSignature(
    interpreter: Interpreter,
    xBatch: Array<Array<FloatArray>>,
    yBatch: Array<FloatArray>,
    lossSize: Int
  ): Float {
    val inputs = HashMap<String, Any>()
    inputs["x"] = xBatch
    inputs["y"] = yBatch

    val loss = FloatArray(lossSize)
    val outputs = HashMap<String, Any>()
    outputs["loss"] = loss

    interpreter.runSignature(inputs, outputs, "train")
    return if (loss.isNotEmpty()) loss[0] else Float.NaN
  }

  private fun windowToTimeMajor(windowObj: JSONObject): Array<FloatArray> {
    val keys = listOf(
      "accel_x", "accel_y", "accel_z",
      "gyro_x", "gyro_y", "gyro_z",
      "mag_x", "mag_y", "mag_z"
    )

    val out = Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) }

    for (t in 0 until WINDOW_SIZE) {
      for (f in keys.indices) {
        val arr = windowObj.getJSONArray(keys[f])
        out[t][f] = arr.getDouble(t).toFloat()
      }
    }

    return out
  }

  private fun windowReadableMapToTimeMajor(windowMap: ReadableMap): Array<FloatArray> {
    val keys = listOf(
      "accel_x", "accel_y", "accel_z",
      "gyro_x", "gyro_y", "gyro_z",
      "mag_x", "mag_y", "mag_z"
    )

    val out = Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) }

    for (t in 0 until WINDOW_SIZE) {
      for (f in keys.indices) {
        val arr = windowMap.getArray(keys[f])
          ?: throw IllegalArgumentException("Missing key: ${keys[f]}")
        out[t][f] = arr.getDouble(t).toFloat()
      }
    }

    return out
  }
}