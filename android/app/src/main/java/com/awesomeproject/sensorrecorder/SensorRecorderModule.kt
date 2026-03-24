package com.awesomeproject.sensorrecorder

import android.content.Context
import android.hardware.*
import android.os.*
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule.RCTDeviceEventEmitter
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import android.content.Intent
import androidx.core.content.FileProvider
import android.content.ClipData
import java.util.Locale

class SensorRecorderModule(private val reactCtx: ReactApplicationContext)
  : ReactContextBaseJavaModule(reactCtx), LifecycleEventListener {

  override fun getName() = "SensorRecorder"

  // --- Sensors & infra
  private val sensorManager = reactCtx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
  private val acc: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
  private val gyr: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
  private val mag: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

  private var handlerThread: HandlerThread? = null
  private var handler: Handler? = null
  private var wakeLock: PowerManager.WakeLock? = null

  private val running = AtomicBoolean(false)
  private val scheduler = Executors.newSingleThreadScheduledExecutor()
  private var task: ScheduledFuture<*>? = null

  private var out: FileOutputStream? = null
  private lateinit var currentFile: File
  private var startNs: Long = 0L
  private var samples = 0
  private var lastEmitNs = 0L

  private data class Vec3(var x: Float=Float.NaN, var y: Float=Float.NaN, var z: Float=Float.NaN)
  private val lastAcc = Vec3()
  private val lastGyr = Vec3()
  private val lastMag = Vec3()

  private val accListener = object: SensorEventListener {
    override fun onSensorChanged(e: SensorEvent) { lastAcc.x=e.values[0]; lastAcc.y=e.values[1]; lastAcc.z=e.values[2] }
    override fun onAccuracyChanged(s: Sensor?, a: Int) {}
  }
  private val gyrListener = object: SensorEventListener {
    override fun onSensorChanged(e: SensorEvent) { lastGyr.x=e.values[0]; lastGyr.y=e.values[1]; lastGyr.z=e.values[2] }
    override fun onAccuracyChanged(s: Sensor?, a: Int) {}
  }
  private val magListener = object: SensorEventListener {
    override fun onSensorChanged(e: SensorEvent) { lastMag.x=e.values[0]; lastMag.y=e.values[1]; lastMag.z=e.values[2] }
    override fun onAccuracyChanged(s: Sensor?, a: Int) {}
  }

  // --- Event emitter ke JS
  private fun sendEvent(name: String, map: WritableMap) {
    reactCtx.getJSModule(RCTDeviceEventEmitter::class.java).emit(name, map)
  }
  @ReactMethod fun addListener(eventName: String?) {}
  @ReactMethod fun removeListeners(count: Int) {}

  private fun hasHighRatePerm(): Boolean =
    Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
      reactCtx.checkSelfPermission(android.Manifest.permission.HIGH_SAMPLING_RATE_SENSORS) ==
        android.content.pm.PackageManager.PERMISSION_GRANTED
  private var maxSamples = 0

  // --- API dipanggil dari JS
  @ReactMethod
  fun startRecording(rateHz: Int, durationSec: Int, label: String, promise: Promise) {
    if (running.get()) { promise.reject("E_RUNNING", "Already running"); return }
    if (acc==null || gyr==null || mag==null) { promise.reject("E_NO_SENSOR", "Missing sensors"); return }

    try {
      // ⚙️ Jika durationSec < 0 → mode manual (tanpa batas waktu)
      val unlimited = durationSec < 0
      maxSamples = if (unlimited) Int.MAX_VALUE else rateHz * durationSec
      samples = 0

      // WakeLock — tahan CPU supaya sensor tidak sleep
      val pm = reactCtx.getSystemService(Context.POWER_SERVICE) as PowerManager
      val lockTime = if (unlimited) 60L * 60L * 1000L else durationSec.toLong() * 1000 + 5000 // max 1 jam kalau manual
      wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SensorRecorder:W")
      wakeLock?.acquire(lockTime)

      // Thread untuk listener
      handlerThread = HandlerThread("SensorRecorderThread", Process.THREAD_PRIORITY_MORE_FAVORABLE)
      handlerThread!!.start()
      handler = Handler(handlerThread!!.looper)

      // Register listener (sampling rate)
      val listenerRateUs = if (hasHighRatePerm()) 0 else maxOf(5_000, 1_000_000 / (rateHz * 2))
      sensorManager.registerListener(accListener, acc, listenerRateUs, handler)
      sensorManager.registerListener(gyrListener, gyr, listenerRateUs, handler)
      sensorManager.registerListener(magListener, mag, listenerRateUs, handler)

      // File output
      val dir = File(reactCtx.getExternalFilesDir(null), "har"); dir.mkdirs()
      currentFile = File(dir, "har_${label}_${rateHz}hz_${System.currentTimeMillis()}.csv")
      out = FileOutputStream(currentFile)
      out!!.write(("# label=$label\n# target_hz=$rateHz\n# duration_s=$durationSec\n").toByteArray())
      out!!.write("no;time;seconds_elapsed;accel_x;accel_y;accel_z;gyro_x;gyro_y;gyro_z;mag_x;mag_y;mag_z;label\n".toByteArray())

      // Scheduler untuk sampling
      samples = 0
      startNs = SystemClock.elapsedRealtimeNanos()
      lastEmitNs = 0
      val periodNs = 1_000_000_000L / rateHz

      running.set(true)
      task = scheduler.scheduleAtFixedRate({
        if (!running.get()) return@scheduleAtFixedRate

        val t = SystemClock.elapsedRealtimeNanos()
        val elapsed = (t - startNs) / 1_000_000_000.0

        val humanLabel = label.replace('_', ' ')
          .split(' ')
          .joinToString(" ") { it.replaceFirstChar { c -> c.titlecase(Locale.getDefault()) } }

        val line = String.format(
          Locale.US,
          "%d;%.2E;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%s\n",
          samples,
          t.toDouble(),
          elapsed,
          lastAcc.x, lastAcc.y, lastAcc.z,
          lastGyr.x, lastGyr.y, lastGyr.z,
          lastMag.x, lastMag.y, lastMag.z,
          humanLabel
        )
        out?.write(line.toByteArray())
        samples++

        // kirim progress tiap 0.5s
        if (t - lastEmitNs >= 500_000_000L) {
          val m = Arguments.createMap().apply {
            putInt("samples", samples)
            putInt("expected", maxSamples)
            putDouble("elapsedSec", elapsed)
            putString("path", currentFile.absolutePath)
          }
          sendEvent("SensorRecorderProgress", m)
          lastEmitNs = t
        }

        // auto stop kalau bukan mode manual
        if (!unlimited && samples >= maxSamples) {
          stopInternal()
        }

      }, 0, periodNs, TimeUnit.NANOSECONDS)

      promise.resolve(currentFile.absolutePath)
    } catch (e: Exception) {
      stopInternal()
      promise.reject("E_START", e)
    }
  }

  @ReactMethod
  fun stopRecording(promise: Promise) {
    stopInternal()
    promise.resolve(null)
  }
  @ReactMethod
    fun shareLast(promise: Promise) {
    try {
        if (!this::currentFile.isInitialized) {
        promise.reject("E_NO_FILE", "No file"); return
        }
        val uri = FileProvider.getUriForFile(
        reactCtx,
        reactCtx.packageName + ".fileprovider",
        currentFile
        )
        val intent = Intent(Intent.ACTION_SEND).apply {
        type = "text/csv"
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_TITLE, currentFile.name)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        clipData = ClipData.newUri(reactCtx.contentResolver, currentFile.name, uri) // ← penting
        }
        val chooser = Intent.createChooser(intent, "Share CSV").apply {
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        reactCtx.startActivity(chooser)
        promise.resolve(uri.toString())
    } catch (e: Exception) {
        promise.reject("E_SHARE", e)
    }
    }
@ReactMethod
fun sharePath(path: String, promise: Promise) {
  try {
    val file = File(path)
    if (!file.exists()) { promise.reject("E_NO_FILE", "File not found"); return }

    val uri = FileProvider.getUriForFile(
      reactCtx,
      reactCtx.packageName + ".fileprovider",
      file
    )

    val intent = Intent(Intent.ACTION_SEND).apply {
      type = "text/csv"
      putExtra(Intent.EXTRA_STREAM, uri)
      putExtra(Intent.EXTRA_TITLE, file.name)
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
      clipData = ClipData.newUri(reactCtx.contentResolver, file.name, uri)
    }

    val chooser = Intent.createChooser(intent, "Share CSV").apply {
      addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
    reactCtx.startActivity(chooser)
    promise.resolve(uri.toString())
  } catch (e: Exception) {
    promise.reject("E_SHARE", e)
  }
}

  private fun stopInternal() {
    if (!running.getAndSet(false)) return
    try { task?.cancel(true) } catch (_:Exception) {}
    task = null

    try {
      sensorManager.unregisterListener(accListener)
      sensorManager.unregisterListener(gyrListener)
      sensorManager.unregisterListener(magListener)
    } catch (_:Exception) {}

    try { handlerThread?.quitSafely() } catch (_:Exception) {}
    handlerThread = null; handler = null

    try { out?.flush(); out?.close() } catch (_:Exception) {}
    out = null

    // emit complete
    try {
      val m = Arguments.createMap().apply {
        putInt("samples", samples)
        putInt("expected", maxSamples)
        putDouble("elapsedSec", (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000_000.0)
        putString("path", if (this@SensorRecorderModule::currentFile.isInitialized) currentFile.absolutePath else null)
      }
      sendEvent("SensorRecorderComplete", m)
    } catch (_:Exception) {}

    try { if (wakeLock?.isHeld == true) wakeLock?.release() } catch (_:Exception) {}
    wakeLock = null
  }

  override fun initialize() { reactApplicationContext.addLifecycleEventListener(this) }
  override fun onCatalystInstanceDestroy() { stopInternal() }
  override fun onHostResume() {}
  override fun onHostPause() {}
  override fun onHostDestroy() { stopInternal() }
}
