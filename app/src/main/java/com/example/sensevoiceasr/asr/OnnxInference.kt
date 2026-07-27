package com.example.sensevoiceasr.asr

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import ai.onnxruntime.OrtException
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.sqrt

/**
 * ONNX Runtime inference wrapper for SenseVoiceSmall.
 * Loads model from internal storage (copied from assets on first run)
 * for memory-mapped file access instead of loading entire 133MB into RAM.
 *
 * Emits detailed diagnostic logs via a callback so the UI can display them.
 */
class OnnxInference(private val context: Context) {
    companion object {
        private const val TAG = "OnnxInference"
        private const val MODEL_FILE = "model_int4.onnx"
    }

    private var session: OrtSession? = null
    private var environment: OrtEnvironment? = null
    private var featureExtractor: FeatureExtractor? = null
    private var tokenizer: Tokenizer? = null

    /** Optional callback to receive human-readable diagnostic log lines. */
    var logCallback: ((String) -> Unit)? = null

    private fun log(msg: String) {
        Log.i(TAG, msg)
        logCallback?.invoke(msg)
    }

    private fun logWarn(msg: String) {
        Log.w(TAG, msg)
        logCallback?.invoke("[WARN] $msg")
    }

    private fun logErr(msg: String) {
        Log.e(TAG, msg)
        logCallback?.invoke("[ERR] $msg")
    }

    @Volatile
    var isLoaded = false
        private set

    @Volatile
    var lastError: String? = null
        private set

    private fun getModelPath(): File {
        val modelDir = File(context.filesDir, "models")
        if (!modelDir.exists()) modelDir.mkdirs()
        val modelFile = File(modelDir, MODEL_FILE)

        if (!modelFile.exists()) {
            log("Copying model from assets to internal storage...")
            val startTime = System.currentTimeMillis()
            context.assets.open(MODEL_FILE).use { input ->
                FileOutputStream(modelFile).use { output ->
                    val buffer = ByteArray(8192)
                    var read: Int
                    var totalBytes = 0L
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        totalBytes += read
                    }
                    log("Copied $totalBytes bytes...")
                }
            }
            log("Model copied in ${System.currentTimeMillis() - startTime}ms (${modelFile.length() / (1024 * 1024)}MB on disk)")
        } else {
            log("Model already exists at ${modelFile.absolutePath} (${modelFile.length() / (1024 * 1024)}MB)")
        }
        return modelFile
    }

    fun loadModel(): Boolean {
        return try {
            log("========== LOADING MODEL ==========")
            val startTime = System.currentTimeMillis()

            environment = OrtEnvironment.getEnvironment()
            log("ONNX Runtime environment created (version ${environment?.javaClass?.`package`?.implementationVersion ?: "?"})")

            val modelPath = getModelPath().absolutePath
            log("Creating session from file: $modelPath")

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.intra_op_num_threads", "2")
                addConfigEntry("session.inter_op_num_threads", "1")
                addConfigEntry("session.execution_mode", "0")
                addConfigEntry("session.graph_optimization_level", "1")
                log("Session options: intra=2, inter=1, exec=SEQUENTIAL, opt=BASIC")
            }
            session = environment?.createSession(modelPath, options)
            log("Session created OK")

            val sess = session ?: throw IllegalStateException("Session is null after creation")
            log("--- Model Input Info ---")
            for ((name, info) in sess.inputInfo) {
                log("  INPUT  name=$name  nodeInfo=${info.toString().replace('\n', ' ')}")
            }
            log("--- Model Output Info ---")
            for ((name, info) in sess.outputInfo) {
                log("  OUTPUT name=$name  nodeInfo=${info.toString().replace('\n', ' ')}")
            }

            log("Initializing FeatureExtractor and loading CMVN...")
            featureExtractor = FeatureExtractor().apply {
                val cmvnOk = loadCmvn(context)
                log("CMVN load result: $cmvnOk")
            }
            log("Initializing Tokenizer...")
            tokenizer = Tokenizer(context)
            val tokLoaded = tokenizer?.load() ?: false
            log("Tokenizer load result: $tokLoaded")

            isLoaded = true
            val loadTime = System.currentTimeMillis() - startTime
            log("========== MODEL LOADED in ${loadTime}ms ==========")
            true
        } catch (e: OutOfMemoryError) {
            val msg = "Out of memory: ${e.message}"
            logErr(msg)
            lastError = msg
            isLoaded = false
            false
        } catch (e: OrtException) {
            val msg = "ONNX error: ${e.message}"
            logErr(msg)
            lastError = msg
            isLoaded = false
            false
        } catch (e: Exception) {
            val msg = "Failed to load: ${e.message}"
            logErr(msg)
            lastError = msg
            isLoaded = false
            false
        }
    }

    /**
     * Transcribe a float audio array (16kHz mono, range [-1, 1]).
     * Returns the transcribed text, and emits extensive diagnostics.
     */
    fun transcribe(audio: FloatArray): String {
        if (!isLoaded) {
            logWarn("transcribe() called but model not loaded")
            return ""
        }

        val totalStart = System.currentTimeMillis()
        log("========== TRANSCRIBE START ==========")
        log("Audio input: ${audio.size} samples = ${String.format("%.2f", audio.size / 16000.0f)}s @ 16kHz")

        // Audio stats
        var minA = Float.MAX_VALUE
        var maxA = Float.MIN_VALUE
        var sumA = 0.0
        var sumSq = 0.0
        for (s in audio) {
            if (s < minA) minA = s
            if (s > maxA) maxA = s
            sumA += s
            sumSq += (s * s).toDouble()
        }
        val meanA = (sumA / audio.size).toFloat()
        val rmsA = sqrt((sumSq / audio.size).toFloat())
        log("Audio stats: min=$minA, max=$maxA, mean=$meanA, rms=$rmsA")
        if (rmsA < 0.001f) {
            logWarn("Audio RMS is extremely low (< 0.001) - input may be silence or mic not capturing")
        }

        try {
            val extractor = featureExtractor ?: return ""
            val sess = session ?: return ""
            val env = environment ?: return ""
            val tok = tokenizer

            // ---- Feature Extraction ----
            log("--- Step 1/5: Feature Extraction ---")
            val feStart = System.currentTimeMillis()
            val (features, _) = extractor.extract(audio)
            val feMs = System.currentTimeMillis() - feStart
            if (features.isEmpty()) {
                logWarn("FeatureExtractor returned EMPTY features! Aborting inference.")
                return ""
            }
            val numFrames = features.size
            val featDim = features[0].size
            log("Features: $numFrames frames x $featDim dims (total=${numFrames * featDim} floats) in ${feMs}ms")
            log("Features[0][0:10]: ${features[0].take(10).map { String.format("%.4f", it) }}")
            log("Features[mid][0:10]: ${features[numFrames / 2].take(10).map { String.format("%.4f", it) }}")
            log("Features[last][0:10]: ${features.last().take(10).map { String.format("%.4f", it) }}")

            // Flatten
            val flatFeatures = FloatArray(1 * numFrames * featDim)
            for (i in 0 until numFrames) {
                System.arraycopy(features[i], 0, flatFeatures, i * featDim, featDim)
            }

            // ---- Inference Inputs ----
            log("--- Step 2/5: Building ONNX Input Tensors ---")
            val speechShape = longArrayOf(1, numFrames.toLong(), featDim.toLong())
            val speechTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatFeatures), speechShape)
            val speechLengths = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(numFrames)), longArrayOf(1))
            val language = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(0)), longArrayOf(1))
            val textnorm = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(1)), longArrayOf(1))
            log("speech shape=${speechShape.contentToString()} dtype=float32")
            log("speech_lengths=[$numFrames] (int32)")
            log("language=[0] (int32, 0=auto)")
            log("textnorm=[1] (int32, 1=enable ITN)")

            val inputs = mapOf(
                "speech" to speechTensor,
                "speech_lengths" to speechLengths,
                "language" to language,
                "textnorm" to textnorm
            )

            // ---- Run Inference ----
            log("--- Step 3/5: Running ONNX session.run() ---")
            val infStart = System.currentTimeMillis()
            val outputs = sess.run(inputs)
            val infMs = System.currentTimeMillis() - infStart
            // OrtSession.Result is Iterable<Map.Entry<...>> — count entries via iteration
            var outputCount = 0
            val outIter = outputs.iterator()
            while (outIter.hasNext()) { outIter.next(); outputCount++ }
            log("session.run() returned $outputCount outputs in ${infMs}ms")

            val outputEntry = outputs.iterator().next()
            val outputTensor = outputEntry.value as? OnnxTensor
            if (outputTensor == null) {
                logErr("Output is not OnnxTensor! Class=${outputEntry.value.javaClass.name}")
                return ""
            }
            val outputInfo = outputTensor.info
            val outputShape = outputInfo.shape
            log("Output tensor: name=${outputEntry.key}  type=${outputInfo.type}  shape=${outputShape?.contentToString()}")

            // ---- Logits / Token Stats ----
            log("--- Step 4/5: CTC Logit Analysis & Token IDs ---")
            val rawValue = outputTensor.value
            val tokenIds: IntArray
            var allZero = true
            var maxNonBlankId = 0
            var maxNonBlankProb = -1.0f

            when (rawValue) {
                is Array<*> -> {
                    if (rawValue.isNotEmpty() && rawValue[0] is Array<*>) {
                        val batch0 = rawValue[0] as Array<*>
                        log("CTC logits: T=${batch0.size} frames, vocab dim check...")
                        tokenIds = IntArray(batch0.size) { frame ->
                            val frameArr = batch0[frame] as FloatArray
                            var maxIdx = 0
                            var maxVal = frameArr[0]
                            for (i in 1 until frameArr.size) {
                                if (frameArr[i] > maxVal) {
                                    maxVal = frameArr[i]
                                    maxIdx = i
                                }
                            }
                            if (maxIdx != 0) allZero = false
                            if (maxIdx != 0 && maxVal > maxNonBlankProb) {
                                maxNonBlankProb = maxVal
                                maxNonBlankId = maxIdx
                            }
                            maxIdx
                        }
                        // Report logits distribution
                        val (lMin, lMax, lMean) = run {
                            var mn = Float.MAX_VALUE
                            var mx = Float.MIN_VALUE
                            var sm = 0.0
                            var cnt = 0
                            for (f in batch0) {
                                val fa = f as FloatArray
                                for (v in fa) {
                                    if (v < mn) mn = v
                                    if (v > mx) mx = v
                                    sm += v
                                    cnt++
                                }
                            }
                            Triple(mn, mx, (sm / cnt).toFloat())
                        }
                        log("Logits stats: min=$lMin, max=$lMax, mean=$lMean over T=${batch0.size} frames")
                        log("Highest-confidence non-blank token: id=$maxNonBlankId prob=$maxNonBlankProb")
                    } else {
                        logWarn("Output is Array but empty or not 2D, size=${rawValue.size}")
                        tokenIds = IntArray(0)
                    }
                }
                else -> {
                    logErr("Unexpected output type: ${rawValue.javaClass.name}")
                    tokenIds = IntArray(0)
                }
            }

            log("Token IDs count: ${tokenIds.size}")
            log("All-blank (all 0): $allZero")
            if (tokenIds.isNotEmpty()) {
                // Print ALL token IDs, not just first 20
                log("ALL Token IDs [${tokenIds.size}]: ${tokenIds.joinToString()}")
                val unique = tokenIds.toSet()
                log("Unique tokens: ${unique.size} ids = ${unique.sorted().joinToString()}")
            }

            // ---- Decoding ----
            log("--- Step 5/5: Decoding ---")
            var rawDecoded = ""
            // Decode WITHOUT stripping special tokens first so we can inspect them
            if (tok != null) {
                val field = Tokenizer::class.java.getDeclaredField("idToToken")
                field.isAccessible = true
                @Suppress("UNCHECKED_CAST")
                val idToTok = field.get(tok) as Map<Int, String>
                val preFilter = mutableListOf<String>()
                var prev = -1
                for (id in tokenIds) {
                    if (id != prev && id != 0) {
                        idToTok[id]?.let { preFilter.add(it) }
                    }
                    prev = id
                }
                val preStr = preFilter.joinToString("").replace("▁", " ")
                rawDecoded = preStr
                log("Decoded (BEFORE special-token filter): \"$preStr\"")
                // Show token-by-token
                val prev2 = mutableListOf<String>()
                var p = -1
                for (id in tokenIds) {
                    if (id != p && id != 0) {
                        idToTok[id]?.let { prev2.add("$id:$it") }
                    }
                    p = id
                }
                log("Tokens after CTC collapse: ${prev2.joinToString(", ")}")
            }

            val result = tok?.decode(tokenIds) ?: run {
                logWarn("Tokenizer not available, using token-IDs fallback")
                rawDecoded
            }
            log("Decoded (AFTER  special-token filter): \"$result\"")
            val totalMs = System.currentTimeMillis() - totalStart
            log("========== TRANSCRIBE END in ${totalMs}ms ==========")

            // Cleanup
            speechTensor.close()
            speechLengths.close()
            language.close()
            textnorm.close()
            outputTensor.close()
            outputs.close()

            return result
        } catch (e: Exception) {
            logErr("Transcription error: ${e.javaClass.simpleName}: ${e.message}")
            val st = e.stackTrace.take(10).joinToString("\n") { "    at $it" }
            Log.e(TAG, "Stack trace:\n$st")
            logErr("See logcat for full stack trace")
            return ""
        }
    }

    fun release() {
        log("Releasing ONNX session...")
        session?.close()
        environment?.close()
        session = null
        environment = null
        isLoaded = false
        log("Released.")
    }
}