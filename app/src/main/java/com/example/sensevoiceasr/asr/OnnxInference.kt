package com.example.sensevoiceasr.asr

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * ONNX Runtime inference wrapper for SenseVoiceSmall.
 * Loads model from internal storage (copied from assets on first run)
 * for memory-mapped file access instead of loading entire 232MB into RAM.
 */
class OnnxInference(private val context: Context) {
    companion object {
        private const val TAG = "OnnxInference"
        private const val MODEL_FILE = "model_quant.onnx"
    }

    private var session: OrtSession? = null
    private var environment: OrtEnvironment? = null
    private var featureExtractor: FeatureExtractor? = null
    private var tokenizer: Tokenizer? = null

    @Volatile
    var isLoaded = false
        private set

    /**
     * Returns the path to the model file in internal storage.
     * Copies from assets on first call.
     */
    private fun getModelPath(): File {
        val modelDir = File(context.filesDir, "models")
        if (!modelDir.exists()) modelDir.mkdirs()
        val modelFile = File(modelDir, MODEL_FILE)

        if (!modelFile.exists()) {
            Log.i(TAG, "Copying model from assets to internal storage...")
            val startTime = System.currentTimeMillis()
            context.assets.open(MODEL_FILE).use { input ->
                FileOutputStream(modelFile).use { output ->
                    val buffer = ByteArray(8192)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                    }
                }
            }
            Log.i(TAG, "Model copied in ${System.currentTimeMillis() - startTime}ms")
        }
        return modelFile
    }

    fun loadModel(): Boolean {
        return try {
            Log.i(TAG, "Loading ONNX model from file...")
            val startTime = System.currentTimeMillis()

            environment = OrtEnvironment.getEnvironment()

            // Load from file path (memory-mapped, not byte array)
            val modelPath = getModelPath().absolutePath
            val options = OrtSession.SessionOptions().apply {
                // Optimize for mobile: use CPU with reduced memory
                addConfigEntry("session.intra_op_num_threads", "2")
                addConfigEntry("session.inter_op_num_threads", "1")
                addConfigEntry("session.execution_mode", "0") // SEQUENTIAL
                addConfigEntry("session.graph_optimization_level", "1") // BASIC
            }
            session = environment?.createSession(modelPath, options)

            featureExtractor = FeatureExtractor()
            tokenizer = Tokenizer(context)
            tokenizer?.load()

            isLoaded = true
            val loadTime = System.currentTimeMillis() - startTime
            Log.i(TAG, "Model loaded in ${loadTime}ms")
            true
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Out of memory loading model: ${e.message}")
            isLoaded = false
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            isLoaded = false
            false
        }
    }

    /**
     * Transcribe a float audio array (16kHz mono, range [-1, 1]).
     * Returns the transcribed text.
     */
    fun transcribe(audio: FloatArray): String {
        if (!isLoaded) return ""

        try {
            val extractor = featureExtractor ?: return ""
            val sess = session ?: return ""
            val env = environment ?: return ""

            val (features, featLen) = extractor.extract(audio)
            if (features.isEmpty()) return ""

            val numFrames = features.size
            val featDim = features[0].size
            val flatFeatures = FloatArray(1 * numFrames * featDim)
            for (i in 0 until numFrames) {
                System.arraycopy(features[i], 0, flatFeatures, i * featDim, featDim)
            }

            val speechShape = longArrayOf(1, numFrames.toLong(), featDim.toLong())
            val speechTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatFeatures), speechShape)

            val speechLengths = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(numFrames.toLong())), longArrayOf(1))
            val language = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(0)), longArrayOf(1))
            val textnorm = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(1)), longArrayOf(1))

            val inputs = mapOf(
                "speech" to speechTensor,
                "speech_lengths" to speechLengths,
                "language" to language,
                "textnorm" to textnorm
            )

            val outputs = sess.run(inputs)
            val outputEntry = outputs.iterator().next()
            val outputTensor = outputEntry.value as? OnnxTensor ?: return ""
            val rawValue = outputTensor.value

            val tokenIds = when (rawValue) {
                is Array<*> -> {
                    if (rawValue.isNotEmpty() && rawValue[0] is Array<*>) {
                        val batch0 = rawValue[0] as Array<*>
                        IntArray(batch0.size) { frame ->
                            val frameArr = batch0[frame] as FloatArray
                            var maxIdx = 0
                            var maxVal = frameArr[0]
                            for (i in 1 until frameArr.size) {
                                if (frameArr[i] > maxVal) {
                                    maxVal = frameArr[i]
                                    maxIdx = i
                                }
                            }
                            maxIdx
                        }
                    } else {
                        IntArray(0)
                    }
                }
                else -> IntArray(0)
            }

            val result = tokenizer?.decode(tokenIds) ?: ""

            // Cleanup
            speechTensor.close()
            speechLengths.close()
            language.close()
            textnorm.close()
            outputTensor.close()
            outputs.close()

            return result
        } catch (e: Exception) {
            Log.e(TAG, "Transcription error: ${e.message}", e)
            return ""
        }
    }

    fun release() {
        session?.close()
        environment?.close()
        session = null
        environment = null
        isLoaded = false
    }
}