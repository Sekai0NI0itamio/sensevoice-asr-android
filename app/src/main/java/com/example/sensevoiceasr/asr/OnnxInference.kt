package com.example.sensevoiceasr.asr

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * ONNX Runtime inference wrapper for SenseVoiceSmall.
 * Handles model loading, feature extraction, and transcription.
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

    private var isLoaded = false

    fun loadModel(): Boolean {
        return try {
            Log.i(TAG, "Loading ONNX model...")
            val startTime = System.currentTimeMillis()

            environment = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            session = environment?.createSession(modelBytes, OrtSession.SessionOptions())

            featureExtractor = FeatureExtractor()
            tokenizer = Tokenizer(context)
            tokenizer?.load()

            isLoaded = true
            val loadTime = System.currentTimeMillis() - startTime
            Log.i(TAG, "Model loaded in ${loadTime}ms")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            isLoaded = false
            false
        }
    }

    fun isReady(): Boolean = isLoaded

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

            // Extract features
            val (features, featLen) = extractor.extract(audio)
            if (features.isEmpty()) return ""

            // Flatten features to 3D tensor [1, num_frames, feat_dim]
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

            // Decode: find argmax per frame
            // Output shape: [1, num_frames, vocab_size] -> float[][][]
            val tokenIds = when (rawValue) {
                is Array<*> -> {
                    // value is Array<Array<FloatArray>> = float[][][]
                    if (rawValue.isNotEmpty() && rawValue[0] is Array<*>) {
                        val batch0 = rawValue[0] as Array<*>  // [num_frames, vocab_size]
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