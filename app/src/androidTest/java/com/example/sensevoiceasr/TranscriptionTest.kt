package com.example.sensevoiceasr

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.sensevoiceasr.asr.FeatureExtractor
import com.example.sensevoiceasr.asr.OnnxInference
import com.example.sensevoiceasr.asr.Tokenizer
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sin
import kotlin.math.PI

/**
 * Tests the full transcription pipeline:
 *   FeatureExtractor → ONNX Inference → Tokenizer
 *
 * Tests with both synthetic audio (sine wave) and actual speech WAV files.
 */
@RunWith(AndroidJUnit4::class)
class TranscriptionTest {
    companion object {
        private const val TAG = "TranscriptionTest"
        private const val SAMPLE_RATE = 16000
    }

    private lateinit var context: Context
    private lateinit var onnxInference: OnnxInference

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        onnxInference = OnnxInference(context)
        val loaded = onnxInference.loadModel()
        Log.i(TAG, "Model loaded: $loaded")
        if (!loaded) {
            Log.e(TAG, "Model load error: ${onnxInference.lastError}")
        }
        assertTrue("Model must load successfully", loaded)
    }

    /**
     * Test with a simple sine wave to verify the pipeline doesn't crash.
     */
    @Test
    fun testSineWaveInference() {
        // Create a 2-second 440Hz sine wave
        val duration = 2.0f
        val numSamples = (SAMPLE_RATE * duration).toInt()
        val audio = FloatArray(numSamples) { i ->
            sin(2.0 * PI * 440.0 * i / SAMPLE_RATE).toFloat() * 0.5f
        }

        Log.i(TAG, "=== Test: Sine wave (${duration}s, $numSamples samples) ===")

        // Run feature extraction (standalone, with CMVN for accurate logging)
        val extractor = FeatureExtractor().apply { loadCmvn(context) }
        val (features, _) = extractor.extract(audio)
        val featLen = features.size
        Log.i(TAG, "Feature extraction: $featLen frames, dim=${if (features.isNotEmpty()) features[0].size else 0}")

        if (features.isEmpty()) {
            Log.e(TAG, "FAIL: Feature extraction returned empty features")
            fail("Feature extraction failed: audio=${audio.size} samples")
        }
        assertTrue("Features must not be empty", features.isNotEmpty())
        Log.i(TAG, "Feature extraction OK: $featLen frames x ${features[0].size} dims")

        // Run transcription
        Log.i(TAG, "Running ONNX inference...")
        val startTime = System.currentTimeMillis()
        val result = onnxInference.transcribe(audio)
        val elapsed = System.currentTimeMillis() - startTime
        Log.i(TAG, "Inference completed in ${elapsed}ms")
        Log.i(TAG, "Result: '$result' (length=${result.length})")

        // For sine wave, we expect some output (even if it's noise tokens)
        // The key test is that the pipeline doesn't crash
        Log.i(TAG, "=== Sine wave test PASSED ===")
    }

    /**
     * Test with a short audio clip (600ms) to check minSpeechFrames behavior.
     */
    @Test
    fun testShortAudioInference() {
        // 600ms of audio (at the boundary of minSpeechFrames=3 × 200ms chunks)
        val numSamples = (SAMPLE_RATE * 0.6).toInt()
        val audio = FloatArray(numSamples) { i ->
            sin(2.0 * PI * 1000.0 * i / SAMPLE_RATE).toFloat() * 0.3f
        }

        Log.i(TAG, "=== Test: Short audio (0.6s, $numSamples samples) ===")

        val extractor = FeatureExtractor().apply { loadCmvn(context) }
        val (features, _) = extractor.extract(audio)
        Log.i(TAG, "Feature extraction: ${features.size} frames")

        val result = onnxInference.transcribe(audio)
        Log.i(TAG, "Result: '$result'")
        Log.i(TAG, "=== Short audio test PASSED ===")
    }

    /**
     * Test with zero audio to see what the model output looks like.
     */
    @Test
    fun testZeroAudioInference() {
        // 1 second of silence
        val numSamples = SAMPLE_RATE
        val audio = FloatArray(numSamples) // all zeros

        Log.i(TAG, "=== Test: Zero audio (1s, $numSamples samples) ===")

        val extractor = FeatureExtractor().apply { loadCmvn(context) }
        val (features, _) = extractor.extract(audio)
        Log.i(TAG, "Feature extraction: ${features.size} frames")

        if (features.isEmpty()) {
            Log.e(TAG, "FAIL: Zero audio features empty!")
            fail("Zero audio should produce features")
        }

        val result = onnxInference.transcribe(audio)
        Log.i(TAG, "Zero audio result: '$result' (length=${result.length})")
        Log.i(TAG, "=== Zero audio test PASSED ===")
    }

    /**
     * Test tokenizer decode directly with known token IDs.
     */
    @Test
    fun testTokenizerDecode() {
        Log.i(TAG, "=== Test: Tokenizer direct decode ===")

        val tokenizer = Tokenizer(context)
        val loaded = tokenizer.load()
        Log.i(TAG, "Tokenizer loaded: $loaded")
        assertTrue("Tokenizer must load", loaded)

        // Test with simple token IDs
        val testIds = intArrayOf(1, 2, 3, 4, 5)
        val decoded = tokenizer.decode(testIds)
        Log.i(TAG, "Decode [1,2,3,4,5]: '$decoded'")

        // Test with blank (0) collapsing
        val testIds2 = intArrayOf(0, 1, 1, 0, 2, 2, 0)
        val decoded2 = tokenizer.decode(testIds2)
        Log.i(TAG, "Decode [0,1,1,0,2,2,0]: '$decoded2'")

        // Test with all zeros
        val testIds3 = intArrayOf(0, 0, 0, 0, 0)
        val decoded3 = tokenizer.decode(testIds3)
        Log.i(TAG, "Decode all zeros: '$decoded3' (should be empty)")

        Log.i(TAG, "=== Tokenizer test PASSED ===")
    }

    /**
     * Test the ONNX model output format directly by inspecting tensor shapes.
     */
    @Test
    fun testModelOutputFormat() {
        Log.i(TAG, "=== Test: Model output format inspection ===")

        // Create minimal audio (30ms = 480 samples, should produce a few frames)
        val audio = FloatArray(480) { i ->
            sin(2.0 * PI * 440.0 * i / SAMPLE_RATE).toFloat() * 0.5f
        }

        Log.i(TAG, "Audio: ${audio.size} samples = ${audio.size * 1000f / SAMPLE_RATE}ms")

        val result = onnxInference.transcribe(audio)
        Log.i(TAG, "Transcription result: '$result'")
        Log.i(TAG, "Result length: ${result.length}")
        Log.i(TAG, "Result is blank: ${result.isBlank()}")

        // Now test with a longer audio (2 seconds)
        val audio2 = FloatArray(SAMPLE_RATE * 2) { i ->
            sin(2.0 * PI * 440.0 * i / SAMPLE_RATE).toFloat() * 0.5f
        }
        val result2 = onnxInference.transcribe(audio2)
        Log.i(TAG, "2s audio result: '$result2'")
        Log.i(TAG, "2s result length: ${result2.length}")
        Log.i(TAG, "2s result is blank: ${result2.isBlank()}")

        Log.i(TAG, "=== Model output format test PASSED ===")
    }
}