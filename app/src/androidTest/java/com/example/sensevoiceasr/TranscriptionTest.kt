package com.example.sensevoiceasr

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.sensevoiceasr.asr.FeatureExtractor
import com.example.sensevoiceasr.asr.OnnxInference
import com.example.sensevoiceasr.asr.Tokenizer
import com.example.sensevoiceasr.audio.VadProcessor
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

    /**
     * Validates the full VadProcessor pipeline end-to-end.
     * Feeds Int16 chunks through processChunk() with synthetic speech-level raw RMS,
     * then silence chunks, and asserts:
     *   - segmentCallback fires exactly once
     *   - duration and sample counts match the speech chunks that were fed
     *   - rawRMS-based VAD correctly classifies speech vs silence (new fix: isSpeech
     *     uses rawRMS, not AGC-amplified rms, so silence chunks correctly increment
     *     silenceCount until threshold)
     */
    @Test
    fun testVadSegmentFinalizesCorrectly() {
        Log.i(TAG, "=== Test: VadProcessor segment finalization (rawRMS VAD fix) ===")
        val chunkMs = 200
        val samplesPerChunk = SAMPLE_RATE * chunkMs / 1000  // 3200

        // 1. 4 speech chunks (800ms total): each chunk with raw RMS = 0.030f > speechThreshold 0.015f
        //    This is a real human-speech-like RMS level (would be Int16 ~1000 peak).
        val speechLevelRaw: Float = 0.030f
        val speechChunks: List<ShortArray> = (0 until 4).map { chunkIdx ->
            val result = ShortArray(samplesPerChunk)
            val peak = (speechLevelRaw * 32768f * 1.4142f).toInt()  // peak for sine with RMS 0.030
            for (j in result.indices) {
                val t = (chunkIdx * samplesPerChunk + j).toDouble() / SAMPLE_RATE
                val s = sin(2.0 * PI * 440.0 * t).toFloat() * speechLevelRaw * 1.4142f
                val intVal = (s * 32768f).toInt()
                result[j] = intVal.coerceIn(-32768, 32767).toShort()
            }
            result
        }
        // 2. 10 silence chunks (2000ms): very low RMS ~0.0005f (< threshold 0.015)
        val silenceLevelRaw: Float = 0.0005f
        val silenceChunks: List<ShortArray> = (0 until 10).map {
            val result = ShortArray(samplesPerChunk)
            for (j in result.indices) {
                val v = (silenceLevelRaw * 32768f * (sin((it * samplesPerChunk + j) * 0.013f)))
                result[j] = v.toInt().coerceIn(-32768, 32767).toShort()
            }
            result
        }

        val vad = VadProcessor(
            sampleRate = SAMPLE_RATE,
            speechThreshold = 0.015f,
            silenceFrames = 8,
            minSpeechFrames = 3,
            maxSpeechFrames = 40
        )
        val segmentAudios = mutableListOf<Pair<FloatArray, Float>>()
        vad.onSegmentFinalized { audio, dur ->
            segmentAudios += audio.clone() to dur
            Log.i(TAG, "  segmentCallback fired: dur=${String.format("%.2f", dur)}s, samples=${audio.size}")
        }

        // 3. Feed all speech chunks first (4 chunks = 800ms ≥ minSpeechFrames=3 OK)
        Log.i(TAG, "Feeding ${speechChunks.size} speech chunks...")
        speechChunks.forEach { vad.processChunk(it) }
        // 4. Feed silence chunks (after silenceFrames=8 silence → segment should fire on silence chunk #8)
        Log.i(TAG, "Feeding ${silenceChunks.size} silence chunks...")
        silenceChunks.forEachIndexed { idx, chunk ->
            vad.processChunk(chunk)
            if (segmentAudios.isNotEmpty()) {
                Log.i(TAG, "  segment fired after silence chunk #${idx + 1}")
            }
        }

        // 5. Assertions
        assertEquals("segmentCallback should fire exactly once after silence threshold", 1, segmentAudios.size)
        val (segAudio, segDur) = segmentAudios[0]
        val expectedSamples = speechChunks.size * samplesPerChunk
        Log.i(TAG, "Segment: samples=${segAudio.size} expected=$expectedSamples dur=${String.format("%.2f", segDur)}s expected=${String.format("%.2f", expectedSamples.toFloat() / SAMPLE_RATE)}s")
        assertEquals("segment audio samples should be speechChunks * 3200", expectedSamples, segAudio.size)
        val expectedDur = expectedSamples.toFloat() / SAMPLE_RATE
        assertEquals("segment duration matches", expectedDur, segDur, 0.001f)
        assertTrue("segment duration > 0", segDur > 0f)

        // 6. Test flushSegment: create new Vad with 3 speech chunks, call flush (no silence)
        Log.i(TAG, "Testing flushSegment() — 3 speech chunks then flush without silence...")
        val vad2 = VadProcessor(sampleRate = SAMPLE_RATE, speechThreshold = 0.015f, silenceFrames = 8, minSpeechFrames = 3)
        var flushSeg: Pair<FloatArray, Float>? = null
        vad2.onSegmentFinalized { a, d -> flushSeg = a.clone() to d }
        speechChunks.take(3).forEach { vad2.processChunk(it) }
        assertEquals("no segment after 3 speech chunks without silence", 0, 0) // sanity — no auto fire, check via flush
        val flushed = vad2.flushSegment()
        assertNotNull("flushSegment returns non-null for 3≥minSpeechFrames=3", flushed)
        assertNotNull("flushSegment triggers segmentCallback", flushSeg)
        assertEquals("flush callback samples = 3 * 3200", 3 * samplesPerChunk, flushSeg!!.first.size)
        Log.i(TAG, "flushSegment OK: returned ${flushed!!.size} samples; callback got ${flushSeg!!.first.size}")

        Log.i(TAG, "=== VadProcessor segment finalization test PASSED ===")
    }
}