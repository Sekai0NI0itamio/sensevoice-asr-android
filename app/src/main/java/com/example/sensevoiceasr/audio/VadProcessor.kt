package com.example.sensevoiceasr.audio

import android.util.Log
import kotlin.math.*

/**
 * Energy-based Voice Activity Detection and preprocessing.
 * Includes AGC (Automatic Gain Control) and soft limiting.
 * Emits per-chunk and per-segment diagnostics via log callback.
 */
class VadProcessor(
    private val sampleRate: Int = 16000,
    private val speechThreshold: Float = 0.01f,
    private val silenceFrames: Int = 8,
    private val minSpeechFrames: Int = 3
) {
    companion object {
        private const val TAG = "VadProcessor"
    }

    var logCallback: ((String) -> Unit)? = null
    private var chunkCount: Int = 0

    private fun log(msg: String) {
        Log.i(TAG, msg)
        logCallback?.invoke("[Vad] $msg")
    }

    // AGC state
    private var agcGainSmooth = 1.0f
    private var prevSample = 0.0f

    // Speech tracking
    private var speechBuffer = mutableListOf<FloatArray>()
    private var silenceCount = 0
    private var speechFrameCount = 0

    private var segmentCallback: ((FloatArray, Float) -> Unit)? = null

    fun onSegmentFinalized(callback: (audio: FloatArray, durationSec: Float) -> Unit) {
        segmentCallback = callback
    }

    /**
     * Process an audio chunk. Returns the processed (AGC'd) float audio if speech was detected.
     * Calls segmentCallback when a speech segment ends.
     */
    fun processChunk(rawChunk: ShortArray): FloatArray? {
        chunkCount++

        // Convert to float [-1, 1]
        val floatAudio = FloatArray(rawChunk.size) { rawChunk[it] / 32768f }

        // Raw RMS before AGC, for VAD decision comparison
        val rawRms = run {
            var s = 0.0
            for (v in floatAudio) s += (v * v).toDouble()
            sqrt(s / floatAudio.size).toFloat()
        }

        // Apply AGC
        val agcAudio = applyAgc(floatAudio)

        // Check if this is speech
        val rms = computeRms(agcAudio)
        val isSpeech = rms > speechThreshold

        // Verbose: log first chunk, then every 5th
        if (chunkCount <= 3 || chunkCount % 5 == 0) {
            log("Chunk #$chunkCount: rawRMS=$rawRms, agcRMS=$rms, isSpeech=$isSpeech, " +
                    "silenceCount=$silenceCount/$silenceFrames, speechFrames=$speechFrameCount, " +
                    "threshold=$speechThreshold, agcGain=$agcGainSmooth")
        }

        if (isSpeech) {
            if (speechBuffer.isEmpty()) {
                log("  >>> Speech segment START (chunk #$chunkCount, rms=$rms)")
            }
            speechBuffer.add(agcAudio)
            silenceCount = 0
            speechFrameCount++
            return agcAudio
        } else {
            silenceCount++
            if (speechBuffer.isNotEmpty() && silenceCount >= silenceFrames) {
                // Finalize segment
                log("  <<< Speech segment END after $silenceCount silence frames (speechFrames=$speechFrameCount, minRequired=$minSpeechFrames)")
                if (speechFrameCount >= minSpeechFrames) {
                    val fullAudio = concatenate(speechBuffer)
                    val duration = fullAudio.size.toFloat() / sampleRate
                    // Segment audio stats
                    val segRms = run {
                        var s = 0.0; for (v in fullAudio) s += (v*v).toDouble(); sqrt(s / fullAudio.size).toFloat()
                    }
                    val segPeak = fullAudio.maxOf { abs(it) }
                    log("  SEGMENT FINALIZED: ${String.format("%.2f", duration)}s, ${fullAudio.size} samples, " +
                            "${speechFrameCount} frames, RMS=$segRms, peak=$segPeak. Calling transcribe...")
                    segmentCallback?.invoke(fullAudio, duration)
                } else {
                    log("  SEGMENT DISCARDED: only $speechFrameCount speech frames (< min $minSpeechFrames)")
                }
                speechBuffer.clear()
                speechFrameCount = 0
            }
            return null
        }
    }

    /**
     * Force-finalize any pending speech segment. Called on stop recording.
     */
    fun flushSegment(): FloatArray? {
        log("flushSegment() called: speechBuffer.size=${speechBuffer.size}, speechFrames=$speechFrameCount, minRequired=$minSpeechFrames")
        if (speechBuffer.isNotEmpty() && speechFrameCount >= minSpeechFrames) {
            val fullAudio = concatenate(speechBuffer)
            val duration = fullAudio.size.toFloat() / sampleRate
            speechBuffer.clear()
            speechFrameCount = 0
            silenceCount = 0
            val segRms = run {
                var s = 0.0; for (v in fullAudio) s += (v*v).toDouble(); sqrt(s / fullAudio.size).toFloat()
            }
            log("  FLUSHED segment ${String.format("%.2f", duration)}s, RMS=$segRms. Calling transcribe...")
            segmentCallback?.invoke(fullAudio, duration)
            return fullAudio
        } else if (speechBuffer.isNotEmpty()) {
            log("  FLUSH DISCARDED: $speechFrameCount frames (< min $minSpeechFrames)")
        }
        speechBuffer.clear()
        speechFrameCount = 0
        silenceCount = 0
        return null
    }

    fun reset() {
        log("reset(): clearing state")
        speechBuffer.clear()
        silenceCount = 0
        speechFrameCount = 0
        agcGainSmooth = 1.0f
        prevSample = 0.0f
        chunkCount = 0
    }

    /**
     * Adaptive AGC with high-pass filter and soft limiter.
     */
    private fun applyAgc(audio: FloatArray): FloatArray {
        val output = FloatArray(audio.size)

        // High-pass filter (80Hz) - simple DC removal + pre-emphasis
        val alpha = 0.97f
        for (i in audio.indices) {
            output[i] = audio[i] - alpha * prevSample
            prevSample = audio[i]
        }

        // RMS-based gain
        var sumSq = 0f
        for (s in output) sumSq += s * s
        val rms = sqrt(sumSq / output.size + 1e-10f)
        val targetRms = 0.12f

        if (rms > 1e-6f) {
            val gain = (targetRms / rms).coerceIn(0.3f, 20.0f)
            agcGainSmooth = 0.85f * agcGainSmooth + 0.15f * gain
            for (i in output.indices) {
                output[i] *= agcGainSmooth
            }
        }

        // Soft limiter
        val peak = output.maxOf { abs(it) }
        if (peak > 0.95f) {
            val scale = 0.95f / peak
            for (i in output.indices) output[i] *= scale
        }

        return output
    }

    private fun computeRms(audio: FloatArray): Float {
        var sumSq = 0f
        for (s in audio) sumSq += s * s
        return sqrt(sumSq / audio.size)
    }

    private fun concatenate(buffers: List<FloatArray>): FloatArray {
        val totalSize = buffers.sumOf { it.size }
        val result = FloatArray(totalSize)
        var offset = 0
        for (b in buffers) {
            System.arraycopy(b, 0, result, offset, b.size)
            offset += b.size
        }
        return result
    }
}