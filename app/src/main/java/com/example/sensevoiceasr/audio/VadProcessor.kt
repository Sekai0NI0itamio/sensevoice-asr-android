package com.example.sensevoiceasr.audio

import android.util.Log
import kotlin.math.*

/**
 * Energy-based Voice Activity Detection and preprocessing.
 * Includes AGC (Automatic Gain Control) and soft limiting.
 */
class VadProcessor(
    private val sampleRate: Int = 16000,
    private val speechThreshold: Float = 0.01f,   // RMS threshold for speech
    private val silenceFrames: Int = 8,            // Consecutive silent frames to end segment
    private val minSpeechFrames: Int = 3            // Minimum speech frames for a valid segment
) {
    companion object {
        private const val TAG = "VadProcessor"
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
     * Process an audio chunk. Returns true if speech detected.
     * Calls segmentCallback when a speech segment ends.
     */
    fun processChunk(rawChunk: ShortArray): FloatArray? {
        // Convert to float [-1, 1]
        val floatAudio = FloatArray(rawChunk.size) { rawChunk[it] / 32768f }

        // Apply AGC
        val agcAudio = applyAgc(floatAudio)

        // Check if this is speech
        val rms = computeRms(agcAudio)
        val isSpeech = rms > speechThreshold

        if (isSpeech) {
            speechBuffer.add(agcAudio)
            silenceCount = 0
            speechFrameCount++
            return agcAudio
        } else {
            silenceCount++
            if (speechBuffer.isNotEmpty() && silenceCount >= silenceFrames) {
                // Finalize segment
                if (speechFrameCount >= minSpeechFrames) {
                    val fullAudio = concatenate(speechBuffer)
                    val duration = fullAudio.size.toFloat() / sampleRate
                    Log.d(TAG, "Segment finalized: ${duration}s, ${speechFrameCount} frames")
                    segmentCallback?.invoke(fullAudio, duration)
                }
                speechBuffer.clear()
                speechFrameCount = 0
            }
            return null
        }
    }

    /**
     * Force-finalize any pending speech segment.
     */
    fun flushSegment(): FloatArray? {
        if (speechBuffer.isNotEmpty() && speechFrameCount >= minSpeechFrames) {
            val fullAudio = concatenate(speechBuffer)
            val duration = fullAudio.size.toFloat() / sampleRate
            speechBuffer.clear()
            speechFrameCount = 0
            silenceCount = 0
            segmentCallback?.invoke(fullAudio, duration)
            return fullAudio
        }
        speechBuffer.clear()
        speechFrameCount = 0
        silenceCount = 0
        return null
    }

    fun reset() {
        speechBuffer.clear()
        silenceCount = 0
        speechFrameCount = 0
        agcGainSmooth = 1.0f
        prevSample = 0.0f
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