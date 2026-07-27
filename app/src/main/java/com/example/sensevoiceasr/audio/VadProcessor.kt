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
    private val speechThreshold: Float = 0.015f,
    private val silenceFrames: Int = 8,
    private val minSpeechFrames: Int = 3,
    private val maxSpeechFrames: Int = 40
) {
    companion object {
        private const val TAG = "VadProcessor"
        private const val AGC_MAX_GAIN = 8.0f
        private const val AGC_MIN_GAIN = 0.3f
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

    private fun finalizeSegment(reason: String): FloatArray? {
        if (speechBuffer.isEmpty()) return null
        log("  <<< Speech segment END ($reason) after $silenceCount silence frames (speechFrames=$speechFrameCount, minRequired=$minSpeechFrames)")
        var audio: FloatArray? = null
        if (speechFrameCount >= minSpeechFrames) {
            val fullAudio = concatenate(speechBuffer)
            val duration = fullAudio.size.toFloat() / sampleRate
            val segRms = run {
                var s = 0.0; for (v in fullAudio) s += (v*v).toDouble(); sqrt(s / fullAudio.size).toFloat()
            }
            val segPeak = fullAudio.maxOf { abs(it) }
            log("  SEGMENT FINALIZED ($reason): ${String.format("%.2f", duration)}s, ${fullAudio.size} samples, " +
                    "${speechFrameCount} frames, RMS=$segRms, peak=$segPeak. Calling transcribe...")
            try {
                segmentCallback?.invoke(fullAudio, duration)
                log("  segmentCallback returned OK for ${String.format("%.2f", duration)}s segment")
            } catch (t: Throwable) {
                Log.e(TAG, "segmentCallback threw", t)
                logCallback?.invoke("[Vad][ERR] segmentCallback crashed: ${t.javaClass.simpleName}: ${t.message}")
            }
            audio = fullAudio
        } else {
            log("  SEGMENT DISCARDED: only $speechFrameCount speech frames (< min $minSpeechFrames)")
        }
        speechBuffer.clear()
        speechFrameCount = 0
        silenceCount = 0
        return audio
    }

    /**
     * Process an audio chunk. Returns the processed (AGC'd) float audio if speech was detected.
     * Calls segmentCallback when a speech segment ends (silence threshold OR max-duration cutoff).
     * VAD uses rawRMS (before AGC) to avoid AGC-gain-amplified noise being misclassified as speech.
     */
    fun processChunk(rawChunk: ShortArray): FloatArray? {
        chunkCount++

        // Convert to float [-1, 1]
        val floatAudio = FloatArray(rawChunk.size) { rawChunk[it] / 32768f }

        // Raw RMS before AGC — USE THIS for speech detection (defect fix: don't use post-AGC RMS)
        val rawRms = run {
            var s = 0.0
            for (v in floatAudio) s += (v * v).toDouble()
            sqrt(s / floatAudio.size).toFloat()
        }

        // Apply AGC — output is still fed to downstream model / feature extraction for normalized input
        val agcAudio = applyAgc(floatAudio)
        val agcRms = computeRms(agcAudio)

        // VAD decision on rawRMS (NOT AGC-amplified rms)
        val isSpeech = rawRms >= speechThreshold

        // Verbose: log first chunk, then every 5th
        if (chunkCount <= 3 || chunkCount % 5 == 0) {
            log("Chunk #$chunkCount: rawRMS=$rawRms, agcRMS=$agcRms, isSpeech=$isSpeech " +
                    "(VAD on rawRMS >= $speechThreshold), " +
                    "silenceCount=$silenceCount/$silenceFrames, speechFrames=$speechFrameCount/$maxSpeechFrames, " +
                    "agcGain=$agcGainSmooth (min=$AGC_MIN_GAIN max=$AGC_MAX_GAIN)")
        }

        if (isSpeech) {
            if (speechBuffer.isEmpty()) {
                log("  >>> Speech segment START (chunk #$chunkCount, rawRMS=$rawRms, agcRMS=$agcRms)")
            }
            speechBuffer.add(agcAudio)
            silenceCount = 0
            speechFrameCount++

            // Max-duration auto-cutoff: finalize segment after 8s (40 × 200ms) of continuous speech
            if (speechFrameCount >= maxSpeechFrames && speechBuffer.isNotEmpty()) {
                log("  MAX-DURATION CUTOFF after $speechFrameCount frames (${String.format("%.1f", speechFrameCount * 0.2f)}s) — finalizing segment + starting fresh buffer")
                finalizeSegment(reason = "maxDuration ${String.format("%.1f", maxSpeechFrames * 0.2f)}s cutoff")
            }
            return agcAudio
        } else {
            silenceCount++
            if (speechBuffer.isNotEmpty() && silenceCount >= silenceFrames) {
                finalizeSegment(reason = "silenceThreshold ${silenceFrames}frames")
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
     * Gain range reduced to [0.3×, 8×] to avoid 20× amplification of ambient
     * room noise which caused VAD misclassification (bug: every silence chunk
     * passed threshold after AGC).
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
            val gain = (targetRms / rms).coerceIn(AGC_MIN_GAIN, AGC_MAX_GAIN)
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