package com.example.sensevoiceasr.asr

import android.util.Log
import kotlin.math.*

/**
 * Mel filterbank feature extraction with LFR stacking.
 * Matches the WavFrontend config from SenseVoiceSmall:
 *   fs=16000, n_mels=80, frame_length=25ms, frame_shift=10ms
 *   lfr_m=7, lfr_n=6
 */
class FeatureExtractor(
    private val sampleRate: Int = 16000,
    private val nMels: Int = 80,
    private val frameLengthMs: Float = 25f,
    private val frameShiftMs: Float = 10f,
    private val nFft: Int = 512,
    private val lfrM: Int = 7,
    private val lfrN: Int = 6,
    private val preEmphasis: Float = 0.97f
) {
    companion object {
        private const val TAG = "FeatureExtractor"
    }

    private val frameLength: Int = (sampleRate * frameLengthMs / 1000f).toInt()
    private val frameShift: Int = (sampleRate * frameShiftMs / 1000f).toInt()

    // Hamming window
    private val hammingWindow: FloatArray = FloatArray(frameLength) {
        0.54f - 0.46f * cos(2.0 * PI * it / (frameLength - 1)).toFloat()
    }

    // Mel filterbank
    private val melFilterbank: Array<FloatArray> by lazy { createMelFilterbank() }

    // FFT workspace
    private val fftReal = FloatArray(nFft)
    private val fftImag = FloatArray(nFft)

    private fun hzToMel(hz: Float): Float = 1127f * ln(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (exp(mel / 1127f) - 1f)

    private fun createMelFilterbank(): Array<FloatArray> {
        val fftBinFreq = FloatArray(nFft / 2 + 1) { it * sampleRate.toFloat() / nFft }
        val melLow = hzToMel(0f)
        val melHigh = hzToMel(sampleRate / 2f)
        val melPoints = FloatArray(nMels + 2) { melLow + it * (melHigh - melLow) / (nMels + 1) }
        val hzPoints = FloatArray(nMels + 2) { melToHz(melPoints[it]) }
        val bin = IntArray(nMels + 2) { ((nFft + 1) * hzPoints[it] / sampleRate).toInt().coerceIn(0, nFft / 2) }

        val filters = Array(nMels) { FloatArray(nFft / 2 + 1) }
        for (m in 1..nMels) {
            for (k in bin[m - 1] until bin[m]) {
                filters[m - 1][k] = (k - bin[m - 1]).toFloat() / (bin[m] - bin[m - 1]).toFloat()
            }
            for (k in bin[m] until bin[m + 1]) {
                filters[m - 1][k] = (bin[m + 1] - k).toFloat() / (bin[m + 1] - bin[m]).toFloat()
            }
        }
        return filters
    }

    /**
     * Extract LFR-stacked fbank features from raw audio.
     * Returns: features array [num_lfr_frames, lfr_m * nMels]
     */
    fun extract(audio: FloatArray): Pair<Array<FloatArray>, Int> {
        val frames = frameAudio(audio)
        if (frames.isEmpty()) {
            return Pair(emptyArray(), 0)
        }

        // Apply window and compute FFT for each frame
        val fbankFeatures = Array(frames.size) { FloatArray(nMels) }
        for (i in frames.indices) {
            val frame = frames[i]
            // Apply pre-emphasis
            val preEmphed = FloatArray(frame.size)
            preEmphed[0] = frame[0]
            for (j in 1 until frame.size) {
                preEmphed[j] = frame[j] - preEmphasis * frame[j - 1]
            }
            // Apply Hamming window
            for (j in preEmphed.indices) {
                fftReal[j] = preEmphed[j] * hammingWindow[j]
            }
            for (j in preEmphed.size until nFft) {
                fftReal[j] = 0f
                fftImag[j] = 0f
            }
            // Compute FFT
            realFft(fftReal, fftImag, nFft)
            // Power spectrum
            val powerSpec = FloatArray(nFft / 2 + 1) { k ->
                (fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k]) / nFft
            }
            // Apply mel filterbank
            for (m in 0 until nMels) {
                var sum = 0f
                for (k in 0..nFft / 2) {
                    sum += melFilterbank[m][k] * powerSpec[k]
                }
                fbankFeatures[i][m] = ln(max(sum, 1e-10f))
            }
        }

        // LFR stacking: stack lfrM frames, shift by lfrN
        val lfrFeatures = mutableListOf<FloatArray>()
        var i = 0
        while (i + lfrM <= fbankFeatures.size) {
            val stacked = FloatArray(lfrM * nMels)
            for (j in 0 until lfrM) {
                System.arraycopy(fbankFeatures[i + j], 0, stacked, j * nMels, nMels)
            }
            lfrFeatures.add(stacked)
            i += lfrN
        }

        return Pair(lfrFeatures.toTypedArray(), lfrFeatures.size)
    }

    private fun frameAudio(audio: FloatArray): List<FloatArray> {
        val frames = mutableListOf<FloatArray>()
        var start = 0
        while (start + frameLength <= audio.size) {
            frames.add(audio.copyOfRange(start, start + frameLength))
            start += frameShift
        }
        return frames
    }

    /** Simple real FFT (radix-2 DIT). n must be power of 2. */
    private fun realFft(real: FloatArray, imag: FloatArray, n: Int) {
        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                val tmpR = real[i]; real[i] = real[j]; real[j] = tmpR
            }
            var m = n shr 1
            while (m >= 1 && j >= m) {
                j -= m
                m = m shr 1
            }
            j += m
        }
        // Cooley-Tukey
        var len = 2
        while (len <= n) {
            val halfLen = len shr 1
            val angle = -2.0 * PI / len
            val wReal = cos(angle).toFloat()
            val wImag = sin(angle).toFloat()
            for (i in 0 until n step len) {
                var curReal = 1f
                var curImag = 0f
                for (k in 0 until halfLen) {
                    val tReal = curReal * real[i + k + halfLen] - curImag * imag[i + k + halfLen]
                    val tImag = curReal * imag[i + k + halfLen] + curImag * real[i + k + halfLen]
                    real[i + k + halfLen] = real[i + k] - tReal
                    imag[i + k + halfLen] = imag[i + k] - tImag
                    real[i + k] += tReal
                    imag[i + k] += tImag
                    val newCurReal = curReal * wReal - curImag * wImag
                    curImag = curReal * wImag + curImag * wReal
                    curReal = newCurReal
                }
            }
            len = len shl 1
        }
    }
}