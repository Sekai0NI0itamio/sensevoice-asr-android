package com.example.sensevoiceasr.asr

import android.content.Context
import android.util.Log
import kotlin.math.*

/**
 * Mel filterbank feature extraction with LFR stacking and CMVN normalization.
 * Matches the WavFrontend config from SenseVoiceSmall:
 *   fs=16000, n_mels=80, frame_length=25ms, frame_shift=10ms
 *   lfr_m=7, lfr_n=6, with CMVN from am.mvn
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
        private const val AUDIO_SCALE = 32768f  // 1 << 15, matches Python upsacle_samples
        private const val CMVN_FILE = "am.mvn"
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

    // CMVN parameters (loaded from am.mvn)
    private var cmvnAddShift: FloatArray? = null
    private var cmvnRescale: FloatArray? = null
    private var cmvnLoaded = false

    private fun hzToMel(hz: Float): Float = 1127f * ln(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (exp(mel / 1127f) - 1f)

    private fun createMelFilterbank(): Array<FloatArray> {
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
     * Load CMVN parameters from am.mvn file in assets.
     * Parses the Kaldi nnet1 format to extract AddShift (means) and Rescale (vars).
     */
    fun loadCmvn(context: Context): Boolean {
        if (cmvnLoaded) return true
        return try {
            val content = context.assets.open(CMVN_FILE).bufferedReader().use { it.readText() }
            val lines = content.lines()

            var addShiftValues: FloatArray? = null
            var rescaleValues: FloatArray? = null

            for (i in lines.indices) {
                val parts = lines[i].trim().split("\\s+".toRegex())
                if (parts.isEmpty()) continue

                when (parts[0]) {
                    "<AddShift>" -> {
                        if (i + 1 < lines.size) {
                            val nextParts = lines[i + 1].trim().split("\\s+".toRegex())
                            if (nextParts.isNotEmpty() && nextParts[0] == "<LearnRateCoef>") {
                                // Format: <LearnRateCoef> 0 [ val1 val2 ... valN ]
                                val startIdx = nextParts.indexOf("[") + 1
                                val endIdx = nextParts.lastIndexOf("]")
                                if (startIdx > 0 && endIdx > startIdx) {
                                    val valList = nextParts.subList(startIdx, endIdx)
                                    addShiftValues = FloatArray(valList.size) { valList[it].toFloat() }
                                }
                            }
                        }
                    }
                    "<Rescale>" -> {
                        if (i + 1 < lines.size) {
                            val nextParts = lines[i + 1].trim().split("\\s+".toRegex())
                            if (nextParts.isNotEmpty() && nextParts[0] == "<LearnRateCoef>") {
                                val startIdx = nextParts.indexOf("[") + 1
                                val endIdx = nextParts.lastIndexOf("]")
                                if (startIdx > 0 && endIdx > startIdx) {
                                    val valList = nextParts.subList(startIdx, endIdx)
                                    rescaleValues = FloatArray(valList.size) { valList[it].toFloat() }
                                }
                            }
                        }
                    }
                }
            }

            if (addShiftValues != null && rescaleValues != null) {
                cmvnAddShift = addShiftValues
                cmvnRescale = rescaleValues
                cmvnLoaded = true
                Log.i(TAG, "CMVN loaded: addShift=${addShiftValues.size}, rescale=${rescaleValues.size}")
                Log.i(TAG, "CMVN addShift first 5: ${addShiftValues.take(5).joinToString()}")
                Log.i(TAG, "CMVN rescale first 5: ${rescaleValues.take(5).joinToString()}")
                true
            } else {
                Log.w(TAG, "CMVN file parsed but missing AddShift or Rescale")
                false
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load CMVN: ${e.message}")
            false
        }
    }

    /**
     * Extract LFR-stacked fbank features with CMVN normalization from raw audio.
     * Matches the Python WavFrontend pipeline:
     *   1. Scale audio by 32768 (upsacle_samples)
     *   2. Compute mel fbank features
     *   3. Apply LFR stacking with left-padding
     *   4. Apply CMVN normalization: (feature + addShift) * rescale
     *
     * Returns: features array [num_lfr_frames, lfr_m * nMels]
     */
    fun extract(audio: FloatArray): Pair<Array<FloatArray>, Int> {
        // Scale audio by 32768 to match Python upsacle_samples
        val scaledAudio = FloatArray(audio.size) { audio[it] * AUDIO_SCALE }

        val frames = frameAudio(scaledAudio)
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

        // LFR stacking with left-padding to match Python WavFrontend.apply_lfr
        // Python prepends (lfr_m - 1) // 2 copies of the first frame as left context
        val leftPad = (lfrM - 1) / 2
        val paddedFeatures = mutableListOf<FloatArray>()
        // Add left padding: repeat first frame
        repeat(leftPad) { paddedFeatures.add(fbankFeatures[0].copyOf()) }
        paddedFeatures.addAll(fbankFeatures)

        // Compute number of LFR frames: ceil(T_original / lfr_n)
        val T = fbankFeatures.size
        val T_lfr = ceil(T.toFloat() / lfrN).toInt()

        val lfrFeatures = mutableListOf<FloatArray>()
        for (k in 0 until T_lfr) {
            val startIdx = k * lfrN
            val stacked = FloatArray(lfrM * nMels)

            for (j in 0 until lfrM) {
                val srcIdx = startIdx + j
                if (srcIdx < paddedFeatures.size) {
                    System.arraycopy(paddedFeatures[srcIdx], 0, stacked, j * nMels, nMels)
                } else {
                    // Right-pad with last frame (matches Python behavior)
                    System.arraycopy(paddedFeatures.last(), 0, stacked, j * nMels, nMels)
                }
            }
            lfrFeatures.add(stacked)
        }

        // Apply CMVN normalization: (feature + addShift) * rescale
        if (cmvnLoaded && cmvnAddShift != null && cmvnRescale != null) {
            val addShift = cmvnAddShift!!
            val rescale = cmvnRescale!!
            val featDim = lfrM * nMels
            if (addShift.size >= featDim && rescale.size >= featDim) {
                for (frame in lfrFeatures) {
                    for (d in 0 until featDim) {
                        frame[d] = (frame[d] + addShift[d]) * rescale[d]
                    }
                }
            } else {
                Log.w(TAG, "CMVN dims mismatch: addShift=${addShift.size}, rescale=${rescale.size}, expected=$featDim")
            }
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