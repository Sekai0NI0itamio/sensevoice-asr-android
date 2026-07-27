package com.example.sensevoiceasr.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlin.math.sqrt

/**
 * Captures 16kHz mono PCM audio from the device microphone.
 * Emits audio chunks as ShortArrays via a SharedFlow.
 * Supports a diagnostic log callback for UI visibility.
 */
class AudioRecorder(
    private val sampleRate: Int = 16000,
    private val channelConfig: Int = AudioFormat.CHANNEL_IN_MONO,
    private val audioFormat: Int = AudioFormat.ENCODING_PCM_16BIT,
    private val chunkMs: Int = 200
) {
    companion object {
        private const val TAG = "AudioRecorder"
    }

    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _audioChunks = MutableSharedFlow<ShortArray>(replay = 0, extraBufferCapacity = 64)
    val audioChunks: SharedFlow<ShortArray> = _audioChunks

    /** Diagnostic log callback (one line per event) */
    var logCallback: ((String) -> Unit)? = null
    private var chunkCount: Int = 0

    private fun log(msg: String) {
        Log.i(TAG, msg)
        logCallback?.invoke("[AudioRecorder] $msg")
    }

    private fun logErr(msg: String) {
        Log.e(TAG, msg)
        logCallback?.invoke("[AudioRecorder][ERR] $msg")
    }

    private val bufferSize: Int get() = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        .coerceAtLeast(sampleRate * chunkMs / 1000 * 2)

    val chunkSamples: Int get() = sampleRate * chunkMs / 1000

    fun start(): Boolean {
        if (audioRecord != null) {
            Log.w(TAG, "Already recording")
            return false
        }
        chunkCount = 0

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                logErr("AudioRecord failed to initialize (state=${audioRecord?.state})")
                audioRecord?.release()
                audioRecord = null
                return false
            }

            audioRecord?.startRecording()
            val recState = audioRecord?.recordingState
            log("Recording started: ${sampleRate}Hz, ${chunkMs}ms chunks (${chunkSamples} samples/chunk), bufferSize=$bufferSize, recState=$recState")

            recordingJob = scope.launch {
                val buffer = ShortArray(chunkSamples)
                while (isActive) {
                    val read = audioRecord?.read(buffer, 0, chunkSamples) ?: -1
                    if (read > 0) {
                        chunkCount++
                        val chunk = if (read == chunkSamples) {
                            buffer.copyOf()
                        } else {
                            buffer.copyOf(read)
                        }
                        // Compute RMS of this chunk for diagnostics
                        var sumSq = 0.0
                        var mx: Short = Short.MIN_VALUE
                        var mn: Short = Short.MAX_VALUE
                        for (s in chunk) {
                            sumSq += (s * s).toDouble()
                            if (s > mx) mx = s
                            if (s < mn) mn = s
                        }
                        val rms = sqrt(sumSq / chunk.size).toFloat()
                        // Every 10th chunk (2s at 200ms), report stats
                        if (chunkCount == 1 || chunkCount % 10 == 0) {
                            log("Chunk #$chunkCount: $read samples, RMS_int16=$rms, min=$mn, max=$mx")
                        }
                        _audioChunks.emit(chunk)
                    } else if (read < 0) {
                        logErr("AudioRecord read error code=$read after $chunkCount successful chunks")
                        break
                    }
                }
                log("Recording loop exited after $chunkCount chunks")
            }
            return true
        } catch (e: SecurityException) {
            logErr("Microphone permission denied: ${e.message}")
            audioRecord?.release()
            audioRecord = null
            return false
        } catch (e: Exception) {
            logErr("Failed to start recording: ${e.javaClass.simpleName}: ${e.message}")
            audioRecord?.release()
            audioRecord = null
            return false
        }
    }

    fun stop() {
        recordingJob?.cancel()
        recordingJob = null
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            logErr("Error stopping recording: ${e.message}")
        }
        audioRecord = null
        log("Recording stopped. Total chunks emitted: $chunkCount")
    }

    fun isRecording(): Boolean = audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING

    fun release() {
        stop()
        scope.cancel()
    }
}