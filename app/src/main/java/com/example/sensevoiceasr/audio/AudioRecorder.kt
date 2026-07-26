package com.example.sensevoiceasr.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

/**
 * Captures 16kHz mono PCM audio from the device microphone.
 * Emits audio chunks as ShortArrays via a SharedFlow.
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

    private val bufferSize: Int get() = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        .coerceAtLeast(sampleRate * chunkMs / 1000 * 2) // 2 bytes per sample

    val chunkSamples: Int get() = sampleRate * chunkMs / 1000

    fun start(): Boolean {
        if (audioRecord != null) {
            Log.w(TAG, "Already recording")
            return false
        }

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                audioRecord?.release()
                audioRecord = null
                return false
            }

            audioRecord?.startRecording()
            Log.i(TAG, "Recording started: ${sampleRate}Hz, ${chunkMs}ms chunks, buffer=$bufferSize")

            recordingJob = scope.launch {
                val buffer = ShortArray(chunkSamples)
                while (isActive) {
                    val read = audioRecord?.read(buffer, 0, chunkSamples) ?: -1
                    if (read > 0) {
                        val chunk = if (read == chunkSamples) {
                            buffer.copyOf()
                        } else {
                            buffer.copyOf(read)
                        }
                        _audioChunks.emit(chunk)
                    } else if (read < 0) {
                        Log.e(TAG, "AudioRecord read error: $read")
                        break
                    }
                }
            }
            return true
        } catch (e: SecurityException) {
            Log.e(TAG, "Microphone permission denied", e)
            audioRecord?.release()
            audioRecord = null
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
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
            Log.e(TAG, "Error stopping recording", e)
        }
        audioRecord = null
        Log.i(TAG, "Recording stopped")
    }

    fun isRecording(): Boolean = audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING

    fun release() {
        stop()
        scope.cancel()
    }
}