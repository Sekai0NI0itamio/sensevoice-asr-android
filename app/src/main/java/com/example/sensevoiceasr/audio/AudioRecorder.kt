package com.example.sensevoiceasr.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Captures 16kHz mono PCM audio from the device microphone.
 * Emits audio chunks as Float32 ByteArrays (ready for WebSocket send).
 */
class AudioRecorder(
    private val sampleRate: Int = 16000,
    private val chunkMs: Int = 200
) {
    companion object {
        private const val TAG = "AudioRecorder"
    }

    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _audioChunks = MutableSharedFlow<ByteArray>(replay = 0, extraBufferCapacity = 64)
    val audioChunks: SharedFlow<ByteArray> = _audioChunks

    val chunkSamples: Int get() = sampleRate * chunkMs / 1000

    private val bufferSize: Int get() = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    ).coerceAtLeast(chunkSamples * 2)

    fun start(): Boolean {
        if (audioRecord != null) {
            Log.w(TAG, "Already recording")
            return false
        }

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                audioRecord?.release()
                audioRecord = null
                return false
            }

            audioRecord?.startRecording()
            Log.i(TAG, "Recording started: ${sampleRate}Hz, ${chunkMs}ms chunks")

            recordingJob = scope.launch {
                val shortBuffer = ShortArray(chunkSamples)
                while (isActive) {
                    val read = audioRecord?.read(shortBuffer, 0, chunkSamples) ?: -1
                    if (read > 0) {
                        // Convert Int16 to Float32 for the server
                        val floatBytes = ByteBuffer.allocate(read * 4).apply {
                            order(ByteOrder.LITTLE_ENDIAN)
                            for (i in 0 until read) {
                                putFloat(shortBuffer[i] / 32768f)
                            }
                        }.array()
                        _audioChunks.emit(floatBytes)
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

    fun release() {
        stop()
        scope.cancel()
    }
}