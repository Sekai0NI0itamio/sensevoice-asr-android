package com.example.sensevoiceasr

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.example.sensevoiceasr.audio.AudioRecorder
import com.example.sensevoiceasr.audio.VadProcessor
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {
    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
    }

    private lateinit var btnRecord: Button
    private lateinit var btnClear: Button
    private lateinit var transcriptText: TextView
    private lateinit var statusText: TextView

    private var audioRecorder: AudioRecorder? = null
    private var vadProcessor: VadProcessor? = null

    private var recordingJob: Job? = null
    private var isRecording = false

    private val timeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())

    private lateinit var viewModel: AsrViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRecord = findViewById(R.id.btn_record)
        btnClear = findViewById(R.id.btn_clear)
        transcriptText = findViewById(R.id.transcript_text)
        statusText = findViewById(R.id.status)

        // ViewModel survives activity recreation (permission dialogs, rotations)
        viewModel = ViewModelProvider(this)[AsrViewModel::class.java]

        // Observe model state
        viewModel.modelState.observe(this) { state ->
            when (state) {
                is AsrViewModel.ModelState.Loading -> {
                    updateStatus("Loading model...")
                    btnRecord.isEnabled = false
                    btnRecord.text = "Loading..."
                }
                is AsrViewModel.ModelState.Ready -> {
                    updateStatus("Ready")
                    btnRecord.isEnabled = true
                    btnRecord.text = "Hold to\nRecord"
                }
                is AsrViewModel.ModelState.Error -> {
                    updateStatus(state.message)
                    btnRecord.isEnabled = false
                    btnRecord.text = "Error"
                    Toast.makeText(this, state.message, Toast.LENGTH_LONG).show()
                }
            }
        }

        // Hold-to-record button
        btnRecord.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!viewModel.isModelReady) {
                        Toast.makeText(this, "Model not ready yet", Toast.LENGTH_SHORT).show()
                        return@setOnTouchListener false
                    }
                    startRecording()
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    stopRecording()
                    true
                }
                else -> false
            }
        }

        // Clear button
        btnClear.setOnClickListener {
            transcriptText.text = ""
            statusText.text = "Cleared"
        }
    }

    private fun startRecording() {
        if (!hasMicrophonePermission()) {
            requestMicrophonePermission()
            return
        }

        if (isRecording) return
        isRecording = true

        btnRecord.text = "Recording..."
        btnRecord.setBackgroundColor(getColor(R.color.record_red_dark))
        updateStatus("Recording...")

        audioRecorder = AudioRecorder(chunkMs = 200)
        vadProcessor = VadProcessor(
            sampleRate = 16000,
            speechThreshold = 0.01f,
            silenceFrames = 8,
            minSpeechFrames = 3
        )

        vadProcessor?.onSegmentFinalized { audio, duration ->
            lifecycleScope.launch(Dispatchers.IO) {
                val startTime = System.currentTimeMillis()
                val text = viewModel.transcribe(audio)
                val inferenceTime = System.currentTimeMillis() - startTime

                if (text.isNotBlank()) {
                    withContext(Dispatchers.Main) {
                        appendTranscript(text, duration, inferenceTime)
                    }
                }
            }
        }

        val started = audioRecorder?.start() ?: false
        if (!started) {
            isRecording = false
            resetRecordButton()
            Toast.makeText(this, "Failed to start recording", Toast.LENGTH_SHORT).show()
            return
        }

        recordingJob = lifecycleScope.launch(Dispatchers.IO) {
            val recorder = audioRecorder ?: return@launch
            val vad = vadProcessor ?: return@launch
            recorder.audioChunks.collect { chunk ->
                if (!isRecording) return@collect
                vad.processChunk(chunk)
            }
        }
    }

    private fun stopRecording() {
        if (!isRecording) return
        isRecording = false

        recordingJob?.cancel()
        recordingJob = null

        val vad = vadProcessor
        lifecycleScope.launch(Dispatchers.IO) {
            vad?.flushSegment()
        }

        audioRecorder?.stop()
        audioRecorder = null
        vadProcessor = null

        resetRecordButton()
        updateStatus("Ready")
    }

    private fun appendTranscript(text: String, durationSec: Float, inferenceTimeMs: Long) {
        val timestamp = timeFormat.format(Date())
        val currentText = transcriptText.text.toString()
        val newEntry = if (currentText.isEmpty()) {
            "[$timestamp] $text  (${"%.1f".format(durationSec)}s, ${inferenceTimeMs}ms)"
        } else {
            "\n\n[$timestamp] $text  (${"%.1f".format(durationSec)}s, ${inferenceTimeMs}ms)"
        }
        transcriptText.append(newEntry)

        transcriptText.post {
            val scrollView = findViewById<android.widget.ScrollView>(R.id.scroll_view)
            scrollView.fullScroll(android.view.View.FOCUS_DOWN)
        }
    }

    private fun updateStatus(msg: String) {
        runOnUiThread {
            statusText.text = msg
        }
    }

    private fun resetRecordButton() {
        runOnUiThread {
            btnRecord.text = "Hold to\nRecord"
            btnRecord.setBackgroundColor(getColor(R.color.record_red))
        }
    }

    private fun hasMicrophonePermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
    }

    private fun requestMicrophonePermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            PERMISSION_REQUEST_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Microphone permission is required", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        recordingJob?.cancel()
        audioRecorder?.release()
    }
}