package com.example.sensevoiceasr

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Typeface
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
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest
import com.example.sensevoiceasr.audio.AudioRecorder
import com.example.sensevoiceasr.audio.VadProcessor
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

    /** Auto-scroll guard: if user has manually scrolled up, pause auto-scroll. */
    private var userScrolling = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRecord = findViewById(R.id.btn_record)
        btnClear = findViewById(R.id.btn_clear)
        transcriptText = findViewById(R.id.transcript_text)
        statusText = findViewById(R.id.status)

        // Style the log panel: monospace, smaller, line-wrapped so long token IDs are readable
        transcriptText.typeface = Typeface.MONOSPACE
        transcriptText.textSize = 11f
        transcriptText.setLineSpacing(2f, 1f)

        // Detect manual scroll to pause auto-scroll
        val scrollView = findViewById<android.widget.ScrollView>(R.id.scroll_view)
        scrollView.viewTreeObserver.addOnScrollChangedListener {
            val view = scrollView.getChildAt(scrollView.childCount - 1)
            if (view != null) {
                val diff = view.bottom - (scrollView.height + scrollView.scrollY)
                // If bottom diff > 30px, user has scrolled up -> pause auto scroll
                userScrolling = diff > 30
            }
        }
        // Clicking on the log view re-enables auto-scroll
        scrollView.setOnTouchListener { _, _ ->
            userScrolling = false
            false
        }

        // ViewModel survives activity recreation (permission dialogs, rotations)
        viewModel = ViewModelProvider(this)[AsrViewModel::class.java]

        // Observe model state
        lifecycleScope.launch {
            viewModel.modelState.collectLatest { state ->
                when (state) {
                    is AsrViewModel.ModelState.Loading -> {
                        updateStatus("Loading model...")
                        btnRecord.isEnabled = false
                        btnRecord.text = "Loading..."
                    }
                    is AsrViewModel.ModelState.Ready -> {
                        updateStatus("Ready — hold mic to record")
                        btnRecord.isEnabled = true
                        btnRecord.text = "Hold to\nRecord"
                    }
                    is AsrViewModel.ModelState.Error -> {
                        updateStatus(state.message)
                        btnRecord.isEnabled = false
                        btnRecord.text = "Error"
                        Toast.makeText(this@MainActivity, state.message, Toast.LENGTH_LONG).show()
                    }
                }
            }
        }

        // Observe log lines and render to the panel
        lifecycleScope.launch {
            viewModel.logLines.collectLatest { lines ->
                renderLogLines(lines)
            }
        }

        // Hold-to-record button
        btnRecord.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!viewModel.isModelReady) {
                        Toast.makeText(this, "Model not ready yet — wait for 'Ready' status", Toast.LENGTH_SHORT).show()
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

        // Clear button: clears the log view AND tells viewModel to reset its buffer
        btnClear.setOnClickListener {
            val scrollView = findViewById<android.widget.ScrollView>(R.id.scroll_view)
            transcriptText.text = ""
            // Ask ViewModel to clear; we do this by posting a sentinel log line that
            // triggers the display to clear while still allowing future entries.
            // Simpler: just clear the ViewModel's backing list via a helper.
            viewModel.appendLog("--- CLEARED by user at ${timeFormat.format(Date())} ---")
            // Short delay then scroll to bottom: the new line triggers collectLatest which re-renders
            // Actually we want to HARD clear:
            scrollView.post { scrollView.fullScroll(android.view.View.FOCUS_DOWN) }
            updateStatus("Cleared")
        }
    }

    private fun renderLogLines(lines: List<String>) {
        // Check if the last line is our sentinel "--- CLEARED ---": if so, show only from that point onward
        val lastClearIdx = lines.indexOfLast { it.contains("--- CLEARED by user") }
        val visible = if (lastClearIdx >= 0 && lastClearIdx < lines.size - 1) {
            lines.subList(lastClearIdx + 1, lines.size)
        } else {
            lines
        }
        transcriptText.text = visible.joinToString("\n")
        // Auto-scroll to bottom unless user has scrolled up manually
        if (!userScrolling) {
            val scrollView = findViewById<android.widget.ScrollView>(R.id.scroll_view)
            scrollView.post { scrollView.fullScroll(android.view.View.FOCUS_DOWN) }
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
        updateStatus("Recording... (speak, then release)")
        viewModel.appendLog("========== RECORDING STARTED at ${timeFormat.format(Date())} ==========")

        audioRecorder = AudioRecorder(chunkMs = 200).also {
            it.logCallback = { line -> viewModel.appendLog(line) }
        }
        vadProcessor = VadProcessor(
            sampleRate = 16000,
            speechThreshold = 0.015f,
            silenceFrames = 8,
            minSpeechFrames = 3,
            maxSpeechFrames = 40
        ).also {
            it.logCallback = { line -> viewModel.appendLog(line) }
        }

        // Segment callbacks are invoked from processChunk() or flushSegment(), both of which
        // already run on the IO dispatcher (collect coroutine / flush IO job).  DO NOT launch
        // an extra nested lifecycleScope here (was silently losing callbacks to lifecycle races).
        // Call transcribe synchronously from the current IO context; wrap everything in
        // try/catch so every failure surfaces as a visible UI log line.
        vadProcessor?.onSegmentFinalized { audio, duration ->
            try {
                viewModel.appendLog("[MainActivity] Got VAD segment: ${String.format("%.2f", duration)}s, ${audio.size} samples. Running transcribe on IO thread...")
                val startTime = System.currentTimeMillis()
                val text = try {
                    viewModel.transcribe(audio)
                } catch (t: Throwable) {
                    Log.e("MainActivity", "transcribe threw", t)
                    viewModel.appendLog("[MainActivity][ERR] transcribe crashed: ${t.javaClass.simpleName}: ${t.message}")
                    ""
                }
                val inferenceTime = System.currentTimeMillis() - startTime
                viewModel.appendLog("[MainActivity] transcribe() completed in ${inferenceTime}ms. result=\"$text\"")

                if (text.isNotBlank()) {
                    viewModel.appendLog(">>> FINAL TRANSCRIPTION [${String.format("%.2f", duration)}s audio, ${inferenceTime}ms]: \"$text\"")
                    runOnUiThread { appendTranscript(text, duration, inferenceTime) }
                } else {
                    viewModel.appendLog("<<< Transcription returned EMPTY string (scroll UP to see Features/Logits/TokenIDs above)")
                }
            } catch (t: Throwable) {
                Log.e("MainActivity", "onSegmentFinalized handler crashed", t)
                viewModel.appendLog("[MainActivity][ERR] onSegmentFinalized handler crashed: ${t.javaClass.simpleName}: ${t.message}")
            }
        }

        val started = audioRecorder?.start() ?: false
        if (!started) {
            isRecording = false
            resetRecordButton()
            Toast.makeText(this, "Failed to start recording (check log for details)", Toast.LENGTH_SHORT).show()
            return
        }

        recordingJob = lifecycleScope.launch(Dispatchers.IO) {
            val recorder = audioRecorder ?: return@launch
            val vad = vadProcessor ?: return@launch
            var chunksProcessed = 0
            recorder.audioChunks.collect { chunk ->
                if (!isRecording) return@collect
                vad.processChunk(chunk)
                chunksProcessed++
            }
            viewModel.appendLog("[MainActivity] Chunk collector exited after $chunksProcessed chunks")
        }
    }

    private fun stopRecording() {
        if (!isRecording) return
        isRecording = false
        viewModel.appendLog("========== RECORDING STOPPED at ${timeFormat.format(Date())} ==========")

        // 1) Capture local refs BEFORE nulling (critical: we null refs AFTER flush completes, not before)
        val recordingJobSnapshot = recordingJob
        val vadSnapshot = vadProcessor
        val recorderSnapshot = audioRecorder

        // 2) Stop the chunk collector first (cancel, not detach)
        recordingJobSnapshot?.cancel()
        recordingJob = null

        // 3) Flush + transcribe synchronously on IO dispatcher BEFORE nulling refs or resetting UI
        lifecycleScope.launch(Dispatchers.IO) {
            var flushedSamples: Int? = null
            try {
                val flushed = vadSnapshot?.flushSegment()
                flushedSamples = flushed?.size
                if (flushed != null) {
                    viewModel.appendLog("[MainActivity] flushSegment() returned audio segment of ${flushed.size} samples (${String.format("%.2f", flushed.size / 16000.0f)}s)")
                } else {
                    viewModel.appendLog("[MainActivity] flushSegment() returned null (no qualifying segment — too few speech frames / empty buffer)")
                }
            } catch (t: Throwable) {
                Log.e("MainActivity", "flushSegment threw", t)
                viewModel.appendLog("[MainActivity][ERR] flushSegment() crashed: ${t.javaClass.simpleName}: ${t.message}")
            }

            // 4) Now safe to stop recorder, null refs, release hardware
            try {
                recorderSnapshot?.stop()
            } catch (t: Throwable) {
                Log.e("MainActivity", "audioRecorder.stop() threw", t)
                viewModel.appendLog("[MainActivity][ERR] audioRecorder.stop() crashed: ${t.javaClass.simpleName}: ${t.message}")
            }
            audioRecorder = null
            vadProcessor = null

            // 5) Reset UI state only AFTER flush + stop sequence is done
            withContext(Dispatchers.Main) {
                resetRecordButton()
                updateStatus("Ready — flushed ${if (flushedSamples != null) "$flushedSamples samples" else "no segment"}")
            }
        }
    }

    private fun appendTranscript(text: String, durationSec: Float, inferenceTimeMs: Long) {
        val timestamp = timeFormat.format(Date())
        // Also add as a prominent entry in the log panel
        viewModel.appendLog("═══════════════════════════════════════════════════════")
        viewModel.appendLog("✓ TRANSCRIPT @ $timestamp: \"$text\"")
        viewModel.appendLog("   (audio ${String.format("%.1f", durationSec)}s · inference ${inferenceTimeMs}ms)")
        viewModel.appendLog("═══════════════════════════════════════════════════════")

        // And append to the rendered view (still compatible with old logic)
        val currentText = transcriptText.text.toString()
        val newEntry = if (currentText.isEmpty()) {
            "[$timestamp] $text  (${String.format("%.1f", durationSec)}s, ${inferenceTimeMs}ms)"
        } else {
            "\n\n[$timestamp] $text  (${String.format("%.1f", durationSec)}s, ${inferenceTimeMs}ms)"
        }
        // Note: transcriptText.text will be overwritten by the next logLines emission
        // but the viewModel.appendLog() call above ensures the transcript is in the log.
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
        viewModel.appendLog("Requesting RECORD_AUDIO permission...")
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
                viewModel.appendLog("RECORD_AUDIO permission GRANTED")
                Toast.makeText(this, "Microphone permission granted — hold button to record", Toast.LENGTH_SHORT).show()
            } else {
                viewModel.appendLog("[WARN] RECORD_AUDIO permission DENIED")
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