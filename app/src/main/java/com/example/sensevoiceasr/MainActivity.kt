package com.example.sensevoiceasr

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.sensevoiceasr.audio.AudioRecorder
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val PREFS_NAME = "sensevoice_prefs"
        private const val KEY_SERVER_URL = "server_url"
        private const val DEFAULT_URL = "wss://lauren-incident-contract-prepaid.trycloudflare.com"
    }

    private lateinit var btnRecord: Button
    private lateinit var btnClear: Button
    private lateinit var btnConnect: Button
    private lateinit var statusText: TextView
    private lateinit var serverUrlInput: EditText
    private lateinit var transcriptContainer: android.widget.LinearLayout

    private var audioRecorder: AudioRecorder? = null
    private var webSocket: WebSocket? = null
    private var okHttpClient: OkHttpClient? = null

    private var recordingJob: Job? = null
    private var isRecording = false
    private var isConnected = false

    private val timeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRecord = findViewById(R.id.btn_record)
        btnClear = findViewById(R.id.btn_clear)
        btnConnect = findViewById(R.id.btn_connect)
        statusText = findViewById(R.id.status)
        serverUrlInput = findViewById(R.id.server_url)
        transcriptContainer = findViewById(R.id.transcript_container)

        // Load saved server URL
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        serverUrlInput.setText(prefs.getString(KEY_SERVER_URL, DEFAULT_URL))

        // Connect button
        btnConnect.setOnClickListener {
            val url = serverUrlInput.text.toString().trim()
            if (url.isNotEmpty()) {
                prefs.edit().putString(KEY_SERVER_URL, url).apply()
                connectToServer(url)
            }
        }

        // Hold-to-record button
        btnRecord.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!isConnected) {
                        Toast.makeText(this, "Not connected to server", Toast.LENGTH_SHORT).show()
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
            transcriptContainer.removeAllViews()
            partialView = null
            statusText.text = "Cleared"
        }

        // Auto-connect on launch
        val savedUrl = prefs.getString(KEY_SERVER_URL, DEFAULT_URL) ?: DEFAULT_URL
        connectToServer(savedUrl)
    }

    private fun connectToServer(url: String) {
        updateStatus("Connecting...")
        btnConnect.text = "..."

        // Close existing connection
        webSocket?.close(1000, "Reconnecting")

        okHttpClient = OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()

        val request = Request.Builder().url(url).build()
        webSocket = okHttpClient?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                isConnected = true
                runOnUiThread {
                    updateStatus("Connected")
                    btnConnect.text = "Connected"
                    btnConnect.setBackgroundColor(getColor(R.color.connect_green))
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val msg = JSONObject(text)
                    val type = msg.optString("type")
                    when (type) {
                        "partial" -> {
                            val partialText = msg.optString("text", "")
                            runOnUiThread { showPartial(partialText) }
                        }
                        "final" -> {
                            val finalText = msg.optString("text", "")
                            val timestamp = msg.optString("timestamp", "")
                            val duration = msg.optDouble("duration", 0.0)
                            val inferenceMs = msg.optDouble("inference_ms", 0.0)
                            runOnUiThread {
                                clearPartial()
                                appendTranscript(finalText, timestamp, duration, inferenceMs)
                            }
                        }
                        "status" -> {
                            val statusMsg = msg.optString("text", "")
                            runOnUiThread { updateStatus(statusMsg) }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "Parse error", e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
                isConnected = false
                runOnUiThread {
                    updateStatus("Disconnected")
                    btnConnect.text = "Connect"
                    btnConnect.setBackgroundColor(getColor(R.color.button_clear))
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                isConnected = false
                runOnUiThread {
                    updateStatus("Connection failed: ${t.message}")
                    btnConnect.text = "Retry"
                    btnConnect.setBackgroundColor(getColor(R.color.record_red))
                }
            }
        })
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
        val started = audioRecorder?.start() ?: false
        if (!started) {
            isRecording = false
            resetRecordButton()
            Toast.makeText(this, "Failed to start recording", Toast.LENGTH_SHORT).show()
            return
        }

        recordingJob = lifecycleScope.launch(Dispatchers.IO) {
            val recorder = audioRecorder ?: return@launch
            recorder.audioChunks.collect { floatBytes ->
                if (!isRecording) return@collect
                val ws = webSocket
                if (ws != null && isConnected) {
                    ws.send(okio.ByteString.of(*floatBytes))
                }
            }
        }
    }

    private fun stopRecording() {
        if (!isRecording) return
        isRecording = false

        recordingJob?.cancel()
        recordingJob = null

        audioRecorder?.stop()
        audioRecorder = null

        resetRecordButton()
        updateStatus("Connected")
    }

    private var partialView: TextView? = null

    private fun showPartial(text: String) {
        if (partialView == null) {
            partialView = TextView(this).apply {
                setTextColor(getColor(R.color.partial_blue))
                textSize = 15f
                setPadding(16, 4, 16, 8)
            }
            transcriptContainer.addView(partialView)
        }
        partialView?.text = "... $text"
    }

    private fun clearPartial() {
        partialView?.let {
            transcriptContainer.removeView(it)
        }
        partialView = null
    }

    private fun appendTranscript(text: String, timestamp: String, durationSec: Double, inferenceTimeMs: Double) {
        val entry = TextView(this).apply {
            setTextColor(getColor(R.color.text_primary))
            textSize = 14f
            setPadding(16, 6, 16, 2)
            text = "[$timestamp] $text"
        }
        val meta = TextView(this).apply {
            setTextColor(getColor(R.color.text_secondary))
            textSize = 11f
            setPadding(16, 0, 16, 10)
            text = "${"%.1f".format(durationSec)}s · ${"%.0f".format(inferenceTimeMs)}ms"
        }
        transcriptContainer.addView(entry)
        transcriptContainer.addView(meta)

        // Auto-scroll
        findViewById<android.widget.ScrollView>(R.id.scroll_view).post {
            findViewById<android.widget.ScrollView>(R.id.scroll_view).fullScroll(android.view.View.FOCUS_DOWN)
        }
    }

    private fun updateStatus(msg: String) {
        runOnUiThread { statusText.text = msg }
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
        webSocket?.close(1000, "App closed")
        okHttpClient?.dispatcher?.executorService?.shutdown()
    }
}