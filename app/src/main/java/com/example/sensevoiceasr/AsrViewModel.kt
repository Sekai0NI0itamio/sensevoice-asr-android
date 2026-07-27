package com.example.sensevoiceasr

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.sensevoiceasr.asr.OnnxInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

/**
 * Survives activity recreation (permission dialogs, rotations, etc.).
 * Loads the ONNX model once and holds it for the lifetime of the app.
 * Exposes a logFlow that streams diagnostic lines from every pipeline component
 * (model load, audio chunks, VAD segments, inference steps, decoding) so the
 * UI can display them as a live diagnostic console.
 */
class AsrViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "AsrViewModel"
        private const val MAX_LOG_LINES = 500
    }

    sealed class ModelState {
        data object Loading : ModelState()
        data object Ready : ModelState()
        data class Error(val message: String) : ModelState()
    }

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Loading)
    val modelState: StateFlow<ModelState> = _modelState

    /** Channel + Flow for live log lines with a limited buffer, dropping oldest on overflow. */
    private val logChannel = Channel<String>(capacity = 1024, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    private val _logLines = MutableStateFlow<List<String>>(emptyList())
    val logLines: StateFlow<List<String>> = _logLines

    private var onnxInference: OnnxInference? = null

    val isModelReady: Boolean get() = onnxInference?.isLoaded == true

    init {
        // Collect log channel into the StateFlow list, bounded to MAX_LOG_LINES
        viewModelScope.launch(Dispatchers.Default) {
            logChannel.receiveAsFlow().collect { line ->
                val current = _logLines.value
                val updated = if (current.size >= MAX_LOG_LINES) {
                    current.drop(current.size - MAX_LOG_LINES + 1) + line
                } else {
                    current + line
                }
                _logLines.value = updated
            }
        }

        appendLog("App started. AsrViewModel initializing.")
        appendLog("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} (SDK ${android.os.Build.VERSION.SDK_INT})")
        appendLog("ABIs: ${android.os.Build.SUPPORTED_ABIS.joinToString()}")
        loadModel()
    }

    /** Emit a single diagnostic line to both logcat and the UI log flow. */
    fun appendLog(line: String) {
        Log.i(TAG, line)
        viewModelScope.launch {
            logChannel.send(line)
        }
    }

    private fun loadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _modelState.value = ModelState.Loading
                appendLog("Starting ONNX model load on IO thread...")

                val inference = OnnxInference(getApplication())
                // Wire inference logs -> UI flow
                inference.logCallback = { line -> appendLog(line) }

                val loaded = inference.loadModel()
                if (loaded) {
                    onnxInference = inference
                    _modelState.value = ModelState.Ready
                    appendLog("Model load SUCCESS. isModelReady=$isModelReady")
                } else {
                    val errMsg = inference.lastError ?: "Model load failed (unknown error)"
                    _modelState.value = ModelState.Error(errMsg)
                    appendLog("[ERR] Model load FAILED: $errMsg")
                }
            } catch (e: OutOfMemoryError) {
                val msg = "Out of memory during load: ${e.message}"
                Log.e(TAG, msg)
                appendLog("[ERR] $msg")
                _modelState.value = ModelState.Error("Not enough memory. Try closing other apps.")
            } catch (e: Exception) {
                val msg = "Model load exception: ${e.javaClass.simpleName}: ${e.message}"
                Log.e(TAG, msg, e)
                appendLog("[ERR] $msg")
                _modelState.value = ModelState.Error("Model error: ${e.message}")
            }
        }
    }

    fun transcribe(audio: FloatArray): String {
        appendLog("AsrViewModel.transcribe() called: audio=${audio.size} samples (${String.format("%.2f", audio.size / 16000.0f)}s)")
        val start = System.currentTimeMillis()
        val result = onnxInference?.transcribe(audio) ?: run {
            appendLog("[WARN] transcribe: onnxInference is null, returning empty")
            return ""
        }
        appendLog("AsrViewModel.transcribe() returned in ${System.currentTimeMillis() - start}ms: \"$result\"")
        return result
    }

    override fun onCleared() {
        appendLog("AsrViewModel.onCleared() — releasing inference")
        super.onCleared()
        onnxInference?.release()
        onnxInference = null
    }
}