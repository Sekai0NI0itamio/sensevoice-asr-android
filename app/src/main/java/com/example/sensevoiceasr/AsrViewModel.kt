package com.example.sensevoiceasr

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.sensevoiceasr.asr.OnnxInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Survives activity recreation (permission dialogs, rotations, etc.).
 * Loads the ONNX model once and holds it for the lifetime of the app.
 */
class AsrViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "AsrViewModel"
    }

    sealed class ModelState {
        data object Loading : ModelState()
        data object Ready : ModelState()
        data class Error(val message: String) : ModelState()
    }

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Loading)
    val modelState: StateFlow<ModelState> = _modelState

    private var onnxInference: OnnxInference? = null

    val isModelReady: Boolean get() = onnxInference?.isLoaded == true

    init {
        loadModel()
    }

    private fun loadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _modelState.value = ModelState.Loading
                Log.i(TAG, "Starting model load...")
                val inference = OnnxInference(getApplication())
                val loaded = inference.loadModel()
                if (loaded) {
                    onnxInference = inference
                    _modelState.value = ModelState.Ready
                    Log.i(TAG, "Model ready")
                } else {
                    _modelState.value = ModelState.Error("Model load failed")
                    Log.e(TAG, "Model load returned false")
                }
            } catch (e: OutOfMemoryError) {
                Log.e(TAG, "Out of memory: ${e.message}")
                _modelState.value = ModelState.Error("Not enough memory. Try closing other apps.")
            } catch (e: Exception) {
                Log.e(TAG, "Model load error: ${e.message}", e)
                _modelState.value = ModelState.Error("Model error: ${e.message}")
            }
        }
    }

    fun transcribe(audio: FloatArray): String {
        return onnxInference?.transcribe(audio) ?: ""
    }

    override fun onCleared() {
        super.onCleared()
        onnxInference?.release()
        onnxInference = null
    }
}