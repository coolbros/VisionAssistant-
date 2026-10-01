package com.example.visionassistant

import android.graphics.Bitmap
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.util.UUID

sealed class AppState {
    object Idle : AppState()
    object Listening : AppState()
    object Transcribing : AppState()
    object Capturing : AppState()
    object Processing : AppState()
    data class Speaking(val text: String) : AppState()
    data class Error(val message: String) : AppState()
}

class MainViewModel : ViewModel() {
    private val _state = MutableStateFlow<AppState>(AppState.Idle)
    val state: StateFlow<AppState> = _state.asStateFlow()

    private val _captureRequests = MutableSharedFlow<Unit>()
    val captureRequests: SharedFlow<Unit> = _captureRequests.asSharedFlow()

    private val _taskState = MutableStateFlow(TaskState())
    val taskState: StateFlow<TaskState> = _taskState.asStateFlow()

    private val visionModel: VisionModel = CloudVisionModel(BuildConfig.GEMINI_API_KEY)
    private val json = Json { ignoreUnknownKeys = true }
    
    private var tts: TextToSpeech? = null
    
    private var readRetryCount = 0
    private val maxReadRetries = 3

    private var findRetryCount = 0
    private val maxFindRetries = 15

    // Latency Tracking (General / ASK)
    var interactionStartTime: Long = 0
    var speechRecStartTime: Long = 0
    var speechRecCompleteTime: Long = 0
    var captureStartTime: Long = 0
    var captureCompleteTime: Long = 0
    var imageEncodeStartTime: Long = 0
    var imageEncodeCompleteTime: Long = 0

    // Latency Tracking (FIND)
    var findIterationStartTime: Long = 0

    fun setTts(textToSpeech: TextToSpeech) {
        tts = textToSpeech
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                if (utteranceId == "READ_RETRY" || utteranceId == "FIND_RETRY") {
                    viewModelScope.launch {
                        delay(500)
                        if (utteranceId == "FIND_RETRY") {
                            findIterationStartTime = System.currentTimeMillis()
                        }
                        captureStartTime = System.currentTimeMillis()
                        _captureRequests.emit(Unit)
                    }
                }
            }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {}
        })
    }

    fun setMode(mode: TaskMode) {
        _taskState.value = _taskState.value.copy(mode = mode)
        resetState()
    }

    fun onInteractionStart() {
        interactionStartTime = System.currentTimeMillis()
        if (_taskState.value.mode == TaskMode.FIND) {
            findIterationStartTime = interactionStartTime
        }
    }

    fun onSpeechRecognitionStart() {
        speechRecStartTime = System.currentTimeMillis()
        _state.value = AppState.Listening
    }

    fun onSpeechRecognitionComplete(text: String, isAsk: Boolean) {
        speechRecCompleteTime = System.currentTimeMillis()
        _state.value = AppState.Transcribing
        if (isAsk) {
            _taskState.value = _taskState.value.copy(askQuestion = text)
        } else {
            _taskState.value = _taskState.value.copy(findTarget = text)
        }
        
        // Immediately trigger capture after transcription
        captureStartTime = System.currentTimeMillis()
        _state.value = AppState.Capturing
        viewModelScope.launch {
            _captureRequests.emit(Unit)
        }
    }

    fun onSpeechRecognitionFailed() {
        speechRecCompleteTime = System.currentTimeMillis()
        val speech = "I didn't hear that. Please try again."
        _state.value = AppState.Speaking(speech)
        tts?.speak(speech, TextToSpeech.QUEUE_FLUSH, null, UUID.randomUUID().toString())
    }

    fun onCaptureRequested() {
        if (captureStartTime == 0L || _state.value == AppState.Idle) {
            captureStartTime = System.currentTimeMillis()
        }
        _state.value = AppState.Capturing
    }

    fun onImageCaptured(bitmap: Bitmap) {
        captureCompleteTime = System.currentTimeMillis()
        _state.value = AppState.Processing
        viewModelScope.launch {
            val currentMode = _taskState.value.mode
            val currentState = _taskState.value.copy(retryCount = Math.max(readRetryCount, findRetryCount))
            
            val requestStartTime = System.currentTimeMillis()
            val decision = visionModel.analyze(
                image = bitmap,
                state = currentState
            )
            val requestCompleteTime = System.currentTimeMillis()
            val parseCompleteTime = System.currentTimeMillis() 

            var speechToSay = decision.speech
            var utteranceId = UUID.randomUUID().toString()
            var willRetry = false
            
            if (currentMode == TaskMode.READ) {
                if (decision.needsRepositioning) {
                    if (readRetryCount < maxReadRetries) {
                        readRetryCount++
                        utteranceId = "READ_RETRY"
                        willRetry = true
                    } else {
                        readRetryCount = 0
                        speechToSay = "I can't read this clearly. Try moving closer or improving the lighting."
                    }
                } else {
                    readRetryCount = 0
                }
            } else if (currentMode == TaskMode.FIND) {
                val decisionCompleteStart = System.currentTimeMillis()
                val findResult = try { json.decodeFromString<FindResult>(decision.debugData) } catch(e: Exception) { null }
                
                if (findResult == null || decision.speech.startsWith("I encountered")) {
                    speechToSay = "I encountered an error trying to find that."
                } else {
                    if (findResult.target_visible) {
                        val centerX = findResult.center_x
                        val centerY = findResult.center_y
                        val area = findResult.width * findResult.height
                        
                        val isCenteredX = centerX in 0.35..0.65
                        val isCenteredY = centerY in 0.35..0.65
                        val isLargeEnough = area >= 0.08
                        
                        if (isCenteredX && isCenteredY && isLargeEnough && findResult.confidence > 0.5) {
                            speechToSay = "Found your ${currentState.findTarget}."
                            findRetryCount = 0
                        } else {
                            // Local deterministic policy
                            if (centerX < 0.35) speechToSay = "Move left."
                            else if (centerX > 0.65) speechToSay = "Move right."
                            else if (centerY < 0.35) speechToSay = "Move up."
                            else if (centerY > 0.65) speechToSay = "Move down."
                            else speechToSay = "Move closer."
                            
                            if (findRetryCount < maxFindRetries) {
                                findRetryCount++
                                utteranceId = "FIND_RETRY"
                                willRetry = true
                            } else {
                                findRetryCount = 0
                                speechToSay = "I couldn't find it. Try moving the camera."
                                willRetry = false
                            }
                        }
                    } else {
                        // Bounded search strategy
                        if (findRetryCount < maxFindRetries) {
                            findRetryCount++
                            utteranceId = "FIND_RETRY"
                            willRetry = true
                            // Speak only occasional search hints
                            speechToSay = if (findRetryCount % 3 == 0) "Searching." else ""
                        } else {
                            findRetryCount = 0
                            speechToSay = "I couldn't find it. Try moving the camera."
                            willRetry = false
                        }
                    }
                }
                
                val decisionCompleteTime = System.currentTimeMillis()
                
                _state.value = AppState.Speaking(speechToSay)
                val ttsStartTime = System.currentTimeMillis()
                if (speechToSay.isNotEmpty()) {
                    tts?.speak(speechToSay, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
                } else {
                    // Simulate fast TTS end to re-trigger
                    if (willRetry) {
                         viewModelScope.launch {
                             delay(500)
                             findIterationStartTime = System.currentTimeMillis()
                             captureStartTime = System.currentTimeMillis()
                             _captureRequests.emit(Unit)
                         }
                    }
                }
                logFindLatencies(requestStartTime, requestCompleteTime, parseCompleteTime, decisionCompleteTime, ttsStartTime)
                return@launch
            }
            
            _state.value = AppState.Speaking(speechToSay)
            val ttsStartTime = System.currentTimeMillis()
            tts?.speak(speechToSay, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
            
            if (currentMode == TaskMode.ASK || currentMode == TaskMode.GENERAL) {
                logAskLatencies(requestStartTime, requestCompleteTime, parseCompleteTime, ttsStartTime)
            } else if (currentMode == TaskMode.READ) {
                logAskLatencies(requestStartTime, requestCompleteTime, parseCompleteTime, ttsStartTime) // Reuse
            }
        }
    }

    private fun logAskLatencies(reqStart: Long, reqComplete: Long, parseComplete: Long, ttsStart: Long) {
        val total = ttsStart - interactionStartTime
        val captureDur = captureCompleteTime - captureStartTime
        val encodeDur = imageEncodeCompleteTime - imageEncodeStartTime
        val reqDur = reqComplete - reqStart
        val parseDur = parseComplete - reqComplete
        
        Log.i("VisionAssistantLatency", """
            LATENCY REPORT (ASK/GENERAL/READ):
            ask_button_press -> tts_start (total_latency): ${total}ms
            speech_recognition_start -> speech_recognition_complete: ${speechRecCompleteTime - speechRecStartTime}ms
            capture_start -> capture_complete: ${captureDur}ms
            image_encode_start -> image_encode_complete: ${encodeDur}ms
            request_start -> request_complete: ${reqDur}ms
            response_parse_complete: ${parseDur}ms
        """.trimIndent())
        
        captureStartTime = 0L
    }

    private fun logFindLatencies(reqStart: Long, reqComplete: Long, parseComplete: Long, decisionComplete: Long, ttsStart: Long) {
        val total = ttsStart - findIterationStartTime
        val captureDur = captureCompleteTime - captureStartTime
        val reqDur = reqComplete - reqStart
        val parseDur = parseComplete - reqComplete
        val decisionDur = decisionComplete - parseComplete
        
        Log.i("VisionAssistantLatency", """
            LATENCY REPORT (FIND ITERATION):
            find_iteration_start -> tts_start (iteration_total): ${total}ms
            capture_start -> capture_complete: ${captureDur}ms
            request_start -> request_complete: ${reqDur}ms
            response_parse_complete: ${parseDur}ms
            decision_complete: ${decisionDur}ms
        """.trimIndent())
    }

    fun onImageCaptureFailed(e: Exception) {
        _state.value = AppState.Error("Failed to capture image.")
        readRetryCount = 0
        findRetryCount = 0
    }

    fun resetState() {
        _state.value = AppState.Idle
        readRetryCount = 0
        findRetryCount = 0
        tts?.stop()
    }
}
