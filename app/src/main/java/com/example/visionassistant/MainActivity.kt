package com.example.visionassistant

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.visionassistant.theme.VisionAssistantTheme
import java.util.Locale

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    private val viewModel: MainViewModel by viewModels()
    private var tts: TextToSpeech? = null
    private var speechRecognizer: SpeechRecognizer? = null
    
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            if (permissions[Manifest.permission.CAMERA] == false) {
                Log.e("VisionAssistant", "Camera permission denied.")
            }
            if (permissions[Manifest.permission.RECORD_AUDIO] == false) {
                Log.e("VisionAssistant", "Audio permission denied.")
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        tts = TextToSpeech(this, this)
        
        val permissions = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.CAMERA)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        if (permissions.isNotEmpty()) {
            requestPermissionLauncher.launch(permissions.toTypedArray())
        }

        setupSpeechRecognizer()

        enableEdgeToEdge()
        setContent {
            VisionAssistantTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    MainScreen(viewModel, ::startListening, ::stopAction)
                }
            }
        }
    }

    private fun setupSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(this)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
            speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onError(error: Int) {
                    viewModel.onSpeechRecognitionFailed()
                }
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val text = matches[0]
                        val isAsk = viewModel.taskState.value.mode == TaskMode.ASK
                        viewModel.onSpeechRecognitionComplete(text, isAsk)
                    } else {
                        viewModel.onSpeechRecognitionFailed()
                    }
                }
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }

    private fun startListening() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            viewModel.onInteractionStart()
            viewModel.onSpeechRecognitionStart()
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            }
            
            // TTS "Listening." or "What should I find?" before starting recognition
            val prompt = if (viewModel.taskState.value.mode == TaskMode.ASK) "Listening." else "What should I find?"
            tts?.speak(prompt, TextToSpeech.QUEUE_FLUSH, null, "PROMPT")
            
            // Give TTS a small buffer before starting the microphone, but since we can't easily await TTS in this synchronous call, we just start it.
            // Android's recognizer will beep anyway.
            speechRecognizer?.startListening(intent)
        } else {
            tts?.speak("Microphone permission is required.", TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun stopAction() {
        speechRecognizer?.cancel()
        viewModel.resetState()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.US
            tts?.let { viewModel.setTts(it) }
        }
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        speechRecognizer?.destroy()
        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel, 
    onStartListening: () -> Unit,
    onStopAction: () -> Unit
) {
    val state by viewModel.state.collectAsState()
    val taskState by viewModel.taskState.collectAsState()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val imageCapture = remember { ImageCapture.Builder().build() }

    fun captureImage() {
        viewModel.onCaptureRequested()
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    viewModel.imageEncodeStartTime = System.currentTimeMillis()
                    val bitmap = imageProxyToBitmap(image)
                    viewModel.imageEncodeCompleteTime = System.currentTimeMillis()
                    
                    viewModel.onImageCaptured(bitmap)
                    image.close()
                }

                override fun onError(exception: ImageCaptureException) {
                    viewModel.onImageCaptureFailed(exception)
                }
            }
        )
    }

    LaunchedEffect(Unit) {
        viewModel.captureRequests.collect {
            captureImage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Vision Assistant") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { innerPadding ->
        Box(modifier = Modifier.fillMaxSize().padding(innerPadding)) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    val previewView = PreviewView(ctx)
                    val executor = ContextCompat.getMainExecutor(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = androidx.camera.core.Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        
                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageCapture
                            )
                        } catch (e: Exception) {
                            Log.e("CameraPreview", "Use case binding failed", e)
                        }
                    }, executor)
                    previewView
                }
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.Black.copy(alpha = 0.6f))
                    .padding(8.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                ModeButton(
                    text = "GENERAL",
                    isSelected = taskState.mode == TaskMode.GENERAL,
                    onClick = { viewModel.setMode(TaskMode.GENERAL) }
                )
                ModeButton(
                    text = "READ",
                    isSelected = taskState.mode == TaskMode.READ,
                    onClick = { viewModel.setMode(TaskMode.READ) }
                )
                ModeButton(
                    text = "ASK",
                    isSelected = taskState.mode == TaskMode.ASK,
                    onClick = { viewModel.setMode(TaskMode.ASK) }
                )
                ModeButton(
                    text = "FIND",
                    isSelected = taskState.mode == TaskMode.FIND,
                    onClick = { viewModel.setMode(TaskMode.FIND) }
                )
            }

            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .background(Color.Black.copy(alpha = 0.6f))
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = when (state) {
                        is AppState.Idle -> "Ready"
                        is AppState.Listening -> "Listening..."
                        is AppState.Transcribing -> "Transcribing..."
                        is AppState.Capturing -> "Capturing..."
                        is AppState.Processing -> "Thinking..."
                        is AppState.Speaking -> "Speaking: ${(state as AppState.Speaking).text}"
                        is AppState.Error -> "Error: ${(state as AppState.Error).message}"
                    },
                    style = MaterialTheme.typography.titleMedium,
                    color = Color.White,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                // Show debug data during FIND
                if (taskState.mode == TaskMode.FIND && taskState.findTarget.isNotEmpty()) {
                     Text(
                        text = "Target: ${taskState.findTarget}",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.Yellow,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                }

                if (taskState.mode == TaskMode.ASK || taskState.mode == TaskMode.FIND) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            Button(
                                onClick = { onStartListening() },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(60.dp)
                                    .semantics { 
                                        contentDescription = if (taskState.mode == TaskMode.ASK) "Ask a question" else "Find an object"
                                    },
                                enabled = state is AppState.Idle || state is AppState.Error || state is AppState.Speaking
                            ) {
                                Text(if (taskState.mode == TaskMode.ASK) "Ask" else "Find", style = MaterialTheme.typography.titleMedium)
                            }
                            Button(
                                onClick = { onStopAction() },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(60.dp)
                                    .semantics { contentDescription = "Stop" },
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                            ) {
                                Text("Stop", style = MaterialTheme.typography.titleMedium)
                            }
                        }
                    }
                } else {
                    Button(
                        onClick = { 
                            viewModel.onInteractionStart()
                            captureImage() 
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(80.dp)
                            .semantics { 
                                contentDescription = when(taskState.mode) {
                                    TaskMode.GENERAL -> "Capture general scene"
                                    TaskMode.READ -> "Read text in front of camera"
                                    else -> ""
                                }
                            }
                    ) {
                        Text(
                            text = when(taskState.mode) {
                                TaskMode.GENERAL -> "What's in front of me?"
                                TaskMode.READ -> "Read text"
                                else -> ""
                            }, 
                            style = MaterialTheme.typography.titleLarge
                        )
                    }
                    if (state is AppState.Speaking || state is AppState.Error) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = { onStopAction() },
                            modifier = Modifier.fillMaxWidth().height(60.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                        ) {
                            Text("Stop", style = MaterialTheme.typography.titleMedium)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ModeButton(text: String, isSelected: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected) MaterialTheme.colorScheme.primary else Color.Gray
        ),
        modifier = Modifier.semantics { 
            contentDescription = "Switch to $text mode. Currently ${if(isSelected) "selected" else "not selected"}." 
        }
    ) {
        Text(text)
    }
}

fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val buffer = image.planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    val bitmap = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size, null)
    
    val matrix = Matrix()
    matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}
