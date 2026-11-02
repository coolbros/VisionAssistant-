package com.example.visionassistant

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class LocalVisionEngine(private val context: Context) : VisionModel {

    companion object {
        init {
            System.loadLibrary("vision_jni")
        }
    }

    private val TAG = "LocalVisionEngine"
    private val modelName = "LFM2-VL-450M-Q4_0.gguf"
    private val projName = "mmproj-LFM2-VL-450M-Q8_0.gguf"
    
    private var isInitialized = false

    private external fun nativeInitialize(modelPath: String, mmprojPath: String, threads: Int, nativeLibDir: String): Boolean
    private external fun nativeInfer(bitmap: Bitmap, prompt: String): String
    private external fun nativeClose()

    private suspend fun ensureModels() = withContext(Dispatchers.IO) {
        if (isInitialized) return@withContext
        
        copyAssetIfNotExists(modelName)
        copyAssetIfNotExists(projName)
        
        val modelFile = File(context.filesDir, modelName).absolutePath
        val projFile = File(context.filesDir, projName).absolutePath
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        
        // Pass 4 threads for baseline CPU inference (adjust later if needed)
        val success = nativeInitialize(modelFile, projFile, 4, nativeLibDir)
        if (!success) {
            Log.e(TAG, "Failed to initialize native engine")
        }
        
        isInitialized = true
    }

    private fun copyAssetIfNotExists(filename: String) {
        val outFile = File(context.filesDir, filename)
        if (outFile.exists()) return
        
        Log.d(TAG, "Copying $filename from assets...")
        context.assets.open(filename).use { ins ->
            FileOutputStream(outFile).use { outs ->
                val buffer = ByteArray(1024 * 1024)
                var read: Int
                while (ins.read(buffer).also { read = it } != -1) {
                    outs.write(buffer, 0, read)
                }
            }
        }
        Log.d(TAG, "Copied $filename.")
    }

    override suspend fun analyze(image: Bitmap, state: TaskState): VisionDecision = withContext(Dispatchers.IO) {
        if (state.mode != TaskMode.FIND) {
            return@withContext VisionDecision("Local engine does not support this mode yet.")
        }
        
        val target = state.findTarget.ifBlank { "target" }
        ensureModels()
        
        // 1. Resize image (retained from Phase 3.1)
        val maxDim = 336
        val scale = Math.min(maxDim.toFloat() / image.width, maxDim.toFloat() / image.height)
        val scaledWidth = Math.round(image.width * scale)
        val scaledHeight = Math.round(image.height * scale)
        val resizedImage = Bitmap.createScaledBitmap(image, scaledWidth, scaledHeight, true)

        val prompt = "You are a visual target localizer. Target: $target. Look at the image. Return EXACTLY ONE label from this list: NOT_VISIBLE, UNCERTAIN, UP_LEFT, UP, UP_RIGHT, LEFT, CENTER, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT, CENTER_FAR, FOUND. Do not output any other words. Do not explain. Do not use punctuation."
        
        // 2. Inference via JNI
        val output = nativeInfer(resizedImage, prompt)
        
        // 4. Parse label
        val words = output.split(Regex("\\s+"))
        val possibleLabels = setOf("NOT_VISIBLE", "UNCERTAIN", "UP_LEFT", "UP", "UP_RIGHT", "LEFT", "CENTER", "RIGHT", "DOWN_LEFT", "DOWN", "DOWN_RIGHT", "CENTER_FAR", "FOUND")
        
        var parsedLabel = "UNCERTAIN" // default
        for (word in words.reversed()) {
            val cleanWord = word.replace(Regex("[^A-Z_]"), "")
            if (possibleLabels.contains(cleanWord)) {
                parsedLabel = cleanWord
                break
            }
        }
        
        Log.d(TAG, "Parsed Label: $parsedLabel")
        
        // 5. Convert to Policy Action
        val action = when (parsedLabel) {
            "NOT_VISIBLE", "UNCERTAIN" -> FastPolicyAction.REACQUIRE
            "LEFT", "UP_LEFT", "DOWN_LEFT" -> FastPolicyAction.LEFT
            "RIGHT", "UP_RIGHT", "DOWN_RIGHT" -> FastPolicyAction.RIGHT
            "CENTER_FAR", "CENTER", "UP", "DOWN" -> FastPolicyAction.MOVE_CLOSER
            "FOUND" -> FastPolicyAction.TASK_COMPLETE
            else -> FastPolicyAction.REACQUIRE
        }
        
        val speech = when(action) {
            FastPolicyAction.LEFT -> "Move left"
            FastPolicyAction.RIGHT -> "Move right"
            FastPolicyAction.MOVE_CLOSER -> "Move closer"
            FastPolicyAction.REACQUIRE -> "Look around"
            FastPolicyAction.TASK_COMPLETE -> "Found your $target."
            else -> ""
        }
        
        return@withContext VisionDecision(
            speech = speech,
            action = action,
            taskComplete = action == FastPolicyAction.TASK_COMPLETE,
            needsRepositioning = action != FastPolicyAction.TASK_COMPLETE,
            debugData = "LFM2-VL output: $parsedLabel"
        )
    }
}
