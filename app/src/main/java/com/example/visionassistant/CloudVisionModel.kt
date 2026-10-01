package com.example.visionassistant

import android.graphics.Bitmap
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import com.google.ai.client.generativeai.type.generationConfig
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class ReadResult(
    val text_found: Boolean = false,
    val confidence: Double = 0.0,
    val text: String = "",
    val needs_repositioning: Boolean = false,
    val guidance: String = ""
)

@Serializable
data class AskResult(
    val answer: String = "",
    val confidence: Double = 0.0,
    val uncertain: Boolean = false
)

@Serializable
data class FindResult(
    val target_visible: Boolean = false,
    val confidence: Double = 0.0,
    val center_x: Double = 0.5,
    val center_y: Double = 0.5,
    val width: Double = 0.0,
    val height: Double = 0.0,
    val speech_hint: String = ""
)

class CloudVisionModel(apiKey: String) : VisionModel {
    private val json = Json { ignoreUnknownKeys = true }
    
    private val generativeModelGeneral = GenerativeModel(
        modelName = "gemini-3.5-flash",
        apiKey = apiKey
    )

    private val generativeModelJson = GenerativeModel(
        modelName = "gemini-3.5-flash",
        apiKey = apiKey,
        generationConfig = generationConfig {
            responseMimeType = "application/json"
        }
    )

    override suspend fun analyze(
        image: Bitmap,
        state: TaskState
    ): VisionDecision {
        return try {
            when (state.mode) {
                TaskMode.GENERAL -> analyzeGeneral(image)
                TaskMode.READ -> analyzeRead(image)
                TaskMode.ASK -> analyzeAsk(image, state.askQuestion)
                TaskMode.FIND -> analyzeFind(image, state.findTarget)
            }
        } catch (e: Exception) {
            VisionDecision(
                speech = "Sorry, I had trouble connecting to the cloud.",
                taskComplete = true
            )
        }
    }

    private suspend fun analyzeGeneral(image: Bitmap): VisionDecision {
        val prompt = """
            You are a closed-loop visual guidance agent for a blind user.
            The user's goal is: "What is in front of me?"
            
            Describe what you see in the frame in 1-2 short, concise sentences. 
            Be direct and helpful. 
        """.trimIndent()
        
        val response = generativeModelGeneral.generateContent(
            content {
                image(image)
                text(prompt)
            }
        )
        return VisionDecision(
            speech = response.text ?: "I am not sure what I see.",
            taskComplete = true 
        )
    }
    
    private suspend fun analyzeRead(image: Bitmap): VisionDecision {
        val prompt = """
            Extract text from this image for a blind user.
            Return a JSON object with:
            - text_found (boolean)
            - confidence (number 0.0 to 1.0)
            - text (string): the exact text found, maintaining reading order.
            - needs_repositioning (boolean): true if the text is cut off, blurry, too small, or poorly framed.
            - guidance (string): one short instruction to fix framing (e.g., "Move the camera closer.", "Move the camera slightly down.", "Hold the camera steady.") or empty if perfectly framed.
            
            If NO useful text is visible, text_found=false and provide guidance like "Move the camera toward the page."
            Do not invent text.
        """.trimIndent()
        
        val response = generativeModelJson.generateContent(
            content {
                image(image)
                text(prompt)
            }
        )
        
        val jsonString = response.text ?: "{}"
        val result = try { json.decodeFromString<ReadResult>(jsonString) } catch(e: Exception) { null }
        
        if (result == null) {
             return VisionDecision("I couldn't read that clearly.", taskComplete = true, debugData = jsonString)
        }
        
        if (!result.text_found) {
            return VisionDecision(
                speech = result.guidance.ifEmpty { "No readable text found." },
                taskComplete = !result.needs_repositioning,
                needsRepositioning = result.needs_repositioning,
                debugData = jsonString
            )
        }
        
        if (result.needs_repositioning) {
            return VisionDecision(
                speech = result.guidance,
                taskComplete = false,
                needsRepositioning = true,
                debugData = jsonString
            )
        }
        
        return VisionDecision(
            speech = result.text,
            taskComplete = true,
            needsRepositioning = false,
            debugData = jsonString
        )
    }

    private suspend fun analyzeAsk(image: Bitmap, question: String): VisionDecision {
        val prompt = """
            You are a closed-loop visual guidance agent for a blind user.
            The user asks: "$question"
            
            Return a JSON object with:
            - answer (string): A concise answer to the question based ONLY on visible evidence. Use spatial language (left/right/above) if useful. 
            - confidence (number 0.0 to 1.0)
            - uncertain (boolean): true if ambiguous, obscured, or unsure.
            
            If uncertain, the answer should state "I can see what appears to be..." or "I am not sure." Do not invent details.
        """.trimIndent()
        
        val response = generativeModelJson.generateContent(
            content {
                image(image)
                text(prompt)
            }
        )
        
        val jsonString = response.text ?: "{}"
        val result = try { json.decodeFromString<AskResult>(jsonString) } catch(e: Exception) { null }
        
        if (result == null) {
             return VisionDecision("I couldn't determine the answer.", taskComplete = true, debugData = jsonString)
        }
        
        return VisionDecision(
            speech = result.answer,
            taskComplete = true,
            debugData = jsonString
        )
    }

    private suspend fun analyzeFind(image: Bitmap, target: String): VisionDecision {
        val prompt = """
            You are a visual perception module for a blind user looking for: "$target"
            
            Return a strict JSON object with:
            - target_visible (boolean): Is the target visible in the image?
            - confidence (number 0.0 to 1.0): Confidence of detection.
            - center_x (number 0.0 to 1.0): Normalized X center coordinate (0.0 is left, 1.0 is right).
            - center_y (number 0.0 to 1.0): Normalized Y center coordinate (0.0 is top, 1.0 is bottom).
            - width (number 0.0 to 1.0): Normalized width of the object box.
            - height (number 0.0 to 1.0): Normalized height of the object box.
            - speech_hint (string): A short generic hint like "Move right" or "Move closer". This will only be used as a fallback if the local policy needs it.
            
            Do not invent locations. If target_visible is false, center coordinates, width, and height should be 0.5 and 0.0.
        """.trimIndent()
        
        val response = generativeModelJson.generateContent(
            content {
                image(image)
                text(prompt)
            }
        )
        
        val jsonString = response.text ?: "{}"
        val result = try { json.decodeFromString<FindResult>(jsonString) } catch(e: Exception) { null }
        
        if (result == null) {
             return VisionDecision("I encountered an error trying to find that.", taskComplete = true, debugData = jsonString)
        }
        
        // Pass JSON directly into debugData so that MainViewModel can parse and run the deterministic policy
        // We will just return the serialized FindResult in debugData. We don't make the speech decision here.
        // Actually, to make it clean, we can make VisionDecision more structured or we can just parse the json string again in the ViewModel.
        // The user says: "Do NOT ask Gemini to directly decide arbitrary movement instructions... The Android application determines the actual action using deterministic local logic."
        
        return VisionDecision(
            speech = "", // Decision made in ViewModel
            taskComplete = false, // Handled in ViewModel
            debugData = jsonString
        )
    }
}
