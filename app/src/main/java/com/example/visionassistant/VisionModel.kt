package com.example.visionassistant

import android.graphics.Bitmap

enum class TaskMode {
    GENERAL, READ, ASK, FIND
}

data class TaskState(
    val mode: TaskMode = TaskMode.GENERAL,
    val askQuestion: String = "",
    val findTarget: String = "",
    val retryCount: Int = 0
)

enum class FastPolicyAction {
    NONE, LEFT, RIGHT, MOVE_CLOSER, MOVE_AWAY, STOP, REACQUIRE, WAIT, TASK_COMPLETE
}

data class VisionDecision(
    val speech: String,
    val action: FastPolicyAction = FastPolicyAction.NONE,
    val taskComplete: Boolean = true,
    val needsRepositioning: Boolean = false,
    val debugData: String = ""
)

interface VisionModel {
    suspend fun analyze(
        image: Bitmap,
        state: TaskState
    ): VisionDecision
}
