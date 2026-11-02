# 👁️ VisionAssistant

> **A Closed-Loop Real-Time Visual Guidance Agent for Blind and Low-Vision Users**

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin%20%7C%20C%2B%2B20-blue.svg)](https://kotlinlang.org)
[![On-Device VLM](https://img.shields.io/badge/Edge%20AI-llama.cpp%20%2B%20LFM2--VL-orange.svg)](https://github.com/ggml-org/llama.cpp)
[![Architecture](https://img.shields.io/badge/Architecture-Jev--Style%20System%20One-purple.svg)](#)
[![Cloud VLM](https://img.shields.io/badge/Cloud%20AI-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

---

## 💡 The Problem & Core Thesis

Most existing visual assistants for visually impaired users are **passive**: they take a snapshot, send it to the cloud, wait several seconds, and dump a verbose description of the scene (*"I see a room with a table, chairs, and a cup"*).

For a blind person trying to locate their keys or read a medication label, a passive 4-second delay is unusable.

**VisionAssistant flips this paradigm into an active, closed-loop visual guidance agent.** 

Instead of treating real-time guidance like an autoregressive chatbot, VisionAssistant adopts a **Jev-style "System One" decision model** approach:
- **System One (Reactive Perception-Action Loop):** For time-critical spatial tracking, the model operates non-autoregressively—evaluating visual states directly against a schema-constrained decision space (`FastPolicyAction`) via **candidate scoring** rather than streaming free-form prose.
- **System Two (Deliberative Multimodal Reasoning):** Delegates complex, open-ended visual queries to cloud frontier models when deep contextual understanding is actually required.

---

## 🚀 Key Modes & Capabilities

| Mode | Engine | Description |
|---|---|---|
| 🎯 **FIND** | **On-Device VLM** (Jev-Style System One) | **Closed-loop spatial tracking.** Directs the user in real time (*"Turn slightly left"*, *"Move closer"*, *"Target reached"*) using non-autoregressive action selection with zero cloud dependency. |
| 📖 **READ** | **Cloud VLM** (Gemini) | **Smart OCR & Document Assistance.** Detects text, instructs the user on framing (*"Tilt down slightly"*), and speaks summaries or verbatim content. |
| ❓ **ASK** | **Cloud VLM** (Gemini) | **Visual Question Answering.** Open-ended conversational reasoning (*"Does this tie match my suit?"*, *"What temperature is the dial set to?"*). |
| 🌐 **GENERAL** | **Cloud VLM** (Gemini) | **Instant Situational Awareness.** Concise, prioritized scene understanding focused on navigation hazards and immediate surroundings. |

---

## 🏗️ Architecture: Hybrid Edge + Cloud

VisionAssistant pairs a **Jev-style System One edge engine** for fast, high-frequency closed loops with **System Two cloud multimodal reasoning** for complex tasks.

```
                    ┌──────────────────────────────────────────────┐
                    │               User Voice / UI                │
                    └──────────────────────┬───────────────────────┘
                                           │
                                    [MVI Controller]
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      [FIND Mode: Jev-Style System One]                [ASK / READ / GENERAL]
                     │                                           │
             CameraX Analyzer                           CameraX Capture
                     │                                           │
          In-Memory Persistent JNI                        Gemini Flash API
                     │                                           │
        llama.cpp + LFM2-VL-450M GGUF                   Structured JSON Decision
         (Non-Autoregressive Scoring)                            │
           (~1.1s warm on-device CPU)                            │
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
                            [Speech System (TTS) & Earcons]
```

### 🧠 Engineering Highlights: Jev-Style Decision Engine & Persistent JNI
1. **Jev-Style Non-Autoregressive Decision Evaluation:**
   Instead of generating open-ended text tokens that invite latency and format hallucinations, our high-frequency loop uses a **schema-constrained decision head**. The model evaluates candidate logits directly into a typed `FastPolicyAction` (`LEFT`, `RIGHT`, `MOVE_CLOSER`, `STOP`, `REACQUIRE`), dropping decode time to **0–1 ms**.

2. **Persistent In-Process JNI (Eliminating Process Overhead):**
   Early prototypes spawned a subprocess per frame, causing **~3.9s per-frame latency** due to constantly reloading weights from disk. We re-engineered the engine into a persistent native JNI runtime:
   - **Dynamic Backend Loading:** Dynamically binds ARMv8 NEON CPU routines (`libggml-cpu-android_armv8.so`) via `ggml_backend_load_all_from_path`.
   - **Zero Disk Re-read:** Keeps model weights and multimodal projection (`mmproj`) resident in RAM across iterations.
   - **Latency Cut by ~70%:** Reduced iteration latency from **~3900ms down to ~1160ms (P50)** on CPU.

---

## 📊 On-Device Performance Benchmark (FIND Mode)

Tested on physical ARM64 Android device (Snapdragon, 4 CPU threads):

| Metric | First Frame (Cold) | Warm (P50) | Warm (P95) |
|---|---|---|---|
| **Vision Encoder** | 1,628 ms | **672 ms** | 679 ms |
| **Prompt Processing** | 870 ms | **488 ms** | 529 ms |
| **Total Inference** | 2,504 ms | **~1,169 ms** | **~1,215 ms** |

---

## ♿ Accessibility-First Design

- **TalkBack & Screen Reader First:** Every UI surface has high-contrast elements, semantic labels, and zero dependency on visual touch-targets.
- **Voice & Auditory Feedback:** Hands-free voice recognition loop combined with Text-To-Speech (TTS) and auditory earcons to mask processing intervals.
- **Privacy-Preserving Camera:** Video frames analyzed in FIND mode are processed directly in device RAM and never leave the device.

---

## 🛠️ Tech Stack

- **Platform:** Android 14+ (SDK 34), Min SDK 26
- **UI & Architecture:** Jetpack Compose, Material 3, MVI Unidirectional Data Flow, Kotlin Coroutines & Flow
- **Hardware Abstraction:** CameraX (`ImageAnalysis` + `ImageCapture`), Android `SpeechRecognizer`, `TextToSpeech`
- **Edge AI:** C++20, Android NDK (CMake), `llama.cpp`, `LFM2-VL-450M` (Q4_0 + Q8_0 mmproj)
- **Cloud AI:** Google Gemini 1.5 / 2.0 Flash API

---

## ⚡ Getting Started

### 1. Prerequisites
- Android Studio Ladybug / Meerkat or later
- Android NDK (r26+ or r27+) & CMake 3.22.1+
- A Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/coolbros/VisionAssistant-.git
   cd VisionAssistant-
   ```

2. Add your Gemini API key in `local.properties`:
   ```properties
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

3. Ensure model weights are present in `app/src/main/assets/`:
   - `LFM2-VL-450M-Q4_0.gguf`
   - `mmproj-LFM2-VL-450M-Q8_0.gguf`

4. Build and install:
   ```bash
   ./gradlew assembleDebug
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

---

## 🔮 What's Next

- [ ] **NPU / Hexagon DSP Acceleration:** Offload vision encoder to Qualcomm HTP / Adreno GPU via QNN/OpenCL for sub-400ms closed-loop feedback.
- [ ] **GUIDE Mode:** Multi-step guided interactions for complex appliances (microwaves, washing machines, elevator buttons).
- [ ] **Offline Spatial Memory:** Local object persistence using spatial anchors to remember where personal items were last seen.
